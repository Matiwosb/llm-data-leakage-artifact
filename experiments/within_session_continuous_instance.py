"""
within_session_continuous_instance.py
──────────────────────────────────
Within-session KV cache leakage test using a CONTINUOUS Llama instance.

Approach:
  - A single Llama instance is created per model and stays alive for the
    entire experiment.
  - Session 0 (Store): PII is fed into the model. The KV cache is populated
    naturally in RAM — no save_state() is called.
  - Sessions 1-N (Probe): Probe prompts are issued on the SAME instance.
    The model attends over the still-active KV cache from the store phase.

Threat model:
  Models a long-running LLM server where a prior user's context is never
  explicitly cleared before the next user's request arrives.

Contrast with experiment_save_load.py:
  That file serializes the KV cache to disk and loads it into a fresh
  instance — a stronger isolation boundary. The difference in leak rates
  between the two files is itself an experimental finding.
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Package root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force line-buffered stdout so every print() reaches the terminal immediately,
# even before llama-cpp-python redirects the stdout file descriptor during
# model loading (verbose=False uses os.dup2 to send fd-1 to /dev/null).
sys.stdout.reconfigure(line_buffering=True)
# print("[EXP] Script starting ...", flush=True)

import pandas as pd
from llama_cpp import Llama

import kv_leakage.config as config
from kv_leakage.drive_upload import upload_results
from kv_leakage.session_logger import SessionLogger
from kv_leakage.prompts import (
    STORE_PROMPTS,
    RETRIEVE_PROMPTS,
    evaluate_leak,
    get_prompt_metadata,
)
from kv_leakage.utils import (
    batch_models,
    build_result_row,
    reset_runtime_state,
    resolve_models,
    sort_models_by_size,
)

# ── RUN_ID persistence (same pattern as Exp 3) ────────────────────────────────
_RUN_ID_FILE = config.CONTINUOUS_DIR / ".run_id"

def _load_run_id() -> str | None:
    return _RUN_ID_FILE.read_text().strip() if _RUN_ID_FILE.exists() else None

def _save_run_id(run_id: str) -> None:
    config.CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_ID_FILE.write_text(run_id)


# ── Countdown helper ──────────────────────────────────────────────────────────

def _wait_with_countdown(seconds: int, next_session: int) -> None:
    resume_at = datetime.fromtimestamp(time.time() + seconds)
    print(f"\n[WAIT] {seconds // 3600}h {(seconds % 3600) // 60}m until Session {next_session}")
    print(f"       Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"       Resume  : {resume_at.strftime('%Y-%m-%d %H:%M:%S')}")
    remaining = seconds
    while remaining > 0:
        chunk = min(60, remaining)
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            h, r = divmod(remaining, 3600)
            m, s = divmod(r, 60)
            parts = ([f"{h}h"] if h else []) + ([f"{m}m"] if m else []) + ([f"{s}s"] if s else [])
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{', '.join(parts)} remaining until Session {next_session}")
    print(f"  [READY] Starting Session {next_session} — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ── Low-level generation ──────────────────────────────────────────────────────

def _generate(llm: Llama, prompt: str, stop: list[str] | None = None) -> str:
    """Single completion call. Uses config.GEN_CONFIG for all parameters."""
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=config.GEN_CONFIG["max_tokens"],
        temperature=config.GEN_CONFIG["temperature"],
        # top_k=config.GEN_CONFIG["top_k"],
        # top_p=config.GEN_CONFIG["top_p"],
        # repeat_penalty=config.GEN_CONFIG["repeat_penalty"],
        stop=stop or ["\nUser:", "\n\nUser:"],
    )
    return out["choices"][0]["text"].strip()


# ── Session 0 — store ALL PII into the live instance ─────────────────────────

def run_session_0(
    chat_models: list, run_id: str
) -> tuple[dict[str, Llama], dict[str, str], pd.DataFrame]:
    """
    Load each model, feed all STORE_PROMPTS in sequence, and keep the
    Llama instances alive (returned in llm_registry for probe sessions).

    Returns (llm_registry, history_registry, df_store) where:
      llm_registry     : {model_name: Llama}  — live instances, cache populated
      history_registry : {model_name: str}    — full PII conversation text;
                         used as the prompt prefix in every probe so the model
                         can attend over all stored PII. llama.cpp only reuses
                         KV cache tokens whose prefix matches the new prompt,
                         so the history must be included in each probe call.
      df_store         : DataFrame of store-phase rows
    """
    llm_registry: dict[str, Llama] = {}
    history_registry: dict[str, str] = {}
    results = []

    for m in chat_models:
        name, filename, model_path = m["name"], m["filename"], m["path"]
        print("\n" + "=" * 80, flush=True)
        print(f"SESSION 0 — STORE (continuous instance, no save_state): {name}", flush=True)
        print("=" * 80, flush=True)

        logger = SessionLogger(experiment="exp4_continuous", session_id=0, model_name=name)
        try:
            logger.snapshot_memory("pre_load")
            t0 = time.time()
            print(f"  [LOADING] {filename} — this may take 10-60s ...", flush=True)
            sys.stdout.flush()  # ensure all prints reach terminal before fd redirect
            llm = Llama(
                model_path=model_path,
                n_ctx=config.N_CTX,
                n_gpu_layers=config.N_GPU_LAYERS,
                verbose=config.VERBOSE,
            )
            device_used = "gpu" if config.N_GPU_LAYERS != 0 else "cpu"
            load_ms = round((time.time() - t0) * 1000, 1)
            logger.set_load_time_ms(load_ms)
            logger.record_model_arch(llm)
            logger.snapshot_memory("post_load")
            print(f"  [LOAD] {filename}  ({device_used})  load_time={load_ms}ms", flush=True)

            # Feed ALL store prompts one by one, growing the conversation history.
            # history_text accumulates every User/Assistant turn so that:
            #   (a) each store call attends over all prior PII turns
            #   (b) probe calls can send the full history as prefix, allowing
            #       llama.cpp to reuse already-computed KV cache entries for
            #       those tokens rather than recomputing from scratch
            history_text = ""
            for key, prompt in STORE_PROMPTS.items():
                full_prompt = history_text + f"User: {prompt}\nAssistant:"
                gen = _generate(llm, full_prompt)
                history_text += f"User: {prompt}\nAssistant: {gen}\n\n"
                logger.increment("store_turns")
                print(f"  [store] {key} → {gen[:70]}", flush=True)

                results.append(build_result_row(
                    session_id=0, phase="store",
                    model_name=name, filename=filename, model_path=model_path,
                    device_used=device_used, key=key, prompt=prompt, generation=gen,
                    cache_state="continuous_in_ram", kv_state_restored=False,
                    load_time_ms=load_ms,
                ))

            logger.snapshot_memory("post_store")
            print(f"\n  [OK] {len(STORE_PROMPTS)} PII items stored — "
                  f"instance kept alive, KV cache active in RAM", flush=True)
            print(f"  [INFO] history_text: {len(history_text)} chars, "
                  f"~{len(history_text)//4} tokens", flush=True)
            llm_registry[name]     = llm
            history_registry[name] = history_text

        except Exception as e:
            print(f"[ERROR] Session 0 — {name}: {e}")
            traceback.print_exc()
        finally:
            logger.print_summary()
            logger.save()

    df = pd.DataFrame(results)
    # out = config.CONTINUOUS_DIR / f"session0_results_{run_id}.csv"
    # df.to_csv(out, index=False)
    # print(f"\n[OK] Session 0 saved → {out}  ({len(df)} rows)")
    
    # Include model name in filename — run_session_0 is called once per model
    # (batch_size=1), so without the model name each call overwrites the last.
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_"
                        for m in chat_models for c in [m["name"]])[: 60]
    if len(chat_models) == 1:
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_"
                            for c in chat_models[0]["name"])[:60]
    else:
        safe_name = "multi"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.CONTINUOUS_DIR / f"session0_results_{safe_name}_{run_id}_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] Session 0 saved → {out}  ({len(df)} rows)")
    
    return llm_registry, history_registry, df


# ── Probe session — same live instance, KV cache still in RAM ────────────────

def run_probe_session(
    session_id: int,
    chat_models: list,
    llm_registry: dict[str, Llama],
    history_registry: dict[str, str],
    run_id: str,
) -> pd.DataFrame:
    """
    Run all RETRIEVE_PROMPTS on the live instances from Session 0.

    Each probe is sent as:
        <full PII history from Session 0>  +  "User: <probe>\nAssistant:"

    This matches the token prefix already in the KV cache, so llama.cpp
    reuses the cached K/V values for the history tokens and only computes
    new tokens for the probe. The model can then attend over all stored PII.
    """
    results = []

    for m in chat_models:
        name, filename, model_path = m["name"], m["filename"], m["path"]
        print("\n" + "=" * 80, flush=True)
        print(f"SESSION {session_id} — PROBE (continuous instance, KV in RAM): {name}", flush=True)
        print("=" * 80, flush=True)

        llm = llm_registry.get(name)
        if llm is None:
            print(f"[SKIP] No live instance for {name} — Session 0 may have failed.")
            continue

        device_used = "gpu" if config.N_GPU_LAYERS != 0 else "cpu"
        logger = SessionLogger(experiment="exp4_continuous", session_id=session_id, model_name=name)
        logger.record_model_arch(llm)

        # Retrieve this model's stored history — prepend it to every probe
        # so the model sees the full PII context and can recall it
        history_text = history_registry.get(name, "")
        logger.snapshot_memory("session_start")

        for key, prompt in RETRIEVE_PROMPTS.items():
            try:
                # Prepend history so the prompt prefix matches the KV cache
                probe_payload = history_text + f"User: {prompt}\nAssistant:"
                gen = _generate(llm, probe_payload)
                category, *_ = get_prompt_metadata(key)
                leaked, trigger, match_count = evaluate_leak(category, gen)
                logger.increment("probe_turns")
                if leaked:
                    logger.increment("leaks")

                results.append(build_result_row(
                    session_id=session_id, phase="probe",
                    model_name=name, filename=filename, model_path=model_path,
                    device_used=device_used, key=key, prompt=prompt, generation=gen,
                    cache_state="continuous_in_ram", kv_state_restored=False,
                    leak_detected=leaked, leak_trigger=trigger, leak_match_count=match_count,
                ))
                if leaked:
                    print(f"  [LEAK] {key} | {trigger} | {gen[:80]}", flush=True)

            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
                traceback.print_exc()

        logger.snapshot_memory("session_end")
        logger.print_summary()
        logger.save()

    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.CONTINUOUS_DIR / f"session{session_id}_results_{ts}.csv"
    df.to_csv(out, index=False)

    probe_rows = df[df["phase"] == "probe"]
    total = len(probe_rows)
    leaks = int(probe_rows["leak_detected"].sum()) if total else 0
    print(f"\n[OK] Session {session_id} → {out}  ({len(df)} rows)")
    print(f"     Leaks: {leaks}/{total}  ({round(100*leaks/total, 2) if total else 0}%)")
    return df


# ── Combine & analyse ─────────────────────────────────────────────────────────

def combine_and_analyse(run_id: str) -> pd.DataFrame:
    all_dfs = []
    for csv_file in sorted(config.CONTINUOUS_DIR.glob("session*_results*.csv")):
        try:
            tmp = pd.read_csv(csv_file)
            if "run_id" in tmp.columns:
                tmp = tmp[tmp["run_id"] == run_id]
            if len(tmp):
                all_dfs.append(tmp)
                print(f"  Loaded: {csv_file.name}  ({len(tmp)} rows)")
        except Exception as e:
            print(f"  [SKIP] {csv_file.name}: {e}")

    if not all_dfs:
        print("[WARN] No session CSVs found.")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    out_path = config.CONTINUOUS_DIR / f"combined_all_sessions_{run_id}.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nCombined → {out_path}  ({len(df_all)} rows)")

    probe_df = df_all[df_all["phase"] == "probe"].copy()
    if probe_df.empty:
        print("[INFO] No probe rows to analyse yet.")
        return df_all

    total = len(probe_df)
    leaks = int(probe_df["leak_detected"].sum())
    print(f"\nOverall | probes: {total} | leaks: {leaks} | "
          f"rate: {round(100*leaks/total, 2) if total else 0}%\n")

    def agg(grp):
        return (grp["leak_detected"]
                .agg(total="count", leaks="sum")
                .assign(leak_rate_pct=lambda x: (100 * x["leaks"] / x["total"]).round(2)))

    for label, grouper in [
        ("Leak rate by session (cache degradation over time)", "session_id"),
        ("By technique", "technique"),
        ("By category", "category"),
        ("By sensitivity", "sensitivity_level"),
        ("By leak trigger clause", "leak_trigger"),
    ]:
        print(f"── {label} ──")
        print(agg(probe_df.groupby(grouper)).to_string())
        print()

    # print("── Interpretation ──")
    # print("  High leak rate across all sessions → in-RAM KV cache persists strongly")
    # print("  Declining leak rate over sessions  → cache degrades under memory pressure")
    # print("  Compare Exp 4 rate vs Exp 3 rate   → cost of server restart on leakage")
    return df_all


# ── Full automated pipeline ───────────────────────────────────────────────────

def run_full_pipeline(chat_models: list, run_id: str) -> None:
    pipeline_start = datetime.now()

    chat_models  = sort_models_by_size(chat_models)
    total_models = len(chat_models)

    # Exp 4 MUST process one model at a time — each model stays loaded in
    # VRAM for all probe sessions before the next model loads. Loading
    # multiple models simultaneously would exhaust VRAM because llm_registry
    # keeps them all alive. MODEL_BATCH_SIZE is intentionally ignored here.
    batches = batch_models(chat_models, 1)

    print("=" * 60, flush=True)
    print("CONTINUOUS INSTANCE LEAKAGE (full pipeline)", flush=True)
    print(f"RUN_ID   : {run_id}", flush=True)
    print(f"Models   : {total_models} total (processed one at a time)", flush=True)
    print(f"Sessions : 0 (store) + {config.CONTINUOUS_SESSIONS} (probe)", flush=True)
    print(f"Wait     : {config.CONTINUOUS_WAIT_SECONDS}s between probe sessions (sessions 2+)", flush=True)
    print(f"NOTE     : Each model stays loaded for all sessions then unloads", flush=True)
    print(f"NOTE     : MODEL_BATCH_SIZE ignored — VRAM holds only 1 live instance", flush=True)
    print("=" * 60, flush=True)

    for model_num, batch in enumerate(batches, 1):
        m = batch[0]
        print(f"\n{'═'*70}", flush=True)
        print(f"MODEL {model_num}/{total_models}: {m['filename']}", flush=True)
        print(f"{'═'*70}", flush=True)

        llm_registry, history_registry, _ = run_session_0(batch, run_id)

        for session_id in config.CONTINUOUS_SESSIONS:
            # Session 1 probes immediately after store — no wait.
            # All subsequent sessions wait CONTINUOUS_WAIT_SECONDS to test
            # whether the in-RAM KV cache degrades under memory pressure over time.
            if session_id != config.CONTINUOUS_SESSIONS[0]:
                _wait_with_countdown(config.CONTINUOUS_WAIT_SECONDS, session_id)

            print(f"\n{'#'*60}", flush=True)
            print(f"  MODEL {model_num}/{total_models} | SESSION {session_id}", flush=True)
            print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"  Elapsed : {str(datetime.now() - pipeline_start).split('.')[0]}", flush=True)
            print(f"{'#'*60}", flush=True)

            run_probe_session(session_id, batch, llm_registry, history_registry, run_id)

        # Unload this model before loading the next — frees VRAM immediately
        print(f"\n[INFO] Model {model_num}/{total_models} complete — unloading ...", flush=True)
        for name, llm in llm_registry.items():
            try:
                reset_runtime_state(llm)
                print(f"  [OK] Unloaded {name}", flush=True)
            except Exception as e:
                print(f"  [WARN] Could not cleanly unload {name}: {e}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("ALL MODELS AND SESSIONS COMPLETE — combining results ...", flush=True)
    print(f"{'='*60}", flush=True)
    combine_and_analyse(run_id)

    duration = str(datetime.now() - pipeline_start).split(".")[0]
    print(f"\nTotal duration: {duration}", flush=True)

    # ── Upload results to Google Drive ────────────────────────────────────────
    # folder_id    = os.environ.get("GDRIVE_FOLDER_ID")
    folder_id    = getattr(config, "GDRIVE_FOLDER_ID", os.environ.get("GDRIVE_FOLDER_ID"))
    result_files = sorted(config.CONTINUOUS_DIR.glob("session*_results*.csv")) + \
                   sorted(config.CONTINUOUS_DIR.glob(f"combined_all_sessions_{run_id}.csv"))
    seen = set()
    unique_files = [f for f in result_files if not (f in seen or seen.add(f))]
    try:
        upload_results(unique_files, folder_id=folder_id)
    except FileNotFoundError as e:
        print(f"[DRIVE] Skipping upload — {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Continuous Instance KV Cache Leakage"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--session", type=int, metavar="N",
        help="Run a single session (0 = store, 1-5 = probe). "
             "Probe sessions require the model to still be in memory — "
             "use the full pipeline mode for automated runs.",
    )
    group.add_argument(
        "--combine", action="store_true",
        help="Merge all session CSVs and print analysis (no inference).",
    )
    args = parser.parse_args()

    chat_models, missing = resolve_models(config.MODEL_DIR, config.TARGET_FILENAMES)
    if missing:
        print(f"[WARN] Missing models (skipped): {missing}")
    if not chat_models:
        sys.exit("[ERROR] No models found. Check config.MODEL_DIR and TARGET_FILENAMES.")

    # ── GPU memory guard ──────────────────────────────────────────────────────
    # Experiment 4 keeps models alive for the whole pipeline. If a previous run
    # is still holding VRAM, the first model load will fail with a cryptic
    # "Failed to load model from file" error. Warn the user up front.
    if config.N_GPU_LAYERS != 0:
        from kv_leakage.utils import get_nvidia_smi_memory
        for g in get_nvidia_smi_memory():
            if "error" in g:
                break
            used_pct = round(g["used_mb"] / g["total_mb"] * 100, 1)
            if used_pct > 20:
                print(f"\n[WARN] GPU {g['gpu_index']} is {used_pct}% full "
                      f"({g['used_mb']:.0f}/{g['total_mb']:.0f} MB) before any model loads.")
                print("       A previous experiment may still be holding VRAM.")
                print("       Run: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
                print("       Then: kill <PID>  to release GPU memory before continuing.\n")
                ans = input("Continue anyway? [y/N]: ").strip().lower()
                if ans != "y":
                    sys.exit("[ABORT] Free GPU memory and rerun.")

    config.CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.combine:
        run_id = _load_run_id()
        if not run_id:
            sys.exit("[ERROR] No .run_id file found. Run Session 0 first.")
        print(f"[INFO] Using RUN_ID: {run_id}")
        combine_and_analyse(run_id)
        return

    if args.session is not None:
        if args.session == 0:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            _save_run_id(run_id)
            print(f"[INFO] New RUN_ID: {run_id}")
            print("[WARN] --session 0 stores PII but the model will be unloaded")
            print("       after this call. For continuous probing, use the")
            print("       full pipeline (no --session flag) instead.")
            llm_registry, history_registry, _ = run_session_0(chat_models, run_id)
            for llm in llm_registry.values():
                reset_runtime_state(llm)
        else:
            print("[WARN] --session N for probe sessions requires the model to")
            print("       remain in memory from Session 0. Use the full pipeline.")
            sys.exit(1)
        return

    # Full automated pipeline (recommended)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_run_id(run_id)
    run_full_pipeline(chat_models, run_id)


if __name__ == "__main__":
    main()