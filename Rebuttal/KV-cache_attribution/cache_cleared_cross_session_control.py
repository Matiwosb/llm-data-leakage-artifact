"""
cache_cleared_cross_session_control.py
─────────────────────────────────────
Cache-Cleared Cross-Session Control

Establishes the no-KV-cache baseline. Any leakage here signals weight
memorization or probe design artefacts, NOT KV persistence.

Flow
----
  Session 0   : Inject all STORE_PROMPTS into context.
                save_state() is intentionally NOT called.
  Cache clear : KV files, CUDA, OS page cache, __pycache__ all wiped.
  Sessions 1–5: Each loads a fresh model, runs all RETRIEVE_PROMPTS
                as one-shot probes (no conversation history), scores results.

Usage
-----
    python cache_cleared_cross_session_control.py

Optionally export SUDO_PASSWORD env var for OS page-cache drop:
    SUDO_PASSWORD=mypass python cache_cleared_cross_session_control.py
"""

import gc
import getpass
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

import kv_leakage.config as config
from kv_leakage.drive_upload import upload_results
from kv_leakage.prompts import RETRIEVE_PROMPTS, STORE_PROMPTS, evaluate_leak, get_prompt_metadata
from kv_leakage.utils import (
    build_result_row,
    clear_all_caches,
    generate_one_shot,
    generate_with_history,
    load_model,
    reset_runtime_state,
    resolve_models,
    verify_clean_state,
)


# ── Session 0 — store (no save_state) ────────────────────────────────────────

def run_session_0(chat_models: list) -> pd.DataFrame:
    results = []

    for m in chat_models:
        name, filename, model_path = m["name"], m["filename"], m["path"]
        print("\n" + "=" * 90)
        print(f"SESSION 0 — STORE (control, no save_state): {name}")
        print("=" * 90)

        llm, history = None, []
        try:
            t0 = time.time()
            llm, device_used = load_model(model_path, "Session 0")
            load_ms = round((time.time() - t0) * 1000, 1)

            for key, prompt in STORE_PROMPTS.items():
                print(f"  > Storing [{key}]")
                gen, _ = generate_with_history(llm, history, prompt)
                results.append(build_result_row(
                    session_id=0, phase="store",
                    model_name=name, filename=filename, model_path=model_path,
                    device_used=device_used, key=key, prompt=prompt, generation=gen,
                    cache_state="no_kv_save", kv_state_restored=False,
                    load_time_ms=load_ms,
                ))

            print("\n  > NOTE: save_state() intentionally NOT called — this is the control")

        except Exception as e:
            print(f"[ERROR] Session 0 — {name}: {e}")
            traceback.print_exc()
        finally:
            reset_runtime_state(llm)
            llm = None

    df = pd.DataFrame(results)
    out = config.CLEARED_DIR / "session0_results.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] Session 0 saved → {out}  ({len(df)} rows)")
    return df


# ── Sessions 1–N — probe with cleared cache ───────────────────────────────────

def run_probe_sessions(chat_models: list) -> list[dict]:
    all_results = []

    for session_id in config.PROBE_SESSIONS:
        print("\n" + "#" * 90)
        print(f"SESSION {session_id} — PROBE (cache cleared, no KV state)")
        print("#" * 90)

        try:
            verify_clean_state(session_id)
        except RuntimeError as e:
            print(f"[ABORT] Pre-session verification failed: {e}")
            continue

        session_results = []

        for m in chat_models:
            name, filename, model_path = m["name"], m["filename"], m["path"]
            print("\n" + "=" * 90)
            print(f"  SESSION {session_id}: {name}")
            print("=" * 90)

            llm = None
            try:
                t0 = time.time()
                llm, device_used = load_model(model_path, f"Session {session_id}")
                load_ms = round((time.time() - t0) * 1000, 1)
                print(f"  [INFO] Load time: {load_ms}ms")

                verify_clean_state(session_id, expected_load_time_ms=load_ms)

                for key, prompt in RETRIEVE_PROMPTS.items():
                    gen, _ = generate_one_shot(llm, prompt)
                    category, *_ = get_prompt_metadata(key)
                    leaked, trigger, match_count = evaluate_leak(category, gen)

                    row = build_result_row(
                        session_id=session_id, phase="probe",
                        model_name=name, filename=filename, model_path=model_path,
                        device_used=device_used, key=key, prompt=prompt, generation=gen,
                        cache_state="fully_cleared", kv_state_restored=False,
                        load_time_ms=load_ms,
                        leak_detected=leaked, leak_trigger=trigger, leak_match_count=match_count,
                    )
                    session_results.append(row)
                    if leaked:
                        print(f"    [LEAK] {key} | {trigger} | {gen[:80]}")

            except RuntimeError as e:
                print(f"[ABORT] Verification failed mid-session: {e}")
                break
            except Exception as e:
                print(f"[ERROR] Session {session_id} — {name}: {e}")
                traceback.print_exc()
            finally:
                reset_runtime_state(llm)
                llm = None

        # Save per-session CSV
        df_session = pd.DataFrame(session_results)
        session_csv = config.CLEARED_DIR / f"session{session_id}_results.csv"
        df_session.to_csv(session_csv, index=False)
        print(f"\n[OK] Session {session_id} → {session_csv}  ({len(df_session)} rows)")

        probe_rows = df_session[df_session["phase"] == "probe"]
        total = len(probe_rows)
        leaks = int(probe_rows["leak_detected"].sum()) if total else 0
        print(f"     Leak rate: {leaks}/{total}  ({round(100*leaks/total,2) if total else 0}%)")

        all_results.extend(session_results)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ── Combine & analyse ─────────────────────────────────────────────────────────

def combine_and_analyse(probe_results: list[dict]) -> pd.DataFrame:
    dfs = []
    s0_path = config.CLEARED_DIR / "session0_results.csv"
    if s0_path.exists():
        dfs.append(pd.read_csv(s0_path))

    if probe_results:
        dfs.append(pd.DataFrame(probe_results))

    if not dfs:
        print("[WARN] Nothing to combine.")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    out_path = config.CLEARED_DIR / "exp2_combined_results.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nCombined → {out_path}  ({len(df_all)} rows)")

    probe_df = df_all[(df_all["phase"] == "probe") & (df_all["session_id"] > 0)].copy()
    if probe_df.empty:
        print("[INFO] No probe rows to analyse.")
        return df_all

    total = len(probe_df)
    leaks = int(probe_df["leak_detected"].sum())
    print(f"\nOverall baseline | probes: {total} | leaks: {leaks} | "
          f"rate: {round(100*leaks/total,2) if total else 0}%")
    print()

    def agg(grp):
        return (grp["leak_detected"]
                .agg(total="count", leaks="sum")
                .assign(leak_rate_pct=lambda x: (100 * x["leaks"] / x["total"]).round(2)))

    for label, grouper in [
        ("By session", "session_id"),
        ("By technique", "technique"),
        ("By category", "category"),
        ("By sensitivity", "sensitivity_level"),
        ("By leak trigger", "leak_trigger"),
    ]:
        print(f"── {label} ──")
        print(agg(probe_df.groupby(grouper)).to_string())
        print()

    # print("── Interpretation guide ──")
    # print("  Zero recall          → KV cache is the sole leakage vector ✓")
    # print("  Consistent recall    → Weight memorization (not KV cache)")
    # print("  Inconsistent recall  → Model hallucination / probe bleed")
    # print("  Fast load times      → OS page cache not fully cleared")

    return df_all


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    chat_models, missing = resolve_models(config.MODEL_DIR, config.TARGET_FILENAMES)

    print("=" * 60)
    print("EXPERIMENT 2 — CACHE-CLEARED CROSS-SESSION CONTROL")
    print(f"Models  : {[m['filename'] for m in chat_models]}")
    print(f"Sessions: 0 (store) + {config.PROBE_SESSIONS} (probe)")
    if missing:
        print(f"Missing : {missing}")
    print("=" * 60)

    if not chat_models:
        sys.exit("[ERROR] No models found. Check config.MODEL_DIR and TARGET_FILENAMES.")

    # ── Sudo password for OS page cache (optional) ────────────────────────────
    sudo_password = os.environ.get("SUDO_PASSWORD")
    if sudo_password is None:
        try:
            test = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5)
            if test.returncode == 0:
                print("[OK] Passwordless sudo available")
            else:
                ans = input("Enter sudo password for OS page cache drop (Enter to skip): ").strip()
                sudo_password = ans or None
        except Exception:
            pass

    # ── Session 0 — store ────────────────────────────────────────────────────
    run_session_0(chat_models)

    # ── Clear all caches ──────────────────────────────────────────────────────
    print("\n")
    clear_all_caches(
        log_path=config.CLEARED_DIR / "cache_clear_log.json",
        sudo_password=sudo_password,
    )

    remaining = list(config.KV_STATE_DIR.rglob("*"))
    print("KV_STATE_DIR after clear:", "[OK] Empty" if not remaining
          else f"[WARN] {len(remaining)} files remain")

    # ── Sessions 1–N — probe ─────────────────────────────────────────────────
    probe_results = run_probe_sessions(chat_models)

    # ── Combine & analyse ─────────────────────────────────────────────────────
    print("\n" + "#" * 90)
    print("ALL PROBE SESSIONS COMPLETE — combining results ...")
    print("#" * 90)
    combine_and_analyse(probe_results)

    # ── Upload results to Google Drive ────────────────────────────────────────
    # Set GDRIVE_FOLDER_ID env var to upload into a specific Drive folder,
    # or leave unset to upload to the root of My Drive.
    # Upload is skipped silently if credentials.json is not present.
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    result_files = sorted(config.CLEARED_DIR.glob("session*_results.csv")) + \
                   [config.CLEARED_DIR / "exp2_combined_results.csv",
                    config.CLEARED_DIR / "cache_clear_log.json"]
    try:
        upload_results(result_files, folder_id=folder_id)
    except FileNotFoundError as e:
        print(f"[DRIVE] Skipping upload — {e}")


if __name__ == "__main__":
    main()
