"""
utils.py — Shared helpers for model management, generation, memory monitoring,
           KV state persistence, and cache clearing.
"""

import gc
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil
import torch
from llama_cpp import Llama

from kv_leakage.config import (
    CACHE_DIRS_TO_CLEAR,
    GEN_CONFIG,
    KV_STATE_DIR,
    N_CTX,
    N_GPU_LAYERS,
    PROCESS_MEMORY_THRESHOLD_MB,
)
from kv_leakage.prompts import get_prompt_metadata

# Set to e.g. "llama-2" for chat-formatted models; None = raw completion mode.
CHAT_FORMAT = None

# ── Memory ────────────────────────────────────────────────────────────────────

def get_process_memory_mb() -> float | None:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def get_nvidia_smi_memory() -> list[dict]:
    """Read GPU memory via nvidia-smi (captures llama.cpp CUDA allocations)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        gpus = []
        for i, line in enumerate(result.stdout.strip().splitlines()):
            used, total = [v.strip() for v in line.split(",")]
            gpus.append({"gpu_index": i, "used_mb": float(used), "total_mb": float(total)})
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


# ── Model lifecycle ───────────────────────────────────────────────────────────

def resolve_models(model_dir: Path, target_filenames: set) -> tuple[list, list]:
    found, missing = [], []
    if not target_filenames:
        # Empty set → discover all .gguf files in MODEL_DIR
        for p in sorted(model_dir.glob("*.gguf")):
            found.append({"name": p.stem, "filename": p.name, "path": str(p)})
    else:
        for fn in sorted(target_filenames):
            p = model_dir / fn
            if p.exists():
                found.append({"name": p.stem, "filename": fn, "path": str(p)})
            else:
                missing.append(fn)
    return found, missing

# def _parse_param_size(filename: str) -> float:
#     """
#     Extract the parameter count (in billions) from a GGUF filename.
#     Handles patterns like: 7b, 13b, 7B, 13B, 1.1b, 34b, 70b, 0.5b
#     Returns float('inf') if no match so unknown sizes sort to the end.
 
#     Examples:
#         "nous-hermes-2-mistral-7b-dpo.Q4_0.gguf"  → 7.0
#         "orca-2-13b.Q4_0.gguf"                    → 13.0
#         "tinyllama-1.1b.Q4_0.gguf"                → 1.1
#         "mixtral-8x7b.Q4_0.gguf"                  → 56.0  (8*7)
#         "gpt4all-falcon-newbpe-q4_0.gguf"          → 7.0   (alias)
#         "unknown-model.gguf"                       → inf
#     """
#     import re
#     name = filename.lower()
 
#     # ── Known model aliases ──────────────────────────────────────────────────
#     # Models whose filenames do not contain a standard Nb size token.
#     # Key = substring to match (case-insensitive), Value = parameter count.
#     # Add any models from your folder that parse incorrectly.
#     _ALIASES: dict[str, float] = {
#         # gpt4all bundled models
#         "falcon":           7.0,
#         "gpt4all":          7.0,
#         "mpt":              7.0,
#         # Phi family (Microsoft) — no "b" suffix in standard releases
#         "phi-1":            1.3,
#         "phi-2":            2.7,
#         "phi-3-mini":       3.8,
#         "phi-3-small":      7.0,
#         "phi-3-medium":    14.0,
#         "phi-3.5-mini":     3.8,
#         # Gemma family (Google) — sometimes written without "b" suffix
#         "gemma-2-2b":       2.0,
#         "gemma-2-9b":       9.0,
#         "gemma-2-27b":     27.0,
#     }
#     for alias, size in _ALIASES.items():
#         if alias in name:
#             return size
 
#     # ── Mixture-of-experts pattern: NxMb (e.g. 8x7b → 56B) ──────────────────
#     moe = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", name)
#     if moe:
#         return float(moe.group(1)) * float(moe.group(2))
 
#     # ── Standard pattern: 7b, 13b, 1.1b, 0.5b, 6.7b ─────────────────────────
#     # Requires "b" immediately after the number so version strings like
#     # "hermes-2-pro" or "llava-v1.5" don't trigger a false match.
#     m = re.search(r"(\d+(?:\.\d+)?)b", name)
#     if m:
#         return float(m.group(1))
 
#     # Unknown — sorts to end so large/unknown models run last
#     return float("inf")
 
 
# def sort_models_by_size(models: list) -> list:
#     """
#     Sort models by ascending parameter size so smaller models run first
#     and larger models (e.g. 13B) run last.
 
#     Models with unrecognisable sizes sort to the very end.
 
#     Example ordering for a mixed folder:
#         tinyllama-1.1b   →  1.1B  (batch 1)
#         mistral-7b       →  7.0B  (batch 1 or 2)
#         nous-hermes-7b   →  7.0B  (batch 1 or 2)
#         orca-2-13b       → 13.0B  (last batch)
#         wizardlm-13b     → 13.0B  (last batch)
 
#     Usage:
#         models, _ = resolve_models(config.MODEL_DIR, config.TARGET_FILENAMES)
#         models     = sort_models_by_size(models)
#         batches    = batch_models(models, config.MODEL_BATCH_SIZE)
#     """
#     return sorted(models, key=lambda m: (_parse_param_size(m["filename"]), m["filename"]))
 
    def sort_models_by_size(models: list) -> list:
        """
        Sort models by ascending file size on disk so smaller models run first
        and larger models (e.g. 13B) run last — with no filename parsing or
        hardcoded aliases needed.
    
        Why file size works:
            For Q4_0 quantized models the file size scales linearly with the
            parameter count. A 7B Q4_0 model is always a smaller file than a
            13B Q4_0 model. This holds across model families and naming schemes
            because quantization is applied uniformly — no regex, no aliases,
            no hardcoding required.
    
            Approximate file sizes at Q4_0:
                1B  →  ~0.7 GB
                3B  →  ~2.0 GB
                7B  →  ~4.0 GB
                8B  →  ~5.0 GB
            13B  →  ~7.5 GB
            34B  →  ~20  GB
            70B  →  ~38  GB
    
        Models that cannot be stat-ed (missing path) sort to the end.
    
        Usage:
            models, _ = resolve_models(config.MODEL_DIR, config.TARGET_FILENAMES)
            models     = sort_models_by_size(models)
            batches    = batch_models(models, config.MODEL_BATCH_SIZE)
        """
        def _file_size(m: dict) -> int:
            try:
                return Path(m["path"]).stat().st_size
            except (OSError, KeyError):
                return 10 ** 15   # unreachable path sorts to the very end
    
        return sorted(models, key=lambda m: (_file_size(m), m["filename"]))
    
    
def get_probe_order(
    retrieve_prompts: dict,
    repeat_id: int,
    shuffle: bool = True,
    seed: int | None = 42,
) -> list[tuple[str, str]]:
    """
    Return the probe prompt list for one repeat, optionally shuffled.

    Args:
        retrieve_prompts : The full RETRIEVE_PROMPTS dict.
        repeat_id        : Which repeat this is (0-indexed). Combined with
                           seed so each repeat gets a different but
                           reproducible order across runs.
        shuffle          : If False, always returns prompts in original order.
        seed             : Base random seed. None = truly random every run.

    Returns:
        List of (key, prompt) tuples in the order to run them.
    """
    import random
    items = list(retrieve_prompts.items())
    if shuffle:
        rng = random.Random(None if seed is None else seed + repeat_id)
        rng.shuffle(items)
    return items


def batch_models(models: list, batch_size: int) -> list[list]:
        """
        Split a list of models into consecutive batches of at most batch_size.
    
        Example with 9 models and batch_size=4:
        batch 1: models 1-4
        batch 2: models 5-8
        batch 3: model  9
    
        Usage in an experiment runner:
            all_models, _ = resolve_models(config.MODEL_DIR, config.TARGET_FILENAMES)
            for batch_num, batch in enumerate(batch_models(all_models, config.MODEL_BATCH_SIZE), 1):
                print(f"Batch {batch_num}/{total_batches}: {[m['filename'] for m in batch]}")
                run_experiment(batch, ...)
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        return [models[i : i + batch_size] for i in range(0, len(models), batch_size)]



def sort_models_by_size(models: list) -> list:
    """
    Sort models by ascending file size on disk — no filename parsing needed.
    File size scales linearly with parameter count for same quantization level.
    Smaller models run first; larger models (13B+) run last automatically.
    Models with missing paths sort to the very end.
    """
    def _file_size(m: dict) -> int:
        try:
            return Path(m["path"]).stat().st_size
        except (OSError, KeyError):
            return 10 ** 15

    return sorted(models, key=lambda m: (_file_size(m), m["filename"]))


def batch_models(models: list, batch_size: int) -> list[list]:
    """
    Split a list of models into consecutive batches of at most batch_size.
    e.g. 21 models, batch_size=4 → 5 batches of 4 + 1 batch of 1.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return [models[i : i + batch_size] for i in range(0, len(models), batch_size)]


def load_model(model_path: str, session_label: str = ""):
    # from llama_cpp import Llama
    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            chat_format=CHAT_FORMAT,
            verbose=False,
        )
        device_used = "gpu" if N_GPU_LAYERS != 0 else "cpu"
        print(f"[LOAD] {Path(model_path).name}  ({device_used})  [{session_label}]")
        return llm, device_used
    except Exception as e:
        print(f"[ERROR] Could not load {model_path}: {e}")
        raise


def safe_close(model_obj) -> None:
    try:
        if model_obj is not None and hasattr(model_obj, "close"):
            model_obj.close()
    except Exception as e:
        print(f"[WARN] model close failed: {e}")


def reset_runtime_state(model_obj=None, pause_seconds: int = 0) -> None:
    """Unload model, free GPU memory, optionally wait."""
    safe_close(model_obj)
    try:
        del model_obj
    except Exception:
        pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    # Report GPU state
    for g in get_nvidia_smi_memory():
        if "error" in g:
            print(f"[INFO] nvidia-smi unavailable: {g['error']}")
            continue
        used_pct = round(g["used_mb"] / g["total_mb"] * 100, 1)
        tag = "[OK]" if used_pct <= 15 else "[WARN]"
        print(f"{tag} GPU {g['gpu_index']} after reset: "
              f"{g['used_mb']}MB / {g['total_mb']}MB ({used_pct}%)")
    if pause_seconds:
        time.sleep(pause_seconds)


# ── KV state persistence ──────────────────────────────────────────────────────

def _state_path(model_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    d = KV_STATE_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d / "kv_state.bin"


def save_kv_state(llm, model_name: str) -> str | None:
    path = _state_path(model_name)
    if not hasattr(llm, "save_state"):
        print("[ERROR] save_state() not available — upgrade llama-cpp-python")
        return None
    try:
        state = llm.save_state()
        if state is None:
            print("[ERROR] save_state() returned None")
            return None
        with open(path, "wb") as f:
            pickle.dump(state, f)
        size_mb = path.stat().st_size / 1e6
        print(f"[OK] KV state saved → {path}  ({size_mb:.1f} MB)")
        return str(path)
    except Exception as e:
        print(f"[ERROR] save_state() failed: {e}")
        traceback.print_exc()
        return None


def load_kv_state(llm, model_name: str) -> bool:
    path = _state_path(model_name)
    print(f"[INFO] KV state path: {path}  [exists={path.exists()}]")
    if not path.exists():
        print("[WARN] kv_state.bin not found — running without restored context")
        return False
    if not hasattr(llm, "load_state"):
        print("[ERROR] load_state() not available — upgrade llama-cpp-python")
        return False
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        llm.load_state(state)
        size_mb = path.stat().st_size / 1e6
        print(f"[OK] KV state restored ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] load_state() failed: {e}")
        traceback.print_exc()
        return False


# ── Cache clearing ────────────────────────────────────────────────────────────

def clear_all_caches(log_path: Path | None = None, sudo_password: str | None = None) -> dict:
    log = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actions": []}

    def record(msg, status="ok"):
        print(f"[{status.upper()}] {msg}")
        log["actions"].append({"msg": msg, "status": status})

    print("=" * 60)
    print("CLEARING ALL CACHES")
    print("=" * 60)

    gc.collect()
    record("gc.collect() done")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            record("torch CUDA cleared")
    except Exception as e:
        record(f"torch CUDA failed: {e}", "warn")

    for cdir in CACHE_DIRS_TO_CLEAR:
        cdir = Path(cdir)
        if cdir.exists():
            try:
                shutil.rmtree(cdir)
                cdir.mkdir(parents=True, exist_ok=True)
                record(f"Cleared: {cdir}")
            except Exception as e:
                record(f"Failed to clear {cdir}: {e}", "warn")
        else:
            record(f"Not found (already clean): {cdir}", "skip")

    # OS page cache
    try:
        subprocess.run(["sync"], check=True)
        cmd = ["sudo", "-S" if sudo_password else "-n", "tee", "/proc/sys/vm/drop_caches"]
        inp = ((sudo_password + "\n3") if sudo_password else "3")
        result = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            record("OS page cache dropped")
        else:
            record(f"OS page cache drop failed: {result.stderr.strip()}", "warn")
    except Exception as e:
        record(f"OS page cache skipped: {e}", "warn")

    # __pycache__
    for pycache in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            record(f"Removed {pycache}")
        except Exception as e:
            record(f"Could not remove {pycache}: {e}", "warn")

    kv_files = list(KV_STATE_DIR.rglob("*.bin"))
    if kv_files:
        record(f"WARNING: {len(kv_files)} .bin file(s) still in KV_STATE_DIR!", "warn")
    else:
        record("KV_STATE_DIR is clean")

    print("=" * 60)
    print("CACHE CLEAR COMPLETE")
    print("=" * 60)

    if log_path:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    return log


# ── Pre-session verification ──────────────────────────────────────────────────

def verify_clean_state(session_id: int, expected_load_time_ms: float | None = None) -> None:
    """
    Raise RuntimeError if caches are not clean before a probe session.
    When called AFTER model load (expected_load_time_ms provided), only warns
    about load-time warmth — never raises.
    """
    print("=" * 60)
    print(f"PRE-SESSION {session_id} VERIFICATION")
    print("=" * 60)

    if expected_load_time_ms is not None:
        if expected_load_time_ms < 5000:
            print(f"[WARN] Load time {expected_load_time_ms}ms — OS page cache may be warm.")
        else:
            print(f"[OK] Load time {expected_load_time_ms}ms — consistent with cold cache")
        print("=" * 60)
        return

    issues = []

    kv_files = list(KV_STATE_DIR.rglob("*.bin"))
    if kv_files:
        issues.append(f"{len(kv_files)} KV .bin file(s) still present")
    else:
        print("[OK] KV state dir is empty")

    prompt_cache = KV_STATE_DIR.parent / "llama_prompt_cache"
    if prompt_cache.exists() and any(prompt_cache.iterdir()):
        issues.append("llama_prompt_cache is not empty")
    else:
        print("[OK] llama_prompt_cache is empty")

    for g in get_nvidia_smi_memory():
        if "error" in g:
            print(f"[SKIP] nvidia-smi: {g['error']}")
            continue
        used_pct = round(g["used_mb"] / g["total_mb"] * 100, 1)
        if used_pct > 15:
            issues.append(f"GPU {g['gpu_index']} still {used_pct}% occupied")
        else:
            print(f"[OK] GPU {g['gpu_index']}: {used_pct}% ({g['used_mb']}MB / {g['total_mb']}MB)")

    mem = get_process_memory_mb()
    if mem and mem > PROCESS_MEMORY_THRESHOLD_MB:
        issues.append(
            f"Process memory too high: {mem}MB "
            f"(threshold: {PROCESS_MEMORY_THRESHOLD_MB}MB) — model may not be unloaded"
        )
    else:
        print(f"[OK] Process memory: {mem}MB")

    print("=" * 60)
    if issues:
        for issue in issues:
            print(f"[FAIL] {issue}")
        raise RuntimeError(f"Session {session_id} verification failed: {issues}")
    print(f"[OK] All checks passed — session {session_id} safe to run")
    print("=" * 60)


# ── Prompt rendering ──────────────────────────────────────────────────────────

def render_raw_prompt(history: list[dict], user_prompt: str) -> str:
    """Build a plain-text multi-turn prompt from conversation history."""
    blocks = [f"User: {t['user']}\nAssistant: {t['assistant']}" for t in history]
    blocks.append(f"User: {user_prompt}\nAssistant:")
    return "\n\n".join(blocks)


def generate_with_history(llm, history: list[dict], user_prompt: str) -> tuple[str, str]:
    """Generate a reply and append to history. Used in store phases."""
    prompt_text = render_raw_prompt(history, user_prompt)
    out = llm.create_completion(
        prompt=prompt_text,
        max_tokens=GEN_CONFIG["max_tokens"],
        temperature=GEN_CONFIG["temperature"],
    )
    text = out["choices"][0]["text"].strip()
    history.append({"user": user_prompt, "assistant": text})
    return text, prompt_text


def generate_one_shot(llm, user_prompt: str) -> tuple[str, str]:
    """Single-turn generation with no conversation context. Used in probe phases."""
    prompt_payload = f"User: {user_prompt}\nAssistant:"
    out = llm.create_completion(
        prompt=prompt_payload,
        max_tokens=GEN_CONFIG["max_tokens"],
        temperature=GEN_CONFIG["temperature"],
        stop=["\nUser:", "\n\nUser:"],
    )
    return out["choices"][0]["text"].strip(), prompt_payload


def generate_probe_frozen(llm, store_history: list[dict], probe_prompt: str) -> tuple[str, str]:
    """
    Run a single probe against a FROZEN store history.
    store_history is never mutated — every probe sees identical context.
    Used in the within-session experiment.
    """
    payload = render_raw_prompt(store_history, probe_prompt)
    out = llm.create_completion(
        prompt=payload,
        max_tokens=GEN_CONFIG["max_tokens"],
        temperature=GEN_CONFIG["temperature"],
        stop=["\nUser:", "\n\nUser:"],
    )
    return out["choices"][0]["text"].strip(), payload


# ── Result row builder ────────────────────────────────────────────────────────

def build_result_row(
    session_id, phase, model_name, filename, model_path,
    device_used, key, prompt, generation,
    cache_state, kv_state_restored=None, load_time_ms=None,
    leak_detected=None, leak_trigger=None, leak_match_count=None,
    repeat_id: int = 0,
) -> dict:
    category, test_type, technique, sensitivity = get_prompt_metadata(key)
    row = {
        "session_id"       : session_id,
        "repeat_id"        : repeat_id,
        "phase"            : phase,
        "model"            : model_name,
        "filename"         : filename,
        "model_path"       : model_path,
        "type"             : "llama.cpp",
        "device_used"      : device_used,
        "n_gpu_layers"     : N_GPU_LAYERS,
        "n_ctx"            : N_CTX,
        "test"             : key,
        "prompt"           : prompt,
        "generation"       : generation,
        "category"         : category,
        "test_type"        : test_type,
        "technique"        : technique,
        "sensitivity_level": sensitivity,
        "cache_state"      : cache_state,
        "kv_state_restored": kv_state_restored,
        "load_time_ms"     : load_time_ms,
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "process_memory_mb": get_process_memory_mb(),
        "gpu_memory"       : json.dumps(get_nvidia_smi_memory()),
    }
    if leak_detected is not None:
        row["leak_detected"]    = leak_detected
        row["leak_trigger"]     = leak_trigger
        row["leak_match_count"] = leak_match_count
    return row
