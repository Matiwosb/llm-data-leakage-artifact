"""
session_logger.py — Structured session logger for KV cache leakage experiments.

Captures and persists:
  - Model architectural parameters  (from GGUF metadata)
  - Memory snapshots                (RAM + GPU at pre-load / post-load / post-unload)
  - Session timing                  (start, load duration, session duration)
  - Generation counters             (store turns, probe turns, leaks detected)
  - Config snapshot                 (n_ctx, n_gpu_layers, GEN_CONFIG values)

Usage
-----
    from kv_leakage.session_logger import SessionLogger

    logger = SessionLogger(experiment="exp1_within", session_id=1, model_name=name)
    logger.snapshot_memory("pre_load")

    llm, device = load_model(model_path, ...)
    logger.record_model_arch(llm)
    logger.snapshot_memory("post_load")
    logger.set_load_time_ms(load_ms)

    # ... run store / probe phases ...
    logger.increment("store_turns")
    logger.increment("probe_turns")
    logger.increment("leaks")

    reset_runtime_state(llm)
    logger.snapshot_memory("post_unload")
    logger.save()          # writes JSON to BASE_OUTPUT_DIR/session_logs/
"""

import ctypes
import json
import os
import time
from datetime import datetime
from pathlib import Path

from kv_leakage.config import BASE_OUTPUT_DIR, GEN_CONFIG, N_CTX, N_GPU_LAYERS
from kv_leakage.utils import get_nvidia_smi_memory, get_process_memory_mb

# LOG_DIR = BASE_OUTPUT_DIR / "session_logs"
LOG_DIR = BASE_OUTPUT_DIR / "within_session_all_model_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class SessionLogger:
    def __init__(self, experiment: str, session_id: int, model_name: str):
        self.experiment  = experiment
        self.session_id  = session_id
        self.model_name  = model_name
        self._start_time = time.time()
        self._ts         = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._data: dict = {
            "experiment"   : experiment,
            "session_id"   : session_id,
            "model_name"   : model_name,
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # Config snapshot
            "config": {
                "n_ctx"       : N_CTX,
                "n_gpu_layers": N_GPU_LAYERS,
                "max_tokens"  : GEN_CONFIG.get("max_tokens"),
                "temperature" : GEN_CONFIG.get("temperature"),
            },

            # Filled by record_model_arch()
            "model_arch": {},

            # Filled by snapshot_memory()
            "memory": {},

            # Filled by set_load_time_ms()
            "load_time_ms": None,

            # Filled by increment()
            "counters": {
                "store_turns": 0,
                "probe_turns": 0,
                "leaks"      : 0,
            },

            # Filled at save()
            "session_duration_s": None,
        }

    # ── Model architecture ────────────────────────────────────────────────────

    def record_model_arch(self, llm) -> None:
        """Extract architecture parameters from a loaded Llama instance."""
        try:
            from llama_cpp import llama_cpp

            arch: dict = {}

            # Per-field API
            try: arch["n_ctx_active"]  = llm.n_ctx()
            except Exception: pass
            try: arch["n_embd"]        = llm.n_embd()
            except Exception: pass
            try: arch["n_vocab"]       = llm.n_vocab()
            except Exception: pass
            try: arch["n_params"]      = llama_cpp.llama_model_n_params(llm.model)
            except Exception: pass

            # Full GGUF metadata
            buf    = ctypes.create_string_buffer(512)
            n_meta = llama_cpp.llama_model_meta_count(llm.model)
            meta   = {}
            for i in range(n_meta):
                llama_cpp.llama_model_meta_key_by_index(llm.model, i, buf, 512)
                key = buf.value.decode("utf-8", errors="replace")
                llama_cpp.llama_model_meta_val_str_by_index(llm.model, i, buf, 512)
                val = buf.value.decode("utf-8", errors="replace")
                meta[key] = val

            # Pull out the most useful architectural fields explicitly
            arch_keys = {
                "general.architecture"               : "architecture",
                "general.name"                       : "model_name_meta",
                "general.file_type"                  : "file_type",
                "general.quantization_version"       : "quant_version",
                "llama.context_length"               : "context_length",
                "llama.embedding_length"             : "embedding_length",
                "llama.feed_forward_length"          : "feed_forward_length",
                "llama.block_count"                  : "n_layers",
                "llama.attention.head_count"         : "n_heads",
                "llama.attention.head_count_kv"      : "n_kv_heads",
                "llama.rope.freq_base"               : "rope_freq_base",
                "llama.rope.dimension_count"         : "rope_dim",
                "llama.attention.layer_norm_rms_epsilon": "rms_norm_epsilon",
                "tokenizer.ggml.model"               : "tokenizer_model",
            }
            for gguf_key, friendly_key in arch_keys.items():
                if gguf_key in meta:
                    arch[friendly_key] = meta[gguf_key]

            arch["raw_metadata"] = meta
            self._data["model_arch"] = arch
            print(f"  [LOG] Architecture recorded: "
                  f"{arch.get('architecture','?')} | "
                  f"layers={arch.get('n_layers','?')} | "
                  f"heads={arch.get('n_heads','?')} | "
                  f"embd={arch.get('embedding_length','?')} | "
                  f"vocab={arch.get('n_vocab','?')}")
        except Exception as e:
            print(f"  [LOG] Architecture capture failed: {e}")
            self._data["model_arch"] = {"error": str(e)}

    # ── Memory snapshots ──────────────────────────────────────────────────────

    def snapshot_memory(self, label: str) -> None:
        """
        Take a memory snapshot tagged with `label`.
        Typical labels: 'pre_load', 'post_load', 'post_unload'.
        """
        snapshot = {
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "process_ram_mb"  : get_process_memory_mb(),
            "gpu"             : get_nvidia_smi_memory(),
        }
        self._data["memory"][label] = snapshot

        ram = snapshot["process_ram_mb"]
        gpu_info = ", ".join(
            f"GPU{g['gpu_index']}={g['used_mb']}/{g['total_mb']}MB"
            for g in snapshot["gpu"] if "gpu_index" in g
        ) or "n/a"
        print(f"  [LOG] Memory [{label}] RAM={ram}MB  {gpu_info}")

    # ── Load time ─────────────────────────────────────────────────────────────

    def set_load_time_ms(self, ms: float) -> None:
        self._data["load_time_ms"] = ms
        print(f"  [LOG] Load time: {ms}ms")

    # ── Counters ──────────────────────────────────────────────────────────────

    def increment(self, counter: str, by: int = 1) -> None:
        """Increment a named counter (store_turns / probe_turns / leaks)."""
        if counter not in self._data["counters"]:
            self._data["counters"][counter] = 0
        self._data["counters"][counter] += by

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self) -> Path:
        """
        Write the log to a JSON file and return the path.

        If the GDRIVE_FOLDER_ID environment variable is set, the JSON is also
        uploaded to Google Drive automatically. Upload is skipped silently if
        credentials.json is not present.
        """
        self._data["session_duration_s"] = round(time.time() - self._start_time, 2)

        safe_model = "".join(c if c.isalnum() or c in "-_." else "_"
                             for c in self.model_name)
        filename = (f"{self.experiment}_session{self.session_id}"
                    f"_{safe_model}_{self._ts}.json")
        path = LOG_DIR / filename

        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)

        print(f"  [LOG] Session log saved → {path}")

        # Auto-upload to Google Drive if GDRIVE_FOLDER_ID is set
        folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        # folder_id = getattr(config, "GDRIVE_FOLDER_ID", os.environ.get("GDRIVE_FOLDER_ID"))
        try:
            from kv_leakage.drive_upload import upload_file
            upload_file(path, folder_id=folder_id)
        except FileNotFoundError as e:
            print(f"  [LOG] Drive upload skipped — {e}")
        except Exception as e:
            print(f"  [LOG] Drive upload failed — {e}")

        return path

    # ── Convenience: pretty summary ───────────────────────────────────────────

    def print_summary(self) -> None:
        arch = self._data.get("model_arch", {})
        mem  = self._data.get("memory", {})
        ctr  = self._data.get("counters", {})

        print("\n" + "─" * 60)
        print(f"  SESSION LOG SUMMARY  [{self.experiment} | session {self.session_id}]")
        print("─" * 60)
        print(f"  Model      : {self.model_name}")
        print(f"  Arch       : {arch.get('architecture','?')}  "
              f"layers={arch.get('n_layers','?')}  "
              f"heads={arch.get('n_heads','?')}  "
              f"kv_heads={arch.get('n_kv_heads','?')}")
        print(f"  Embedding  : {arch.get('embedding_length','?')}  "
              f"FFN={arch.get('feed_forward_length','?')}  "
              f"vocab={arch.get('n_vocab','?')}")
        print(f"  Params     : {arch.get('n_params', '?')}")
        print(f"  Context    : n_ctx={self._data['config']['n_ctx']}  "
              f"(model max={arch.get('context_length','?')})")
        print(f"  Quant      : file_type={arch.get('file_type','?')}")
        print(f"  Load time  : {self._data.get('load_time_ms','?')}ms")

        for label in ("pre_load", "post_load", "post_unload"):
            snap = mem.get(label)
            if snap:
                gpu_str = "  ".join(
                    f"GPU{g['gpu_index']}={g['used_mb']}MB"
                    for g in snap["gpu"] if "gpu_index" in g
                ) or "n/a"
                print(f"  RAM [{label:<13}]: {snap['process_ram_mb']}MB   {gpu_str}")

        print(f"  Counters   : store={ctr.get('store_turns',0)}  "
              f"probes={ctr.get('probe_turns',0)}  "
              f"leaks={ctr.get('leaks',0)}")
        print(f"  Duration   : {self._data.get('session_duration_s','?')}s")
        print("─" * 60)
