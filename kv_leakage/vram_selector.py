"""
vram_selector.py — Automatic model selection based on detected GPU VRAM.

Used by experiment4 (continuous instance) which keeps ALL selected models
alive in VRAM simultaneously. The correct group is chosen at startup so
the total model size always fits within available VRAM.

Configuration
-------------
Edit VRAM_MODEL_GROUPS below to define which models to run at each VRAM tier.
The highest threshold that fits detected VRAM is selected automatically.

Usage
-----
    from kv_leakage.vram_selector import auto_select_models
    filenames = auto_select_models()          # returns set of filenames
    models, missing = resolve_models(MODEL_DIR, filenames)
"""

from pathlib import Path

# ── VRAM groups ───────────────────────────────────────────────────────────────
# Key   = minimum VRAM in GB required for this group.
# Value = set of model filenames to load simultaneously.
# Total file sizes within each group must fit in the corresponding VRAM tier.
#
# Approximate Q4_0 sizes:
#   1B  ≈ 0.7 GB   3B  ≈ 2 GB   7/8B ≈ 4–5 GB   13B ≈ 7 GB
#
# Add KV cache overhead: n_ctx=2048 per model adds ~0.3–1 GB per model.

VRAM_MODEL_GROUPS: dict[int, set[str]] = {
    # ≥ 28 GB effective — two 13B models  (~14 GB weights + ~2 GB KV = ~16 GB total)
    # Fires on 32 GB GPU (32 - 4 headroom = 28 GB effective)
    28: {
        "nous-hermes-llama2-13b.Q4_0.gguf",
        "wizardlm-13b-v1.2.Q4_0.gguf",
    },
    # ≥ 16 GB effective — four 7/8B models  (~17 GB weights + ~2 GB KV = ~19 GB total)
    # Fires on 24 GB GPU (24 - 4 headroom = 20 GB effective ≥ 16)
    16: {
        "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        "mistral-7b-instruct-v0.1.Q4_0.gguf",
        "ghost-7b-v0.9.1-Q4_0.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf",
    },
    # ≥ 7 GB effective — three small/medium models  (~8 GB weights + ~1 GB KV = ~9 GB total)
    7: {
        "Phi-3-mini-4k-instruct.Q4_0.gguf",
        "Llama-3.2-3B-Instruct-Q4_0.gguf",
        "orca-2-7b.Q4_0.gguf",
    },
    # < 7 GB / CPU fallback — two small models  (~3 GB total)
    0: {
        "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "Phi-3-mini-4k-instruct.Q4_0.gguf",
    },
}


# ── Core function ─────────────────────────────────────────────────────────────

def auto_select_models(
    vram_groups: dict[int, set[str]] = None,
    headroom_gb: float = 4.0,
) -> set[str]:
    """
    Detect available GPU VRAM and return the appropriate model set.

    Args:
        vram_groups:  Override the default VRAM_MODEL_GROUPS table.
        headroom_gb:  Reserve this many GB of VRAM as safety margin.
                      Effective VRAM = detected_total - headroom_gb.
                      Default 4 GB leaves room for KV caches and CUDA overhead.
    Returns:
        Set of model filenames matching the detected VRAM tier.
    """
    groups = vram_groups or VRAM_MODEL_GROUPS

    # ── Detect VRAM ───────────────────────────────────────────────────────────
    total_vram_gb = _detect_vram_gb()
    effective_gb  = max(0.0, total_vram_gb - headroom_gb)

    print(f"[VRAM] Detected: {total_vram_gb:.1f} GB total  "
          f"({headroom_gb:.1f} GB reserved → {effective_gb:.1f} GB effective)")

    # ── Pick highest matching threshold ───────────────────────────────────────
    threshold = max(
        (k for k in groups if k <= effective_gb),
        default=0,
    )
    selected = groups[threshold]

    print(f"[VRAM] Selected group (≥{threshold} GB tier): "
          f"{len(selected)} model(s)")
    for name in sorted(selected):
        size = _model_size_gb(name)
        size_str = f"{size:.1f} GB" if size else "size unknown"
        print(f"         {name}  ({size_str})")

    return selected


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_vram_gb() -> float:
    """Return total VRAM of GPU 0 in GB, or 0 if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            return float(lines[0].strip()) / 1024  # MiB → GB
    except Exception:
        pass
    return 0.0


def _model_size_gb(filename: str) -> float | None:
    """Return file size in GB for a model filename, or None if not found."""
    try:
        from kv_leakage.config import MODEL_DIR
        p = Path(MODEL_DIR) / filename
        if p.exists():
            return p.stat().st_size / (1024 ** 3)
    except Exception:
        pass
    return None
