"""
config.py — Central configuration for KV cache leakage experiments.
Edit this file to change models, generation parameters, and output paths.
"""

from pathlib import Path

# Root of the Oraginized_version/ folder — always correct regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Generation ────────────────────────────────────────────────────────────────
GEN_CONFIG = {
    "max_tokens":     150,
    "temperature":    0.0,   # deterministic / greedy
    "top_k":          40,
    "top_p":          0.9,
    "repeat_penalty": 1.1,
}

N_CTX        = 2048
N_GPU_LAYERS = -1   # -1 = full GPU offload (CUDA build); 0 = CPU only
VERBOSE      = False

# ── Model discovery ───────────────────────────────────────────────────────────
MODEL_DIR = Path("/home/mbirbo1/gpt4all_models")

# Leave empty set to run ALL .gguf files found in MODEL_DIR.
# TARGET_FILENAMES = set()
# 
TARGET_FILENAMES = {
    # "LFM2-1.2B-Extract-Q4_0.gguf",
    # "DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf",
    "ghost-7b-v0.9.1-Q4_0.gguf",
    # "gpt4all-13b-snoozy-q4_0.gguf",
    # "gpt4all-falcon-newbpe-q4_0.gguf",
    # "Llama-3.2-1B-Instruct-Q4_0.gguf",
    # "Llama-3.2-3B-Instruct-Q4_0.gguf",
    # "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
    # "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "mistral-7b-instruct-v0.1.Q4_0.gguf",
    # "mistral-7b-openorca.gguf2.Q4_0.gguf",
    # "mpt-7b-chat.gguf4.Q4_0.gguf",
    # "mpt-7b-chat-newbpe-q4_0.gguf",
    "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
    # "nous-hermes-llama2-13b.Q4_0.gguf",
    "orca-2-7b.Q4_0.gguf",
    "orca-2-13b.Q4_0.gguf",
    # "orca-mini-3b-gguf2-q4_0.gguf",
    # "Phi-3-mini-4k-instruct.Q4_0.gguf",
    # "qwen2.5-coder-7b-instruct-q4_0.gguf",
    # "qwen2-1_5b-instruct-q4_0.gguf",
    # "wizardlm-13b-v1.2.Q4_0.gguf",
}

# ── Output directories ────────────────────────────────────────────────────────
BASE_OUTPUT_DIR  = _PROJECT_ROOT / "results"
KV_STATE_DIR     = BASE_OUTPUT_DIR / "kv_states"
CLEARED_DIR      = BASE_OUTPUT_DIR / "cleared_cache_experiment"
# WITHIN_DIR       = BASE_OUTPUT_DIR / "within_session_experiment"
WITHIN_DIR       = BASE_OUTPUT_DIR / "models_within_session_experiment"
KV_CROSS_DIR     = BASE_OUTPUT_DIR / "kv_cross_session_experiment"
# CONTINUOUS_DIR   = BASE_OUTPUT_DIR / "continuous_instance_experiment"
CONTINUOUS_DIR   = BASE_OUTPUT_DIR / "one_continuous_instance_experiment"

for _d in (BASE_OUTPUT_DIR, KV_STATE_DIR, CLEARED_DIR, WITHIN_DIR, KV_CROSS_DIR, CONTINUOUS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
# Testing different session leakages uncomment the below part

# Experiment 1 (within-session)
# WITHIN_NUM_SESSIONS = 1
# WITHIN_WAIT_SECONDS = 60   # 1 hour between sessions; set to 60 for quick test for 1 hour 3600

# # Experiment 2 (cross-session / cache-cleared control)
# PROBE_SESSIONS = [1]

# # Experiment 3 (KV cross_session leakage with cache clearing)
# KV_PROBE_SESSIONS = [1]
# KV_WAIT_SECONDS = 60   # 1 hour between sessions; set to 60 for quick test

# CONTINUOUS_SESSIONS   = [1, 2, 3, 4, 5]   # probe session IDs after store
# CONTINUOUS_WAIT_SECONDS = 3600             # wait between probe sessions; 60 for quick test

#Testing different session leakages with one digit session IDs for quick test

# Change WITHIN_NUM_SESSIONS to control the number of sessions for ALL
# experiments at once. All session lists are derived from this single value.
WITHIN_NUM_SESSIONS = 5   # set to e.g. 3 for a shorter run across all experiments
 
# Derived session lists — all experiments run the same number of sessions.
# Edit WITHIN_NUM_SESSIONS above; do not edit these directly.
PROBE_SESSIONS      = list(range(1, WITHIN_NUM_SESSIONS + 1))  # Exp 2
KV_PROBE_SESSIONS   = list(range(1, WITHIN_NUM_SESSIONS + 1))  # Exp 3
CONTINUOUS_SESSIONS = list(range(1, WITHIN_NUM_SESSIONS + 1))  # Exp 4
 
# Wait times between probe sessions (seconds).
# Session 1 always runs immediately in Exp 4; this applies to sessions 2-N.
WITHIN_WAIT_SECONDS     = 3600   # Exp 1 — set to 60 for quick test
KV_WAIT_SECONDS         = 3600   # Exp 3 — set to 60 for quick test
CONTINUOUS_WAIT_SECONDS = 3600     # Exp 4 — set to 60 for quick test

# Directories wiped by clear_all_caches()
CACHE_DIRS_TO_CLEAR = [
    KV_STATE_DIR,
    BASE_OUTPUT_DIR / "llama_prompt_cache",
    Path.home() / ".cache" / "llama_cpp",
    Path("/tmp/__pycache__"),
]

# GDRIVE_FOLDER_ID = "127PDPJPXnquH-RXlVAltK1LOkUwLB_gX"  # "Within_Session_KV_Leakage_Experiment_Results folder"
GDRIVE_FOLDER_ID = "1OIYD1nMpioox6-k_qCBkyREQaqbNO7fm"

# ── Model batching ───────────────────────────────────────────────────────────
# Maximum number of models to load and run at once.
# Set to match how many models fit in your VRAM simultaneously.
# e.g. 24 GB VRAM handles ~4 x 7B Q4_0 models, or ~2 x 13B Q4_0 models.
MODEL_BATCH_SIZE = 4

# ── Probe stability settings ─────────────────────────────────────────────────
# Number of times to repeat the full probe set within each session.
# Set to 1 for a single pass (original behaviour).
PROBE_REPEATS   = 1

# Whether to randomise probe order on each repeat.
SHUFFLE_PROBES  = True

# Fixed random seed for reproducibility. Set to None for truly random order.
PROBE_SEED      = 42

# Process memory threshold for pre-session verification (MB).
# Raise this on machines with large baseline RAM footprints.
PROCESS_MEMORY_THRESHOLD_MB = 12_000
