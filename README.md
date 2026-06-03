# KV Cache Session Leakage — Experiments

## Project structure

```
kv_leakage/
├── config.py                          # Models, paths, generation settings
├── prompts.py                         # Full prompt dataset + leak detection
├── utils.py                           # Shared helpers (model load/unload, generation,
│                                      #   KV state save/load, cache clearing, verification)
├── experiments/
│   ├── experiment1_within_session.py  # Exp 1 — within-session leakage
│   └── experiment2_cross_session_control.py  # Exp 2 — cache-cleared baseline
└── results/                           # Auto-created; CSVs written here
    ├── within_session_experiment/
    ├── cleared_cache_experiment/
    └── kv_states/
```

## Setup

```bash
pip install llama-cpp-python pandas psutil torch

# For CUDA (run in a fresh environment, then restart):
CMAKE_ARGS="-DGGML_CUDA=ON" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## Running

### 1. Configure models

Open `kv_leakage/config.py` and set:

```python
MODEL_DIR = Path("/path/to/your/gguf/models")   # folder containing .gguf files
TARGET_FILENAMES = {
    "mistral-7b-instruct-v0.1.Q4_0.gguf",       # add/remove models here
    ...
}
```

### 2. Run experiments

All scripts are run from the `Oraginized_version/` directory.

```bash
cd Oraginized_version/

# Experiment 1 — Within-session leakage
# Loads the model fresh each session; store phase then probe phase in the same context.
# Runs NUM_SESSIONS sessions spaced WITHIN_WAIT_SECONDS apart (default: 5 sessions × 1 hour).
python experiments/experiment1_within_session.py

# Experiment 2 — Cache-cleared cross-session control (baseline)
# Session 0 stores PII, then ALL caches are wiped; sessions 1–5 probe with a fresh model.
# Any leakage here points to weight memorization, not KV persistence.
python experiments/experiment2_cross_session_control.py

# Optional: drop OS page cache between sessions for stronger isolation
SUDO_PASSWORD=mypass python experiments/experiment2_cross_session_control.py

# Experiment 3 — KV state saved and restored across sessions
# Session 0: store PII and save KV state to disk.
# Sessions 1–5: load saved KV state into a fresh model instance and probe.
python experiments/experiment3_kv_cross_session.py              # all sessions
python experiments/experiment3_kv_cross_session.py --session 0  # store only
python experiments/experiment3_kv_cross_session.py --session 1  # probe session 1
python experiments/experiment3_kv_cross_session.py --combine    # merge CSVs

# Experiment 4 — Continuous model instance leakage
# A single model instance stays alive for all sessions; no state serialisation.
# Models a long-running LLM server that never clears context between users.
python experiments/experiment4_continuous_instance.py
```

### 3. Results

CSVs are written automatically to `results/` subdirectories:

| Experiment | Output directory |
|---|---|
| Exp 1 | `results/models_within_session_experiment/` |
| Exp 2 | `results/cleared_cache_experiment/` |
| Exp 3 | `results/kv_cross_session_experiment/` |
| Exp 4 | `results/one_continuous_instance_experiment/` |

A combined `*_combined_<run_id>.csv` is written at the end of each run.

### 4. Google Drive upload (optional)

Place `credentials.json` (Google OAuth 2.0) in the `Oraginized_version/` directory.
Results are uploaded automatically at the end of each run.
To target a specific Drive folder, set the `GDRIVE_FOLDER_ID` environment variable:

```bash
GDRIVE_FOLDER_ID=your_folder_id python experiments/experiment1_within_session.py
```

## Interpreting results

| Exp 2 outcome | Interpretation |
|---|---|
| Sessions 1–5 show no recall | ✅ KV cache is the sole leakage vector |
| Consistent exact-value recall | ⚠️ Weight memorization |
| Inconsistent / plausible-wrong | ⚠️ Model hallucination / probe contamination |
| Load time < 5 s | ⚠️ OS page cache not fully cleared — note as caveat |

## Quick test (skip waiting)

In `config.py`:
```python
WITHIN_WAIT_SECONDS = 60   # 1 minute instead of 1 hour
PROBE_SESSIONS = [1, 2]    # only 2 probe sessions
```
