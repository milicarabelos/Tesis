#!/bin/bash

set -e

export HF_HOME=/home/mcarabelos/tesis/models_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"

if [ "$#" -eq 0 ]; then
    echo "Usage:"
    echo "  $0 <model> [model ...] --tasks <task> [task ...]"
    echo
    echo "Example:"
    echo "  $0 meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct --tasks gsm8k hellaswag"
    exit 1
fi

MODELS=()
TASKS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASKS+=("$1")
                shift
            done
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done


# ============================================================
# Download models
# ============================================================

for model in "${MODELS[@]}"; do

    repo_id="${model#https://huggingface.co/}"
    repo_id="${repo_id%/}"

    echo
    echo "========================================"
    echo "Downloading model: $repo_id"
    echo "========================================"

    python - "$repo_id" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]

path = snapshot_download(
    repo_id=repo_id,
)

print(f"Model ready at: {path}")
PY

done


# ============================================================
# Download lm-eval datasets
# ============================================================

for task in "${TASKS[@]}"; do

    echo
    echo "========================================"
    echo "Preparing task: $task"
    echo "========================================"

    python - "$task" <<'PY'
import sys
import lm_eval
from lm_eval.tasks import TaskManager

task = sys.argv[1]

print(f"Loading task: {task}")

task_manager = TaskManager()

task_manager.load_task_or_group(task)

print(f"Task ready: {task}")
PY

done


echo
echo "========================================"
echo "All models and datasets are ready."
echo "HF_HOME: $HF_HOME"
echo "========================================"
