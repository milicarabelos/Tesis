#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=short



set -euo pipefail
mkdir -p logs

# ---------- configuración ----------
export HF_HOME=/home/mcarabelos/tesis/models_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Cada entrada: id/path de modelo, o model_args completo SIN espacios
# (ej: pretrained=/ruta/base,peft=/ruta/adapter,load_in_4bit=True)
MODEL_LIST=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "pretrained=/ruta/local/llama3-awq"
)
BENCH_LIST=(boolq wic openbookqa)

export MODELS="${MODEL_LIST[*]}"
export BENCHMARKS="${BENCH_LIST[*]}"
export OUTPUT_DIR="results/${SLURM_JOB_ID:-manual}"
export NUM_FEWSHOT=0
export BATCH_SIZE=8
export APPLY_CHAT_TEMPLATE=0
export LIMIT=20        # descomentar para un smoke test rápido

# Si los nodos de cómputo no tienen internet, descargar antes en el login node
# y activar estas dos líneas:
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------- entorno ----------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medir_inferencia     # o: module load / conda activate

python run_eval.py

