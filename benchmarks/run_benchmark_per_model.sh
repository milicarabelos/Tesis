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
# Un task por modelo: rango = 0 .. (nº de modelos) - 1
# Ahora: 2 modelos -> 0-1. Si cambiás MODEL_LIST, ajustá este rango.
# Para limitar cuántos corren a la vez: --array=0-1%1
#SBATCH --array=0-1

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
  # "pretrained=/ruta/local/llama3-awq"
)
BENCH_LIST=(boolq wic openbookqa)

# ---------- selección del modelo según el índice del array ----------
NM=${#MODEL_LIST[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}

if [[ "${SLURM_ARRAY_TASK_MAX:-$((NM - 1))}" -ne $((NM - 1)) ]]; then
  echo "AVISO: el --array termina en ${SLURM_ARRAY_TASK_MAX} pero hay $NM modelos (debería ser 0-$((NM - 1)))" >&2
fi
if (( IDX >= NM )); then
  echo "Índice $IDX fuera de rango (hay $NM modelos), nada que hacer." >&2
  exit 0
fi

export MODELS="${MODEL_LIST[$IDX]}"
export BENCHMARKS="${BENCH_LIST[*]}"

# Carpeta independiente por modelo: results/<modelo>/<jobid>/ (con su propio Excel)
MODEL_SLUG=$(echo "$MODELS" | sed 's/[^A-Za-z0-9._-]\+/_/g' | cut -c1-100)
export OUTPUT_DIR="results/${MODEL_SLUG}/${SLURM_ARRAY_JOB_ID:-manual}"
export NUM_FEWSHOT=0
export BATCH_SIZE=8
export APPLY_CHAT_TEMPLATE=0
# export LIMIT=20        # comentar esta línea para la corrida completa

# Si los nodos de cómputo no tienen internet, descargar antes en el login node
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Task $IDX/$((NM - 1)): MODELS=$MODELS BENCHMARKS=$BENCHMARKS OUTPUT_DIR=$OUTPUT_DIR"

# ---------- entorno ----------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medir_inferencia

python run_eval.py
