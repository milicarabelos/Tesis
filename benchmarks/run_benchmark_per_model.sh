#!/bin/bash
#SBATCH --job-name=lm-eval-20
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=short
# Un task por modelo: rango = 0 .. (nº de modelos) - 1
# Ahora: 2 modelos -> 0-1. Si cambiás MODEL_LIST, ajustá este rango.
# %1 = una task a la vez (mediciones comparables)
#SBATCH --array=0-1%1

set -euo pipefail
mkdir -p logs

# ---------- configuración ----------
export HF_HOME=/home/mcarabelos/tesis/models_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"

MODEL_LIST=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
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

MODEL_SLUG=$(echo "$MODELS" | sed 's/[^A-Za-z0-9._-]\+/_/g' | cut -c1-100)
export OUTPUT_DIR="results/${MODEL_SLUG}/${SLURM_ARRAY_JOB_ID:-manual}"
export NUM_FEWSHOT=0
# CAMBIO: antes era 1. "auto" (o 16/32) acelera mucho los loglikelihood.
export BATCH_SIZE=${BATCH_SIZE:-auto}
export APPLY_CHAT_TEMPLATE=0
export LIMIT=20        # comentar esta línea para la corrida completa

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Nodo: $(hostname)"
echo "Task $IDX/$((NM - 1)): MODELS=$MODELS BENCHMARKS=$BENCHMARKS OUTPUT_DIR=$OUTPUT_DIR BATCH_SIZE=$BATCH_SIZE"

# ---------- copiar modelo a /scratch (sin fallback) ----------
die() { echo "[stage] ERROR: $*" >&2; exit 1; }

if [[ "$MODELS" == *"="* || "$MODELS" == /* ]]; then
  die "'$MODELS' no es un id de HF; solo se soportan ids (org/modelo)"
fi

[[ -d /scratch && -w /scratch ]] || die "/scratch no existe o no es escribible en $(hostname)"

LOCAL_BASE="/scratch/${USER}/${SLURM_JOB_ID}"
mkdir -p "$LOCAL_BASE/hub" "$LOCAL_BASE/datasets" || die "no se pudo crear $LOCAL_BASE"
trap 'rm -rf "$LOCAL_BASE"' EXIT      # limpia al terminar o si el job muere

src="$HF_HOME/hub/models--${MODELS//\//--}"
dst="$LOCAL_BASE/hub/models--${MODELS//\//--}"
[[ -d "$src/snapshots" ]] || die "$MODELS no está en $HF_HOME/hub (¿falta descargarlo?)"

if [[ -f "$src/refs/main" ]]; then
  rev=$(cat "$src/refs/main")
else
  rev=$(basename "$(ls -d "$src"/snapshots/* | head -1)")
fi
snap="$src/snapshots/$rev"
[[ -d "$snap" ]] || die "no existe el snapshot $snap"

mapfile -t FILES < <(find -L "$snap" -maxdepth 1 -type f)
(( ${#FILES[@]} > 0 )) || die "snapshot vacío: $snap"

need=0
for f in "${FILES[@]}"; do need=$(( need + $(stat -L -c %s "$f") )); done
avail=$(df --output=avail -B1 /scratch | tail -1 | tr -d ' ')
echo "[stage] necesito $(( need / 1024**3 )) GiB, disponibles $(( avail / 1024**3 )) GiB en /scratch"
(( avail > need + 2*1024**3 )) || die "espacio insuficiente en /scratch"

mkdir -p "$dst/snapshots/$rev" "$dst/refs"
if [[ -f "$src/refs/main" ]]; then cp "$src/refs/main" "$dst/refs/main"; fi

echo "[stage] copiando $MODELS -> $dst ..."
T_COPY=$(date +%s)
for f in "${FILES[@]}"; do
  cp -L "$f" "$dst/snapshots/$rev/" || die "falló la copia de $f"
done

for f in "${FILES[@]}"; do
  b=$(basename "$f")
  [[ "$(stat -L -c %s "$f")" == "$(stat -c %s "$dst/snapshots/$rev/$b")" ]] \
    || die "tamaño distinto tras copiar: $b"
done
[[ -f "$dst/snapshots/$rev/config.json" ]] || die "falta config.json en la copia"
ls "$dst/snapshots/$rev"/*.safetensors >/dev/null 2>&1 || die "no hay .safetensors en la copia"
echo "[stage] modelo listo en $(( $(date +%s) - T_COPY ))s"

# ---------- NUEVO: copiar datasets a /scratch ----------
# Copia el cache procesado (arrow) y los repos de datasets del hub.
T_DS=$(date +%s)
if [[ -d "$HF_HOME/datasets" ]]; then
  cp -a "$HF_HOME/datasets/." "$LOCAL_BASE/datasets/" || die "falló la copia de datasets"
else
  echo "[stage] AVISO: no existe $HF_HOME/datasets" >&2
fi
shopt -s nullglob
for d in "$HF_HOME"/hub/datasets--*; do
  cp -a "$d" "$LOCAL_BASE/hub/" || die "falló la copia de $d"
done
shopt -u nullglob
echo "[stage] datasets copiados en $(( $(date +%s) - T_DS ))s"

export HF_HUB_CACHE="$LOCAL_BASE/hub"
export HF_DATASETS_CACHE="$LOCAL_BASE/datasets"   # ahora apunta al disco local
du -sh "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# ---------- entorno ----------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate benchmarks311
echo "using benchmarks311 env"

python run_eval.py
