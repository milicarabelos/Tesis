#!/usr/bin/env bash
# Evalúa los checkpoints cuantizados (AWQ y RTN) de test_methods/quantized con run_eval.py.
# Resultados en ${OUTPUT_DIR}_quantized/resultados_quantized.xlsx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDIR="$ROOT/test_methods/quantized"

# NF4 igual que bnb-nf4 en test_methods/quant_compare.py (double quant, compute en DTYPE)
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
NF4="pretrained=$BASE_MODEL,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True"

export MODELS="${MODELS:-$QDIR/Llama-3.1-8B-awq $QDIR/Llama-3.1-8B-rtn $NF4}"
export BENCHMARKS="${BENCHMARKS:-boolq wic openbookqa sqac mmlu gsm8k afrimgsm}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/benchmarks/results}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
export APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
# los kernels AWQ (gemm) están pensados para fp16
export DTYPE="${DTYPE:-float16}"
# la carga en paralelo de transformers (ThreadPool) da segfault con Python 3.15 en checkpoints
# pre-cuantizados; NF4 ya carga secuencial, así que los tiempos de carga quedan comparables
export HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"

python "$ROOT/benchmarks/run_eval.py" --exp quantized
