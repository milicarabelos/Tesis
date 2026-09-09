 #!/bin/bash

set -u

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

run_model() {
    local name="$1"
    local model="$2"
    local gguf="${3:-}"

    local output_file="$RESULTS_DIR/${name}.txt"

    echo
    echo "========================================"
    echo "Running: $name"
    echo "Model:   $model"

    if [ -n "$gguf" ]; then
        echo "GGUF:    $gguf"
    fi

    echo "Output:  $output_file"
    echo "========================================"

    if [ -n "$gguf" ]; then
        model_name="$model" \
        gguf_file="$gguf" \
        python inference.py > "$output_file" 2>&1
    else
        model_name="$model" \
        env -u gguf_file \
        python inference.py > "$output_file" 2>&1
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ $name finished successfully"
    else
        echo "✗ $name FAILED (exit code $exit_code)"
    fi
}


# Qwen 3.5
run_model "Qwen3.5-27B" \
    "unsloth/Qwen3.5-27B-GGUF" \
    "Qwen3.5-27B-Q4_K_M.gguf"

run_model "Qwen3.5-9B" \
    "unsloth/Qwen3.5-9B-GGUF" \
    "Qwen3.5-9B-Q4_K_M.gguf"

run_model "Qwen3.5-4B" \
    "unsloth/Qwen3.5-4B-MTP-GGUF" \
    "Qwen3.5-4B-Q4_K_M.gguf"


# Nemotron
run_model "Nemotron-3-Nano-4B-GGUF" \
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF" \
    "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

run_model "Nemotron-3-Nano-4B-FP8" \
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"


# Llama
run_model "Llama-3.2-3B-bnb-4bit" \
    "unsloth/Llama-3.2-3B-bnb-4bit"

run_model "Llama-3.2-3B-Instruct-QLORA-INT4-EO8" \
    "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"

run_model "Llama-3.1-8B-bnb-4bit" \
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


# Phi
run_model "Phi-4" \
    "microsoft/phi-4-gguf" \
    "phi-4-Q4_K_S.gguf"


echo
echo "========================================"
echo "ALL MODELS FINISHED"
echo "Results saved in: $RESULTS_DIR/"
echo "========================================"

