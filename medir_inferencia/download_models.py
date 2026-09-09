#!/usr/bin/env python3
"""
Script para descargar TODOS los modelos LLM en login node
Usa HF_HOME local para evitar descargas en nodos de cómputo
"""
import os
import sys
from pathlib import Path

# ============================================
# CONFIGURACIÓN
# ============================================
MODELS_DIR = Path.home() / "tesis" / "medir_inferencia" / "models_cache"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar HF_HOME para descargar aquí
os.environ["HF_HOME"] = str(MODELS_DIR)

print(f"📁 Descargando modelos en: {MODELS_DIR}")
print(f"📌 Asegúrate de agregar esto a tu sbatch:")
print(f"   export HF_HOME={MODELS_DIR}")
print()

# ============================================
# MODELOS A DESCARGAR
# ============================================

# GGUF models
gguf_models = [
    ("unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-MTP-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"),
    ("microsoft/phi-4-gguf", "phi-4-Q4_K_S.gguf"),
]

# Transformer models (full downloads)
transformer_models = [
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
]

# ============================================
# DESCARGAR GGUF FILES
# ============================================
print("=" * 60)
print("Descargando GGUF files...")
print("=" * 60)

from huggingface_hub import hf_hub_download

for repo_id, filename in gguf_models:
    try:
        print(f"\n📥 {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"   ✓ Guardado en: {path}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        sys.exit(1)

# ============================================
# DESCARGAR TRANSFORMER MODELS
# ============================================
print("\n" + "=" * 60)
print("Descargando Transformer models...")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM

for model_name in transformer_models:
    try:
        print(f"\n📥 {model_name}")
        print("   Descargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✓ Tokenizer OK")
        
        print("   Descargando modelo...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"   ✓ Modelo OK")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        sys.exit(1)

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 60)
print("✓ TODOS LOS MODELOS DESCARGADOS")
print("=" * 60)
print(f"\n📁 Ubicación: {MODELS_DIR}")
print(f"\n📝 Agregar esto a tu sbatch (submit_job.sbatch):")
print(f"\n   export HF_HOME={MODELS_DIR}\n")
