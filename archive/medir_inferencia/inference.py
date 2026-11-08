import os
import time
import torch

# --------------------------------------------------
# Config vía variables de entorno
# --------------------------------------------------
# Ejemplos de uso:
#   model_name="Doub7e/llama-3-8b-dpo-distilabel-epfl-sft-bnb8bit" python inference.py
#   model_name="unsloth/Qwen3.5-27B-GGUF" gguf_file="Qwen3.5-27B-Q4_K_M.gguf" python inference.py
#
# Si "gguf_file" viene vacío/no seteado -> se carga con transformers (comportamiento original).
# Si "gguf_file" tiene valor            -> se carga con llama-cpp-python.

MODEL_NAME = os.environ.get(
    "model_name",
    "Doub7e/llama-3-8b-dpo-distilabel-epfl-sft-bnb8bit",
)
GGUF_FILE = os.environ.get("gguf_file", "").strip()

PROMPT = "Explain what a supercomputer is in simple terms."
MAX_NEW_TOKENS = 100
WARMUP_TOKENS = 20


# --------------------------------------------------
# Chequeo de GPU (sin fallback a CPU)
# --------------------------------------------------
def require_gpu():
    # torch.cuda.is_available() también es True en builds de torch con ROCm/HIP,
    # ya que ROCm expone el mismo namespace "cuda" en PyTorch.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No se detectó GPU (torch.cuda.is_available() == False). "
            "Abortando: este script no debe correr en CPU."
        )


# --------------------------------------------------
# Loaders
# --------------------------------------------------
def load_model_transformers(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    require_gpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},  # fuerza TODO a cuda:0; si no entra, tira OOM en vez de offloadear a CPU
    )
    model.eval()

    # Verificación explícita: si algún parámetro terminó en CPU, abortar.
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cuda"}:
        raise RuntimeError(
            f"El modelo no quedó completamente en GPU (dispositivos encontrados: {devices}). "
            "Abortando en vez de correr parcialmente en CPU."
        )

    return {"backend": "transformers", "tokenizer": tokenizer, "model": model}


def load_model_llama_cpp(model_name, gguf_file):
    # Requiere: pip install llama-cpp-python huggingface_hub
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    require_gpu()

    # Chequeo de que el binario de llama.cpp fue compilado con soporte GPU
    # (CUDA/HIP/Metal). Si esto da False, el build no tiene GPU offload real
    # y n_gpu_layers=-1 correría todo en CPU silenciosamente.
    try:
        from llama_cpp import llama_cpp as _llama_cpp_low

        if not _llama_cpp_low.llama_supports_gpu_offload():
            raise RuntimeError(
                "llama-cpp-python fue instalado SIN soporte GPU "
                "(llama_supports_gpu_offload() == False). "
                "Reinstalalo con CMAKE_ARGS=\"-DGGML_HIP=on ...\" (ver pasos anteriores)."
            )
    except ImportError:
        raise RuntimeError(
            "No se pudo verificar el binding de bajo nivel de llama-cpp-python "
            "para confirmar soporte GPU. Abortando por seguridad."
        )

    model_path = hf_hub_download(repo_id=model_name, filename=gguf_file)
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # forzado: todas las capas a GPU, sin fallback parcial
        verbose=False,     # deja ver en el log cuántas capas se offloadearon
    )
    return {"backend": "llama_cpp", "llm": llm}


# --------------------------------------------------
# Inference
# --------------------------------------------------
def run_inference_transformers(handle):
    tokenizer = handle["tokenizer"]
    model = handle["model"]

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    inference_time = end - start
    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.shape[1] - input_tokens
    tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0.0
    response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)

    return {
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "inference_time": inference_time,
        "tokens_per_second": tokens_per_second,
    }


def run_inference_llama_cpp(handle):
    llm = handle["llm"]

    # Warm-up
    _ = llm(PROMPT, max_tokens=WARMUP_TOKENS, echo=False)

    start = time.perf_counter()
    output = llm(PROMPT, max_tokens=MAX_NEW_TOKENS, echo=False)
    end = time.perf_counter()

    inference_time = end - start
    response = output["choices"][0]["text"]
    usage = output.get("usage", {})
    input_tokens = usage.get("prompt_tokens", len(llm.tokenize(PROMPT.encode())))
    output_tokens = usage.get("completion_tokens", 0)
    tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0.0

    return {
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "inference_time": inference_time,
        "tokens_per_second": tokens_per_second,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print(f"Modelo:     {MODEL_NAME}")
    print(f"GGUF file:  {GGUF_FILE if GGUF_FILE else '(no aplica, backend=transformers)'}")
    print("Loading model...")

    if GGUF_FILE:
        handle = load_model_llama_cpp(MODEL_NAME, GGUF_FILE)
    else:
        handle = load_model_transformers(MODEL_NAME)

    print("Model loaded.")

    if handle["backend"] == "transformers":
        stats = run_inference_transformers(handle)
    else:
        stats = run_inference_llama_cpp(handle)

    print("\n" + "=" * 50)
    print("Response:")
    print("=" * 50)
    print(stats["response"])

    print("\n" + "=" * 50)
    print("Inference statistics")
    print("=" * 50)
    print(f"Model:             {MODEL_NAME}")
    print(f"Backend:           {handle['backend']}")
    print(f"Input tokens:      {stats['input_tokens']}")
    print(f"Output tokens:     {stats['output_tokens']}")
    print(f"Inference time:    {stats['inference_time']:.4f} s")
    print(f"Tokens / second:   {stats['tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()
