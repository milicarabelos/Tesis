#!/usr/bin/env python
"""
Diagnóstico de cómo transformers ejecuta un checkpoint AWQ en esta máquina.

transformers 5.x carga AWQ con gptqmodel, que elige un QuantLinear según el backend
(config "backend", default "auto"). Este script prueba cada backend de gptqmodel con el
checkpoint dado y muestra qué clase de capa quedó y cuánto tarda el decode.
Que un backend falle acá es información, no un error del script.

Ejemplo:
  python test_methods/check_awq_env.py --model test_methods/quantized/Llama-2-7b-hf-rtn
"""
import argparse
import gc
import importlib.metadata as md
import time

import torch

# backends que transformers acepta para AWQ (según su ValueError en 5.17/5.19)
AWQ_BACKENDS = ["auto", "gemm", "gemm_triton", "gemv", "gemv_fast", "torch_awq", "torch_fused_awq",
                "exllama_v2", "exllama_v1", "marlin", "machete"]


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def linear_impl(model):
    m = model.model.layers[0].self_attn.q_proj
    return f"{type(m).__module__}.{type(m).__name__}"


@torch.inference_mode()
def decode_tok_s(model, tokenizer, n=16):
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(model.device)
    kw = dict(do_sample=False, pad_token_id=tokenizer.eos_token_id)
    model.generate(ids, max_new_tokens=2, min_new_tokens=2, **kw)  # warmup
    torch.cuda.synchronize()
    t = time.perf_counter()
    model.generate(ids, max_new_tokens=n, min_new_tokens=n, **kw)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint AWQ (ej. quantized/<modelo>-rtn)")
    p.add_argument("--backends", default=",".join(AWQ_BACKENDS), help="lista separada por comas")
    args = p.parse_args()

    print("torch", torch.__version__, "| hip", torch.version.hip, "| cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    if torch.version.hip:
        print("gcnArchName", torch.cuda.get_device_properties(0).gcnArchName)
    for pkg in ["transformers", "gptqmodel", "triton", "pytorch-triton-rocm", "autoawq", "autoawq-kernels",
                "optimum", "accelerate", "kernels"]:
        print(f"{pkg:>20}: {version(pkg)}")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AwqConfig

    qc = getattr(AutoConfig.from_pretrained(args.model), "quantization_config", None) or {}
    qc = qc if isinstance(qc, dict) else qc.to_dict()
    if qc.get("quant_method") != "awq":
        raise ValueError(f"{args.model} no es un checkpoint AWQ (quant_method={qc.get('quant_method')}); "
                         "pasá quantized/<modelo>-rtn o -awq")
    backends = args.backends.split(",")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for b in backends:
        print(f"\n--- backend={b}", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, device_map={"": 0},
                quantization_config=AwqConfig(bits=4, backend=b))
            print("  capa:", linear_impl(model))
            print(f"  decode: {decode_tok_s(model, tokenizer):.2f} tok/s")
        except Exception as e:
            print(f"  FALLA: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
