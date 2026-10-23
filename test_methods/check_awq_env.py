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
    p.add_argument("--backends", help="lista separada por comas; default: todos los de gptqmodel.BACKEND")
    args = p.parse_args()

    print("torch", torch.__version__, "| hip", torch.version.hip, "| cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    if torch.version.hip:
        print("gcnArchName", torch.cuda.get_device_properties(0).gcnArchName)
    for pkg in ["transformers", "gptqmodel", "triton", "pytorch-triton-rocm", "autoawq", "autoawq-kernels",
                "optimum", "accelerate", "kernels"]:
        print(f"{pkg:>20}: {version(pkg)}")

    from gptqmodel.utils.backend import BACKEND
    from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

    backends = args.backends.split(",") if args.backends else [b.value for b in BACKEND]
    print("\nbackends de gptqmodel:", backends)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for b in backends:
        print(f"\n--- backend={b}")
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
