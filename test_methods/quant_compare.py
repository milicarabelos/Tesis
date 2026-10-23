#!/usr/bin/env python
"""
Carga un modelo de Hugging Face con distintos métodos de cuantización, le hace una
pregunta simple y guarda memoria, tokens/s y la configuración usada.

Métodos (--methods, separados por coma):
  base      sin cuantizar (fp16, como el paper de AWQ)
  bnb-nf4   bitsandbytes 4 bits NF4 (blockwise 64, double quant, compute en fp16)
  bnb-int8  bitsandbytes LLM.int8() (absmax por fila + descomposición de outliers)
  rtn       checkpoint de quantize_awq.py: INT4 g128 con zero point, round-to-nearest (formato AWQ GEMM)
  awq       checkpoint de quantize_awq.py: misma cuantización + escalas/clipping calibrados como el paper

Cada método corre en un subproceso propio para que la memoria de uno no contamine al otro.

Ejemplo:
  python test_methods/quantize_awq.py --model meta-llama/Llama-2-7b-hf
  python test_methods/quant_compare.py --model meta-llama/Llama-2-7b-hf --methods base,bnb-nf4,rtn,awq
"""
import argparse
import gc
import importlib.metadata as md
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

GB = 1024**3
METHODS = ["base", "bnb-nf4", "bnb-int8", "rtn", "awq"]


class DeviceMemMonitor(threading.Thread):
    """Pico de memoria a nivel dispositivo (≈ nvidia-smi / rocm-smi) menos el baseline. Igual que en run_eval.py."""

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.baseline = self._used()
        self.peak = 0
        self._stop_evt = threading.Event()

    @staticmethod
    def _used():
        return sum(t - f for f, t in (torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())))

    def sample(self):
        self.peak = max(self.peak, self._used() - self.baseline)

    def run(self):
        while not self._stop_evt.wait(self.interval):
            self.sample()

    def stop(self):
        self._stop_evt.set()
        self.join()


def sync():
    torch.cuda.synchronize()


def alloc_gb():
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / GB


def peak_alloc_gb():
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / GB


def reset_peak(monitor):
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    monitor.sample()
    monitor.peak = monitor._used() - monitor.baseline


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def quant_config(method, awq_backend):
    from transformers import AwqConfig, BitsAndBytesConfig
    if method in ("rtn", "awq"):
        # el resto de la config viene del checkpoint; esto solo elige el kernel de gptqmodel
        return AwqConfig(bits=4, backend=awq_backend) if awq_backend else None
    if method == "bnb-nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    if method == "bnb-int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


@torch.inference_mode()
def run_one(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method = args.method
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm no disponible.")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if (total - free) / GB > args.max_busy_gb:
            raise RuntimeError(f"GPU {i} ya tiene {(total - free) / GB:.1f} GiB ocupados de {total / GB:.1f} GiB "
                               "(otro proceso); las medidas de memoria no serían válidas. "
                               "Revisá rocm-smi --showpids / nvidia-smi o usá una GPU exclusiva.")
    monitor = DeviceMemMonitor()
    monitor.start()

    model_id = args.model
    if method in ("rtn", "awq"):
        model_id = str(Path(args.quantized_dir) / f"{args.model.rstrip('/').split('/')[-1]}-{method}")
        if not Path(model_id).is_dir():
            raise FileNotFoundError(f"{model_id} no existe; corré quantize_awq.py --model {args.model}")
    qc = quant_config(method, args.awq_backend)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": 0},
        **({"quantization_config": qc} if qc is not None else {}),
    )
    model.eval()
    qconf = getattr(model.config, "quantization_config", None)
    qconf = None if qconf is None else json.loads(json.dumps(
        qconf if isinstance(qconf, dict) else qconf.to_dict(), default=str))
    q_proj = model.model.layers[0].self_attn.q_proj
    linear_impl = f"{type(q_proj).__module__}.{type(q_proj).__name__}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sync()
    load_time = time.perf_counter() - t0
    monitor.sample()
    mem_after_load = {
        "footprint_gb": model.get_memory_footprint() / GB,
        "alloc_after_load_gb": alloc_gb(),
        "peak_device_load_gb": monitor.peak / GB,
    }

    if args.chat:
        if not tokenizer.chat_template:
            raise ValueError(f"--chat pero {model_id} no tiene chat template")
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                  return_dict=True).input_ids
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n_new = args.max_new_tokens

    def gen(n, fixed=True):
        return model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=n,
                              min_new_tokens=n if fixed else None, do_sample=False, pad_token_id=pad_id)

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        return time.perf_counter() - t, out

    def median(fn):
        return sorted(timed(fn)[0] for _ in range(args.repeats))[args.repeats // 2]

    timed(lambda: gen(n_new))  # warmup (compilación de kernels, autotune)
    reset_peak(monitor)

    # respuesta natural (puede cortar por EOS) para guardar el texto
    t_answer, out = timed(lambda: gen(n_new, fixed=False))
    answer_ids = out[0, input_ids.shape[1]:]
    # largo fijo para que tok/s sea comparable entre métodos
    t_gen1 = median(lambda: gen(1))
    t_genN = median(lambda: gen(n_new))
    monitor.sample()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "model": model_id,
        "dtype": "float16",
        "quantization_config": qconf,
        "linear_impl": linear_impl,
        "prompt": args.prompt,
        "chat": args.chat,
        "prompt_tokens": input_ids.shape[1],
        "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
        "answer_tokens": len(answer_ids),
        "answer_tok_per_s": len(answer_ids) / t_answer,
        "new_tokens": n_new,
        "ttft_ms": t_gen1 * 1000,
        "decode_tok_per_s": (n_new - 1) / (t_genN - t_gen1),
        "e2e_tok_per_s": n_new / t_genN,
        "load_time_s": load_time,
        **mem_after_load,
        "peak_alloc_gen_gb": peak_alloc_gb(),
        "peak_device_gen_gb": monitor.peak / GB,
        "device_name": torch.cuda.get_device_name(0),
        "backend": f"rocm {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}",
        "versions": {p: version(p) for p in ["torch", "transformers", "accelerate", "bitsandbytes",
                                             "gptqmodel", "triton", "pytorch-triton-rocm"]},
    }
    monitor.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("versions", "quantization_config")},
                     ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quantized-dir", default=str(Path(__file__).resolve().parent / "quantized"),
                   help="donde quantize_awq.py dejó <modelo>-rtn y <modelo>-awq")
    p.add_argument("--methods", default="base,bnb-nf4,rtn,awq")
    p.add_argument("--method", help=argparse.SUPPRESS)  # uso interno: un método por subproceso
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--chat", action="store_true", help="aplica el chat template (modelos instruct)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--awq-backend", help="fuerza el backend de gptqmodel para rtn/awq (ver check_awq_env.py)")
    p.add_argument("--max-busy-gb", type=float, default=1.5,
                   help="memoria de GPU ya ocupada al arrancar (incluye el contexto propio) a partir de la cual se aborta")
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "quant_compare.jsonl"))
    args = p.parse_args()

    if args.method:
        return run_one(args)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        p.error(f"métodos desconocidos: {unknown}; válidos: {METHODS}")
    failed = []
    for m in methods:
        print(f"\n=== {m} ===", flush=True)
        if subprocess.run([sys.executable, __file__, *sys.argv[1:], "--method", m]).returncode:
            failed.append(m)
    if failed:
        sys.exit(f"fallaron: {failed}")


if __name__ == "__main__":
    main()
