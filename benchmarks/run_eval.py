#!/usr/bin/env python
"""
Evalúa una lista de modelos en una lista de benchmarks con lm-evaluation-harness
y deja un Excel con: tiempo de carga, tiempo de inferencia y valor de cada métrica.

Variables de entorno (las define el sbatch):
  MODELS        lista separada por espacios. Cada entrada puede ser:
                  - un id/path de modelo:   meta-llama/Llama-3.1-8B-Instruct
                  - un model_args completo: pretrained=/ruta/base,peft=/ruta/adapter,load_in_4bit=True
                    (sin espacios; se reconoce porque contiene "=")
                El modelo se carga con transformers (ver load_hf) y se le pasa ya
                instanciado a lm-eval. Las claves que no son de HFLM_KEYS van directo
                a from_pretrained (ej. attn_implementation=sdpa).
  BENCHMARKS    nombres de tareas de lm-eval, separados por espacios (boolq wic openbookqa sqac)
                incluye las tasks propias de benchmarks/tasks/ (ej. sqac)
  OUTPUT_DIR    carpeta de salida (default: results)
  NUM_FEWSHOT   default 0
  BATCH_SIZE    default 8 (puede ser "auto")
  DTYPE         default bfloat16 (se ignora si el model_args ya trae dtype=)
  APPLY_CHAT_TEMPLATE  "1" para activarlo (default 0)
  LIMIT         nº de ejemplos por tarea, solo para smoke tests (default: todos)
  PERF          "0" para saltear el micro-benchmark de inferencia (default 1)
  PERF_PROMPT_TOKENS / PERF_NEW_TOKENS / PERF_REPEATS
                largo del prompt, tokens generados y repeticiones del
                micro-benchmark (default 512 / 128 / 3)

Argumentos:
  --force NAME  ejecuta una nueva variante en <OUTPUT_DIR>_<NAME> y la guarda
                como resultados_<NAME>.xlsx
"""
import argparse
import gc
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string

# tasks propias (ej. sqac); se resuelve relativo al script, no al cwd
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:150]


def to_model_args(entry: str) -> str:
    return entry if "=" in entry else f"pretrained={entry}"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


GB = 1024**3


class DeviceMemMonitor(threading.Thread):
    """Muestrea la memoria usada de las GPUs a nivel dispositivo y guarda el pico.

    torch.cuda.mem_get_info usa cudaMemGetInfo en NVIDIA y hipMemGetInfo en AMD,
    así que es la misma medida en ambos. Se resta el baseline tomado al arrancar
    (contexto CUDA/HIP + otros procesos) para quedarse con lo que agrega el modelo.
    """

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

    def reset(self):
        self.peak = 0
        self.sample()

    def stop(self):
        self._stop_evt.set()
        self.join()


DEVICE_MEM = None  # se inicializa en main si hay GPU


def reset_peak_mem():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()  # si no, la caché de la task anterior infla el pico de dispositivo
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        if DEVICE_MEM:
            DEVICE_MEM.reset()


def mem_stats():
    """Memoria en GB sumada sobre todas las GPUs.

    - alloc: tensores vivos según PyTorch; comparable entre NVIDIA y AMD.
    - device: memoria usada del dispositivo menos baseline (≈ nvidia-smi / rocm-smi);
      incluye caché del allocator y workspaces de cuBLAS/hipBLAS.
    """
    if not torch.cuda.is_available():
        return {}
    n = range(torch.cuda.device_count())
    if DEVICE_MEM:
        DEVICE_MEM.sample()
    return {
        "mem_alloc_gb": sum(torch.cuda.memory_allocated(i) for i in n) / GB,
        "peak_mem_alloc_gb": sum(torch.cuda.max_memory_allocated(i) for i in n) / GB,
        "peak_mem_device_gb": DEVICE_MEM.peak / GB if DEVICE_MEM else None,
    }


# claves del model_args que van a HFLM y no a from_pretrained
HFLM_KEYS = {"max_length", "add_bos_token", "truncation", "max_batch_size"}


def load_hf(entry, cfg):
    """Carga modelo + tokenizer con transformers. Acá se puede customizar la config de carga."""
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = simple_parse_args_string(to_model_args(entry))
    pretrained = args.pop("pretrained")
    # si hay un adapter PEFT como Lora, se carga después de cargar el modelo base
    peft_path = args.pop("peft", None)
    tok_name = args.pop("tokenizer", pretrained)
    hflm_kwargs = {k: args.pop(k) for k in list(args) if k in HFLM_KEYS}

    dtype = getattr(torch, args.pop("dtype", cfg["dtype"]))
    if args.pop("load_in_4bit", False):
        args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    elif args.pop("load_in_8bit", False):
        args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    args.setdefault("device_map", {"": 0} if cfg["device"] == "cuda" else "cpu")

    model = AutoModelForCausalLM.from_pretrained(pretrained, **{dtype_kw: dtype}, **args)
    if peft_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=args.get("trust_remote_code", False))
    return model, tokenizer, hflm_kwargs


def gpu_info():
    if not torch.cuda.is_available():
        return {"backend": "cpu", "device_name": None}
    # torch con ROCm expone la misma API torch.cuda; torch.version.hip distingue el build
    return {
        "backend": f"rocm {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}",
        "device_name": torch.cuda.get_device_name(0),
    }


@torch.inference_mode()
def micro_benchmark(model, tokenizer, cfg):
    """Inferencia con prompt y largo de salida fijos, batch 1, greedy.

    - prefill_tok_per_s: prompt_tokens / tiempo de un forward sobre el prompt.
    - decode_tok_per_s: (N-1) / (t_generate(N) - t_generate(1)); aísla el decode del prefill.
    Se fuerza min_new_tokens=N para que un EOS temprano no acorte la salida.
    """
    n_prompt, n_new, reps = cfg["perf_prompt_tokens"], cfg["perf_new_tokens"], cfg["perf_repeats"]
    device = next(model.parameters()).device
    base = tokenizer("El procesamiento de lenguaje natural estudia cómo las computadoras entienden texto. ",
                     add_special_tokens=False).input_ids
    ids = (base * (n_prompt // len(base) + 1))[:n_prompt]
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def timed(fn):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0

    def gen(n):
        return lambda: model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=n,
                                      min_new_tokens=n, do_sample=False, pad_token_id=pad_id)

    forward = lambda: model(input_ids=input_ids, attention_mask=attn)
    timed(forward)  # warmup
    timed(gen(n_new))
    reset_peak_mem()

    def median(fn):
        return sorted(timed(fn) for _ in range(reps))[reps // 2]

    t_prefill = median(forward)
    t_gen1 = median(gen(1))
    t_genN = median(gen(n_new))
    mem = mem_stats()
    return {
        **gpu_info(),
        "perf_prompt_tokens": n_prompt,
        "perf_new_tokens": n_new,
        "ttft_ms": t_gen1 * 1000,
        "prefill_tok_per_s": n_prompt / t_prefill,
        "decode_tok_per_s": (n_new - 1) / (t_genN - t_gen1),
        "e2e_tok_per_s": n_new / t_genN,
        "perf_peak_mem_alloc_gb": mem.get("peak_mem_alloc_gb"),
        "perf_peak_mem_device_gb": mem.get("peak_mem_device_gb"),
    }

def load_all_raw(raw_dir):
    """Lee todos los JSON de raw/ y junta sus filas y sus resultados de performance."""
    rows, perfs = [], []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            rows += data.get("rows", [])
            if data.get("perf"):
                perfs.append(data["perf"])
        except Exception as e:
            print(f"[warn] no se pudo leer {p.name}: {e}", file=sys.stderr, flush=True)
    return rows, perfs

def rows_from_results(model_label, task, res, n_samples, load_time, eval_time, extra=None):
    """Convierte el dict de resultados de lm-eval en filas (una por métrica)."""
    benchmark_time_per_sample = (eval_time / n_samples) if eval_time and n_samples else None
    rows = []
    for key, val in res.items():
        if key == "alias" or not isinstance(val, (int, float)):
            continue
        name, _, flt = key.partition(",")
        if name.endswith("_stderr"):
            continue
        rows.append(
            {
                "model": model_label,
                "task": task,
                "metric": name,
                "filter": flt,
                "value": val,
                "stderr": res.get(f"{name}_stderr,{flt}"),
                "n_samples": n_samples,
                "load_time_s": load_time,
                "benchmark_len": n_samples,
                "benchmark_time_s": eval_time,
                "benchmark_time_per_sample_s": benchmark_time_per_sample,
                "total_time_s": eval_time,
                "time_per_sample_s": benchmark_time_per_sample,
                "samples_per_s": (n_samples / eval_time) if eval_time and n_samples else None,
                **(extra or {}),
            }
        )
    return rows


def evaluate_model(entry, tasks, cfg, raw_dir, task_manager, force=False):
    label = entry
    raw_path = raw_dir / f"{slug(label)}.json"
    done = json.loads(raw_path.read_text()) if raw_path.exists() else {"rows": [], "tasks_done": []}

    pending = [t for t in tasks if t not in done["tasks_done"]]
    need_perf = cfg["perf"] and not done.get("perf")
    if not pending and not need_perf:
        print(f"[skip] {label}: ya evaluado", flush=True)
        return done["rows"]

    # ---- carga del modelo (se mide aparte) ----
    print(f"[load] {to_model_args(entry)}", flush=True)
    reset_peak_mem()
    sync()
    t0 = time.perf_counter()
    model, tokenizer, hflm_kwargs = load_hf(entry, cfg)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=cfg["batch_size"], **hflm_kwargs)
    sync()
    load_time = time.perf_counter() - t0
    load_mem = mem_stats()
    load_extra = {
        "load_mem_gb": load_mem.get("mem_alloc_gb"),
        "load_peak_mem_gb": load_mem.get("peak_mem_alloc_gb"),
        "load_peak_mem_device_gb": load_mem.get("peak_mem_device_gb"),
    }
    print(f"model [{label}] charged in [load]:[{load_time:.1f}s] {load_extra}", flush=True)

    # ---- micro-benchmark de inferencia (una vez por modelo) ----
    if need_perf:
        done["perf"] = {"model": label, "load_time_s": load_time, **load_extra,
                        **micro_benchmark(model, tokenizer, cfg)}
        raw_path.write_text(json.dumps(done, indent=2))
        print(f"[perf] {done['perf']}", flush=True)

    # ---- una tarea por vez para medir tiempo de cada benchmark ----
    for task in pending:
        print(f"[eval] {label} :: {task}", flush=True)
        reset_peak_mem()
        sync()
        t0 = time.perf_counter()
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            task_manager=task_manager,
            num_fewshot=cfg["num_fewshot"],
            limit=cfg["limit"],
            apply_chat_template=cfg["chat"],
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
        sync()
        eval_time = time.perf_counter() - t0

        mem = mem_stats()
        extra = {
            **load_extra,
            "peak_mem_alloc_gb": mem.get("peak_mem_alloc_gb"),
            "peak_mem_device_gb": mem.get("peak_mem_device_gb"),
        }
        n = out.get("n-samples", {}).get(task, {}).get("effective")
        done["rows"] += rows_from_results(label, task, out["results"][task], n, load_time, eval_time, extra)
        done["tasks_done"].append(task)
        raw_path.write_text(json.dumps(done, indent=2))  # guardado incremental
        print(f"[eval] {task}: {eval_time:.1f}s {extra}", flush=True)

    del lm, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return done["rows"]


def write_excel(rows, perfs, path):
    df = pd.DataFrame(rows)
    if df.empty:
        print("Sin resultados, no se genera Excel.")
        return

    df["task_metric"] = df["task"] + " | " + df["metric"]
    summary = df.pivot_table(index="model", columns="task_metric", values="value", aggfunc="first")

    benchmark_df = df.drop_duplicates(["model", "task"]).copy()
    benchmark_df["benchmark_time_per_sample_s"] = (
        benchmark_df["benchmark_time_s"] / benchmark_df["benchmark_len"]
    )

    benchmark_summary = benchmark_df.set_index(["model", "task"])[
        ["benchmark_len", "benchmark_time_s", "benchmark_time_per_sample_s"]
    ]
    benchmark_summary = benchmark_summary.unstack("task")
    benchmark_summary.columns = [
        f"{task} | {'sample_len' if metric == 'benchmark_len' else 'time' if metric == 'benchmark_time_s' else 'time_per_sample'}"
        for metric, task in benchmark_summary.columns
    ]

    load = df.drop_duplicates("model").set_index("model")["load_time_s"].rename("load_time_s")
    timing = pd.concat([load, benchmark_summary], axis=1)
    timing["total_eval_s"] = benchmark_df.pivot_table(index="model", values="benchmark_time_s", aggfunc="sum")

    cols = [
        "model", "task", "metric", "filter", "value", "stderr", "n_samples",
        "load_time_s", "benchmark_time_s", "benchmark_len", "benchmark_time_per_sample_s",
        "total_time_s", "time_per_sample_s", "samples_per_s",
        "load_mem_gb", "load_peak_mem_gb", "load_peak_mem_device_gb", "peak_mem_alloc_gb", "peak_mem_device_gb",
    ]
    cols = [c for c in cols if c in df.columns]  # raws viejos no tienen las columnas nuevas
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.reset_index().to_excel(xw, sheet_name="resumen", index=False)
        timing.reset_index().to_excel(xw, sheet_name="tiempos", index=False)
        df[cols].to_excel(xw, sheet_name="detalle", index=False)
        performance_sheet(df, perfs).to_excel(xw, sheet_name="performance", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)

def performance_sheet(df, perfs):
    """Una fila por modelo: memoria, carga y micro-benchmark de inferencia."""
    perf = pd.DataFrame(perfs)
    if "peak_mem_alloc_gb" in df.columns:
        task_peak = (
            df.groupby("model")[["peak_mem_alloc_gb", "peak_mem_device_gb"]].max()
            .rename(columns=lambda c: f"tasks_max_{c}").reset_index()
        )
        perf = task_peak if perf.empty else perf.merge(task_peak, on="model", how="outer")
    return perf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        metavar="NAME",
        help="ejecuta una variante nueva en <OUTPUT_DIR>_<NAME>",
    )
    args = parser.parse_args()

    models = os.environ["MODELS"].split()
    tasks = os.environ["BENCHMARKS"].split()
    out_dir = Path(os.environ.get("OUTPUT_DIR", "results"))
    output_suffix = slug(args.force).replace("_", "") if args.force else None
    if output_suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{output_suffix}"
    excel_name = f"resultados_{output_suffix}.xlsx" if output_suffix else "resultados.xlsx"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        ## borrar los raw de la variante anterior para forzar re-evaluación
        for raw_path in raw_dir.glob("*.json"):
            raw_path.unlink()

    limit = os.environ.get("LIMIT")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; este script requiere una GPU CUDA.")

    cfg = {
        "num_fewshot": int(os.environ.get("NUM_FEWSHOT", 0)),
        "batch_size": os.environ.get("BATCH_SIZE", "8"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "chat": os.environ.get("APPLY_CHAT_TEMPLATE", "0") == "1",
        "limit": int(limit) if limit else None,
        "device": "cuda",
        "perf": os.environ.get("PERF", "1") == "1",
        "perf_prompt_tokens": int(os.environ.get("PERF_PROMPT_TOKENS", 512)),
        "perf_new_tokens": int(os.environ.get("PERF_NEW_TOKENS", 128)),
        "perf_repeats": int(os.environ.get("PERF_REPEATS", 3)),
    }
    print(f"lm_eval {getattr(lm_eval, '__version__', '?')} | cfg={cfg}", flush=True)
    print(f"models={models}\ntasks={tasks}", flush=True)

    global DEVICE_MEM
    DEVICE_MEM = DeviceMemMonitor()
    DEVICE_MEM.start()
    print(f"[gpu] {gpu_info()} baseline={DEVICE_MEM.baseline / GB:.2f}GB", flush=True)

    task_manager = TaskManager(include_path=str(TASKS_DIR))
    for entry in models:
        try:
            evaluate_model(entry, tasks, cfg, raw_dir, task_manager, force=bool(args.force))
        except Exception as e:  # un modelo que falla no tumba el resto
            print(f"[ERROR] {entry}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    DEVICE_MEM.stop()
    rows, perfs = load_all_raw(raw_dir)
    write_excel(rows, perfs, out_dir / excel_name)
    print(f"Listo: {out_dir / excel_name}", flush=True)


if __name__ == "__main__":
    main()
