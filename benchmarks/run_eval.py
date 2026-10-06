#!/usr/bin/env python
"""
Evalúa una lista de modelos en una lista de benchmarks con lm-evaluation-harness
y deja un Excel con: tiempo de carga, tiempo de inferencia y valor de cada métrica.

Variables de entorno (las define el sbatch):
  MODELS        lista separada por espacios. Cada entrada puede ser:
                  - un id/path de modelo:   meta-llama/Llama-3.1-8B-Instruct
                  - un model_args completo: pretrained=/ruta/base,peft=/ruta/adapter,load_in_4bit=True
                    (sin espacios; se reconoce porque contiene "=")
  BENCHMARKS    nombres de tareas de lm-eval, separados por espacios (boolq wic openbookqa)
  OUTPUT_DIR    carpeta de salida (default: results)
  NUM_FEWSHOT   default 0
  BATCH_SIZE    default 8 (puede ser "auto")
  DTYPE         default bfloat16 (se ignora si el model_args ya trae dtype=)
  APPLY_CHAT_TEMPLATE  "1" para activarlo (default 0)
  LIMIT         nº de ejemplos por tarea, solo para smoke tests (default: todos)
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import lm_eval
from lm_eval.api.registry import get_model


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:150]


def to_model_args(entry: str) -> str:
    return entry if "=" in entry else f"pretrained={entry}"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rows_from_results(model_label, task, res, n_samples, load_time, eval_time):
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
            }
        )
    return rows


def evaluate_model(entry, tasks, cfg, raw_dir):
    label = entry
    raw_path = raw_dir / f"{slug(label)}.json"
    done = json.loads(raw_path.read_text()) if raw_path.exists() else {"rows": [], "tasks_done": []}

    pending = [t for t in tasks if t not in done["tasks_done"]]
    if not pending:
        print(f"[skip] {label}: ya evaluado", flush=True)
        return done["rows"]

    # ---- carga del modelo (se mide aparte) ----
    model_args = to_model_args(entry)
    if "dtype=" not in model_args:
        model_args += f",dtype={cfg['dtype']}"
    print(f"[load] {model_args}", flush=True)
    sync()
    t0 = time.perf_counter()
    lm = get_model("hf").create_from_arg_string(
        model_args, {"batch_size": cfg["batch_size"], "device": cfg["device"]}
    )
    sync()
    load_time = time.perf_counter() - t0
    print(f"model [{label}] charged in [load]:[{load_time:.1f}s]", flush=True)

    # ---- una tarea por vez para medir tiempo de cada benchmark ----
    for task in pending:
        print(f"[eval] {label} :: {task}", flush=True)
        sync()
        t0 = time.perf_counter()
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
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

        n = out.get("n-samples", {}).get(task, {}).get("effective")
        done["rows"] += rows_from_results(label, task, out["results"][task], n, load_time, eval_time)
        done["tasks_done"].append(task)
        raw_path.write_text(json.dumps(done, indent=2))  # guardado incremental
        print(f"[eval] {task}: {eval_time:.1f}s", flush=True)

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return done["rows"]


def write_excel(rows, path):
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
        "total_time_s", "time_per_sample_s", "samples_per_s"
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.reset_index().to_excel(xw, sheet_name="resumen", index=False)
        timing.reset_index().to_excel(xw, sheet_name="tiempos", index=False)
        df[cols].to_excel(xw, sheet_name="detalle", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)

def main():
    models = os.environ["MODELS"].split()
    tasks = os.environ["BENCHMARKS"].split()
    out_dir = Path(os.environ.get("OUTPUT_DIR", "results"))
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    limit = os.environ.get("LIMIT")
    cfg = {
        "num_fewshot": int(os.environ.get("NUM_FEWSHOT", 0)),
        "batch_size": os.environ.get("BATCH_SIZE", "8"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "chat": os.environ.get("APPLY_CHAT_TEMPLATE", "0") == "1",
        "limit": int(limit) if limit else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(f"lm_eval {getattr(lm_eval, '__version__', '?')} | cfg={cfg}", flush=True)
    print(f"models={models}\ntasks={tasks}", flush=True)

    all_rows = []
    for entry in models:
        try:
            all_rows += evaluate_model(entry, tasks, cfg, raw_dir)
        except Exception as e:  # un modelo que falla no tumba el resto
            print(f"[ERROR] {entry}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_excel(all_rows, out_dir / "resultados.xlsx")
    print(f"Listo: {out_dir / 'resultados.xlsx'}", flush=True)


if __name__ == "__main__":
    main()
