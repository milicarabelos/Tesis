# run_total.py
import json
import time
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

MODEL_PATH = "./models/llama31-8b-instruct"
INCLUDE_PATH = "./mis_tasks"
SHOTS = [0, 5, 10]
OUT = Path("results/run_total.json")

OUT.parent.mkdir(parents=True, exist_ok=True)

# Reanudar si ya existe un archivo previo
if OUT.exists():
    salida = json.loads(OUT.read_text())
    hechos = {e["nombre"] for e in salida["experimentos"]}
    print("Ya completados:", sorted(hechos))
else:
    salida = {
        "meta": {
            "model": MODEL_PATH,
            "limit": None,
            "inicio": datetime.now().isoformat(timespec="seconds"),
            "seeds": "defaults del harness (random 0, numpy 1234, torch 1234, fewshot 1234)",
        },
        "experimentos": [],
    }
    hechos = set()

lm = HFLM(pretrained=MODEL_PATH, dtype="bfloat16", batch_size=1)
task_manager = TaskManager(include_path=INCLUDE_PATH)

for chat in (False, True):
    for n in SHOTS:
        nombre = f"{'chat' if chat else 'nochat'}_{n}shot"
        if nombre in hechos:
            print(f"--- {nombre}: ya hecho, se salta")
            continue

        print(f"\n>>> {nombre}")
        t0 = time.time()

        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=["sqac"],
            num_fewshot=n,
            limit=None,                  # test completo
            apply_chat_template=chat,
            fewshot_as_multiturn=chat,
            task_manager=task_manager,
            log_samples=False,           # solo métricas
        )

        metricas = {
            k.split(",")[0]: v
            for k, v in res["results"]["sqac"].items()
            if "," in k and "stderr" not in k
        }
        n_docs = res["n-samples"]["sqac"]["effective"]

        salida["experimentos"].append({
            "nombre": nombre,
            "chat_template": chat,
            "num_fewshot": n,
            "n_docs": n_docs,
            "segundos": round(time.time() - t0, 1),
            "metricas": metricas,
        })

        # Guardado incremental
        OUT.write_text(json.dumps(salida, ensure_ascii=False, indent=2))
        print("   ", n_docs, "docs |", metricas)

salida["meta"]["fin"] = datetime.now().isoformat(timespec="seconds")
OUT.write_text(json.dumps(salida, ensure_ascii=False, indent=2))

print("\n" + "=" * 70)
print(f"{'experimento':<16}{'n':>6}{'EM':>9}{'F1':>9}{'contains':>10}{'min':>8}")
for e in salida["experimentos"]:
    m = e["metricas"]
    print(f"{e['nombre']:<16}{e['n_docs']:>6}{m['exact_match']:>9.4f}"
          f"{m['f1']:>9.4f}{m['contains']:>10.4f}{e['segundos']/60:>8.1f}")
print(f"\nGuardado en {OUT}")
