# run_experiments.py
import json
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

MODEL_PATH = "./models/llama31-8b-instruct"
INCLUDE_PATH = "./mis_tasks"
LIMIT = 5
SHOTS = [0, 5, 10]
OUT = Path("results/experimentos_sqac.json")

OUT.parent.mkdir(parents=True, exist_ok=True)

# Modelo cargado una sola vez y reutilizado
lm = HFLM(pretrained=MODEL_PATH, dtype="bfloat16", batch_size=1)
task_manager = TaskManager(include_path=INCLUDE_PATH)

salida = {
    "meta": {
        "model": MODEL_PATH,
        "limit": LIMIT,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "seeds": "defaults del harness (random 0, numpy 1234, torch 1234, fewshot 1234)",
    },
    "experimentos": [],
}

def extraer_prompt(arguments):
    """Devuelve el prompt, sea cual sea la forma en que venga `arguments`."""
    if isinstance(arguments, dict):                      # forma del .jsonl guardado
        return arguments["gen_args_0"]["arg_0"]
    primero = arguments[0]                               # forma en memoria
    if isinstance(primero, (list, tuple)):
        return primero[0]
    return primero

for chat in (False, True):
    for n in SHOTS:
        nombre = f"{'chat' if chat else 'nochat'}_{n}shot"
        print(f"\n>>> {nombre}")

        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=["sqac"],
            num_fewshot=n,
            limit=LIMIT,
            apply_chat_template=chat,
            fewshot_as_multiturn=chat,   # solo válido con chat template
            task_manager=task_manager,
            log_samples=True,
        )

        metricas = {
            k.split(",")[0]: v
            for k, v in res["results"]["sqac"].items()
            if "," in k and not k.startswith("alias") and "stderr" not in k
        }

        muestras = []

        s0 = res["samples"]["sqac"][0]
        print("CLAVES:", list(s0.keys()))
        print("resps:", repr(s0["resps"])[:200])
        print("arguments:", repr(s0["arguments"])[:200])

        for s in res["samples"]["sqac"]:
            muestras.append({
                "doc_id": s["doc_id"],
                "titulo": s["doc"]["title"],
                "contexto": s["doc"]["context"],
                "pregunta": s["doc"]["question"],
                "gold": s["doc"]["answers"]["text"],
                "respuesta": s["resps"][0][0],
                "exact_match": s["exact_match"],
                "f1": round(s["f1"], 4),
                "contains": s["contains"],
                "prompt_completo": extraer_prompt(s["arguments"]),
            })

        salida["experimentos"].append({
            "nombre": nombre,
            "chat_template": chat,
            "num_fewshot": n,
            "metricas": metricas,
            "muestras": muestras,
        })

        # Guardado incremental: si algo falla, no pierdes lo ya corrido
        OUT.write_text(json.dumps(salida, ensure_ascii=False, indent=2))
        print("   ", metricas)

print(f"\nListo -> {OUT}")
