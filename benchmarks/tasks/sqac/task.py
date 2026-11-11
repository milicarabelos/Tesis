import re
import string
from collections import Counter
from copy import deepcopy

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean  # OJO: con np.mean lm-eval no calcula stderr
from lm_eval.api.task import ConfigurableTask

# --- Normalización estilo SQuAD, adaptada al español ---------------------
ARTICULOS_ES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
PUNCT_ES = set(string.punctuation) | set("¿¡«»“”‘’—–…")


import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))

def normalize_es(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(ZERO_WIDTH).lower()
    s = "".join(ch for ch in s if ch not in PUNCT_ES)
    tokens = [t for t in s.split() if t not in ARTICULOS_ES]
    return " ".join(tokens)

def em_score(pred: str, golds: list[str]) -> float:
    p = normalize_es(pred)
    return max(float(p == normalize_es(g)) for g in golds)


def _prf(pred: str, gold: str) -> tuple[float, float, float]:
    """Precision, recall y F1 a nivel token (estilo SQuAD)."""
    p, g = normalize_es(pred).split(), normalize_es(gold).split()
    if not p or not g:
        v = float(p == g)
        return v, v, v
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0, 0.0, 0.0
    precision, recall = n / len(p), n / len(g)
    return precision, recall, 2 * precision * recall / (precision + recall)


def prf_score(pred: str, golds: list[str]) -> tuple[float, float, float]:
    """(P, R, F1) contra la gold que maximiza F1, para que P y R sean coherentes con F1."""
    return max((_prf(pred, g) for g in golds), key=lambda t: t[2])


def f1_score(pred: str, golds: list[str]) -> float:
    return prf_score(pred, golds)[2]


def contains_score(prediction: str, labels: list[str]):
    if not prediction or not labels:
        return 0
    return max(
        int(bool(re.search(re.escape(label), prediction, re.IGNORECASE)))
        for label in labels
    )


# --- Task ----------------------------------------------------------------
class SQAC(ConfigurableTask):
    VERSION = 1
    DATASET_PATH = "PlanTL-GOB-ES/SQAC"
    DATASET_NAME = None  # verifica contra SQAC.py

    def __init__(self, **kwargs):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self): return True
    def has_validation_docs(self): return True
    def has_test_docs(self): return True

    def training_docs(self): return self.dataset["train"]
    def validation_docs(self): return self.dataset["validation"]
    def test_docs(self): return self.dataset["test"]
    
    def download(self, dataset_kwargs=None):
        import datasets
        self.dataset = datasets.load_dataset(
            "PlanTL-GOB-ES/SQAC", revision="refs/convert/parquet"
        )

    def _get_answer_texts(self, doc):
        answers = doc["answers"]["text"]
        return [str(a).strip() for a in answers if str(a).strip()]

    def doc_to_text(self, doc, doc_to_text=None):
        return (
            f"Contexto: {doc['context'].strip()}\n\n"
            f"Pregunta: {doc['question'].strip()}\n"
            f"Respuesta:"
        )

    def doc_to_target(self, doc, doc_to_target=None):
        answers = self._get_answer_texts(doc)
        return answers[0] if answers else ""
    
    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        arguments = deepcopy(self.config.generation_kwargs) or {}
        arguments["until"] = ["\n"]
        arguments["max_gen_toks"] = 64
        arguments["do_sample"] = False
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, arguments),
                idx=0,
                **kwargs,
            )
        ]

    METRICS = ("exact_match", "f1", "precision", "recall", "contains")

    def process_results(self, doc, results):
        pred = results[0].strip()
        golds = self._get_answer_texts(doc)
        if not golds:
            return dict.fromkeys(self.METRICS, 0.0)
        precision, recall, f1 = prf_score(pred, golds)
        return {
            "exact_match": em_score(pred, golds),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "contains": contains_score(pred, golds),
        }

    def aggregation(self):
        return dict.fromkeys(self.METRICS, mean)

    def higher_is_better(self):
        return dict.fromkeys(self.METRICS, True)
