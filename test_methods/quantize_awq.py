#!/usr/bin/env python
"""
Cuantiza un modelo con RTN y AWQ usando la configuración y la calibración del paper
(Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"):
  - W4, grupos de 128, con zero point (= awq/entry.py de llm-awq)
  - calibración: pileval, 128 documentos, bloques de 512 tokens (get_calib_dataset de llm-awq, copiado abajo)
  - búsqueda de escalas solo con activaciones (duo_scaling=False, como llm-awq) + búsqueda de clipping

La búsqueda la hace AutoAWQ (pip install autoawq), que es un port de llm-awq, y los checkpoints quedan en
el formato AWQ GEMM que transformers carga con from_pretrained. RTN se guarda en el mismo formato pero sin
búsqueda, así rtn y awq se cargan y ejecutan igual y solo difieren en la calibración.

Ejemplo:
  python test_methods/quantize_awq.py --model meta-llama/Llama-2-7b-hf
  -> test_methods/quantized/Llama-2-7b-hf-rtn, test_methods/quantized/Llama-2-7b-hf-awq
"""
import argparse
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer
from awq.utils.module import exclude_layers_to_not_quantize, get_named_linears

QUANT_CONFIG = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
N_SAMPLES = 128  # entry.py: run_awq(..., n_samples=128, seqlen=512)
SEQLEN = 512


def get_calib_dataset(tokenizer, n_samples=N_SAMPLES, block_size=SEQLEN):
    """awq/utils/calib_data.py de llm-awq: docs de pileval de <=512 tokens, concatenados y cortados en bloques."""
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    for data in dataset:
        line_encoded = tokenizer.encode(data["text"].strip())
        if len(line_encoded) > 512 or len(line_encoded) == 0:
            continue
        samples.append(line_encoded)
        if len(samples) == n_samples:
            break
    cat = [tok for s in samples for tok in s]
    n_split = len(cat) // block_size
    print(f" * Split into {n_split} blocks")
    return [cat[i * block_size:(i + 1) * block_size] for i in range(n_split)]


class RTNQuantizer(AwqQuantizer):
    """Solo la cuantización final de AwqQuantizer (pseudo_quantize_tensor + WQLinear_GEMM), sin calibración."""

    def init_quant(self, *args, **kwargs):
        return self.awq_model.get_model_layers(self.model), None, None

    def quantize(self):
        for layer in self.modules:
            named_linears = exclude_layers_to_not_quantize(get_named_linears(layer), self.modules_to_not_convert)
            self._apply_quant(layer, named_linears)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--methods", default="rtn,awq")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "quantized"))
    args = p.parse_args()

    methods = args.methods.split(",")
    if set(methods) - {"rtn", "awq"}:
        p.error(f"--methods acepta rtn y awq, no {methods}")

    # entry.py usa el tokenizer lento
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    for method in methods:
        model = AutoAWQForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        if method == "awq":
            blocks = get_calib_dataset(tokenizer)
            model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=blocks, duo_scaling=False,
                           apply_clip=True, max_calib_samples=len(blocks), max_calib_seq_len=SEQLEN)
        else:
            model.quantize(tokenizer, quant_config=QUANT_CONFIG, quantizer_cls=RTNQuantizer)
        out = Path(args.output_dir) / f"{args.model.rstrip('/').split('/')[-1]}-{method}"
        model.save_quantized(str(out))
        tokenizer.save_pretrained(out)
        print(f"{method} guardado en {out}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
