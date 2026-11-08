# Tesis

Código para evaluar LLMs abiertos (Llama, Qwen, etc.) en GPU, comparando calidad en
benchmarks, velocidad de inferencia y uso de memoria, con y sin cuantización
(bitsandbytes NF4/INT8, RTN, AWQ), en NVIDIA (CUDA) y AMD (ROCm).

## Estructura

| Carpeta | Contenido |
|---|---|
| [`benchmarks/`](benchmarks/README.md) | `run_eval.py`: evalúa modelos con `lm-evaluation-harness` (un subproceso por modelo) y genera un Excel con métricas, tiempos, memoria y tok/s. `run_quantized.sh` lo corre sobre los checkpoints cuantizados. `tasks/` tiene tareas propias (SQAC). |
| [`test_methods/`](test_methods/README.md) | Cuantización de modelos (RTN, AWQ) y comparación de métodos y kernels: memoria, velocidad y backend usado. |
| `archive/` | Scripts de experimentos anteriores. Se conservan como referencia. |

## Flujo típico

1. Cuantizar: `python test_methods/quantize_awq.py --model <modelo>`
2. Evaluar en benchmarks: `benchmarks/run_eval.py` o `benchmarks/run_quantized.sh`

El detalle de configuración, salidas y versiones está en el README de cada carpeta.
