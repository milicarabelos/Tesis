# Tesis

## `run_eval.py`

Este script corre una lista de modelos contra una lista de benchmarks de `lm-evaluation-harness` y guarda un Excel con resultados, tiempos, memoria y velocidad de inferencia.

Por cada modelo:

1. Lo carga con `transformers` (función `load_hf`) y mide tiempo y memoria de carga.
2. Corre un micro-benchmark de inferencia (prompt y salida de largo fijo) para medir tokens/s.
3. Le pasa el modelo ya instanciado a `lm-eval` (`HFLM(pretrained=model, tokenizer=tokenizer)`) y corre cada benchmark, midiendo tiempo y pico de memoria por tarea.

### Ejemplo de uso

```bash
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct" \
BENCHMARKS="boolq wic openbookqa" \
OUTPUT_DIR=results \
	BATCH_SIZE=8 NUM_FEWSHOT=0 APPLY_CHAT_TEMPLATE=0 \
	python benchmarks/run_eval.py
```

Para repetir el experimento con otra configuración sin reutilizar los
resultados existentes, usá `--force` con un nombre para la variante:

```bash
MODELS="meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B" \
BENCHMARKS="boolq wic openbookqa" \
OUTPUT_DIR=results BATCH_SIZE=auto NUM_FEWSHOT=0 APPLY_CHAT_TEMPLATE=0 \
python benchmarks/run_eval.py --force auto_batch
```

Esto guarda la ejecución en `results_autobatch/` y genera
`results_autobatch/resultados_autobatch.xlsx`. El nombre se puede cambiar
libremente; los caracteres no válidos para un nombre de archivo se reemplazan
por `_`, y esos `_` se quitan del nombre de la variante.

### Configuraciones

`run_eval.py` se controla con variables de entorno. Estas son todas las configuraciones disponibles:

- `MODELS`: lista de modelos separada por espacios.
	- Formato simple: `meta-llama/Llama-3.1-8B-Instruct`
	- Formato avanzado (model_args completo): `pretrained=/ruta/base,peft=/ruta/adapter,load_in_4bit=True`
	- Si una entrada contiene `=`, el script la toma como `model_args` completo.
	- El modelo se carga con `transformers` en `load_hf()`; ahí se puede customizar la carga.
	- Claves que interpreta el script:
		- `pretrained`, `tokenizer` (default: igual a `pretrained`), `peft` (adapter LoRA), `dtype`
		- `load_in_4bit=True` / `load_in_8bit=True` → `BitsAndBytesConfig`
		- `max_length`, `add_bos_token`, `truncation`, `max_batch_size` → se pasan a `HFLM`
	- Cualquier otra clave va directo a `AutoModelForCausalLM.from_pretrained`.
	  Ejemplo: `pretrained=Qwen/Qwen2.5-7B,attn_implementation=sdpa,trust_remote_code=True`
	- Por default carga todo en una GPU (`device_map={"": 0}`); se puede pisar con `device_map=auto`.

- `BENCHMARKS`: tareas de `lm-evaluation-harness` separadas por espacios.
	- Ejemplo: `boolq wic openbookqa sqac`
	- Además de las tareas del harness, se pueden usar las tareas propias de
	  `benchmarks/tasks/` (por ejemplo `sqac`). Se cargan con un `TaskManager`
	  que apunta a esa carpeta, sin importar desde dónde se corra el script.

- `OUTPUT_DIR`: carpeta donde se guardan los resultados.
	- Default: `results`
	- Se guardan:
		- `resultados.xlsx`
		- `raw/*.json` (avance incremental por modelo/tarea)

- `NUM_FEWSHOT`: cantidad de ejemplos de few-shot por tarea.
	- Default: `0`
	- Subirlo puede mejorar métricas en algunas tareas, pero aumenta tiempo de evaluación.

- `BATCH_SIZE`: cantidad de ejemplos procesados en paralelo por paso de inferencia.
	- Default: `8`
	- Más alto: más velocidad potencial, más uso de VRAM.
	- Más bajo: menos VRAM, más lento.
	- También acepta `auto` para que el harness intente elegirlo automáticamente.

- `DTYPE`: precisión para cargar el modelo (`bfloat16`, `float16`, `float32`, etc.).
	- Default: `bfloat16`
	- Si en `MODELS` ya pasás `dtype=...` dentro de `model_args`, ese valor tiene prioridad y `DTYPE` se ignora.

- `APPLY_CHAT_TEMPLATE`: activa template de chat del tokenizer/modelo.
	- `0` (default): desactivado
	- `1`: activado
	- Recomendado activarlo en modelos instruct/chat cuando la tarea lo requiera.

- `LIMIT`: limita la cantidad de ejemplos por benchmark.
	- Default: sin límite (usa todos los ejemplos)
	- Útil para smoke tests rápidos.
	- Ejemplo: `LIMIT=50`

- `PERF`: activa el micro-benchmark de inferencia.
	- `1` (default): activado
	- `0`: desactivado
	- Se corre una vez por modelo; si las tareas ya estaban hechas pero falta el benchmark,
	  el modelo se carga solo para medirlo.

- `PERF_PROMPT_TOKENS`: largo del prompt del micro-benchmark. Default: `512`
- `PERF_NEW_TOKENS`: tokens generados en el micro-benchmark. Default: `128`
- `PERF_REPEATS`: repeticiones (se reporta la mediana). Default: `3`

### Argumentos

- `--force NAME`: crea una variante independiente usando
  `<OUTPUT_DIR>_<NAME>` y `resultados_<NAME>.xlsx`. Los guiones bajos del nombre
  se quitan, por lo que `--force auto_batch` usa `results_autobatch/` y
  `resultados_autobatch.xlsx`. También ignora y reemplaza los JSON incrementales
  existentes de esa variante, por lo que permite repetir una ejecución aunque
  la carpeta ya exista.

### Métricas de inferencia y memoria

Micro-benchmark (una vez por modelo, batch 1, greedy, `min_new_tokens` forzado para
que un EOS temprano no corte la generación, con warmup previo):

- `ttft_ms`: tiempo hasta el primer token (`generate` con 1 token nuevo).
- `prefill_tok_per_s`: `PERF_PROMPT_TOKENS / tiempo de un forward sobre el prompt`.
- `decode_tok_per_s`: `(N-1) / (t_generate(N) - t_generate(1))`; aísla el decode del prefill.
- `e2e_tok_per_s`: `N / t_generate(N)`.

No se reportan tokens/s por tarea: dependen del tipo de tarea (loglikelihood vs
generación), del batch y del padding, así que no son comparables entre modelos.

Memoria (en GB, sumada entre GPUs), medida en la carga, en el micro-benchmark y en cada tarea:

- `*_mem_alloc_gb` / `load_mem_gb` / `load_peak_mem_gb`: tensores vivos según PyTorch
  (pesos + activaciones + KV cache). **Es la métrica comparable entre NVIDIA y AMD.**
- `*_mem_device_gb`: memoria usada del dispositivo menos el baseline tomado al arrancar
  (≈ `nvidia-smi` / `rocm-smi`). Se mide con `torch.cuda.mem_get_info`, que usa
  `cudaMemGetInfo` en NVIDIA y `hipMemGetInfo` en AMD, muestreando cada 50 ms.
  Incluye caché del allocator y workspaces de cuBLAS/hipBLAS, por lo que puede variar
  entre plataformas; sirve para saber si el modelo entra en una GPU.
	- Mide la GPU entera: si otro proceso la usa, el valor se contamina. En SLURM conviene
	  pedir la GPU de forma exclusiva.

### Salida

`resultados.xlsx` tiene las hojas:

- `resumen`: modelo × (tarea | métrica).
- `tiempos`: tiempo de carga y tiempo por tarea.
- `detalle`: una fila por modelo/tarea/métrica, con tiempos y memoria.
- `performance`: una fila por modelo con backend (`cuda X` / `rocm X`), GPU, tiempo y
  memoria de carga, resultados del micro-benchmark y el pico de memoria máximo entre tareas.

Los `raw/*.json` guardan las filas por tarea (`rows`) y el micro-benchmark (`perf`).
Los raws generados con versiones anteriores del script no tienen las columnas nuevas;
para obtenerlas hay que re-correr con `--force`.

### NVIDIA y ROCm

El script es el mismo para ambas: PyTorch con ROCm expone la misma API `torch.cuda`.
El backend se detecta con `torch.version.hip` y queda registrado en la hoja `performance`.
Depende de la instalación de cada entorno:

- `load_in_4bit` / `load_in_8bit` requieren un `bitsandbytes` con soporte para la plataforma.
- `attn_implementation=flash_attention_2` tiene builds distintos; `sdpa` funciona en ambas.

Los tokens/s miden modelo + kernels + hardware: comparar modelos en la misma GPU es
válido, comparar entre GPUs/fabricantes también compara el stack de software.

Notas útiles:

- El script detecta automáticamente dispositivo:
	- `cuda` si hay GPU disponible (NVIDIA o AMD con ROCm)
	- `cpu` en caso contrario
- El tiempo de carga del modelo se mide aparte del tiempo de benchmark.
- Si falla un modelo, el script sigue con los demás.