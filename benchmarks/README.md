# Tesis

## `run_eval.py`

Este script corre una lista de modelos contra una lista de benchmarks de `lm-evaluation-harness` y guarda un Excel con resultados y tiempos.

### Ejemplo de uso

```bash
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct" \
BENCHMARKS="boolq wic openbookqa" \
OUTPUT_DIR=results \
	BATCH_SIZE=8 NUM_FEWSHOT=0 APPLY_CHAT_TEMPLATE=0 \
	python benchmarks/run_eval.py
```

### Configuraciones

`run_eval.py` se controla con variables de entorno. Estas son todas las configuraciones disponibles:

- `MODELS`: lista de modelos separada por espacios.
	- Formato simple: `meta-llama/Llama-3.1-8B-Instruct`
	- Formato avanzado (model_args completo): `pretrained=/ruta/base,peft=/ruta/adapter,load_in_4bit=True`
	- Si una entrada contiene `=`, el script la toma como `model_args` completo.

- `BENCHMARKS`: tareas de `lm-evaluation-harness` separadas por espacios.
	- Ejemplo: `boolq wic openbookqa`

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

Notas útiles:

- El script detecta automáticamente dispositivo:
	- `cuda` si hay GPU disponible
	- `cpu` en caso contrario
- El tiempo de carga del modelo se mide aparte del tiempo de benchmark.
- Si falla un modelo, el script sigue con los demás.