# test_methods

Scripts para cuantizar modelos y comparar métodos de cuantización (memoria, velocidad y kernel usado).

| Script | Qué hace |
|---|---|
| `quantize_awq.py` | Genera los checkpoints `quantized/<modelo>-rtn` y `quantized/<modelo>-awq` |
| `quant_compare.py` | Carga el modelo con cada método (`base`, `bnb-nf4`, `bnb-int8`, `rtn`, `awq`), genera texto y guarda memoria, tok/s y kernel en `results/quant_compare.jsonl` |
| `check_awq_env.py` | Prueba cada backend AWQ de gptqmodel con un checkpoint y muestra qué capa quedó y su velocidad de decode |

```bash
python test_methods/quantize_awq.py --model meta-llama/Llama-3.1-8B
python test_methods/quant_compare.py --model meta-llama/Llama-3.1-8B --methods base,bnb-nf4,rtn,awq
python test_methods/check_awq_env.py --model test_methods/quantized/Llama-3.1-8B-awq
```

Versiones con las que se verificó lo de abajo: torch 2.13.0+cu129, transformers 5.19.0,
gptqmodel 7.6.0, bitsandbytes 0.50.2, NVIDIA A30 (sm_80).

---

## Métodos

Hay dos tipos de método:

- **Offline** (RTN y AWQ): se cuantiza una vez con `quantize_awq.py` y queda un checkpoint en disco.
- **On-the-fly** (bitsandbytes): los pesos fp16 se cuantizan al cargar, cada vez que se llama a `from_pretrained`.

### `base`

Sin cuantizar; se carga en fp16, como en el paper de AWQ.

### RTN (round-to-nearest) — offline

Se hace en `quantize_awq.py` con la clase `RTNQuantizer`:

1. Carga el modelo en fp16 con `AutoAWQForCausalLM` (AutoAWQ).
2. `RTNQuantizer` hereda de `AwqQuantizer` de AutoAWQ, pero se salta la calibración:
   - `init_quant` no corre datos.
   - `quantize` solo llama a `_apply_quant` en cada capa.
3. `_apply_quant` hace `pseudo_quantize_tensor` y empaqueta cada `nn.Linear` en `WQLinear_GEMM`.
4. La cuantización usa **INT4**, **grupos de 128** columnas de entrada y **zero point** (asimétrica):
   - `scale = (max - min) / 15`
   - `zero = round(-min / scale)`
   - `q = clamp(round(w / scale) + zero, 0, 15)`
5. Se guarda con `save_quantized` en formato **AWQ GEMM**. El `config.json` queda con
   `quant_method: awq, version: gemm, bits: 4, group_size: 128, zero_point: true`.

RTN y AWQ se guardan en el **mismo formato**. Por eso se cargan y ejecutan con el mismo kernel y
solo difieren en los valores de los pesos (la calibración). El `config.json` de RTN dice `awq`; es esperado.

### AWQ — offline

`quantize_awq.py` reproduce la configuración de `llm-awq` (`awq/entry.py`) a través de AutoAWQ:

- **Cuantización final**: idéntica a RTN (W4, g128, zero point, `WQLinear_GEMM`).
- **Calibración**: `get_calib_dataset`, copiada de llm-awq:
  - Toma `mit-han-lab/pile-val-backup` (split `validation`) mezclado con `seed=42`.
  - Elige los primeros 128 documentos de 1 a 512 tokens.
  - Los concatena y los corta en bloques de 512 tokens.
- **Búsqueda de escalas** (`model.quantize(..., duo_scaling=False)`):
  - Por cada grupo de capas que comparten entrada, busca un factor `s = mean|x|^α` con α en una grilla [0, 1].
  - Usa solo la magnitud de las activaciones, como llm-awq.
  - El factor que minimiza el error de salida se fusiona en la capa anterior (LayerNorm / Linear).
- **Búsqueda de clipping** (`apply_clip=True`): por cada linear busca el `max_val` de recorte que minimiza el error.
- **Tokenizer**: lento (`use_fast=False`), como `entry.py`.

Requiere `autoawq` (`from awq import ...`), que solo se usa para cuantizar; para cargar no hace falta.

### bnb-nf4 (NF4, la cuantización de QLoRA) — on-the-fly

En `quant_compare.py`:

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
```

- Cada `nn.Linear` pasa a ser `bnb.nn.Linear4bit` al cargar.
- Los pesos se cuantizan por **bloques de 64** (blocksize default) con el código **NF4**:
  16 niveles con distribución normal.
- **Absmax por bloque**, sin zero point.
- **Double quant**: los absmax (fp32) se cuantizan a 8 bits en bloques de 256. Ahorra ~0.37 bits/param.
- El cómputo se hace en `bnb_4bit_compute_dtype`: fp16.

En `benchmarks/run_eval.py` equivale a:
`pretrained=...,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True`.
**Sin `bnb_4bit_quant_type=nf4`, bitsandbytes usa fp4.**

### bnb-int8 (LLM.int8()) — on-the-fly

- `BitsAndBytesConfig(load_in_8bit=True)` → `bnb.nn.Linear8bitLt`.
- Pesos INT8 con absmax por fila.
- Las columnas de activación con outliers (|x| > 6, `llm_int8_threshold`) se multiplican en fp16 y el resto en INT8.

---

## Backends y cómo se seleccionan

### AWQ / RTN: transformers → gptqmodel

En transformers 5.x, los checkpoints AWQ **ya no se ejecutan con AutoAWQ**; los ejecuta **gptqmodel**.
El flujo dentro de `from_pretrained` es:

1. **Lee el `config.json`**:
   - Ve `quantization_config.quant_method == "awq"` y crea un `AwqConfig`.
   - `version: gemm` pasa a `format=GEMM`.
   - El `backend` default es `auto`.
2. **Ajusta el dtype** (`AwqQuantizer.update_dtype`): si se pide `bfloat16` en CUDA, **lo castea a fp16**
   con un warning. Los kernels AWQ son fp16.
3. **Elige el kernel** (`replace_with_awq_linear`):
   - Llama a `gptqmodel.utils.importer.hf_select_quant_linear_v2(bits, group_size, format, backend, device_map, ...)`.
   - Esa función devuelve una clase `QuantLinear`, que reemplaza a cada `nn.Linear`.
4. **Post-proceso** (`hf_gptqmodel_post_init`): reacomoda los pesos al layout del kernel elegido.
   Por ejemplo, Marlin re-empaqueta.

**Con `backend="auto"`**, gptqmodel recorre esta lista **en orden** y se queda con el primer kernel que
pasa `validate_quant_linear`. Esa validación chequea device, compute capability, bits, group size,
dtype y que la extensión compile o esté instalada.

| Orden | `BACKEND` (gptqmodel) | Clase | `backend=` en `AwqConfig` | En esta A30 |
|---|---|---|---|---|
| 1 | `AWQ_TORCH_ATEN` | `TorchAtenAwqLinear` | — | no (solo CPU) |
| 2 | `AWQ_SWORDFISH` | `AwqSwordfishLinear` | — | no (runtime no disponible) |
| 3 | `AWQ_MACHETE` | `AwqMacheteLinear` | `machete` | no (requiere Hopper sm_90) |
| 4 | `AWQ_MARLIN` | `AwqMarlinLinear` | `marlin` | **sí → el que elige `auto`** |
| 5 | `AWQ_EXLLAMA_V2` | `AwqExllamaV2Linear` | `exllama_v2` | no (falla la compilación JIT) |
| 6 | `AWQ_GEMM` | `AwqGEMMLinear` | `gemm` | sí |
| 7 | `AWQ_GEMM_TRITON` | `AwqGEMMTritonLinear` | `gemm_triton` | no (no hay `triton`) |
| 8 | `AWQ_TORCH_FUSED` | `TorchFusedAwqLinear` | `torch_fused_awq` | no (solo CPU/XPU) |
| 9 | `AWQ_TORCH` | `AwqTorchLinear` | `torch_awq` | sí (dequant + matmul en torch, lento) |

- Los formatos `gemv` y `gemv_fast` usan otro empaquetado (`AwqGEMVLinear`, `AwqGEMVFastLinear`) y
  **no sirven para estos checkpoints GEMM**.
- `exllama_v1` está en el enum de transformers, pero gptqmodel 7.6 no tiene kernel AWQ para él.
- La columna "En esta A30" sale de llamar a `validate_quant_linear` para cada clase. Depende del entorno:
  - Si se instala `triton` o se arregla el compilador para ExLlamaV2, cambia.
  - En una H100, `auto` elegiría Machete.

**Para forzar un backend**, se pasa un `AwqConfig` con solo `backend` (y `bits`). transformers mezcla los
"loading attributes" (`backend`) sobre la config del checkpoint e ignora el resto, con un warning:

```python
AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map={"": 0},
                                     quantization_config=AwqConfig(bits=4, backend="gemm"))
```

- En `quant_compare.py`: `--awq-backend gemm`.
- En `check_awq_env.py`: `--backends marlin,gemm,torch_awq`.
- `benchmarks/run_eval.py` **no permite forzarlo**: usa siempre `auto` (Marlin en la A30).

Para ver qué se eligió:
- gptqmodel loguea `Kernel: selected -> AwqMarlinLinear`.
- `quant_compare.py` guarda la clase en `linear_impl`.
- Con `DEBUG=1`, gptqmodel también loguea por qué descarta cada candidato.

### bitsandbytes (NF4 / INT8)

Para NF4 / INT8 no hay "backend" que se elija desde transformers. bitsandbytes decide en dos niveles:

1. **Librería nativa**:
   - Al importar, carga `libbitsandbytes_cuda<ver>.so` según la versión de CUDA de torch.
   - Si no hay binario para esa versión exacta, usa el más cercano. Acá: *"No prebuilt binary for CUDA 12.9,
     loading CUDA 12.8 instead"*.
   - Se puede forzar con `BNB_CUDA_VERSION` (o `BNB_ROCM_VERSION` en AMD).
2. **Kernel por llamada en 4 bits** (`bitsandbytes.autograd._functions.matmul_4bit`, en inferencia sin grad):
   - Llama a `torch.ops.bitsandbytes.gemm_4bit`.
   - Con double quant, le pasa los absmax de 8 bits para que los descuantice adentro.
   - En CUDA (`backends/cuda/ops.py`), según `M` (= tokens del batch):
     - `M ≤ 4` → kernel propio de bnb (SIMT; el caso de decode de a un token).
     - `5 ≤ M ≤ 1536` → heurística por arquitectura (`_gemm_4bit_use_custom_cuda`). Para sm_80 decide entre
       el kernel `mma.sync` de bnb o descuantizar + `F.linear`, según `M`, `N` y `K`.
     - `M > 1536` → descuantiza el peso a fp16 y usa `F.linear` (cuBLAS).
   - Con grad (entrenamiento QLoRA) usa `MatMul4Bit`: descuantiza + matmul.

---

## Notas del entorno `benchmarks` (Python 3.15)

- **AWQ necesita `gptqmodel`** (y `accelerate`). `autoawq` solo hace falta para cuantizar.
- **Segfault al cargar AWQ**: la carga en paralelo de transformers (`ThreadPoolExecutor` en `core_model_loading`)
  da segfault en `PyThreadState_Clear` con checkpoints pre-cuantizados.
  - Se evita con `HF_DEACTIVATE_ASYNC_LOAD=1`; `benchmarks/run_quantized.sh` ya lo exporta.
  - NF4/INT8 no se ven afectados, porque la cuantización on-the-fly ya carga de forma secuencial.
- **Fuga de memoria con AWQ/RTN**: gptqmodel guarda en un global (`_LLAMA_CPP_IMPORT_ERROR` en
  `gptqmodel/nn_modules/qlinear/gguf_cpp.py`) el `ImportError` de `llama_cpp`, con su traceback incluido.
  - Si gptqmodel se importa por primera vez dentro de `from_pretrained`, ese traceback retiene el primer
    modelo AWQ/RTN en la GPU hasta que termina el proceso.
  - `run_eval.py` importa `gptqmodel` al inicio para evitarlo.
  - Cualquier script que cargue varios modelos AWQ en el mismo proceso tiene que hacer lo mismo.
- **Comparación justa**:
  - Los checkpoints AWQ y RTN corren con el mismo kernel, así que su velocidad y memoria son comparables
    entre sí.
  - Contra NF4 también cambia el kernel (bnb vs Marlin) y el tamaño de grupo (64 vs 128).
