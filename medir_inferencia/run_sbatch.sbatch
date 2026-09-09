#!/bin/bash
#SBATCH --job-name=medir_inferencia
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# Crear dir de logs
mkdir -p logs

# Cargar módulos
module load gcc/11.2.0

# Cargar modelos cacheado
export HF_HOME=/home/mcarabelos/tesis/medir_inferencia/models_cache

# Activar conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medir_inferencia

# Ir al directorio
cd /home/mcarabelos/tesis/medir_inferencia

#ver si tiene soporte gpu
python -c "from llama_cpp.llama_cpp import llama_supports_gpu_offload; print(f'GPU offload: {llama_supports_gpu_offload()}')"

# Ejecutar
bash run.sh
