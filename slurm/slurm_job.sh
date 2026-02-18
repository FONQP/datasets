#!/bin/bash
#SBATCH --job-name=metamizer_dataset
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --array=0-99%10
#SBATCH --chdir=/scratch/21cs30065/datasets

PYTHON_SCRIPT=$1

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pmp

echo "Meep version check:"
python -c "import meep; print(meep.__version__)"

mpirun -np $SLURM_NTASKS $PYTHON_SCRIPT --slurm_job_id $SLURM_ARRAY_TASK_ID
    