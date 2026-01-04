#!/bin/bash
#SBATCH --job-name=metamizer_dataset
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --partition=medium
#SBATCH --time=05:00:00
#SBATCH --array=0-99
#SBATCH --chdir=/scratch/21cs30065/datasets

PYTHON_SCRIPT=$1

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pmp

echo "Meep version check:"
python -c "import meep; print(meep.__version__)"

if [[ "$PYTHON_SCRIPT" == "test" ]]; then
    mpirun -np $SLURM_NTASKS python tests/test_sim.py 
else
    mpirun -np $SLURM_NTASKS python $PYTHON_SCRIPT --slurm_job_id $SLURM_ARRAY_TASK_ID
fi
    