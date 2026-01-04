#!/bin/bash
#SBATCH --job-name=metamizer_dataset
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --partition=shared
#SBATCH --time=05:00:00
#SBATCH --chdir=/scratch/21cs30065/datasets

PYTHON_SCRIPT=$1

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pmp

echo "Meep version check:"
python -c "import meep; print(meep.__version__)"

# if python script is test
if [[ "$PYTHON_SCRIPT" == "test" ]]; then
    mpirun -np $SLURM_NTASKS python tests/test_sim.py 
else
    mpirun -np $SLURM_NTASKS python $PYTHON_SCRIPT
fi
    