#!/bin/bash
# Usage: ./run.sh [NODES] [TIME] [SCRIPT_PATH]

DEFAULT_NODES=1
DEFAULT_TIME="12:00:00"
DEFAULT_SCRIPT="gen_dataset --output_dir /scratch/21cs30065/datasets/simulated_datasets/test_run --config /scratch/21cs30065/datasets/configs/config.toml --log=DEBUG"

NODES=${1:-$DEFAULT_NODES}
TIME=${2:-$DEFAULT_TIME}
SCRIPT_PATH=${3:-$DEFAULT_SCRIPT}

echo "Submitting $SCRIPT_PATH on $NODES nodes for $TIME..."

sbatch --nodes=$NODES --time=$TIME slurm_job.sh "$SCRIPT_PATH"