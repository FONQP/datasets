#!/bin/bash
# Usage: ./run.sh [NODES] [TIME] [SCRIPT_PATH]

DEFAULT_NODES=1
DEFAULT_TIME="24:00:00"

RUN_NAME=${1:-"test_run"}
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../simulated_datasets/$RUN_NAME"
CONFIG_NAME=${2:-"config.toml"}
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../configs/$CONFIG_NAME"
NODES=${3:-$DEFAULT_NODES}
TIME=${4:-$DEFAULT_TIME}
SCRIPT_PATH="gen_dataset --output_dir $OUTPUT_DIR --config $CONFIG_PATH --log=DEBUG"

echo "Submitting $SCRIPT_PATH on $NODES nodes for $TIME..."

sbatch --nodes=$NODES --time=$TIME slurm_job.sh "$SCRIPT_PATH"
