#!/bin/bash
# Usage: ./scp2local.sh [RUN_NAME] [USERNAME@REMOTE_IP] [REMOTE_FOLDER_PATH]

RUN_NAME="${1:-test_run}"
FOLDER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../simulated_datasets/$RUN_NAME"
REMOTE_IP="${2:?Error: Remote IP address not provided}"
REMOTE_FOLDER_PATH="${3:?Error: Remote folder path not provided}"

if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder $FOLDER_PATH does not exist"
    exit 1
fi

uv run consolidate_dataset --root_dir "$FOLDER_PATH" --output_path "$FOLDER_PATH"

scp "$FOLDER_PATH/consolidated.zip" "$REMOTE_IP:$REMOTE_FOLDER_PATH/"

echo "Consolidated and transferred $RUN_NAME"
