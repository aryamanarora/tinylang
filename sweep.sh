#!/usr/bin/env bash

# Default values
FOLDER=""
FILE_PREFIX=""
DONT_EXECUTE="false"
QUEUE="jag"

# Parse named arguments
while getopts "f:p:d:q:" opt; do
    case $opt in
        f) FOLDER="$OPTARG" ;;
        p) FILE_PREFIX="$OPTARG" ;;
        d) DONT_EXECUTE="$OPTARG" ;;
        q) QUEUE="$OPTARG" ;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    esac
done

# Check if required argument is provided
if [[ -z "$FOLDER" ]]; then
    echo "Error: -f (folder) argument is required"
    echo "Usage: $0 -f <folder> [-p <file_prefix>] [-d <dont_execute>] [-q <queue>]"
    exit 1
fi

for file in $FOLDER/$FILE_PREFIX*.yaml; do
    filename=$(basename "$file" .yaml)
    # skip if filename is template.yaml
    if [[ "$filename" == "template" ]]; then
        continue
    fi
    
    # get config path after configs/
    config_path=$(echo "$file" | sed 's/.*configs\///' | sed 's/\.yaml$//')
    log_dir="experiments/logs/$config_path"
    
    # skip if test/SummaryEvaluator.csv already exists
    if [[ -f "$log_dir/test/SummaryEvaluator.csv" ]]; then
        echo "Skipping $file - already has test/SummaryEvaluator.csv"
        continue
    fi
    
    echo "Running $file"
    if [[ "$DONT_EXECUTE" == "false" ]]; then
        nlprun -q "$QUEUE" -g 1 -a boundless -n "$filename" "uv run tinylang '$file' --wandb"
    fi
done