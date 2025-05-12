#!/usr/bin/env bash

# given a folder of yaml files, run the sweep
# folder should be required as an argument

FOLDER=$1
FILE_PREFIX=${2:-}
DONT_EXECUTE=${3:-false}

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
        nlprun -q jag -g 1 -a boundless -n "$filename" "uv run tinylang '$file' --wandb"
    fi
done