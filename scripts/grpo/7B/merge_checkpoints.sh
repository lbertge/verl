#!/bin/bash

# Exit on error
set -e

# --- Configuration ---
START_STAGE=2
END_STAGE=10

# Base path for checkpoints
# Update this to match where your training script saved the files
# Example from your prompt: /home/azureuser/cloudfiles/code/checkpoints/qwen_2_5_7B/verl-or-bench-curriculum_v3-stage-2
CHECKPOINT_BASE="/home/azureuser/cloudfiles/code/checkpoints/qwen_2_5_7B"

# Dataset path
DATASET_PATH="synthetic_dataset/resource_allocation/eval"

# Base path for storing results
RESULTS_BASE="results/grpo"

echo "Starting evaluation for stages $START_STAGE to $END_STAGE..."

for (( stage=$START_STAGE; stage<=$END_STAGE; stage++ ))
do
    echo "============================================================"
    echo "Evaluating Stage $stage"
    echo "============================================================"

    # 1. Construct the experiment directory path
    # NOTE: Add '/grpo' in the path below if your training script saved it under a grpo subdirectory
    EXP_DIR="${CHECKPOINT_BASE}/verl-or-bench-curriculum_v3-stage-${stage}"
    
    if [ ! -d "$EXP_DIR" ]; then
        # Try checking for 'grpo' subdirectory if strict path fails
        EXP_DIR="${CHECKPOINT_BASE}/grpo/verl-or-bench-curriculum_v3-stage-${stage}"
        if [ ! -d "$EXP_DIR" ]; then
            echo "Error: Experiment directory not found for Stage $stage."
            echo "Checked: ${CHECKPOINT_BASE}/verl-or-bench-curriculum_v3-stage-${stage}"
            continue
        fi
    fi

    # 2. Find the latest HF checkpoint (step_*)
    # We look inside the 'hf' directory as per your example
    HF_DIR="${EXP_DIR}/hf"
    
    if [ ! -d "$HF_DIR" ]; then
        echo "Warning: 'hf' directory not found in $EXP_DIR."
        echo "Ensure you have converted the FSDP checkpoints to HF format."
        continue
    fi

    # Find the step directory with the highest number (sort -V handles numbers correctly)
    MODEL_PATH=$(find "$HF_DIR" -maxdepth 1 -name "step_*" -type d | sort -V | tail -n 1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No 'step_*' directories found in $HF_DIR"
        continue
    fi

    echo "Found checkpoint: $MODEL_PATH"

    # 3. Run Evaluation
    OUTPUT_PATH="${RESULTS_BASE}/stage-${stage}"
    
    # Create output directory
    mkdir -p "$OUTPUT_PATH"

    echo "Running evaluation command..."
    # Using the python module command as requested
    python3 -m main.evaluation.run_eval \
        --dataset-path "$DATASET_PATH" \
        --model "$MODEL_PATH" \
        --output-path "$OUTPUT_PATH"

    echo "Stage $stage evaluation completed."
    echo ""
done

echo "All evaluations finished."
