#!/bin/bash
set -e

# Base output directory where PPO training results are stored
# This should match the OUTPUT_BASE used in the training script
OUTPUT_BASE="/cloud_checkpoints"

# List of checkpoints corresponding to the training runs (from SFT)
CHECKPOINTS=(20 40 60 80 100 120 140 153)

# Step number of the PPO checkpoint to merge (e.g. step_15 for 15 epochs)
# Adjust this if your training ran for a different number of steps
TARGET_STEP="15"

echo "=================================================="
echo "Starting Model Merge Sequence"
echo "Target Step: ${TARGET_STEP}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "=================================================="

for CKPT_NUM in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="checkpoint-${CKPT_NUM}"
    EXPERIMENT_NAME="verl-or-bench-sft-${CKPT_NAME}"
    
    # Path to the Actor's FSDP checkpoint
    # Structure: /cloud_checkpoints/verl-or-bench-sft-checkpoint-XX/global_step_YY/actor/
    ACTOR_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/global_step_${TARGET_STEP}/actor/"
    
    # Target directory for the Hugging Face format model
    # Structure: /cloud_checkpoints/verl-or-bench-sft-checkpoint-XX/hf_merged/step_YY/
    TARGET_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/hf_merged/step_${TARGET_STEP}/"

    echo "--------------------------------------------------"
    echo "Processing: ${EXPERIMENT_NAME}"
    
    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "Warning: Actor directory not found at ${ACTOR_DIR}"
        echo "Skipping..."
        continue
    fi

    echo "Source: ${ACTOR_DIR}"
    echo "Target: ${TARGET_DIR}"
    echo "--------------------------------------------------"

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ACTOR_DIR}" \
        --target_dir "${TARGET_DIR}"

    echo "Successfully merged model for ${CKPT_NAME}"
done

echo "=================================================="
echo "All merges complete!"
echo "=================================================="


