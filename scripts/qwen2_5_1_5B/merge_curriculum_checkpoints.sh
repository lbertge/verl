#!/bin/bash
set -e

# Base output directory where PPO training results are stored
# This should match the OUTPUT_BASE used in the training script
OUTPUT_BASE="/cloud_checkpoints"

# List of checkpoint corresponding to the curriculum learning stages
CHECKPOINTS=(2)

# Step number of the PPO checkpoint to merge (e.g. step_15 for 15 epochs)
# Adjust this if your training ran for a different number of steps
TARGET_STEP="30"

echo "=================================================="
echo "Starting Model Merge Sequence"
echo "Target Step: ${TARGET_STEP}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "=================================================="

for CKPT_NUM in "${CHECKPOINTS[@]}"; do
    EXPERIMENT_NAME="qwen_2_5_1_5B/verl-or-bench-curriculum_v3-stage-${CKPT_NUM}/"
    
    # Path to the Actor's FSDP checkpoint
    # Structure: /cloud_checkpoints/qwen_2_5_1_5B/verl-or-bench-curriculum_v3-stage-2/global_step_YY/actor/
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


