#!/bin/bash
set -e  # Exit immediately if any command fails

# --- Configuration ---
START_STAGE=5
END_STAGE=10
BASE_DATA_DIR="$HOME/data/or_bench"
BASE_OUTPUT_DIR="/cloud_checkpoints/qwen_2_5_7B/grpo"
LOG_DIR="/cloud_checkpoints/qwen_2_5_7B"

# Initial starting point (Stage 1 result)
STAGE_1_MODEL_PATH="/cloud_checkpoints/qwen_2_5_7B/stage_1"

echo "Starting Curriculum Training with Auto-Merge: Stage $START_STAGE to $END_STAGE"

# --- Helper Function to Find Previous Checkpoint ---
get_checkpoint_path() {
    local stage=$1
    local direct_path="/cloud_checkpoints/qwen_2_5_7B/verl-or-bench-curriculum_v3-stage-${stage}"
    local grpo_path="/cloud_checkpoints/qwen_2_5_7B/grpo/verl-or-bench-curriculum_v3-stage-${stage}"
    
    local output_dir=""
    if [ -d "$direct_path" ]; then
        output_dir="$direct_path"
    elif [ -d "$grpo_path" ]; then
        output_dir="$grpo_path"
    else
        return 1
    fi
    
    # Priority: Look for the MERGED Hugging Face checkpoint
    # The merge step below saves to .../merged_hf
    if [ -d "$output_dir/merged_hf" ]; then
        echo "$output_dir/merged_hf"
        return 0
    fi

    # Fallback: Look for manually converted hf/step_* folders
    local hf_dir="${output_dir}/hf"
    if [ -d "$hf_dir" ]; then
        local latest_hf_ckpt=$(find "$hf_dir" -maxdepth 1 -name "step_*" -type d 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$latest_hf_ckpt" ]; then
            echo "$latest_hf_ckpt"
            return 0
        fi
    fi
    
    return 1
}

# --- Main Loop ---
for (( stage=$START_STAGE; stage<=$END_STAGE; stage++ ))
do
    echo "============================================================"
    echo "Processing Stage $stage"
    echo "============================================================"

    # --- 1. Determine Previous Model Path ---
    if [ "$stage" -eq 2 ]; then
        PREV_MODEL_PATH="$STAGE_1_MODEL_PATH"
    else
        prev_stage=$((stage - 1))
        echo "Looking for merged HF checkpoint from Stage $prev_stage..."
        
        PREV_MODEL_PATH=$(get_checkpoint_path $prev_stage)
        
        if [ -z "$PREV_MODEL_PATH" ]; then
            echo "Error: Could not find valid HF checkpoint for Stage $prev_stage."
            echo "Expected 'merged_hf' directory or 'hf/step_*'."
            exit 1
        fi
    fi

    echo "Using Previous Model: $PREV_MODEL_PATH"

    # --- 2. Define Paths for Current Stage ---
    TRAIN_FILE="${BASE_DATA_DIR}/train_stage_${stage}.parquet"
    VAL_FILE="${BASE_DATA_DIR}/train_stage_${stage}.parquet"
    
    # Store output in the grpo directory
    CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/verl-or-bench-curriculum_v3-stage-${stage}"
    LOG_FILE="${LOG_DIR}/verl-or-bench-curriculum_v3-stage-${stage}.log"
    PROJECT_NAME="verl-or-bench-7b-sft-stage-${stage}"

    echo "Training on: $TRAIN_FILE"
    echo "Output dir: $CURRENT_OUTPUT_DIR"

    # --- 3. Run Training ---
    # Only run training if the merged output doesn't already exist (resume capability)
    if [ -d "$CURRENT_OUTPUT_DIR/merged_hf" ]; then
        echo "Stage $stage seems to be completed (merged_hf exists). Skipping training."
    else
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files="$TRAIN_FILE" \
            data.val_files="$VAL_FILE" \
            data.train_batch_size=8 \
            data.max_prompt_length=4096 \
            data.max_response_length=4096 \
            actor_rollout_ref.model.path="$PREV_MODEL_PATH" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=8 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.strategy=fsdp2 \
            actor_rollout_ref.model.enable_gradient_checkpointing=False \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.n=5 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.ref.strategy=fsdp2 \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.logger="['console','wandb']" \
            trainer.project_name="$PROJECT_NAME" \
            trainer.experiment_name=grpo \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.save_freq=20 \
            trainer.test_freq=5 \
            trainer.default_local_dir="$CURRENT_OUTPUT_DIR" \
            trainer.total_epochs=15 \
            2>&1 | tee "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Error: Training failed for Stage $stage"
            exit 1
        fi
    fi

    # --- 4. Merge Checkpoints (FSDP -> HF) ---
    echo "Merging checkpoints for Stage $stage..."

    # Find the latest global_step directory
    # Note: Using 'global_step_*' because that's what your output showed
    LATEST_CHECKPOINT=$(find "$CURRENT_OUTPUT_DIR" -maxdepth 1 -name "global_step_*" -type d | sort -V | tail -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoints found in $CURRENT_OUTPUT_DIR to merge!"
        exit 1
    fi

    ACTOR_PATH="${LATEST_CHECKPOINT}/actor"
    MERGE_OUTPUT_PATH="${CURRENT_OUTPUT_DIR}/merged_hf"

    if [ -d "$MERGE_OUTPUT_PATH" ]; then
        echo "Merged checkpoint already exists at $MERGE_OUTPUT_PATH. Skipping merge."
    else
        echo "Converting FSDP checkpoint at $ACTOR_PATH to HF format..."
        
        # Using the verl.model_merger command as requested
        # We need to provide the local path to the raw FSDP checkpoint
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_PATH" \
            --target_dir "$MERGE_OUTPUT_PATH"
            
            # NOTE: If your version of model_merger requires --config or --tokenizer,
            # you might need to append them here. Assuming standard args based on your request.
            # Usually model_merger infers or uses what's in the checkpoint folder if saved correctly.

        if [ $? -ne 0 ]; then
            echo "Error: Merge failed for Stage $stage"
            exit 1
        fi
        
        # Critical: Copy tokenizer files from previous model to the new merged folder
        # The merger might only save weights + config.
        echo "Copying tokenizer files..."
        cp "$PREV_MODEL_PATH/tokenizer"* "$MERGE_OUTPUT_PATH/" 2>/dev/null || true
        cp "$PREV_MODEL_PATH/vocab.json" "$MERGE_OUTPUT_PATH/" 2>/dev/null || true
        cp "$PREV_MODEL_PATH/merges.txt" "$MERGE_OUTPUT_PATH/" 2>/dev/null || true
        cp "$PREV_MODEL_PATH/special_tokens_map.json" "$MERGE_OUTPUT_PATH/" 2>/dev/null || true
        cp "$PREV_MODEL_PATH/added_tokens.json" "$MERGE_OUTPUT_PATH/" 2>/dev/null || true
        
        echo "Merge complete: $MERGE_OUTPUT_PATH"
    fi

    echo "Stage $stage finished successfully."
    echo ""
done

echo "Curriculum training chain complete!"
