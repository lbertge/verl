#!/bin/bash
set -e

# Base configuration
# Ensure these paths are correct inside the container (e.g. /workspace/verl/data/...)
# Note: $HOME in the container is /root
TRAIN_FILE="$HOME/data/or_bench/train_2_11.parquet"
VAL_FILE="$HOME/data/or_bench/test_2_11.parquet"

# Checkpoint base directory inside the container (mapped via docker volume)
# Assumes -v ...:/cloud_checkpoints/or_llm_sft_2_11
CHECKPOINT_BASE="/cloud_checkpoints/or_llm_sft_2_11"

# Output base directory
OUTPUT_BASE="/cloud_checkpoints"

# List of checkpoints to train on (sorted numerically)
# You can customize this list or make it dynamic
CHECKPOINTS=(20 40 60 80 100 120 140 153)

echo "=================================================="
echo "Starting PPO Training Sequence"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "=================================================="

for CKPT_NUM in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="checkpoint-${CKPT_NUM}"
    MODEL_PATH="${CHECKPOINT_BASE}/${CKPT_NAME}"
    EXPERIMENT_NAME="verl-or-bench-sft-${CKPT_NAME}"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
    LOG_FILE="${OUTPUT_BASE}/${EXPERIMENT_NAME}.log"

    echo "--------------------------------------------------"
    echo "Processing Checkpoint: ${CKPT_NAME}"
    echo "Model Path: ${MODEL_PATH}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "--------------------------------------------------"

    # Run PPO Training
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILE}" \
        data.train_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path="${MODEL_PATH}" \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="verl-or-bench-sft" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.total_epochs=15 \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        hydra.run.dir="${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_FILE}"

    echo "Finished training for ${CKPT_NAME}"
    echo "--------------------------------------------------"
done

echo "All training runs complete!"


