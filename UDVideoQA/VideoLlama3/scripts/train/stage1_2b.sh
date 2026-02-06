#!/bin/bash
# ============================
# Stage 1 Training Script - 7B Model
# ============================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Load FFmpeg module
module load ffmpeg-6.0-gcc-12.1.0

# Set HuggingFace cache directory to scratch space
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"


# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Clear GPU cache before training
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Add VideoLlama3 to Python path
export PYTHONPATH="$REPO_ROOT/UDVideoQA/VideoLlama3:$PYTHONPATH"

WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-16667}
RANK=${RANK:-0}

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

GLOBAL_BATCH_SIZE=12
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE)))
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

export WANDB_PROJECT=videollama3_qwen2.5_7b
RUN_NAME=stage1_7b
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
OUTP_DIR="${OUTP_DIR:-$REPO_ROOT/outputs/videollama3_training_output_7b}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/UDVideoQA/VideoLlama3/VideoLLaMA3/weights/videollama3_7b_local}"

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    "$REPO_ROOT/UDVideoQA/VideoLlama3/train.py" \
    --deepspeed '{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}' \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_attn_implementation eager \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATA_DIR}/training_data_filtered.jsonl" \
    --data_folder "${DATA_DIR}" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 16 \
    --model_max_length 4096 \
    --mm_max_length 1024 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir "${OUTP_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 48 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mm_projector_lr 1e-3 \
    --vision_encoder_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --run_name $RUN_NAME
