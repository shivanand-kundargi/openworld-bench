#!/bin/bash
# ==========================================
# Native CL: Class-Incremental Learning (CIL)
# Dataset: DomainNet (Mixed Domains)
# Tasks: 5 tasks, 69 classes each (345 total)
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/native_cl_cil"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# CL Methods (including new ones)
METHODS=("icarl" "ewc" "l2p" "dualprompt" "coda_prompt")

# Backbone selection based on method
get_backbone() {
    case $1 in
        "l2p"|"dualprompt"|"coda_prompt") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Native CL (CIL) on DomainNet"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

for METHOD in "${METHODS[@]}"; do
    BACKBONE=$(get_backbone $METHOD)
    echo ""
    echo ">>> Running: ${METHOD} (Backbone: ${BACKBONE})"
    
    python scripts/train.py \
        --method ${METHOD} \
        --setting cl \
        --dataset domainnet \
        --data_root ${DATA_ROOT} \
        --n_tasks 5 \
        --backbone ${BACKBONE} \
        --epochs 10 \
        --batch_size 32 \
        --seed ${SEED} \
        --gpu ${GPU_ID} \
        --output_dir ${OUTPUT_DIR} \
        # Add method-specific args if needed (defaults in config)
        
    echo ">>> Completed: ${METHOD}"
done
