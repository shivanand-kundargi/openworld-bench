#!/bin/bash
# ==========================================
# Cross-Setting: DG methods on CL setting
# Dataset: DomainNet
# Application: DG methods (e.g. IRM) applied to CL
# Structure: Sequential Tasks
# DG methods usually take multiple domains as input.
# In CL, we get sequential tasks.
# Adapter logic: Treat each task as a "domain"?
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_dg_on_cl"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# DG Methods
METHODS=("pego" "irm" "swad" "vrex" "coral")

get_backbone() {
    case $1 in
        "pego_vit") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Cross DG on CL (DomainNet)"
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
        --output_dir ${OUTPUT_DIR}/${METHOD} \
        
    echo ">>> Completed: ${METHOD}"
done
