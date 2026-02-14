#!/bin/bash
# ==========================================
# Cross-Setting: DA methods on CL setting
# Dataset: DomainNet
# Application: DA methods (feature alignment) applied to CL (sequential tasks)
# Structure: Sequential Tasks of Classes (CIL) or Domains (DIL).
# DA usually expects concurrent Source/Target access. 
# Adapter logic: Treat current task as "Target", previous memory/model as "Source" constraint?
# Or just run DA methods on sequential data stream (often fails without adaptation mechanism).
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_da_on_cl"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# DA Methods
METHODS=("dapl" "pmtrans" "dann" "cdan" "mcd")

get_backbone() {
    case $1 in
        "dapl"|"pmtrans") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Cross DA on CL (DomainNet)"
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
