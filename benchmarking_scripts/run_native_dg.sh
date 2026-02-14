#!/bin/bash
# ==========================================
# Native DG: Domain Generalization
# Dataset: DomainNet (Leave-One-Domain-Out)
# Tasks: 6 tasks (Target domain changes each time)
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/native_dg"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# DG Methods (including new ones)
METHODS=("pego" "irm" "swad" "vrex" "coral")

# Backbone selection
get_backbone() {
    # PEGO might work better with ViT but standard DG usually uses ResNet50
    # Let's assume ResNet50 for all unless specified
    case $1 in
        "pego_vit") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Native DG on DomainNet"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

# Domain list
DOMAINS=("real" "clipart" "infograph" "painting" "quickdraw" "sketch")

for METHOD in "${METHODS[@]}"; do
    BACKBONE=$(get_backbone $METHOD)
    echo ""
    echo ">>> Method: ${METHOD} (Backbone: ${BACKBONE})"
    
    for DOMAIN in "${DOMAINS[@]}"; do
        echo ">>> Target Domain: ${DOMAIN}"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting dg \
            --dataset domainnet \
            --data_root ${DATA_ROOT} \
            --target_domain ${DOMAIN} \
            --backbone ${BACKBONE} \
            --epochs 30 \
            --batch_size 32 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}/${METHOD}/${DOMAIN} \
            
        echo ">>> Completed: ${METHOD} on Target ${DOMAIN}"
    done
done
