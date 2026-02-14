#!/bin/bash
# ==========================================
# Native DA: Domain Adaptation
# Dataset: DomainNet (Leave-One-Domain-Out)
# Tasks: 6 tasks (Target domain changes each time)
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/native_da"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# DA Methods (including new ones)
METHODS=("dapl" "pmtrans" "dann" "cdan" "mcd")

# Backbone selection
get_backbone() {
    case $1 in
        "dapl"|"pmtrans") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Native DA on DomainNet"
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
        
        # Setting da implies standard DA logic (source+target available)
        # Assuming train.py iterates through sources correctly given a target
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting da \
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
