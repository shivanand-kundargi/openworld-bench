#!/bin/bash
# ==========================================
# Cross-Setting: CL methods on DG setting
# Dataset: DomainNet
# Application: CL methods applied to DG problem
# Structure: Multiple Source Domains -> Held-out Target Domain
# In 'dg' setting, methods observe (x_source_domains, y_source_domains).
# CL methods typically expect (x, y) stream. 
# Adapter logic in train.py handles this (e.g., sequential domain training).
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_cl_on_dg"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# CL Methods
METHODS=("icarl" "ewc" "l2p" "dualprompt" "coda_prompt")

get_backbone() {
    case $1 in
        "l2p"|"dualprompt"|"coda_prompt") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Cross CL on DG (DomainNet)"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

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
            --epochs 20 \
            --batch_size 32 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}/${METHOD}/${DOMAIN} \
            
        echo ">>> Completed: ${METHOD} on Target ${DOMAIN}"
    done
done
