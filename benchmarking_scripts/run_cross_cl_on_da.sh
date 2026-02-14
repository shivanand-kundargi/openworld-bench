#!/bin/bash
# ==========================================
# Cross-Setting: CL methods on DA setting
# Dataset: DomainNet
# Application: CL methods (rehearsal/prompting) applied to DA problem
# Structure: Source Domains -> Target Domain (CL methods usually treat this as sequential or joint?)
# Setting 'da' implies Source+Target available. CL methods observe (x_s, y_s, x_t).
# Most CL methods only support (x, y). 
# Adapter logic in train.py handles this (e.g., treating domains as tasks or ignoring target labels).
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_cl_on_da"}
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
echo "Benchmarking: Cross CL on DA (DomainNet)"
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
            --setting da \
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
