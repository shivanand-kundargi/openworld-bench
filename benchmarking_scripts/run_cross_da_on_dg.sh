#!/bin/bash
# ==========================================
# Cross-Setting: DA methods on DG setting
# Dataset: DomainNet
# Application: DA methods (e.g. DANN) applied to DG scheme
# Structure: Source Domains (Available) -> Target (Blind/Held-out)
# In DG, we don't see target data during training.
# DA methods like DANN need target data. 
# How to run? 
# Option 1: Treat one source as "target" and others as "source" during training? (Pseudo-DG)
# Option 2: Run without target data (backbone only)?
# Usually "DA on DG" implies using DA methods but without access to true target.
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_da_on_dg"}
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
echo "Benchmarking: Cross DA on DG (DomainNet)"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

DOMAINS=("real" "clipart" "infograph" "painting" "quickdraw" "sketch")

for METHOD in "${METHODS[@]}"; do
    BACKBONE=$(get_backbone $METHOD)
    echo ""
    echo ">>> Method: ${METHOD} (Backbone: ${BACKBONE})"
    
    for DOMAIN in "${DOMAINS[@]}"; do
        echo ">>> Target Domain (Held-out): ${DOMAIN}"
        
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
