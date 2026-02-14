#!/bin/bash
# ==========================================
# Cross-Setting: DG methods on DA setting
# Dataset: DomainNet
# Application: DG methods (e.g. IRM) applied to DA
# Structure: Source Domains + Target (Unlabeled) -> Target (Labeled)
# DG methods usually ignore Target during training.
# In DA setting, they are given target data but DG methods (pure ones)
# might just treat it as another domain (if unlabeled) or ignore it?
# Standard approach: Train on Sources, Evaluate on Target. Same as DG setting.
# The difference "DG on DA" vs "DG on DG" is subtle:
# Maybe DA setting implies we *can* look at target unlabelled?
# If so, DG methods don't natively use unlabelled target (except maybe via self-training if added).
# We run them as standard DG here. 
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/cross_dg_on_da"}
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
echo "Benchmarking: Cross DG on DA (DomainNet)"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

DOMAINS=("real" "clipart" "infograph" "painting" "quickdraw" "sketch")

for METHOD in "${METHODS[@]}"; do
    BACKBONE=$(get_backbone $METHOD)
    echo ""
    echo ">>> Method: ${METHOD} (Backbone: ${BACKBONE})"
    
    for DOMAIN in "${DOMAINS[@]}"; do
        echo ">>> Target Domain: ${DOMAIN}"
        
        # In DA setting, we have access to target x.
        # DG methods implemented here might ignore it or we feed it.
        # train.py setting=da will pass target data.
        # Base DGMethod might not handle x_target in observe(). 
        # This checks robustness of code or if we need adapter.
        
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
