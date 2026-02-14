#!/bin/bash
# ==========================================
# Cross-Setting: DG methods on DA setting
# ==========================================
# Runs DG methods (IRM, VREx, CORAL) in a Domain Adaptation setting.
# Includes both Office-Home and DomainNet.

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/cross_dg_on_da"}

# DG Methods
DG_METHODS=("irm" "vrex" "coral")

echo "============================================"
echo "CROSS-SETTING: DG methods → DA setting"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

# --- Office-Home ---
OH_DOMAINS=("Art" "Clipart" "Product" "Real_World")
for METHOD in "${DG_METHODS[@]}"; do
    for TARGET in "${OH_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${OH_DOMAINS[@]}"; do
            [ "$D" != "$TARGET" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (DG) → office_home DA (target=${TARGET})"
        python scripts/train.py \
            --method ${METHOD} --setting da --dataset office_home \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${TARGET} \
            --backbone resnet50 --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

# --- DomainNet ---
DN_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
for METHOD in "${DG_METHODS[@]}"; do
    for TARGET in "${DN_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${DN_DOMAINS[@]}"; do
            [ "$D" != "$TARGET" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (DG) → domainnet DA (target=${TARGET})"
        python scripts/train.py \
            --method ${METHOD} --setting da --dataset domainnet \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${TARGET} \
            --backbone resnet50 --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

echo "All DG→DA cross-setting experiments completed!"
