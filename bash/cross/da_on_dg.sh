#!/bin/bash
# ==========================================
# Cross-Setting: DA methods on DG setting
# ==========================================
# Runs DA methods (DANN, CDAN, MCD) in a Domain Generalization setting.
# Includes both Office-Home and DomainNet.

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/cross_da_on_dg"}

# DA Methods
DA_METHODS=("dann" "cdan" "mcd")

echo "============================================"
echo "CROSS-SETTING: DA methods → DG setting"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

# --- Office-Home ---
OH_DOMAINS=("Art" "Clipart" "Product" "Real_World")
for METHOD in "${DA_METHODS[@]}"; do
    for HELD_OUT in "${OH_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${OH_DOMAINS[@]}"; do
            [ "$D" != "$HELD_OUT" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (DA) → office_home DG (held_out=${HELD_OUT})"
        python scripts/train.py \
            --method ${METHOD} --setting dg --dataset office_home \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${HELD_OUT} \
            --backbone resnet50 --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

# --- DomainNet ---
DN_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
for METHOD in "${DA_METHODS[@]}"; do
    for HELD_OUT in "${DN_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${DN_DOMAINS[@]}"; do
            [ "$D" != "$HELD_OUT" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (DA) → domainnet DG (held_out=${HELD_OUT})"
        python scripts/train.py \
            --method ${METHOD} --setting dg --dataset domainnet \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${HELD_OUT} \
            --backbone resnet50 --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

echo "All DA→DG cross-setting experiments completed!"
