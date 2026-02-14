#!/bin/bash
# ==========================================
# Cross-Setting: CL methods on DA setting
# ==========================================
# Runs CL methods (iCaRL, DER, LwF) in a Domain Adaptation setting.
# Includes both Office-Home and DomainNet.

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/cross_cl_on_da"}
BUFFER_SIZE=${BUFFER_SIZE:-500}

# CL Methods
CL_METHODS=("icarl" "der" "lwf")

echo "============================================"
echo "CROSS-SETTING: CL methods → DA setting"
echo "GPU: ${GPU_ID}, Seed: ${SEED}, Buffer: ${BUFFER_SIZE}"
echo "============================================"

# --- Office-Home ---
OH_DOMAINS=("Art" "Clipart" "Product" "Real_World")
for METHOD in "${CL_METHODS[@]}"; do
    for TARGET in "${OH_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${OH_DOMAINS[@]}"; do
            [ "$D" != "$TARGET" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (CL) → office_home DA (target=${TARGET})"
        python scripts/train.py \
            --method ${METHOD} --setting da --dataset office_home \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${TARGET} \
            --buffer_size ${BUFFER_SIZE} --backbone resnet50 \
            --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

# --- DomainNet ---
DN_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
for METHOD in "${CL_METHODS[@]}"; do
    for TARGET in "${DN_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${DN_DOMAINS[@]}"; do
            [ "$D" != "$TARGET" ] && SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
        done
        echo ">>> ${METHOD} (CL) → domainnet DA (target=${TARGET})"
        python scripts/train.py \
            --method ${METHOD} --setting da --dataset domainnet \
            --source_domains ${SOURCE_DOMAINS} --target_domain ${TARGET} \
            --buffer_size ${BUFFER_SIZE} --backbone resnet50 \
            --epochs 50 --batch_size 32 --lr 0.001 \
            --seed ${SEED} --gpu ${GPU_ID} --output_dir ${OUTPUT_DIR}
    done
done

echo "All CL→DA cross-setting experiments completed!"
