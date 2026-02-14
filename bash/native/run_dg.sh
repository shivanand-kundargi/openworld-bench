#!/bin/bash
# ==========================================
# Native DG Experiments
# Run DG methods on DG setting (baseline)
# ==========================================
# Runs on both Office-Home (65 classes) and DomainNet (345 classes)

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/native_dg"}

# DG Methods
DG_METHODS=("irm" "vrex" "coral")

echo "============================================"
echo "Running Native DG Experiments"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

# --- Office-Home (4 domains, 65 classes) ---
OH_DOMAINS=("Art" "Clipart" "Product" "Real_World")

for METHOD in "${DG_METHODS[@]}"; do
    for HELD_OUT in "${OH_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${OH_DOMAINS[@]}"; do
            if [ "$D" != "$HELD_OUT" ]; then
                SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
            fi
        done
        
        echo ""
        echo ">>> Running: ${METHOD} on office_home (held_out=${HELD_OUT})"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting dg \
            --dataset office_home \
            --source_domains ${SOURCE_DOMAINS} \
            --target_domain ${HELD_OUT} \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} on office_home (held_out=${HELD_OUT})"
    done
done

# --- DomainNet (6 domains, 345 classes) ---
DN_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

for METHOD in "${DG_METHODS[@]}"; do
    for HELD_OUT in "${DN_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${DN_DOMAINS[@]}"; do
            if [ "$D" != "$HELD_OUT" ]; then
                SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
            fi
        done
        
        echo ""
        echo ">>> Running: ${METHOD} on domainnet (held_out=${HELD_OUT})"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting dg \
            --dataset domainnet \
            --source_domains ${SOURCE_DOMAINS} \
            --target_domain ${HELD_OUT} \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} on domainnet (held_out=${HELD_OUT})"
    done
done

echo ""
echo "============================================"
echo "All Native DG experiments completed!"
echo "============================================"
