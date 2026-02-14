#!/bin/bash
# ==========================================
# Native DA Experiments
# Run DA methods on DA setting (baseline)
# ==========================================
# Runs on both Office-Home (65 classes) and DomainNet (345 classes)

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/native_da"}

# DA Methods
DA_METHODS=("dann" "cdan" "mcd")

echo "============================================"
echo "Running Native DA Experiments"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

# --- Office-Home (4 domains, 65 classes) ---
OH_DOMAINS=("Art" "Clipart" "Product" "Real_World")

for METHOD in "${DA_METHODS[@]}"; do
    for TARGET in "${OH_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${OH_DOMAINS[@]}"; do
            if [ "$D" != "$TARGET" ]; then
                SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
            fi
        done
        
        echo ""
        echo ">>> Running: ${METHOD} on office_home (target=${TARGET})"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting da \
            --dataset office_home \
            --source_domains ${SOURCE_DOMAINS} \
            --target_domain ${TARGET} \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} on office_home (target=${TARGET})"
    done
done

# --- DomainNet (6 domains, 345 classes) ---
DN_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

for METHOD in "${DA_METHODS[@]}"; do
    for TARGET in "${DN_DOMAINS[@]}"; do
        SOURCE_DOMAINS=""
        for D in "${DN_DOMAINS[@]}"; do
            if [ "$D" != "$TARGET" ]; then
                SOURCE_DOMAINS="$SOURCE_DOMAINS $D"
            fi
        done
        
        echo ""
        echo ">>> Running: ${METHOD} on domainnet (target=${TARGET})"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting da \
            --dataset domainnet \
            --source_domains ${SOURCE_DOMAINS} \
            --target_domain ${TARGET} \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} on domainnet (target=${TARGET})"
    done
done

echo ""
echo "============================================"
echo "All Native DA experiments completed!"
echo "============================================"
