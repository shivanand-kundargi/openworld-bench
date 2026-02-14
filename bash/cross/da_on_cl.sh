#!/bin/bash
# ==========================================
# Cross-Setting: DA methods on CL setting
# ==========================================
# This runs DA methods (DANN, CDAN, MCD) in a Continual Learning setting.
# Data is presented sequentially as tasks, one at a time.
# DA methods receive data WITHOUT simultaneous source-target access.

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/SACK-CL/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/cross_da_on_cl"}

# DA Methods to evaluate
DA_METHODS=("dann" "cdan" "mcd")

# CL Datasets
CL_DATASETS=("imagenet_r" "cub200")

echo "============================================"
echo "CROSS-SETTING: DA methods → CL setting"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"
echo ""
echo "EXPERIMENTAL DESIGN:"
echo "- DA methods are run on CL benchmarks"
echo "- Data presented sequentially as tasks"
echo "- No simultaneous source-target access"
echo "- Metrics: Avg. Accuracy, Forgetting, BWT"
echo "============================================"

# Run experiments
for DATASET in "${CL_DATASETS[@]}"; do
    for METHOD in "${DA_METHODS[@]}"; do
        echo ""
        echo ">>> Running: ${METHOD} (DA) on ${DATASET} (CL setting)"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting cl \
            --dataset ${DATASET} \
            --data_root ${DATA_ROOT} \
            --n_tasks 10 \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} (DA) on ${DATASET} (CL setting)"
    done
done

echo ""
echo "============================================"
echo "All DA→CL cross-setting experiments completed!"
echo "============================================"
