#!/bin/bash
# ==========================================
# Cross-Setting: DG methods on CL setting
# ==========================================
# This runs DG methods (IRM, VREx, CORAL) in a Continual Learning setting.
# Data is presented sequentially. DG methods cannot access past tasks simultaneously.

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/SACK-CL/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/cross_dg_on_cl"}

# DG Methods to evaluate
DG_METHODS=("irm" "vrex" "coral")

# CL Datasets
CL_DATASETS=("imagenet_r" "cub200")

echo "============================================"
echo "CROSS-SETTING: DG methods → CL setting"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"
echo ""
echo "EXPERIMENTAL DESIGN:"
echo "- DG methods run on CL benchmarks"
echo "- Invariance constraints applied within each task"
echo "- No simultaneous multi-domain access"
echo "- Metrics: Avg. Accuracy, Forgetting, BWT"
echo "============================================"

# Run experiments
for DATASET in "${CL_DATASETS[@]}"; do
    for METHOD in "${DG_METHODS[@]}"; do
        echo ""
        echo ">>> Running: ${METHOD} (DG) on ${DATASET} (CL setting)"
        
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
            
        echo ">>> Completed: ${METHOD} (DG) on ${DATASET} (CL setting)"
    done
done

echo ""
echo "============================================"
echo "All DG→CL cross-setting experiments completed!"
echo "============================================"
