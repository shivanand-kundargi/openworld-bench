#!/bin/bash
# ==========================================
# Native CL Experiments
# Run CL methods on CL setting (baseline)
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/SACK-CL/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/native_cl"}
BUFFER_SIZE=${BUFFER_SIZE:-500}

# CL Methods
CL_METHODS=("icarl" "der" "lwf")

# CL Datasets
CL_DATASETS=("imagenet_r" "cub200")

echo "============================================"
echo "Running Native CL Experiments"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "Buffer Size: ${BUFFER_SIZE}"
echo "============================================"

# Run experiments
for DATASET in "${CL_DATASETS[@]}"; do
    for METHOD in "${CL_METHODS[@]}"; do
        echo ""
        echo ">>> Running: ${METHOD} on ${DATASET}"
        
        python scripts/train.py \
            --method ${METHOD} \
            --setting cl \
            --dataset ${DATASET} \
            --data_root ${DATA_ROOT} \
            --n_tasks 10 \
            --buffer_size ${BUFFER_SIZE} \
            --backbone resnet50 \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.001 \
            --seed ${SEED} \
            --gpu ${GPU_ID} \
            --output_dir ${OUTPUT_DIR}
            
        echo ">>> Completed: ${METHOD} on ${DATASET}"
    done
done

echo ""
echo "============================================"
echo "All Native CL experiments completed!"
echo "============================================"
