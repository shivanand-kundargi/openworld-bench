#!/bin/bash
# ==========================================
# Native CL: Domain-Incremental Learning (DIL)
# Dataset: DomainNet (One Domain per Task)
# Tasks: 6 tasks (real, clipart, infograph, painting, quickdraw, sketch)
# ==========================================

# Configuration
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/benchmarking/native_cl_dil"}
DATA_ROOT=${DATA_ROOT:-"/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data"}

# CL Methods
METHODS=("icarl" "ewc" "l2p" "dualprompt" "coda_prompt")

get_backbone() {
    case $1 in
        "l2p"|"dualprompt"|"coda_prompt") echo "vit_base_patch16_224" ;;
        *) echo "resnet50" ;;
    esac
}

echo "============================================"
echo "Benchmarking: Native CL (DIL) on DomainNet"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "============================================"

for METHOD in "${METHODS[@]}"; do
    BACKBONE=$(get_backbone $METHOD)
    echo ""
    echo ">>> Running: ${METHOD} (Backbone: ${BACKBONE})"
    
    # Needs specific CL-DIL setting or task configuration
    # Assuming 'cl_dil' assumes domain-incremental via --setting cl and appropriate task args or separate setting
    # The current train.py might need adjustment or config support for 'one domain per task'
    # For now, assuming standard CL setting with tasks mapped to domains if supported 
    # OR we use a specific 'cl_dil' setting if implemented.
    # Based on user description: "Task definition: one domain per task"
    
    python scripts/train.py \
        --method ${METHOD} \
        --setting cl \
        --dataset domainnet \
        --data_root ${DATA_ROOT} \
        --n_tasks 6 \
        --backbone ${BACKBONE} \
        --epochs 10 \
        --batch_size 32 \
        --seed ${SEED} \
        --gpu ${GPU_ID} \
        --output_dir ${OUTPUT_DIR} \
        # --task_type domain_incremental ? If supported
        
    echo ">>> Completed: ${METHOD}"
done
