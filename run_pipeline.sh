#!/bin/bash

# Model settings
MODEL_TYPE="gpt"  
MODEL_NAME="gpt-4o-mini" 
MODEL_PATH=""  # only needed for local models

# Define list of tasks to run
TASKS=(
    "deceptive_reviews"
    "llamagc_detect"
    "gptgc_detect"
    "persuasive_pairs"
    "dreaddit"
)

# Define methods to run
# METHODS=(
#     "zero_shot"
#     "few_shot"
#     "zero_shot_gen"
#     "hypogenic"
# )

METHODS=(
    "io_refine"
)

# Algorithm settings
MAX_NUM_HYPOTHESES=10
NUM_TRAIN=20
NUM_TEST=300
SEED=42

# Iterate through each task
for TASK_NAME in "${TASKS[@]}"; do
    echo "Running pipeline for task: $TASK_NAME"
    
    # Create command with base required arguments
    CMD="python pipeline.py \
        --model_type ${MODEL_TYPE} \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --seed ${SEED} \
        --max_num_hypotheses ${MAX_NUM_HYPOTHESES} \
        --num_train ${NUM_TRAIN} \
        --num_test ${NUM_TEST}"

    if [ "${MODEL_TYPE}" = "vllm" ]; then
        CMD="${CMD} --model_path ${MODEL_PATH}"
    fi

    # Default methods to run in pipeline
    # Original commented version kept for reference
    # CMD="${CMD} \
    #     --run_zero_shot \
    #     --run_few_shot \
    #     --run_zero_shot_gen \
    #     --run_hypogenic \
    #     --do_train"

    # Add methods dynamically
    for METHOD in "${METHODS[@]}"; do
        CMD="${CMD} --run_${METHOD}"
    done
    CMD="${CMD} 
    --do_train
    "

    # --use_ood

    
    # Additional methods
    # Uncomment if needed
    # --run_only_paper \
    # --run_hyperwrite \
    # --run_notebooklm \
    # --run_hyporefine \
    # --run_union_hypo \
    # --run_union_refine \
    # --run_io_refine \
    # --run_cross_model \
    # --use_val \
    # --multihyp \
    # --use_refine

    echo "Executing command: $CMD"
    eval $CMD
    
    echo "Completed task: $TASK_NAME"
    echo "----------------------------------------"
done
