#!/bin/bash

# Model settings
MODEL_TYPE="gpt"
MODEL_NAME="gpt-4o-mini"

# MODEL_TYPE="vllm"  
# MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/Meta-Llama-3.1-70B-Instruct"  # only needed for local models

# MODEL_TYPE="vllm"  
# MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/Qwen2.5-72B-Instruct"  # only needed for local models

# MODEL_TYPE="vllm"  
# MODEL_NAME="DeepSeek/DeepSeek-R1-Distill-Llama-70B-local" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/DeepSeek-R1-Distill-Llama-70B-local"  # only needed for local models

# Option to generate config before running the pipeline
GENERATE_CONFIG=true

# Define list of tasks to run
TASKS=(
    "dataset"
)

# Define methods to run

METHODS=(
    # "zero_shot"
    # "few_shot"
    # "zero_shot_gen"
    # "only_paper"
    "hypogenic"
    # "hyporefine"
    # "union_hypo"
    # "union_refine"
    # "io_refine"
)

# Algorithm settings
MAX_NUM_HYPOTHESES=20
NUM_TRAIN=200
NUM_TEST=300
SEED=42

# User instructions
RESEARCH_QUESTION=""
INSTRUCTIONS=""

PYTHON="python"

if command -v python &>/dev/null; then
    PYTHON="python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
fi

# Iterate through each task
for TASK_NAME in "${TASKS[@]}"; do
    echo "Running pipeline for task: $TASK_NAME"
    
    # Create command with base required arguments
    CMD="${PYTHON} pipeline.py \
        --model_type ${MODEL_TYPE} \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --seed ${SEED} \
        --max_num_hypotheses ${MAX_NUM_HYPOTHESES} \
        --num_train ${NUM_TRAIN} \
        --num_test ${NUM_TEST}" \

    if [ "${GENERATE_CONFIG}" = true ]; then
        CMD="${CMD} --generate_config"
    fi

    if [ "${RESEARCH_QUESTION}" != "" ]; then
        CMD="${CMD} --research_question ${RESEARCH_QUESTION}"
    fi
    
    if [ "${INSTRUCTIONS}" != "" ]; then
        CMD="${CMD} --instructions ${INSTRUCTIONS}"
    fi


    if [ "${MODEL_TYPE}" = "vllm" ]; then
        CMD="${CMD} --model_path ${MODEL_PATH}"
    fi

    # Add methods dynamically
    for METHOD in "${METHODS[@]}"; do
        CMD="${CMD} --run_${METHOD}"
    done

    # Check if TASK_NAME contains "journal" and add argument
    if [[ "${TASK_NAME}" == *"journal"* ]]; then
        CMD="${CMD} --literature_folder=\"paper_citations\""
    fi

    # IND setup
    CMD="${CMD} 
    --do_train
    "

    # OOD setup
    # CMD="${CMD} 
    # --use_ood
    # "

    echo "Executing command: $CMD"
    eval $CMD
    
    echo "Completed task: $TASK_NAME"
    echo "----------------------------------------"
done