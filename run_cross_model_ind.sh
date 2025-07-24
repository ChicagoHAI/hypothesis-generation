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

# Define tasks with corresponding models and cross folders as a nested dictionary
declare -A TASKS
TASKS["deceptive_reviews,gpt-4o-mini"]="hyp_20_refine_and_paper"
TASKS["deceptive_reviews,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_with_paper"
TASKS["deceptive_reviews,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_refine_and_paper"
TASKS["deceptive_reviews,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_refine_and_paper"
TASKS["llamagc_detect,gpt-4o-mini"]="hyp_20_hypogenic_and_paper"
TASKS["llamagc_detect,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["llamagc_detect,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["llamagc_detect,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_hypogenic_and_paper"
TASKS["gptgc_detect,gpt-4o-mini"]="hyp_20_hypogenic_and_paper"
TASKS["gptgc_detect,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_refine_and_paper"
TASKS["gptgc_detect,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["gptgc_detect,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_hypogenic_and_paper"
TASKS["persuasive_pairs,gpt-4o-mini"]="hyp_20_refine_and_paper"
TASKS["persuasive_pairs,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["persuasive_pairs,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_refine_and_paper"
TASKS["persuasive_pairs,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_hypogenic_and_paper"
TASKS["dreaddit,gpt-4o-mini"]="hyp_20_hypogenic_and_paper"
TASKS["dreaddit,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_with_paper"
TASKS["dreaddit,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_with_paper"
TASKS["dreaddit,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_with_paper"
TASKS["headline_binary,gpt-4o-mini"]="hyp_20_with_paper"
TASKS["headline_binary,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["headline_binary,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_hypogenic_and_paper"
TASKS["headline_binary,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_hypogenic_and_paper"
TASKS["retweet,gpt-4o-mini"]="hyp_20_refine_and_paper"
TASKS["retweet,Qwen/Qwen2.5-72B-Instruct"]="hyp_20_with_paper"
TASKS["retweet,meta-llama/Meta-Llama-3.1-70B-Instruct"]="hyp_20_refine_and_paper"
TASKS["retweet,DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"]="hyp_20_refine_and_paper"

# Algorithm settings
MAX_NUM_HYPOTHESES=20
NUM_TRAIN=200
NUM_TEST=300
SEED=42

# Check version of Python on machine
PYTHON="python"

if command -v python &>/dev/null; then
    PYTHON="python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
fi

# Iterate through each task and model pair
for TASK_AND_MODEL in "${!TASKS[@]}"; do
    IFS=',' read -r TASK_NAME CROSS_MODEL_NAME <<< "$TASK_AND_MODEL"
    CROSS_HYP_FOLDER=${TASKS[$TASK_AND_MODEL]}

    echo "Running pipeline for task: $TASK_NAME"

    # Skip if CROSS_MODEL_NAME equals MODEL_NAME
    if [ "$CROSS_MODEL_NAME" = "$MODEL_NAME" ]; then
        echo "Skipping task: $TASK_NAME with model: $CROSS_MODEL_NAME (same as MODEL_NAME)"
        continue
    fi

    echo "Using cross model: $CROSS_MODEL_NAME with folder: $CROSS_HYP_FOLDER"

    # Create command with required arguments
    CMD="${PYTHON} pipeline.py \
        --model_type ${MODEL_TYPE} \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --seed ${SEED} \
        --max_num_hypotheses ${MAX_NUM_HYPOTHESES} \
        --num_train ${NUM_TRAIN} \
        --num_test ${NUM_TEST} \
        --run_cross_model \
        --cross_model_name ${CROSS_MODEL_NAME} \
        --cross_hyp_folder ${CROSS_HYP_FOLDER}"

    if [ "${MODEL_TYPE}" = "vllm" ]; then
        CMD="${CMD} --model_path ${MODEL_PATH}"
    fi

    echo "Executing command: $CMD"
    eval $CMD

    echo "Completed task: $TASK_NAME with model: $CROSS_MODEL_NAME"
    echo "----------------------------------------"
done
