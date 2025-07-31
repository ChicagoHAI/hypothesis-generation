#!/bin/bash

# Model settings
MODEL_TYPE="gpt"
MODEL_NAME="gpt-4o-mini"

# MODEL_TYPE="vllm"  
# MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/Meta-Llama-3.1-70B-Instruct"  # only needed for local models

# MODEL_TYPE="vllm"  
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/Meta-Llama-3-8B-Instruct"  # only needed for local models

# MODEL_TYPE="vllm"  
# MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/Qwen2.5-72B-Instruct"  # only needed for local models

# MODEL_TYPE="vllm"  
# MODEL_NAME="DeepSeek/DeepSeek-R1-Distill-Llama-70B-local" 
# MODEL_PATH="/net/projects/chai-lab/shared_models/DeepSeek-R1-Distill-Llama-70B-local"  # only needed for local models

# Option to generate config before running the pipeline
GENERATE_CONFIG=false

# Define list of tasks to run

# Synthetic tasks
TASKS=(
    "admission/level_1/base"
    # "admission/level_2/depth_2"
    # "admission/level_2/distractor_3"
    # "admission/level_2/noise_10"
    # "admission/level_2/size_5"
    # "shoe"
    # "shoe_two_level/simple"
    # "shoe_two_level/hard"
    # "election/level0"
    # "preference/level0"
    # "election/level1"
    # "preference/level1"
    # "election/level2"
    # "preference/level2"
    # "election/level3"
    # "preference/level3"
    # "election/level4"
    # "preference/level4"
    # "election/level5"
    # "preference/level5"
    # "election/level0_nosubtlety"
    # "preference/level0_nosubtlety"
    # "election/level1_nosubtlety"
    # "preference/level1_nosubtlety"
    # "election/level2_nosubtlety"
    # "preference/level2_nosubtlety"
    # "election/level3_nosubtlety"
    # "preference/level3_nosubtlety"
    # "election/level4_nosubtlety"
    # "preference/level4_nosubtlety"
    # "election/level5_nosubtlety"
    # "preference/level5_nosubtlety"
    # 'election_controlled/5_0_0'
    # 'election_controlled/10_0_0'
    # 'election_controlled/15_0_0'
    # 'election_controlled/20_0_0'
    # 'election_controlled/20_0.1_0'
    # 'election_controlled/20_0.2_0'
    # 'election_controlled/20_0.3_0'
    # 'election_controlled/20_0_0.1'
    # 'election_controlled/20_0_0.2'
    # 'election_controlled/20_0_0.3'
    # 'election_controlled/20_0.1_0.1'
    # 'election_controlled/20_0.2_0.2'
    # 'election_controlled/20_0.3_0.3'
    # 'preference_controlled/5_0_0'
    # 'preference_controlled/10_0_0'
    # 'preference_controlled/15_0_0'
    # 'preference_controlled/20_0_0'
    # 'preference_controlled/20_0.1_0'
    # 'preference_controlled/20_0.2_0'
    # 'preference_controlled/20_0.3_0'
    # 'preference_controlled/20_0_0.1'
    # 'preference_controlled/20_0_0.2'
    # 'preference_controlled/20_0_0.3'
    # 'preference_controlled/20_0.1_0.1'
    # 'preference_controlled/20_0.2_0.2'
    # 'preference_controlled/20_0.3_0.3'
    # 'admission/level_3/depth_3'
    # 'admission/level_3/distractor_6'
    # 'admission/level_3/noise_20'
    # 'admission/level_3/size_10'
    # 'admission/level_4/depth_4'
    # 'admission/level_4/distractor_10'
    # 'admission/level_4/noise_30'
    # 'admission/level_4/size_15'
    # "admission_adv/level_1/base"
    # "admission_adv/level_2/depth_2"
    # "admission_adv/level_2/distractor_3"
    # "admission_adv/level_2/noise_10"
    # "admission_adv/level_2/size_5"
    # 'admission_adv/level_3/depth_3'
    # 'admission_adv/level_3/distractor_6'
    # 'admission_adv/level_3/noise_20'
    # 'admission_adv/level_3/size_10'
    # 'admission_adv/level_4/depth_4'
    # 'admission_adv/level_4/distractor_10'
    # 'admission_adv/level_4/noise_30'
    # 'admission_adv/level_4/size_15'
    # 'election/counterfactual/normal'
    # 'election/counterfactual/counterfactual'
)

# Define methods to run

METHODS=(
    # "zero_shot"
    # "few_shot"
    # "zero_shot_gen"
    "hypogenic"
    # "io_refine"
)

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
        --num_test ${NUM_TEST}"

    if [ "${GENERATE_CONFIG}" = true ]; then
        CMD="${CMD} --generate_config"
    fi

    if [ "${MODEL_TYPE}" = "vllm" ]; then
        CMD="${CMD} --model_path ${MODEL_PATH}"
    fi

    # Add methods dynamically
    for METHOD in "${METHODS[@]}"; do
        CMD="${CMD} --run_${METHOD}"
    done
    # IND setup
    CMD="${CMD} 
    --do_train
    "

    echo "Executing command: $CMD"
    eval $CMD
    
    echo "Completed task: $TASK_NAME"
    echo "----------------------------------------"
done