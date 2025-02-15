#!/bin/bash

# Model settings
MODEL_TYPE="gpt"  
MODEL_NAME="gpt-4o-mini" 
TASK_NAME="admission/level_2/size_5" 
MODEL_PATH=""  # only needed for local models

# Algorithm settings
MAX_NUM_HYPOTHESES=10
NUM_TRAIN=20
NUM_TEST=20
SEED=42

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
# CMD="${CMD} \
#     --run_zero_shot \
#     --run_few_shot \
#     --run_zero_shot_gen \
#     --run_hypogenic \
#     --do_train"


CMD="${CMD} \
    --run_io_refine
    --do_train"
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

eval $CMD
