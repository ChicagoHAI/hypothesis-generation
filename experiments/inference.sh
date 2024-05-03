#!/bin/bash

# Enter your path for the code repository
export PATH_PREFIX=/net/scratch/tejess
export CODE_REPO_PATH=${PATH_PREFIX}/hypothesis_generation

# Optional: specify port number and change directory to for redis server
USE_CACHE=0
export PORT=6380
if [ "$USE_CACHE" -eq 1 ]; then
    cd ${PATH_PREFIX}/redis-stable/src/
    ./redis-server --port $PORT &
fi

MODEL_PATH=/net/projects/chai-lab/tejes/Mixtral-8x7B-Instruct-v0.1
# Set experiment parameters
MODEL=claude_2
TASK=headline_binary
SEEDS=49
HYP_SIZE=20
INFERENCE=default
NUM_TRAIN=75
NUM_TEST=25
NUM_VAL=10
EPOCH=0
FILE_LOAD_NUM_TRAIN=final

mkdir -p ${CODE_REPO_PATH}/logs/${MODEL}/${INFERENCE}

# Set file path for generated hypotheses
FILE=${PATH_PREFIX}/hypothesis_generation/outputs/$TASK/$MODEL/hyp_${HYP_SIZE}/hypotheses_training_sample_${FILE_LOAD_NUM_TRAIN}_seed_${SEEDS}_epoch_${EPOCH}.json
python ${PATH_PREFIX}/hypothesis_generation/code/algorithm/algorithm_inference.py \
    --seeds $SEEDS \
    --task $TASK \
    --verbose True\
    --model $MODEL \
    --model_path $MODEL_PATH \
    --use_cache $USE_CACHE \
    --num_train $NUM_TRAIN \
    --num_test $NUM_TEST \
    --inference_style $INFERENCE \
    --hypothesis_file $FILE \
    > ${CODE_REPO_PATH}/logs/${MODEL}/${INFERENCE}/${TASK}_train_${NUM_TRAIN}_hyp_${HYP_SIZE}.txt
