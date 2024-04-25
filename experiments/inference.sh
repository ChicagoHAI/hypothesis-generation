#!/bin/bash

# Enter your path for the code repository
export PATH_PREFIX=/home/haokunliu
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
MODEL=Mixtral-8x7B
TASK=hotel_reviews
SEEDS=49
HYP_SIZE=20
INFERENCE=default
NUM_TRAIN=200
NUM_TEST=300
NUM_VAL=100
EPOCH=0

# Set file path for generated hypotheses
FILE=${PATH_PREFIX}/hypothesis_generation/outputs/$TASK/$MODEL/hyp_${HYP_SIZE}/hypotheses_training_sample_${NUM_TRAIN}_seed_${SEEDS}_epoch_${EPOCH}.json
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
    > ${CODE_REPO_PATH}/logs/${INFERENCE}/${model}/${TASK}_train_${NUM_TRAIN}_hyp_${HYP_SIZE}.txt
