#!/bin/bash

# Enter your path for the code repository
export PATH_PREFIX=/net/scratch/tejess
export CODE_REPO_PATH=${PATH_PREFIX}/hypothesis_generation

# Optional: specify port number and change directory to for redis server
USE_CACHE=0
export PORT=6382
if [ "$USE_CACHE" -eq 1 ]; then
    cd ${PATH_PREFIX}/redis-stable/src/
    ./redis-server --port $PORT &
fi

MODEL_PATH=/net/projects/chai-lab/tejes/Mixtral-8x7B-Instruct-v0.1
MODEL=claude_2
TASK=headline_binary
SEEDS=49
HYP_SIZE=20
DIR=${PATH_PREFIX}/hypothesis_generation/outputs/$TASK/$MODEL/hyp_${HYP_SIZE}/
UPDATE=sampling
INFERENCE=default
REPLACE=default
GENERATION=default

NUM_TRAIN=75
NUM_INIT=10
INIT_BATCH_SIZE=10
INIT_HYPOTHESES_PER_BATCH=10
K=10
SAVE_EVERY_N_EXAMPLES=10
NUM_HYPOTHESES_TO_UPDATE=1

UPDATE_BATCH_SIZE=10
UPDATE_HYPOTHESES_PER_BATCH=5

mkdir -p ${CODE_REPO_PATH}/logs/${MODEL}/

python ${PATH_PREFIX}/hypothesis_generation/code/algorithm/algorithm_generation.py \
    --seeds 49\
    --task $TASK \
    --model $MODEL \
    --model_path $MODEL_PATH \
    --use_cache $USE_CACHE \
    --num_train $NUM_TRAIN \
    --num_init $NUM_INIT \
    --init_batch_size $INIT_BATCH_SIZE \
    --init_hypotheses_per_batch $INIT_HYPOTHESES_PER_BATCH \
    --k $K \
    --save_every_n_examples $SAVE_EVERY_N_EXAMPLES \
    --max_num_hypotheses $HYP_SIZE \
    --num_hypotheses_to_update $NUM_HYPOTHESES_TO_UPDATE \
    --update_batch_size $UPDATE_BATCH_SIZE \
    --update_hypotheses_per_batch $UPDATE_HYPOTHESES_PER_BATCH \
    --update_style $UPDATE \
    --inference_style $INFERENCE \
    --replace_style $REPLACE \
    --generation_style $GENERATION \
    --output_folder $DIR \
    > ${CODE_REPO_PATH}/logs/${MODEL}/generation_${TASK}_train_${NUM_TRAIN}_hyp_${HYP_SIZE}.txt