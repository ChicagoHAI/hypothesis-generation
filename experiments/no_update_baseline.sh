#!/bin/bash

# Enter your path for the code repository
export PATH_PREFIX=/home/haokunliu
export CODE_REPO_PATH=${PATH_PREFIX}/hypothesis_generation

# Optional: specify port number and change directory to for redis server
cd ${PATH_PREFIX}/redis-stable/src/
./redis-server --port 6381 &

MODEL=Mixtral-8x7B
TASK=hotel_reviews
SEEDS=49
HYP_SIZE=20
INFERENCE=default
NUM_TRAIN=200
NUM_TEST=300
NUM_VAL=100
EPOCH=0
NUM_HYPOTHESIS=5

# create directories
print_directory=${CODE_REPO_PATH}/logs/${MODEL}/batched_hypotheses/${TASK}


python ${CODE_REPO_PATH}/code/baselines/batched_learning_generation.py \
    --seed $SEED \
    --num_train $NUM_TRAIN \
    --task $TASK \
    --model $MODEL \
    --num_hypothesis $NUM_HYPOTHESIS \
    > ${print_directory}/batched_gen_${MODEL}_train${NUM_TRAIN}_seed${SEED}_hypothesis${NUM_HYPOTHESIS}.txt

python ${CODE_REPO_PATH}/code/baselines/batched_learning_inference.py \
    --seed $SEED \
    --num_train $NUM_TRAIN \
    --task $TASK \
    --generation_model $MODEL \
    --inference_model $MODEL \
    --num_hypothesis $NUM_HYPOTHESIS \
    --num_test $NUM_TEST \
    --hypothesis_file ${print_directory}/batched_gen_${MODEL}_train${NUM_TRAIN}_seed${SEED}_hypothesis${NUM_HYPOTHESIS}.txt \
    > ${print_directory}/batched_inf_${MODEL}_train${NUM_TRAIN}.txt
