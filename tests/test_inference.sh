#!/bin/bash
set -e

task_list=("headline_binary" "hotel_reviews" "retweet" "shoe")
inference_list=("default" "filter_and_weight" "one_step_adaptive" "two_step_adaptive")

for task in "${task_list[@]}"; do
    for inference in "${inference_list[@]}"; do

        CUDA_VISIBLE_DEVICES=0 hypogenic_inference \
            --task_config_path ./data/${task}/config.yaml \
            --inference_style ${inference}

    done
done
