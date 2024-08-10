#!/bin/bash
set -e

task_list=("headline_binary" "hotel_reviews" "retweet" "shoe")
inference_list=("default" "filter_and_weight" "one_step_adaptive" "two_step_adaptive")
update_list=("default" "sampling")

for task in "${task_list[@]}"; do
    for inference in "${inference_list[@]}"; do
        for update in "${update_list[@]}"; do

            CUDA_VISIBLE_DEVICES=0 hypogenic_generation \
                --task_config_path ./data/${task}/config.yaml \
                --inference_style ${inference} \
                --update_style ${update}

        done
    done
done
