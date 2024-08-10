import argparse
import re
import time
import pickle
import sys
import os
import math
import json

import random
from typing import Union
import torch
import numpy as np

from hypogenic.examples.extract_label import extract_label_register

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import (
    get_results,
    set_seed,
)
from hypogenic.LLM_wrapper import llm_wrapper_register
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
    dict_to_summary_information,
)
from hypogenic.algorithm.inference import inference_register


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_config_path", type=str, default="./data/retweet/config.yaml"
    )
    parser.add_argument("--hypothesis_file", type=str, default=None)
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/net/scratch/llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--model_type", type=str, default="vllm")

    parser.add_argument("--seeds", nargs="+", type=int, default=[49])

    parser.add_argument("--num_train", type=int, default=75)
    parser.add_argument("--num_test", type=int, default=25)
    parser.add_argument("--num_val", type=int, default=25)
    parser.add_argument("--use_valid", action="store_true", default=False)

    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--adaptive_threshold", type=float, default=0.0)
    parser.add_argument("--adaptive_num_hypotheses", type=int, default=5)
    parser.add_argument("--adaptive_num_examples", type=int, default=0)

    parser.add_argument("--use_cache", type=int, default=1)

    parser.add_argument("--inference_style", type=str, default="default")

    args = parser.parse_args()

    if args.hypothesis_file is None:
        args.hypothesis_file = f"./outputs/retweet/{args.model_name}/hyp_20/hypotheses_training_sample_final_seed_49_epoch_0.json"

    return args


def main():
    start_time = time.time()

    args = parse_args()

    accuracy_all = []
    f1_all = []

    hyp_dict = load_dict(args.hypothesis_file)
    hyp_bank = {}
    for hypothesis in hyp_dict:
        hyp_bank[hypothesis] = dict_to_summary_information(hyp_dict[hypothesis])

    assert args.adaptive_num_hypotheses <= len(
        hyp_bank
    ), f"The number of hypotheses chosen in adaptive inference must be less than the total number of hypotheses"

    task = BaseTask(args.task_config_path, from_register=extract_label_register)
    api = llm_wrapper_register.build(args.model_type)(args.model_name, args.model_path)
    prompt_class = BasePrompt(task)

    for seed in args.seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(
            args.num_train, args.num_test, args.num_val, seed
        )

        inference_class = inference_register.build(args.inference_style)(
            api, prompt_class, train_data, task
        )

        if args.use_valid:
            print("Using validation data")
            test_data = val_data
        else:
            print("Using test data")

        pred_list, label_list = inference_class.run_inference_final(
            test_data,
            hyp_bank,
            use_cache=args.use_cache,
            k=args.k,
            adaptive_threshold=args.adaptive_threshold,
            adaptive_num_hypotheses=args.adaptive_num_hypotheses,
            adaptive_num_examples=args.adaptive_num_examples,
        )

        results_dict = get_results(pred_list, label_list)

        print(f"Accuracy for seed {seed}: {results_dict['accuracy']}")
        print(f"F1 for seed {seed}: {results_dict['f1']}")

        # print the wrong indices
        wrong_indices = [
            i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
        ]
        print(f"Wrong indices: {wrong_indices}")

    print(f"Averaged accuracy: {sum(accuracy_all)/len(args.seeds)}")
    print(f"Averaged F1: {sum(f1_all)/len(args.seeds)}")

    # print experiment info
    print(f"Total time: {time.time() - start_time} seconds")
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
