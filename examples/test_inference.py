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

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.data_loader import get_data
from hypogenic.utils import (
    get_results,
    set_seed,
)
from hypogenic.LLM_wrapper import LocalModelWrapper
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
    dict_to_summary_information,
)
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main():
    start_time = time.time()

    task_config_path = "../data/retweet/config.yaml"
    hypothesis_file = f"./outputs/retweet/gpt-4o-mini/hyp_20/hypotheses_training_sample_final_seed_49_epoch_0.json"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    adaptive_num_hypotheses = 5
    num_train = 75
    num_test = 25
    num_val = 10
    use_valid = False

    seeds = [49]
    accuracy_all = []
    f1_all = []
    dict = load_dict(hypothesis_file)
    hyp_bank = {}

    def task_extract_label(text: Union[str, None]) -> str:
        """
        `text` follows the format "the <label> tweet got more retweets"
        """
        if text is None:
            return "other"
        text = text.lower()
        pattern = r"answer: the (\w+) tweet"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return "other"

    task = BaseTask(task_extract_label, task_config_path)

    for hypothesis in dict:
        hyp_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

    assert adaptive_num_hypotheses <= len(
        hyp_bank
    ), f"The number of hypotheses chosen in adaptive inference must be less than the total number of hypotheses"

    api = LocalModelWrapper(model_name, model_path, use_vllm=True)

    for seed in seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(
            num_train, num_test, num_val, seed
        )
        prompt_class = BasePrompt(task)
        inference_class = UpperboundInference(api, prompt_class, train_data, task)

        if use_valid:
            print("Using validation data")
            test_data = val_data
        else:
            print("Using test data")

        pred_list, label_list = inference_class.run_inference_final(
            test_data, hyp_bank, adaptive_num_hypotheses=adaptive_num_hypotheses
        )

        results_dict = get_results(pred_list, label_list)

        print(f"Accuracy for seed {seed}: {results_dict['accuracy']}")
        print(f"F1 for seed {seed}: {results_dict['f1']}")

        # print the wrong indices
        wrong_indices = [
            i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
        ]
        print(f"Wrong indices: {wrong_indices}")

    print(f"Averaged accuracy: {sum(accuracy_all)/len(seeds)}")
    print(f"Averaged F1: {sum(f1_all)/len(seeds)}")

    # print experiment info
    print(f"Total time: {time.time() - start_time} seconds")
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
