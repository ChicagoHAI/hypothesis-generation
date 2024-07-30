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

from hypothesis_generation.tasks import BaseTask
from hypothesis_generation.prompt import BasePrompt
from hypothesis_generation.data_loader import get_data
from hypothesis_generation.utils import (
    LLMWrapper,
    get_results,
    set_seed,
    create_directory,
    get_num_examples,
    GPT_MODELS,
    VALID_MODELS,
)
from hypothesis_generation.algorithm.summary_information import (
    SummaryInformation,
    dict_to_summary_information,
)
from hypothesis_generation.algorithm.inference import (
    DefaultInference,
    KNNInference,
    FilterAndWeightInference,
    SeparateStepsKNNInference,
    UpperboundInference,
)


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def setup_LLM(model, model_path, use_cache):
    api = LLMWrapper.from_model(model, path_name=model_path, use_cache=use_cache)
    return api


def main():
    start_time = time.time()

    task_config_path = "./data/retweet/config.yaml"
    hypothesis_file = f"./outputs/retweet/gpt-4o-mini/hyp_20/hypotheses_training_sample_final_seed_49_epoch_0.json"
    knn_hypotheses = 0
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

    api = setup_LLM("gpt-4o-mini", "", 0)

    task = BaseTask(task_extract_label, task_config_path)

    for hypothesis in dict:
        hyp_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

    assert knn_hypotheses <= len(
        hyp_bank
    ), f"The number of hypotheses chosen in KNN must be less than the total number of hypotheses"

    api = setup_LLM("gpt-4o-mini", "", use_cache=0)

    for seed in seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(
            num_train, num_test, num_val, seed
        )
        prompt_class = BasePrompt(task)
        inference_class = DefaultInference(api, prompt_class, train_data)

        if use_valid:
            print("Using validation data")
            test_data = val_data
        else:
            print("Using test data")

        pred_list, label_list = inference_class.run_inference_final(
            test_data, hyp_bank
        )

        if task.task_name == "shoe":
            accuracy = sum(
                [
                    1 if pred_list[i] == label_list[i] else 0
                    for i in range(len(pred_list))
                ]
            ) / len(pred_list)
            accuracy_all.append(accuracy)
            print(f"Accuracy for seed {seed}: {accuracy}")
        else:
            if isinstance(inference_class, UpperboundInference):
                continue
            tp, tn, fp, fn = get_results(task.task_name, pred_list, label_list)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            accuracy_all.append(accuracy)
            f1_all.append(f1)

            print(f"Accuracy for seed {seed}: {accuracy}")
            print(f"F1 for seed {seed}: {f1}")

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