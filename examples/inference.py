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

from hypogenic.extract_label import retweet_extract_label

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import (
    get_results,
    set_seed,
)
from hypogenic.LLM_wrapper import LocalVllmWrapper
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
from hypogenic.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HypoGenic")


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main():
    start_time = time.time()

    task_config_path = "./data/retweet/config.yaml"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    hypothesis_file = f"./outputs/retweet/{model_name}/hyp_20/hypotheses_training_sample_final_seed_49_epoch_0.json"
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

    task = BaseTask(task_config_path, extract_label=retweet_extract_label)

    for hypothesis in dict:
        hyp_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

    assert adaptive_num_hypotheses <= len(
        hyp_bank
    ), f"The number of hypotheses chosen in adaptive inference must be less than the total number of hypotheses"

    api = LocalVllmWrapper(model_name, model_path)

    for seed in seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(
            num_train, num_test, num_val, seed
        )
        prompt_class = BasePrompt(task)
        inference_class = UpperboundInference(api, prompt_class, train_data, task)

        if use_valid:
            logger.info("Using validation data")
            test_data = val_data
        else:
            logger.info("Using test data")

        pred_list, label_list = inference_class.run_inference_final(
            test_data, hyp_bank, adaptive_num_hypotheses=adaptive_num_hypotheses,
            cache_seed=None
        )

        results_dict = get_results(pred_list, label_list)

        logger.info(f"Accuracy for seed {seed}: {results_dict['accuracy']}")
        logger.info(f"F1 for seed {seed}: {results_dict['f1']}")

        # print the wrong indices
        wrong_indices = [
            i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
        ]
        logger.info(f"Wrong indices: {wrong_indices}")

    logger.info(f"Averaged accuracy: {sum(accuracy_all)/len(seeds)}")
    logger.info(f"Averaged F1: {sum(f1_all)/len(seeds)}")

    # print experiment info
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
