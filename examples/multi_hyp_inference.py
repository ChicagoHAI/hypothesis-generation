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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypogenic.extract_label import retweet_extract_label, hotel_reviews_extract_label, persuasive_pairs_extract_label, dreaddit_extract_label
from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import (
    get_results,
    set_seed,
)
from hypogenic.LLM_wrapper import LocalVllmWrapper, GPTWrapper
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)
from hypogenic.algorithm.generation.utils import extract_hypotheses
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)
from hypogenic.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HypoGenic")

from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.inference import MultiHypDefaultInference
from hypothesis_agent.utils import SpecificityBooster

def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def change_missing_labels(pred_list, label_list):
    labels = []
    for i in range(0, len(label_list)):
        if label_list[i] not in labels:
            labels.append(label_list[i])
    for i in range(0, len(pred_list)):
        if pred_list[i] == "other":
            if label_list[i] == labels[0]:
                pred_list[i] = labels[1]
            else:
                pred_list[i] = labels[0]
    return pred_list

def main():
    start_time = time.time()
    model_name = "gpt-4o-mini"
    # model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    use_ood = False # set to True if testing on OOD data
    if use_ood:
        config_version = "_ood"
    else:
        config_version = ""
    task_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), f"data/dreaddit/config{config_version}.yaml"
    )
    if model_name ==  "meta-llama/Meta-Llama-3.1-70B-Instruct":
        model_path = "/net/projects/chai-lab/shared_models/Meta-Llama-3.1-70B-Instruct"

    max_num_hypotheses = 20
    adaptive_num_hypotheses = 5
    num_train = 200
    num_test = 640
    num_val = 640
    use_valid = False
    generation_seed = 42
    seeds = [42]
    epoch = 0
    cache_seed = None
    use_refine = False
    max_concurrent = 32
    temperature = 1e-5
    max_tokens = 4000
    for sample in ['final']:
        # change this file path to test other hypotheses
        hypothesis_file = "./outputs/union/init_both_multi_refine/dreaddit/gpt-4o-mini/hyp_20/refine_6/hypotheses_training_sample_final_seed_42_epoch_0.json"
        accuracy_all = []
        f1_all = []
        dict = load_dict(hypothesis_file)
        hyp_bank = {}

        task = BaseTask(task_config_path, extract_label=dreaddit_extract_label)
        prompt_class = TestPrompt(task)

        for hypothesis in dict:
            tmp_dict = dict[hypothesis].copy()
            tmp_dict.pop("num_select", None)
            hyp_bank[hypothesis] = SummaryInformation.from_dict(tmp_dict)

        assert adaptive_num_hypotheses <= len(
            hyp_bank
        ), f"The number of hypotheses chosen in adaptive inference must be less than the total number of hypotheses"

        if "gpt" in model_name:
            api = GPTWrapper(model_name)
        else:
            api = LocalVllmWrapper(model_name, model_path, gpu_memory_utilization=0.95)

        for seed in seeds:
            set_seed(seed)
            train_data, test_data, val_data = task.get_data(
                num_train, num_test, num_val, seed
            )
            inference_class = MultiHypDefaultInference(api, prompt_class, train_data, task)

            if use_valid:
                logger.info("Using validation data")
                test_data = val_data
            else:
                logger.info("Using test data")

            pred_list, label_list = inference_class.multiple_hypotheses_run_inference_final(
                test_data,
                hyp_bank,
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                generate_kwargs={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

            pred_list = change_missing_labels(pred_list, label_list) # change prediction="other" (cannot extract label) to false labels for F1 calculation

            results_dict = get_results(pred_list, label_list)

            logger.info(f"Accuracy for seed {seed}: {results_dict['accuracy']}")
            logger.info(f"F1 for seed {seed}: {results_dict['f1']}")

            wrong_indices = [
                i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
            ]
            accuracy_all.append(results_dict["accuracy"])
            f1_all.append(results_dict["f1"])

        logger.info(f"Averaged accuracy: {sum(accuracy_all)/len(accuracy_all)}")
        logger.info(f"Averaged F1: {sum(f1_all)/len(f1_all)}")

        logger.info(f"Total time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
