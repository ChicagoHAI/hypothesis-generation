import argparse
import re
import time
import pickle
import sys
import os
import math
import json

import random
from typing import Callable, Tuple, Union
import torch
import numpy as np

from hypothesis_teneration.tasks import BaseTask
from hypothesis_teneration.prompt import BasePrompt
from hypothesis_teneration.data_loader import get_data
from hypothesis_teneration.utils import LLMWrapper, set_seed, create_directory, get_num_examples, GPT_MODELS, VALID_MODELS
from hypothesis_teneration.algorithm.summary_information import SummaryInformation, dict_to_summary_information

from hypothesis_teneration.algorithm.generation import DefaultGeneration
from hypothesis_teneration.algorithm.inference import DefaultInference, KNNInference, FilterAndWeightInference, SeparateStepsKNNInference, UpperboundInference
from hypothesis_teneration.algorithm.replace import Replace
from hypothesis_teneration.algorithm.update import SamplingUpdate, DefaultUpdate

def load_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def setup_LLM(model, model_path, use_cache):
    api = LLMWrapper(model,
                     path_name=model_path,
                     use_cache=use_cache)
    return api


def main():
    # set up tools
    start_time = time.time()

    # TODO: What attributes should be class members?
    task_config_path = "./data/retweet/config.yaml"
    seeds = [49]
    max_num_hypotheses = 20
    output_folder = f"./outputs/retweet/gpt-4o-mini/hyp_{max_num_hypotheses}/"
    only_best_hypothesis = False
    old_hypothesis_file = None
    current_epoch = -1
    num_init = 10
    init_batch_size = 10
    init_hypotheses_per_batch = 10
    epoch_to_start_from = 0
    num_epochs = 1
    num_train = 75
    num_test = 25
    num_val = 25
    sample_num_to_restart_from = -1
    num_wrong_scale = 0.8
    k = 10
    use_system_prompt = True
    alpha = 5e-1
    update_batch_size = 10
    num_hypotheses_to_update = 1
    update_hypotheses_per_batch = 5
    save_every_n_examples = 10

    def task_extract_label(text: Union[str, None]) -> str:
        """
        `text` follows the format "the <label> tweet got more retweets"
        """
        if text is None:
            return 'other'
        text = text.lower()
        pattern = r"answer: the (\w+) tweet"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return 'other'

    create_directory(output_folder)
    api = setup_LLM("gpt-4o-mini", "", 0)
    print(only_best_hypothesis)

    task = BaseTask(task_extract_label, task_config_path)

    for seed in seeds:
        current_seed = seed

        set_seed(seed)
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = BasePrompt(task)
        inference_class = DefaultInference(api, prompt_class, train_data)
        generation_class = DefaultGeneration(api, prompt_class, inference_class)
        update_class = SamplingUpdate(generation_class, inference_class, Replace())

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            current_epoch = 0
            hypotheses_bank = generation_class.initialize_hypotheses(num_init, init_batch_size, init_hypotheses_per_batch, alpha, use_system_prompt)
            update_class.save_to_json(f"{num_init}_seed_{current_seed}", hypotheses_bank, output_folder, current_epoch)
        else:
            dict = load_dict(old_hypothesis_file)
            for hypothesis in dict:
                hypotheses_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

        for epoch in range(epoch_to_start_from, epoch_to_start_from + num_epochs):
            current_epoch = epoch
            hypotheses_bank = update_class.update(
                hypotheses_bank,
                sample_num_to_restart_from,
                num_init,
                current_epoch,
                epoch_to_start_from,
                num_wrong_scale,
                k,
                use_system_prompt,
                alpha,
                update_batch_size,
                num_hypotheses_to_update,
                update_hypotheses_per_batch,
                only_best_hypothesis,
                save_every_n_examples,
                current_seed
            )
            update_class.save_to_json(f"final_seed_{seed}", hypotheses_bank, output_folder, current_epoch)

    # print experiment info
    print(f'Total time: {time.time() - start_time} seconds')
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()
