import argparse
import logging
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

from hypogenic.extract_label import extract_label_register

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import set_seed


from hypogenic.algorithm.summary_information import SummaryInformation
from hypogenic.LLM_wrapper import (
    llm_wrapper_register,
)

from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import SamplingUpdate, DefaultUpdate
from hypogenic.logger_config import LoggerConfig

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("HypoGenic")


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def main():
    # set up tools
    start_time = time.time()

    # For detailed argument descriptions, please run `hypogenic_generation --help` or see `hypogenic_cmd/generation.py`
    task_config_path = "./data/retweet/config.yaml"
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    # model_type = "vllm"

    model_name = "gpt-4o-mini"
    model_path = None
    model_type = "gpt"
    
    max_num_hypotheses = 20
    output_folder = f"./outputs/retweet/{model_name}/hyp_{max_num_hypotheses}/"
    old_hypothesis_file = None
    num_init = 10
    num_train = 75
    num_test = 25
    num_val = 25
    k = 10
    alpha = 5e-1
    update_batch_size = 10
    num_hypotheses_to_update = 1
    save_every_10_examples = 10
    init_batch_size = 10
    init_hypotheses_per_batch = 10
    cache_seed = None
    temperature = 1e-5
    max_tokens = 1000
    seeds = [42]

    os.makedirs(output_folder, exist_ok=True)
    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=model_path)

    # If implementing a new task, you need to create a new extract_label function and pass in the Task constructor.
    # For existing tasks (shoe, hotel_reviews, retweet, headline_binary), you can use the extract_label_register.
    task = BaseTask(
        task_config_path, extract_label=None, from_register=extract_label_register
    )

    for seed in seeds:
        set_seed(seed)
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = BasePrompt(task)
        inference_class = DefaultInference(api, prompt_class, train_data, task)
        generation_class = DefaultGeneration(api, prompt_class, inference_class, task)

        update_class = DefaultUpdate(
            generation_class=generation_class,
            inference_class=inference_class,
            replace_class=DefaultReplace(max_num_hypotheses),
            save_path=output_folder,
            num_init=num_init,
            k=k,
            alpha=alpha,
            update_batch_size=update_batch_size,
            num_hypotheses_to_update=num_hypotheses_to_update,
            save_every_n_examples=save_every_10_examples,
        )

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            hypotheses_bank = update_class.batched_initialize_hypotheses(
                num_init,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                cache_seed=cache_seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            update_class.save_to_json(
                hypotheses_bank,
                sample=num_init,
                seed=seed,
                epoch=0,
            )
        else:
            dict = load_dict(old_hypothesis_file)
            for hypothesis in dict:
                hypotheses_bank[hypothesis] = SummaryInformation.from_dict(
                    dict[hypothesis]
                )
        for epoch in range(1):
            hypotheses_bank = update_class.update(
                current_epoch=epoch,
                hypotheses_bank=hypotheses_bank,
                current_seed=seed,
                cache_seed=cache_seed,
            )
            update_class.save_to_json(
                hypotheses_bank,
                sample="final",
                seed=seed,
                epoch=epoch,
            )

    # print experiment info
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
