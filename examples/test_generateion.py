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

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.data_loader import get_data
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LocalModelWrapper
from hypogenic.algorithm.summary_information import (
    dict_to_summary_information,
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


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main():
    # set up tools
    start_time = time.time()

    task_config_path = "../data/retweet/config.yaml"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    max_num_hypotheses = 20
    output_folder = f"./outputs/retweet/{model_name}/hyp_{max_num_hypotheses}/"
    old_hypothesis_file = None
    num_init = 10
    num_train = 75
    num_test = 25
    num_val = 25

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

    os.makedirs(output_folder, exist_ok=True)
    api = LocalModelWrapper(model_name, model_path, use_vllm=True)

    task = BaseTask(task_extract_label, task_config_path)

    for seed in [49]:
        set_seed(seed)
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = BasePrompt(task)
        inference_class = UpperboundInference(api, prompt_class, train_data, task)
        generation_class = DefaultGeneration(api, prompt_class, inference_class, task)

        update_class = DefaultUpdate(
            generation_class=generation_class,
            inference_class=inference_class,
            replace_class=DefaultReplace(max_num_hypotheses),
            save_path=output_folder,
            num_init=num_init,
            k=10,
            alpha=5e-1,
            update_batch_size=10,
            num_hypotheses_to_update=1,
            save_every_n_examples=10,
        )

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            hypotheses_bank = update_class.batched_initialize_hypotheses(
                num_init,
                init_batch_size=10,
                init_hypotheses_per_batch=10,
                use_cache=0,
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
                hypotheses_bank[hypothesis] = dict_to_summary_information(
                    dict[hypothesis]
                )
        for epoch in range(1):
            hypotheses_bank = update_class.update(
                current_epoch=epoch,
                hypotheses_bank=hypotheses_bank,
                current_seed=seed,
            )
            update_class.save_to_json(
                hypotheses_bank,
                sample="final",
                seed=seed,
                epoch=epoch,
            )

    # print experiment info
    print(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
