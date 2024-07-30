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

from hypothesis_generation.tasks import BaseTask
from hypothesis_generation.prompt import BasePrompt
from hypothesis_generation.data_loader import get_data
from hypothesis_generation.utils import (
    LLMWrapper,
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

from hypothesis_generation.algorithm.generation import DefaultGeneration
from hypothesis_generation.algorithm.inference import (
    DefaultInference,
    KNNInference,
    FilterAndWeightInference,
    SeparateStepsKNNInference,
    UpperboundInference,
)
from hypothesis_generation.algorithm.replace import Replace
from hypothesis_generation.algorithm.update import SamplingUpdate, DefaultUpdate


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def setup_LLM(model, model_path, use_cache):
    api = LLMWrapper.from_model(model, path_name=model_path, use_cache=use_cache)
    return api


def main():
    # set up tools
    start_time = time.time()

    task_config_path = "./data/retweet/config.yaml"
    max_num_hypotheses = 20
    output_folder = f"./outputs/retweet/gpt-4o-mini/hyp_{max_num_hypotheses}/"
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

    create_directory(output_folder)
    api = setup_LLM("gpt-4o-mini", "", 0)

    task = BaseTask(task_extract_label, task_config_path)

    for seed in [49]:
        set_seed(seed)
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = BasePrompt(task)
        inference_class = DefaultInference(api, prompt_class, train_data)
        generation_class = DefaultGeneration(api, prompt_class, inference_class)

        update_class = SamplingUpdate(
            generation_class=generation_class,
            inference_class=inference_class,
            replace_class=Replace(),
            num_init=num_init,
            k=10,
            alpha=5e-1,
            update_batch_size=10,
            num_hypotheses_to_update=1,
            save_every_n_examples=10,
        )

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            hypotheses_bank = update_class.initialize_hypotheses(
                num_init,
                init_batch_size=10,
                init_hypotheses_per_batch=10,
            )
            update_class.save_to_json(
                hypotheses_bank,
                file_name=os.path.join(
                    output_folder,
                    f"hypotheses_training_sample_{num_init}_seed_{seed}_epoch_0.json",
                ),
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
                file_name=os.path.join(
                    output_folder,
                    f"hypotheses_training_sample_final_seed_{seed}_epoch_{epoch}.json",
                ),
            )

    # print experiment info
    print(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
