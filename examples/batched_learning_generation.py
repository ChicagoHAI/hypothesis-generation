# generate some hypothesis based on a batch of examples (in a low data regime)

import re
import time
import random
import pickle
import sys
import math
import argparse
import os
import sys
from typing import Union

from hypogenic.extract_label import retweet_extract_label

from hypogenic.LLM_wrapper import LocalVllmWrapper
from hypogenic.tasks import BaseTask
from hypogenic.utils import (
    set_seed,
)
from hypogenic.prompt import BasePrompt
from hypogenic.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HypoGenic")


def main():
    start_time = time.time()

    num_train = 25
    num_test = 0
    num_val = 0
    seed = 49
    task_config_path = "./data/retweet/config.yaml"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    num_hypothesis = 5
    cache_seed = None
    hypothesis_file = f"./outputs/retweet/batched_gen_{model_name}_train_{num_train}_seed_{seed}_hypothesis_{num_hypothesis}.txt"

    set_seed(seed)
    logger.info("Getting data ...")
    task = BaseTask(task_config_path, extract_label=retweet_extract_label)

    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
    logger.info("Initialize LLM api ...")

    api = LocalVllmWrapper(model_name, model_path)
    prompt_class = BasePrompt(task)

    prompt_input = prompt_class.batched_generation(train_data, num_hypothesis)
    logger.info("Prompt: ")
    logger.info(prompt_input)
    response = api.generate(prompt_input, cache_seed=cache_seed)
    logger.info(f"prompt length: {len(prompt_input)}")
    logger.info("Response: ")
    logger.info(response)
    os.makedirs(os.path.dirname(hypothesis_file), exist_ok=True)
    with open(hypothesis_file, "w") as f:
        f.write(response)
    logger.info(f"response length: {len(response)}")
    logger.info("************************************************")

    logger.info(f"Time: {time.time() - start_time} seconds")
    # if model in GPT_MODELS:
    #     logger.info(f"Estimated cost: {api.api.costs}")


if __name__ == "__main__":
    main()
