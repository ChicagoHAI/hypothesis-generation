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

from hypogenic.tasks import BaseTask
from hypogenic.utils import (
    set_seed,
)
from hypogenic.prompt import BasePrompt
from hypogenic.LLM_wrapper import LocalModelWrapper
from hypogenic.data_loader import get_data


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
    use_cache = 0
    hypothesis_file = f"./outputs/retweet/batched_gen_{model_name}_train_{num_train}_seed_{seed}_hypothesis_{num_hypothesis}.txt"

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

    set_seed(seed)
    print("Getting data ...")
    task = BaseTask(task_extract_label, task_config_path)
    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
    print("Initialize LLM api ...")
    api = LocalModelWrapper(model_name, model_path, use_vllm=True)
    prompt_class = BasePrompt(task)
    prompt_input = prompt_class.batched_generation(train_data, num_hypothesis)
    print("Prompt: ")
    print(prompt_input)
    response = api.generate(prompt_input)
    print("prompt length: ", len(prompt_input))
    print("Response: ")
    print(response)
    with open(hypothesis_file, "w") as f:
        f.write(response)
    print("response length: ", len(response))
    print("************************************************")

    print(f"Time: {time.time() - start_time} seconds")
    # if model in GPT_MODELS:
    #     print(f"Estimated cost: {api.api.costs}")


if __name__ == "__main__":
    main()
