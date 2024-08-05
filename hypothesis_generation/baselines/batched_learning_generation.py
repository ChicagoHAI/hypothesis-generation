# generate some hypothesis based on a batch of examples (in a low data regime)

import time
import random
import pickle
import sys
import math
import argparse
import os
import sys

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f"{code_repo_path}/code/")
from tasks import BaseTask
from utils import (
    LLMWrapper,
    get_num_examples,
    create_directory,
    VALID_MODELS,
    GPT_MODELS,
    set_seed,
)
from prompt import BasePrompt
from data_loader import get_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_train", type=int, default=25, help="Number of training examples."
    )
    parser.add_argument(
        "--num_test", type=int, default=0, help="Number of test examples."
    )
    parser.add_argument(
        "--num_val", type=int, default=0, help="Number of validation examples."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # TODO: config path instead of task name
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "binary_original_sst",
            "shoe",
            "retweet",
            "hotel_reviews",
            "headline_binary",
        ],
        help="task to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt",
        choices=VALID_MODELS,
        help="Model to use.",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="no_message",
        help="A note on the experiment setting.",
    )
    parser.add_argument(
        "--num_hypothesis",
        type=int,
        default=5,
        help="Number of hypotheses to generate.",
    )
    parser.add_argument(
        "--use_ood_reviews",
        type=str,
        default="None",
        help="Use out-of-distribution hotel reviews.",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path for loading models locally."
    )
    # argument for using api cache, default true (1)
    parser.add_argument(
        "--use_cache", type=int, default=1, help="Use cache for API calls."
    )

    args = parser.parse_args()

    return args


def main():
    start_time = time.time()

    args = parse_args()
    set_seed(args)
    print("Getting data ...")
    train_data, _, _ = get_data(args)
    print("Initialize LLM api ...")
    api = LLMWrapper(args.model, path_name=args.model_path, use_cache=args.use_cache)


    print("**** batched_learning_hypothesis_generation ****")
    prompt_input = prompt_class.batched_generation(train_data, args.num_hypothesis)
    print("Prompt: \n", prompt_input)
    response = api.generate(prompt_input)
    print("prompt length: ", len(prompt_input))
    print("Response: \n", response)
    print("response length: ", len(response))
    print("************************************************")

    print(f"Time: {time.time() - start_time} seconds")
    if api.model in GPT_MODELS:
        print(f"Estimated cost: {api.api.costs}")


if __name__ == "__main__":
    main()
