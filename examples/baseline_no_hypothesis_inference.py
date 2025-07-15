# call LLM to predict label without taking in a hypothesis
# zero-shot learning, with instruction in the prompt

import argparse
import re
import time
import sys
import os
from typing import Union

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypogenic.extract_label import retweet_extract_label

from hypogenic.tasks import BaseTask
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LLMWrapper, GPTWrapper
# from hypogenic.LLM_wrapper import LocalVllmWrapper
from hypogenic.prompt import BasePrompt
from hypogenic.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HypoGenic")

def compute_accuracy(results):
    labels = [result["label"] for result in results]
    preds = [result["pred"] for result in results]
    safety_mode = 0
    x = []
    for label, pred in zip(labels, preds):
        if pred == "other":
            safety_mode += 1
        if pred == label:
            x.append(1)
        else:
            x.append(0)
    acc = sum(x) / len(x)
    logger.info(f"non-safety mode record: {len(x) - safety_mode}")
    logger.info(f"Accuracy: {acc}")
    return acc


def few_shot(
    api: LLMWrapper,
    train_data,
    test_data,
    prompt_class: BasePrompt,
    task,
    few_shot_k,
    cache_seed,
):
    """
    Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
    """
    results = []
    prompt_inputs = [
        prompt_class.few_shot_baseline(train_data, few_shot_k, test_data, i)
        for i in range(len(test_data))
    ]
    responses = api.batched_generate(prompt_inputs, cache_seed=cache_seed)
    for i in range(len(test_data)):
        logger.info(f"********** Example {i} **********")
        pred = task.extract_label(responses[i])
        label = test_data[task.label_name][i]

        # logger.info(f"Prompt: {prompt_inputs[i]}")
        logger.info(f"Response: {responses[i]}")
        logger.info(f"Label: {label}")
        logger.info(f"Prediction: {pred}")
        results.append(
            {
                "prompt": prompt_inputs[i],
                "response": responses[i],
                "label": label,
                "pred": pred,
            }
        )
        logger.info("**********************************")

    return results


def preprocess(task, train_data, k):
    num_examples = len(train_data)

    data = []

    label_nunique = train_data[task.label_name].nunique()
    label_unique = train_data[task.label_name].unique()
    for i in range(k):
        data.append(
            train_data[train_data[task.label_name] == label_unique[i % label_nunique]].iloc[
                i // label_nunique
            ]
        )

    return pd.DataFrame(data).reset_index(drop=True)


def main():
    start_time = time.time()

    seed = 42
    task_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), f"data/retweet/config.yaml"
    )
    task = "retweet"
    model_name = "gpt-4o-mini"
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    num_test = 100
    num_train = 100
    num_val = 100
    few_shot_k = 3
    cache_seed = None

    set_seed(seed)

    task = BaseTask(task_config_path, extract_label=retweet_extract_label)

    prompt_class = BasePrompt(task)
    api = GPTWrapper(model_name)
    # api = LocalVllmWrapper(model_name, model_path)

    train_data, test_data, _ = task.get_data(num_train, num_test, num_val, seed)

    if few_shot_k > 0:
        train_data = preprocess(task, train_data, few_shot_k)

    results = few_shot(
        api, train_data, test_data, prompt_class, task, few_shot_k, cache_seed
    )
    test_accuracy = compute_accuracy(results)

    logger.info(f"Test accuracy: {test_accuracy}")
    logger.info(f"Total time (seconds): {round(time.time() - start_time, 2)}")


if __name__ == "__main__":
    main()
