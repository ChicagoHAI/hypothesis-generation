# call LLM to predict label without taking in a hypothesis
# zero-shot learning, with instruction in the prompt

import argparse
import re
import time
import sys
import os
from typing import Union

import pandas as pd

from hypogenic.tasks import BaseTask
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LocalModelWrapper, LLMWrapper
from hypogenic.data_loader import get_data
from hypogenic.prompt import BasePrompt


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
    print("non-safety mode record:", len(x) - safety_mode)
    print(f"Accuracy: {acc}")
    return acc


def few_shot(
    api: LLMWrapper, train_data, test_data, prompt_class: BasePrompt, task, few_shot_k
):
    """
    Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
    """
    results = []
    for i in range(len(test_data)):
        prompt_input = prompt_class.few_shot_baseline(
            train_data, few_shot_k, test_data, i
        )
        response = api.generate(prompt_input)
        print(f"********** Example {i} **********")
        pred = task.extract_label(response)
        label = test_data["label"][i]

        # print(f"Prompt: {prompt_input}")
        print(f"Response: {response}")
        print(f"Label: {label}")
        print(f"Prediction: {pred}")
        results.append(
            {"prompt": prompt_input, "response": response, "label": label, "pred": pred}
        )
        print("**********************************")

    return results


def preprocess(train_data, k):
    num_examples = len(train_data)

    data = []

    label_nunique = train_data["label"].nunique()
    label_unique = train_data["label"].unique()
    for i in range(k):
        data.append(
            train_data[train_data["label"] == label_unique[i % label_nunique]].iloc[
                i // label_nunique
            ]
        )

    return pd.DataFrame(data)


def main():
    start_time = time.time()

    seed = 42
    task_config_path = "./data/retweet/config.yaml"
    task = "retweet"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    num_test = 100
    num_train = 100
    num_val = 100
    few_shot_k = 3
    use_cache = 0

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

    task = BaseTask(task_extract_label, task_config_path)

    prompt_class = BasePrompt(task)
    api = LocalModelWrapper(model_name, model_path, use_vllm=True)

    train_data, test_data, _ = task.get_data(num_train, num_test, num_val, seed)

    if few_shot_k > 0:
        train_data = preprocess(train_data, few_shot_k)

    results = few_shot(api, train_data, test_data, prompt_class, task, few_shot_k)
    test_accuracy = compute_accuracy(results)

    print("Test accuracy: ", test_accuracy)
    print("Total time (seconds): ", round(time.time() - start_time, 2))


if __name__ == "__main__":
    main()
