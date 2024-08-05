# Assume we generated a list of hypotheses from the training data.
# Here, we want to get their training accuracy, and use the best one as the final hypothesis.
# And then we want to get the test accuracy of the final hypothesis.

import argparse
import re
import sys
import os
from typing import Union

from hypothesis_generation.tasks import BaseTask
from hypothesis_generation.utils import (
    LLMWrapper,
    VALID_MODELS,
    extract_label,
    extract_hypotheses,
    set_seed,
)
from hypothesis_generation.data_loader import get_data
from hypothesis_generation.prompt import BasePrompt


def get_accuracy(api: LLMWrapper, hypothesis, data, prompt_class, task):
    """
    Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
    """
    correct = 0
    hypothesis_dict = {hypothesis: None}
    prompt_input = [
        prompt_class.inference(hypothesis_dict, data, i)
        for i in range(len(data["label"]))
    ]
    responses = api.batched_generate(prompt_input)
    for i, response in enumerate(responses):
        print("*** get_accuracy ***")
        print(response)
        pred = task.extract_label(response)
        print("pred:", pred)
        print("label:", data["label"][i])
        print("*********************")
        if pred == data["label"][i]:
            correct += 1
    # for i in range(len(data["label"])):
    #     hypothesis_dict = {hypothesis: None}
    #     prompt_input = prompt_class.inference(hypothesis_dict, data, i)
    #     response = api.generate(prompt_input)
    #     print("*** get_accuracy ***")
    #     print(response)
    #     pred = task.extract_label(response)
    #     print("pred:", pred)
    #     print("label:", data["label"][i])
    #     print("*********************")
    #     if pred == data["label"][i]:
    #         correct += 1
    accuracy = correct / len(data["label"])
    return accuracy


def main():
    num_train = 25
    num_test = 300
    num_val = 0
    seed = 49
    task_config_path = "./data/retweet/config.yaml"
    task = "retweet"
    model = "Meta-Llama-3.1-8B-Instruct"
    num_hypothesis = 5
    model_path = "/net/scratch/llama/Meta-Llama-3.1-8B-Instruct"
    use_cache = 0
    hypothesis_file = f"./outputs/retweet/batched_gen_{model}_train_{num_train}_seed_{seed}_hypothesis_{num_hypothesis}.txt"

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

    # load the output text file
    with open(hypothesis_file) as f:
        text = f.read()

    # Use regex to extract the hypotheses
    hypotheses = extract_hypotheses(num_hypothesis, text)
    print("Hypotheses: ", hypotheses)
    if len(hypotheses) == 0:
        print("No hypotheses found.")
        return

    # load training data
    task = BaseTask(task_extract_label, task_config_path)
    train_data, test_data, _ = task.get_data(num_train, num_test, num_val, seed)

    # initialization
    prompt_class = BasePrompt(task)
    api = LLMWrapper.from_model(
        model, path_name=model_path, use_cache=use_cache, use_vllm=True
    )

    # get the training accuracy of each hypothesis
    training_accuracies = []
    for hypothesis in hypotheses:
        # get the training accuracy of the hypothesis
        accuracy = get_accuracy(api, hypothesis, train_data, prompt_class, task)
        training_accuracies.append(accuracy)

    # get the test accuracy of the best hypothesis
    best_hypothesis = hypotheses[training_accuracies.index(max(training_accuracies))]
    test_accuracy = get_accuracy(api, best_hypothesis, test_data, prompt_class, task)

    print("Best hypothesis: ", best_hypothesis)
    print("Test accuracy of best hypothesis: ", test_accuracy)
    print("Training accuracy of best hypothesis: ", max(training_accuracies))
    print("Training accuracies: ", training_accuracies)


if __name__ == "__main__":
    main()
