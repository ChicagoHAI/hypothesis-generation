from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import pulp
import random
import re

from .base import Inference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask
from ...utils import get_num_examples


class UpperboundInference(Inference):
    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def predict(self, data, index, hyp_bank):
        assert (
            len(hyp_bank.keys()) == 1
        ), "default inference only supports one hypothesis at a time"
        prompt_input = self.prompt_class.inference(hyp_bank, data, index)
        print(f"Prompt: {prompt_input}\n")
        response = self.api.generate(prompt_input)
        print(f"Response: {response}")
        prediction = self.prompt_class.task.extract_label(response)
        print(f"Prediction: {prediction}")
        actual_label = data["label"][index]
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def _run_inference_final(self, data, hyp_bank, k=1):
        arg_k = k

        # sort hyp_bank by training accuracy from high to low
        hyp_bank = {
            k: v
            for k, v in sorted(
                hyp_bank.items(), key=lambda item: item[1].acc, reverse=True
            )
        }
        # keep top args.k hypotheses
        hyp_bank = {k: v for k, v in list(hyp_bank.items())[:arg_k]}

        # run inference for each hypothesis
        num_samples = get_num_examples(data)
        pred_list = {hyp: [] for hyp in hyp_bank}
        label_list = []
        count = 1
        for hyp in hyp_bank:
            print(f"The {count}th hypothesis")
            for i in range(num_samples):
                pred, label = self.predict(data, i, {hyp: hyp_bank[hyp]})
                pred_list[hyp].append(pred)
                label_list.append(label)
            count += 1

        # compute accuracy for each hypothesis
        correct_list = {hyp: [] for hyp in hyp_bank}
        accuracy_list = {hyp: 0 for hyp in hyp_bank}
        for hyp in hyp_bank:
            for i in range(num_samples):
                if pred_list[hyp][i] == label_list[i]:
                    correct_list[hyp].append(1)
                else:
                    correct_list[hyp].append(0)
            accuracy_list[hyp] = sum(correct_list[hyp]) / num_samples

        # print the correctness of each hypothesis (in matrix form)
        print("Correctness:")
        for hyp in hyp_bank:
            print(f"{correct_list[hyp]}")

        # print accuracy for each hypothesis
        for hyp in hyp_bank:
            print(f"Hypothesis: {hyp}, Accuracy: {accuracy_list[hyp]}")

        # count as correct if one of the hypotheses is correct
        correct = 0
        for i in range(num_samples):
            for hyp in hyp_bank:
                if pred_list[hyp][i] == label_list[i]:
                    correct += 1
                    break
        accuracy = correct / num_samples
        print(f"Upperbound accuracy (if one hyp is correct): {accuracy}")

        return pred_list, label_list

    def run_inference_final(self, data, hyp_bank, **kwargs):
        return self._run_inference_final(data, hyp_bank, **kwargs)
