from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import numpy as np
import pulp
import random
import re

from .base import Inference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask
from ...utils import get_num_examples


class DefaultInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

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

    def run_inference_final(self, data, hyp_bank, **kwargs):
        top_hypothesis = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[
            0
        ]
        num_samples = get_num_examples(data)

        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(
                data, i, {top_hypothesis: hyp_bank[top_hypothesis]}
            )
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list
