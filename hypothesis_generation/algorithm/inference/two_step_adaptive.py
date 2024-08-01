from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import numpy as np
import pulp
import random
import re

from .one_step_adaptive import OneStepAdaptiveInference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask
from ...utils import get_num_examples


class TwoStepAdaptiveInference(OneStepAdaptiveInference):
    """
    This class separate adaptive inference with separate calls for
    selecting hypotheses and making predictions.
    """

    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def default_predict(self, data, index, hyp_bank, use_system_prompt):
        assert (
            len(hyp_bank.keys()) == 1
        ), "default inference only supports one hypothesis at a time"

        prompt_input = self.prompt_class.inference(hyp_bank, data, index)

        response = self.api.generate(prompt_input, use_system_prompt)
        prediction = self.prompt_class.task.extract_label(response)
        actual_label = data["label"][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def select_hypotheses(self, data, index, hyp_bank, use_system_prompt):
        prompt_input = self.prompt_class.adaptive_selection(
            hyp_bank, self.train_data, data, index
        )
        response = self.api.generate(prompt_input, use_system_prompt)

        print("Prompt:", prompt_input[0], prompt_input[1])
        print("Response:", response)

        hyp_idx = re.search(r"Chosen Pattern:\s*Pattern\s*(\d+)", response)

        if hyp_idx == None:
            print(
                f"Could not find chosen hypothesis in response: {response}\n\nHyp_bank: {hyp_bank.keys()}"
            )
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            print(f"Use Hypothesis: {hyp}")
            return hyp

        hyp_idx = hyp_idx.group(1)
        hyp_idx = hyp_idx.strip()
        hyp_idx = int(hyp_idx) - 1

        if hyp_idx >= len(list(hyp_bank.items())):
            print(f"No hypothesis chosen, return to default.")
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            print(f"Use Hypothesis: {hyp}")
            return hyp

        print(f"Extracted Hypothesis Index: {hyp_idx}")
        items = list(hyp_bank.items())
        hyp = items[hyp_idx][0]
        print(f"Extracted Hypothesis: {hyp}")

        return hyp

    def predict(self, data, index, hyp_bank, use_system_prompt):
        # select one hypothesis that is most relevant to the sample
        hyp = self.select_hypotheses(data, index, hyp_bank, use_system_prompt)

        # make prediction using default_predict
        return self.default_predict(
            data, index, {hyp: hyp_bank[hyp]}, use_system_prompt
        )
