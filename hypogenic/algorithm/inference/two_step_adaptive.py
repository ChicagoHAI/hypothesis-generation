from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pulp
import random
import re

from . import inference_register
from .one_step_adaptive import OneStepAdaptiveInference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask


@inference_register.register("two_step_adaptive")
class TwoStepAdaptiveInference(OneStepAdaptiveInference):
    """
    This class separate adaptive inference with separate calls for
    selecting hypotheses and making predictions.
    """

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def default_predict(self, data, index, hyp_bank, use_cache=1):
        assert (
            len(hyp_bank.keys()) == 1
        ), "default inference only supports one hypothesis at a time"

        prompt_input = self.prompt_class.inference(hyp_bank, data, index)

        response = self.api.generate(prompt_input, use_cache=use_cache)
        prediction = self.prompt_class.task.extract_label(response)
        actual_label = data["label"][index]
        print(f"Prompt: {prompt_input}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def select_hypotheses(self, hyp_bank, response):
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

    def batched_predict(
        self,
        data,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        use_cache=1,
    ):
        prompt_inputs = [
            self.prompt_class.adaptive_selection(hyp_bank, self.train_data, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        responses: List[str] = self.api.batched_generate(
            prompt_inputs, use_cache=use_cache
        )
        responses = responses[::-1]

        prompt_inputs = []
        for index, hyp_bank in idx_hyp_pair:
            hyp = self.select_hypotheses(hyp_bank, responses.pop(-1))
            prompt_inputs.append(
                self.prompt_class.inference({hyp: hyp_bank[hyp]}, data, index)
            )
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)
        predictions = [self.task.extract_label(response) for response in responses]
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]
        return predictions, actual_labels

    def predict(self, data, index, hyp_bank, use_cache=1):
        prompt_input = self.prompt_class.adaptive_selection(
            hyp_bank, self.train_data, data, index
        )
        response = self.api.generate(prompt_input, use_cache=use_cache)

        print("Prompt:", prompt_input)
        print("Response:", response)

        # select one hypothesis that is most relevant to the sample
        hyp = self.select_hypotheses(hyp_bank, response)

        # make prediction using default_predict
        return self.default_predict(
            data, index, {hyp: hyp_bank[hyp]}, use_cache=use_cache
        )
