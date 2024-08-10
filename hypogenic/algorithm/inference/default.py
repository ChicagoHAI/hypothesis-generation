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
from .base import Inference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask


@inference_register.register("default")
class DefaultInference(Inference):
    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def batched_predict(
        self,
        data: pd.DataFrame,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        use_cache=1,
    ):
        assert all(
            [len(hyp_bank.keys()) == 1 for _, hyp_bank in idx_hyp_pair]
        ), "default inference only supports one hypothesis at a time"

        prompt_inputs = [
            self.prompt_class.inference(hyp_bank, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)
        predictions = [self.task.extract_label(response) for response in responses]
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]
        return predictions, actual_labels

    def predict(self, data, index, hyp_bank, use_cache=1):
        assert (
            len(hyp_bank.keys()) == 1
        ), "default inference only supports one hypothesis at a time"

        prompt_input = self.prompt_class.inference(hyp_bank, data, index)
        print(f"Prompt: {prompt_input}\n")
        response = self.api.generate(prompt_input, use_cache=use_cache)
        print(f"Response: {response}")
        prediction = self.task.extract_label(response)
        print(f"Prediction: {prediction}")
        actual_label = data["label"][index]
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def run_inference_final(self, data, hyp_bank, use_cache=1, **kwargs):
        top_hypothesis = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[
            0
        ]
        num_samples = len(data)

        return self.batched_predict(
            data,
            [
                (i, {top_hypothesis: hyp_bank[top_hypothesis]})
                for i in range(num_samples)
            ],
            use_cache=use_cache,
        )
