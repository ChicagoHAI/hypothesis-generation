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
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """
        Makes a batch of preductions on a hypothesis.

        Parameters:
            data: the data to predict on
            idx_hyp_pair: a list of tuples of indices and hypothesis banks
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        assert all(
            [len(hyp_bank.keys()) == 1 for _, hyp_bank in idx_hyp_pair]
        ), "default inference only supports one hypothesis at a time"

        # we use the prompt class in order to create the batch of prompts
        prompt_inputs = [
            self.prompt_class.inference(hyp_bank, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        predictions = [self.task.extract_label(response) for response in responses]

        # and once we get the actual labels
        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]

        # we can return our predictions along with the labels
        return predictions, actual_labels

    def run_inference_final(
        self,
        data,
        hyp_bank,
        cache_seed=None,
        max_concurrent=3,
        generate_kwargs={},
        **kwargs,
    ):
        """
        Function for testing the best hypothesis

        Prameters:
            data: the data to predict on
            hyp_bank: the hypotheses that we want to predict from
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """

        # getting the top hypothesis
        top_hypothesis = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[
            0
        ]
        num_samples = len(data)

        # running the batched predict with the top hypothesis
        return self.batched_predict(
            data,
            [
                (i, {top_hypothesis: hyp_bank[top_hypothesis]})
                for i in range(num_samples)
            ],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
