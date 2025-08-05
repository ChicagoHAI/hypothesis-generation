from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import random
import re

from hypogenic.algorithm.generation import Generation, DefaultGeneration
from hypogenic.algorithm.inference import Inference, DefaultInference
from hypogenic.prompt import BasePrompt
from hypogenic.tasks import BaseTask
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)
from hypogenic.LLM_wrapper import LocalVllmWrapper, GPTWrapper, LLMWrapper
from hypogenic.algorithm.generation.utils import extract_hypotheses
from hypogenic.algorithm.update import Update
from hypogenic.extract_label import extract_label_register
from hypogenic.utils import set_seed, get_results
from hypogenic.logger_config import LoggerConfig
from hypogenic.algorithm.replace import DefaultReplace, Replace

from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.task import TestTask
from hypothesis_agent.data_analysis_agent.utils import extract_relevance_results
import matplotlib.pyplot as plt
import pandas as pd

class MultiHypDefaultInference(DefaultInference):
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def multiple_hypotheses_batched_predict(
        self,
        data: pd.DataFrame,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):

        prompt_inputs = [
            self.prompt_class.multiple_hypotheses_inference(hyp_bank, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]
        predictions = [self.task.extract_label(responses[i]) for i in range(len(responses))]

        return predictions, actual_labels

    def run_inference_final(
        self,
        data,
        hyp_bank,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):

        num_samples = len(data)

        return self.multiple_hypotheses_batched_predict(
            data,
            [
                (i, hyp_bank)
                for i in range(num_samples)
            ],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

class MultiHypInferenceWithRank(DefaultInference):
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        train_data: pd.DataFrame,
        task: TestTask,
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

        Returns:
            (predictions, actual_labels, idx_hyp_pair):
            predictions: the predictions

            actual_labels: the actual labels

            idx_hyp_pair: list of tuples of indices and hypothesis banks (key: hyp, value: (is_relevant, rank))
        """
        prompt_inputs = [
            self.prompt_class.is_relevant({hyp: None}, data, index)
            for index, hyp_bank in idx_hyp_pair
            for hyp in hyp_bank
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        responses = responses[::-1]
        for idx, (index, hyp_bank) in enumerate(idx_hyp_pair):
            for hyp in hyp_bank:
                hyp_bank[hyp] = (extract_relevance_results(responses.pop(-1)), -1)
            idx_hyp_pair[idx] = (index, hyp_bank)

        prompt_inputs = [
            self.prompt_class.multiple_hypotheses_inference(
                {hyp: hyp_bank[hyp] for hyp in hyp_bank if hyp_bank[hyp][0]},
                data,
                index,
            )
            for index, hyp_bank in idx_hyp_pair
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        pred_rank_pair = [
            self.task.extract_label_with_rank(response) for response in responses
        ]
        responses = responses[::-1]
        for idx, (index, hyp_bank) in enumerate(idx_hyp_pair):
            hyps = list(hyp_bank.keys())
            hyp_contributions = pred_rank_pair[idx][1]
            if hyp_contributions is None:
                for hyp in hyp_bank:
                    hyp_bank[hyp] = (True, 0) if hyp_bank[hyp][0] else (False, -1)
            else:
                for i, hyp_idx in enumerate(hyp_contributions):
                    hyp_bank[hyps[hyp_idx - 1]] = (True, i)
            idx_hyp_pair[idx] = (index, hyp_bank)

        predictions = [pred for pred, _ in pred_rank_pair]
        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]

        return predictions, actual_labels, idx_hyp_pair
