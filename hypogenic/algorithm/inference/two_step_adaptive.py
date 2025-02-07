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
from ...logger_config import LoggerConfig

logger_name = "HypoGenic - Two Step Adaptive Inference"


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

    def select_hypotheses(self, hyp_bank, response):
        """
        Select the hypothesis to use for the next step of inference.

        Parameters:
            hyp_bank: the hypothesis bank
            response: the response from the model
        """
        logger = LoggerConfig.get_logger(logger_name)

        hyp_idx = re.search(r"Chosen Pattern:\s*Pattern\s*(\d+)", response)

        if hyp_idx == None:
            logger.info(
                f"Could not find chosen hypothesis in response: {response}\n\nHyp_bank: {hyp_bank.keys()}"
            )
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            logger.info(f"Use Hypothesis: {hyp}")
            return hyp

        hyp_idx = hyp_idx.group(1)
        hyp_idx = hyp_idx.strip()
        hyp_idx = int(hyp_idx) - 1

        if hyp_idx >= len(list(hyp_bank.items())):
            logger.info(f"No hypothesis chosen, return to default.")
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            logger.info(f"Use Hypothesis: {hyp}")
            return hyp

        logger.info(f"Extracted Hypothesis Index: {hyp_idx}")
        items = list(hyp_bank.items())
        hyp = items[hyp_idx][0]
        logger.info(f"Extracted Hypothesis: {hyp}")

        return hyp

    def batched_predict(
        self,
        data,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """
        Make predictions on a batch of data.

        Parameters:
            data: the data to predict on
            idx_hyp_pair: a list of tuples of indices and hypothesis banks
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        prompt_inputs = [
            self.prompt_class.adaptive_selection(hyp_bank, self.train_data, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        responses: List[str] = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        responses = responses[::-1]

        prompt_inputs = []
        for index, hyp_bank in idx_hyp_pair:
            hyp = self.select_hypotheses(hyp_bank, responses.pop(-1))
            prompt_inputs.append(
                self.prompt_class.inference({hyp: hyp_bank[hyp]}, data, index)
            )
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        predictions = [self.task.extract_label(response) for response in responses]
        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]
        return predictions, actual_labels
