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
from ...logger_config import LoggerConfig

logger_name = "HypoGenic - Filter and Weight Inference"


@inference_register.register("filter_and_weight")
class FilterAndWeightInference(Inference):
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
        Make predictions on a batch of data.
        Use the hypotheses in hyp_bank to make a weighted-vote prediction.

        Note this function may be called in generation as well.
        Therefore, I only implement it to perform weighted-vote prediction (but not filtering).

        Parameters:
            data: the data to predict on
            idx_hyp_pair: a list of tuples of indices and hypothesis banks
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        logger = LoggerConfig.get_logger(logger_name)
        assert all(
            [len(hyp_bank.keys()) >= 1 for _, hyp_bank in idx_hyp_pair]
        ), "Filter and weight inference requires at least one hypothesis"

        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]
        prompt_inputs = [
            self.prompt_class.inference({hypothesis: hyp_bank[hypothesis]}, data, index)
            for index, hyp_bank in idx_hyp_pair
            for hypothesis in hyp_bank
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        logger.info(f"Responses: {responses}")
        responses = responses[::-1]
        predictions = []
        for _, hyp_bank in idx_hyp_pair:
            pred_dict = {}
            for hypothesis in hyp_bank:
                response = responses.pop(-1)
                pred = self.prompt_class.task.extract_label(response)
                weight = hyp_bank[hypothesis].acc
                if pred in pred_dict:
                    pred_dict[pred] += weight
                else:
                    pred_dict[pred] = weight
            predictions.append(max(pred_dict, key=pred_dict.get))

        return predictions, actual_labels

    def filter_hypotheses(
        self, hyp_bank, responses, indices
    ) -> Dict[str, SummaryInformation]:
        """
        Filter the hypotheses in hyp_bank to only include relevant hypotheses for the sample at index.

        Parameters
            data: the specific dataset
            indices: the specific indices to filter for
            hyp_bank: a dictionary of hypotheses

        Returns
            relevant_hypotheses: a dictionary of relevant hypotheses
        """
        logger = LoggerConfig.get_logger(logger_name)
        relevant_hypotheses_banks = []
        responses = responses[::-1]
        for _ in indices:
            relevant_hypotheses = {}
            for hypothesis in hyp_bank:
                response = responses.pop(-1)

                # only keep the part after "Final answer:"
                if "Final answer:" in response:
                    response = response[
                        response.index("Final answer:") + len("Final answer:") :
                    ]
                    response = response[:5]
                    response = response.lower()

                logger.info(f"Response (truncated): {response}")

                if "yes" in response and "no" in response:
                    if "yes or no" in response:
                        logger.info(f"Hypothsis is not relevant")
                    else:
                        raise ValueError(
                            f'The response should not contain both "yes" and "no". Response: {response}'
                        )
                elif "yes" in response:
                    relevant_hypotheses[hypothesis] = hyp_bank[hypothesis]
                    logger.info("Hypothesis is relevant")
                else:
                    logger.info(f"Hypothsis is not relevant")
            relevant_hypotheses_banks.append(relevant_hypotheses)

        return relevant_hypotheses_banks

    def _run_inference_final(
        self,
        data,
        hyp_bank,
        k=1,
        cache_seed=None,
        max_concurrent=3,
        generate_kwargs={},
        **kwargs,
    ):
        """
        Run over the entire dataset and make predictions.
        For each sample, prompt LLM to determine whether a hypothesis is relevant.
        Use the relevant hypotheses to make a weighted-vote prediction.

        Parameters:
            data: the data to predict on
            hyp_bank: the hypotheses that we want to predict from
            k: the number of hypotheses to keep
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        # get the top k hypotheses by reward (save as dictionary)
        if k > len(hyp_bank):
            k = len(hyp_bank)
        top_hypotheses = {}
        for hypothesis in sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[
            :k
        ]:
            top_hypotheses[hypothesis] = hyp_bank[hypothesis]

        # iterate over the dataset and make predictions
        num_samples = len(data)

        prompt_inputs = [
            self.prompt_class.is_relevant({hypothesis: hyp_bank[hypothesis]}, data, i)
            for i in range(num_samples)
            for hypothesis in top_hypotheses
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        filtered_hypotheses_banks = self.filter_hypotheses(
            top_hypotheses, responses, list(range(num_samples))
        )
        best_hypotheses = max(top_hypotheses, key=lambda x: top_hypotheses[x].acc)
        best_hypotheses_bank = {best_hypotheses: top_hypotheses[best_hypotheses]}

        return self.batched_predict(
            data,
            [
                (
                    i,
                    (
                        filtered_hypotheses
                        if len(filtered_hypotheses) > 0
                        else best_hypotheses_bank
                    ),
                )
                for i, filtered_hypotheses in zip(
                    range(num_samples), filtered_hypotheses_banks
                )
            ],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

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
        Run over the entire dataset and make predictions.
        For each sample, prompt LLM to determine whether a hypothesis is relevant.
        Use the relevant hypotheses to make a weighted-vote prediction.

        Prameters:
            data: the data to predict on
            hyp_bank: the hypotheses that we want to predict from
            k: the number of hypotheses to keep
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        return self._run_inference_final(
            data,
            hyp_bank,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            generate_kwargs=generate_kwargs,
            **kwargs,
        )
