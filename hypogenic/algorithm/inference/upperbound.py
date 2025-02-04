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

logger_name = "HypoGenic - Upperbound Inference"


@inference_register.register("upperbound")
class UpperboundInference(Inference):
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
        assert all(
            [len(hyp_bank.keys()) == 1 for _, hyp_bank in idx_hyp_pair]
        ), "default inference only supports one hypothesis at a time"

        predictions = []
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
        actual_labels = [data[self.task.label_name][index] for index, _ in idx_hyp_pair]
        return predictions, actual_labels

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
        Run inference for each hypothesis in the hypothesis bank and return the predictions.
        We regard the final prediction as correct if at least one of the hypotheses is correct.

        Parameters:
            data: the data to predict on
            hyp_bank: the hypothesis bank
            k: the number of hypotheses to keep
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        logger = LoggerConfig.get_logger(logger_name)

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
        num_samples = len(data)
        pred_list = {hyp: [] for hyp in hyp_bank}
        count = 1
        preds, label_list = self.batched_predict(
            data,
            [(i, {hyp: hyp_bank[hyp]}) for hyp in hyp_bank for i in range(num_samples)],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        preds = preds[::-1]
        for hyp in hyp_bank:
            for i in range(num_samples):
                pred_list[hyp].append(preds.pop(-1))
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
        logger.info("Correctness:")
        for hyp in hyp_bank:
            logger.info(f"{correct_list[hyp]}")

        # print accuracy for each hypothesis
        for hyp in hyp_bank:
            logger.info(f"Hypothesis: {hyp}, Accuracy: {accuracy_list[hyp]}")

        # count as correct if one of the hypotheses is correct
        correct = 0
        for i in range(num_samples):
            for hyp in hyp_bank:
                if pred_list[hyp][i] == label_list[i]:
                    correct += 1
                    break
        accuracy = correct / num_samples
        logger.info(f"Upperbound accuracy (if one hyp is correct): {accuracy}")

        return [
            pred_list[hyp][i] for hyp in hyp_bank for i in range(num_samples)
        ], label_list

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
        Run inference for each hypothesis in the hypothesis bank and return the predictions.
        We regard the final prediction as correct if at least one of the hypotheses is correct.

        Parameters:
            data: the data to predict on
            hyp_bank: the hypothesis bank
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
