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
        use_cache=1,
    ):
        assert all(
            [len(hyp_bank.keys()) >= 1 for _, hyp_bank in idx_hyp_pair]
        ), "Filter and weight inference requires at least one hypothesis"

        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]
        prompt_inputs = [
            self.prompt_class.inference({hypothesis: hyp_bank[hypothesis]}, data, index)
            for index, hyp_bank in idx_hyp_pair
            for hypothesis in hyp_bank
        ]
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)
        print(f"Responses: {responses}")
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

    def predict(self, data, index, hyp_bank, use_cache=1):
        """
        Make prediction on one sample (index) of the dataset.
        Use the hypotheses in hyp_bank to make a weighted-vote prediction.

        Note this function may be called in generation as well.
        Therefore, I only implement it to perform weighted-vote prediction (but not filtering).
        """
        assert (
            len(hyp_bank.keys()) >= 1
        ), "Filter and weight inference requires at least one hypothesis"

        actual_label = data["label"][index]
        pred_dict = {}
        prompt_inputs = [
            self.prompt_class.inference({hypothesis: hyp_bank[hypothesis]}, data, index)
            for hypothesis in hyp_bank
        ]
        responses = self.api.batched_generate(prompt_inputsuse_cache=use_cache)
        preds = [
            self.prompt_class.task.extract_label(response) for response in responses
        ]
        for hypothesis, pred in zip(hyp_bank, preds):
            weight = hyp_bank[hypothesis].acc
            if pred in pred_dict:
                pred_dict[pred] += weight
            else:
                pred_dict[pred] = weight
        prediction = max(pred_dict, key=pred_dict.get)

        print(f"Predictions (weights): {pred_dict}")
        print(f"Prediction (final): {prediction}")
        print(f"Ground truth: {actual_label}")

        return prediction, actual_label

    def filter_hypotheses(
        self, hyp_bank, responses, indices
    ) -> Dict[str, SummaryInformation]:
        """
        Filter the hypotheses in hyp_bank to only include relevant hypotheses for the sample at index.

        Parameters
        __________
        data: the specific dataset
        index: the specific index to filter for
        hyp_bank: a dictionary of hypotheses

        Returns
        __________
        relevant_hypotheses: a dictionary of relevant hypotheses
        """
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

                print(f"Response (truncated): {response}")

                if "yes" in response and "no" in response:
                    if "yes or no" in response:
                        print(f"Hypothsis is not relevant")
                    else:
                        raise ValueError(
                            f'The response should not contain both "yes" and "no". Response: {response}'
                        )
                elif "yes" in response:
                    relevant_hypotheses[hypothesis] = hyp_bank[hypothesis]
                    print("Hypothesis is relevant")
                else:
                    print(f"Hypothsis is not relevant")
            relevant_hypotheses_banks.append(relevant_hypotheses)

        return relevant_hypotheses_banks

    def _run_inference_final(self, data, hyp_bank, k=1, use_cache=1, **kwargs):
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
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)
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
            use_cache=use_cache,
        )

    def run_inference_final(self, data, hyp_bank, use_cache=1, **kwargs):
        """
        Run over the entire dataset and make predictions.
        For each sample, prompt LLM to determine whether a hypothesis is relevant.
        Use the relevant hypotheses to make a weighted-vote prediction.
        """
        return self._run_inference_final(data, hyp_bank, use_cache=use_cache, **kwargs)
