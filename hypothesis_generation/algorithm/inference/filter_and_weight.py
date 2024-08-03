from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import pulp
import random
import re

from .base import Inference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask
from ...utils import get_num_examples


class FilterAndWeightInference(Inference):
    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def predict(self, data, index, hyp_bank):
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
        for hypothesis in hyp_bank:
            hypothesis_dict = {hypothesis: hyp_bank[hypothesis]}
            prompt_input = self.prompt_class.inference(hypothesis_dict, data, index)
            response = self.api.generate(prompt_input)
            pred = self.prompt_class.task.extract_label(response)
            weight = hyp_bank[hypothesis].acc
            if pred in pred_dict:
                pred_dict[pred] += weight
            else:
                pred_dict[pred] = weight
        prediction = max(pred_dict, key=pred_dict.get)

        print(f"Prompt: {prompt_input}\n")
        print(f"Response: {response}")
        print(f"Predictions (weights): {pred_dict}")
        print(f"Prediction (final): {prediction}")
        print(f"Ground truth: {actual_label}")

        return prediction, actual_label

    def filter_hypotheses(self, data, index, hyp_bank):
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
        relevant_hypotheses = {}
        for hypothesis in hyp_bank:
            temp_hyp_bank = {hypothesis: hyp_bank[hypothesis]}
            prompt_input = self.prompt_class.is_relevant(temp_hyp_bank, data, index)
            response = self.api.generate(prompt_input)

            print(f"Prompt: {prompt_input}\n")
            print(f"Response: {response}")

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

        return relevant_hypotheses

    def _run_inference_final(self, data, hyp_bank, k=1):
        # get the top k hypotheses by reward (save as dictionary)
        if k > len(hyp_bank):
            k = len(hyp_bank)
        top_hypotheses = {}
        for hypothesis in sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[
            :k
        ]:
            top_hypotheses[hypothesis] = hyp_bank[hypothesis]

        # iterate over the dataset and make predictions
        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            filtered_hypotheses = self.filter_hypotheses(
                data, i, top_hypotheses
            )
            # if no hypothesis is relevant, use the hypothesis with the highest accuracy
            if len(filtered_hypotheses) == 0:
                best_hypothesis = max(
                    top_hypotheses, key=lambda x: top_hypotheses[x].acc
                )
                filtered_hypotheses[best_hypothesis] = top_hypotheses[best_hypothesis]
            pred, label = self.predict(data, i, filtered_hypotheses)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list

    def run_inference_final(self, data, hyp_bank, **kwargs):
        """
        Run over the entire dataset and make predictions.
        For each sample, prompt LLM to determine whether a hypothesis is relevant.
        Use the relevant hypotheses to make a weighted-vote prediction.
        """
        return self._run_inference_final(data, hyp_bank, **kwargs)
