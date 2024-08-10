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
from .default import DefaultInference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask


@inference_register.register("one_step_adaptive")
class OneStepAdaptiveInference(Inference):
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
        use_cache=1,
    ):
        prompt_inputs = [
            self.prompt_class.one_step_adaptive_inference(
                hyp_bank, self.train_data, data, index
            )
            for index, hyp_bank in idx_hyp_pair
        ]
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)
        predictions = [self.task.extract_label(response) for response in responses]
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]
        return predictions, actual_labels

    def predict(self, data, index, hyp_bank, use_cache=1):
        prompt_input = self.prompt_class.one_step_adaptive_inference(
            hyp_bank, self.train_data, data, index
        )
        response = self.api.generate(prompt_input, use_cache=use_cache)
        prediction = self.prompt_class.task.extract_label(response)
        actual_label = data["label"][index]
        print(f"Prompt: {prompt_input}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def _run_inference_final(
        self,
        data,
        hyp_bank,
        adaptive_threshold=0.0,
        adaptive_num_hypotheses=0,
        adaptive_num_examples=0,
        use_cache=1,
        **kwargs,
    ):
        num_train_data_samples = len(self.train_data)
        similarity_matrix, one_hot_encoded_dict = self.compute_similarity_matrix(
            hyp_bank, num_train_data_samples
        )
        assert list(one_hot_encoded_dict.keys()) == list(
            hyp_bank.keys()
        ), "The keys of the one hot encoded dict and the hyp_bank should be the same"
        similarity_per_hypothesis = [
            np.sum(similarity_matrix[i])
            for i, _ in enumerate(one_hot_encoded_dict.keys())
        ]
        accuracy_per_hypothesis = [hyp_bank[hyp].acc for hyp in one_hot_encoded_dict]
        print("Initial examples per hyp:")
        for hyp in hyp_bank:
            print(f"Hypothesis {hyp}, Examples: {hyp_bank[hyp].correct_examples}")

        print()
        print("One hot encoded dict:")
        for hyp in one_hot_encoded_dict:
            print(f"Hypothesis {hyp}, Encoded Examples: {one_hot_encoded_dict[hyp]}")
        print()
        print("Similarity matrix:\n", similarity_matrix, "\n")

        # choose hypotheses with the least similarities
        selected_indices = self.select_hypotheses_ilp(
            similarity_matrix,
            accuracy_per_hypothesis,
            similarity_per_hypothesis,
            adaptive_threshold,
        )
        key_list = list(one_hot_encoded_dict.keys())
        selected_hypotheses = [key_list[idx] for idx in selected_indices]
        print("Selected hypotheses based upon non-similarity:", selected_hypotheses)

        top_k_hypotheses = sorted(
            selected_hypotheses, key=lambda x: hyp_bank[x].acc, reverse=True
        )[:adaptive_num_hypotheses]

        selected_hyp_bank = {}
        for hypothesis in top_k_hypotheses:
            selected_hyp_bank[hypothesis] = hyp_bank[hypothesis]
        for hyp in selected_hyp_bank:
            selected_hyp_bank[hyp].set_hypothesis(hyp)
            if len(selected_hyp_bank[hyp].correct_examples) > adaptive_num_examples:
                selected_hyp_bank[hyp].set_example(
                    random.sample(
                        selected_hyp_bank[hyp].correct_examples, adaptive_num_examples
                    )
                )

        num_samples = len(data)
        return self.batched_predict(
            data,
            [(i, selected_hyp_bank) for i in range(num_samples)],
            use_cache=use_cache,
        )

    def run_inference_final(self, data, hyp_bank, use_cache=1, **kwargs):
        return self._run_inference_final(data, hyp_bank, use_cache=use_cache, **kwargs)

    def compute_similarity_matrix(self, hyp_bank, num_train_data_samples):
        one_hot_encoded_dict = OrderedDict()

        for hypothesis in hyp_bank:
            indices = [ex[0] for ex in hyp_bank[hypothesis].correct_examples]
            result = [0] * num_train_data_samples  # Initialize array with zeros
            for idx in indices:
                result[idx] = 1  # Set elements at specified indices to 1
            one_hot_encoded_dict[hypothesis] = result

        similarity_matrix = np.zeros((len(hyp_bank), len(hyp_bank)))
        for i, hypothesis_one in enumerate(one_hot_encoded_dict.keys()):
            for j, hypothesis_two in enumerate(one_hot_encoded_dict.keys()):
                if hypothesis_one != hypothesis_two:
                    similarity_matrix[i][j] = np.dot(
                        one_hot_encoded_dict[hypothesis_one],
                        one_hot_encoded_dict[hypothesis_two],
                    ) / (
                        np.linalg.norm(one_hot_encoded_dict[hypothesis_one])
                        * np.linalg.norm(one_hot_encoded_dict[hypothesis_two])
                    )

        return similarity_matrix, one_hot_encoded_dict

    def select_hypotheses_ilp(
        self, similarity_matrix, accuracies, similarities, threshold
    ):
        num_hypotheses = similarity_matrix.shape[0]
        problem = pulp.LpProblem("Hypothesis_Selection", pulp.LpMaximize)

        # Create a binary variable for each hypothesis, indicating whether it's selected
        selection_vars = [
            pulp.LpVariable(f"select_{i}", cat="Binary") for i in range(num_hypotheses)
        ]

        # Objective: Maximize the number of training accuracy of selected hypotheses
        problem += pulp.lpSum(
            [(selection_vars[i] * accuracies[i]) for i in range(num_hypotheses)]
        )

        # Constraints: For each pair of hypotheses, if the similarity is above the threshold,
        # at least one hypothesis must not be selected.
        for i in range(num_hypotheses):
            for j in range(i + 1, num_hypotheses):
                if similarity_matrix[i, j] >= threshold:
                    problem += selection_vars[i] + selection_vars[j] <= 1

        # Solve the problem
        problem.solve()

        # Get the indices of the selected hypotheses
        selected_indices = [
            i for i, var in enumerate(selection_vars) if var.value() == 1
        ]

        return selected_indices
