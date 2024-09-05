from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
# import pulp
import random
import re
import pdb

from . import inference_register
from .base import Inference
from .default import DefaultInference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask
from ...logger_config import LoggerConfig
import re

logger_name = "HypoGenic - Relevance Inference"

def extract_label_persuasion(text):
    # Extract the final answer from the data
    try:
        find = re.findall(r"\[ANS\](.+)\[ANS\]", text)[0]
        find = find.replace("\[ANS\]", "").strip()
    except:
        find = ""

    return find


@inference_register.register("relevance")
class RelevanceInference(DefaultInference):
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
        target_idx = -1,
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

        prompt_inputs_inference = []
        prompt_inputs_relevance = []
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]

        # ----------------------------------------------------------------------
        # TODO: Filter the relevant hypotheses using the helper functions
        # ----------------------------------------------------------------------

        if target_idx == -1:
            for (idx, hypothesis_dict) in idx_hyp_pair:
                prompt_inputs_relevance.append((idx, self.prompt_class.is_relevant(hypothesis_dict, data, idx)))
                prompt_inputs_inference.append((idx, self.prompt_class.inference(hypothesis_dict, data, idx)))
        else:
            for (idx, hypothesis_dict) in idx_hyp_pair:
                prompt_inputs_relevance.append((idx, self.prompt_class.is_relevant_rankings(hypothesis_dict, data, target_idx)))

        relevance_responses = self.api.batched_generate(
            [prompt for _, prompt in prompt_inputs_relevance],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        predictions = []

        accept_den = 0
        accept_num = 0

        # TODO: make the predictions list, should be `None` for irrelevant hypotheses
        # maybe don't make it linear
        for idx2, rel_response in enumerate(relevance_responses):
            rel_rank_hyp = self.extract_relevance_ranking(relevance_responses[0])
            ans = self.extract_relevance(rel_response)

            if target_idx == -1 and ans:
                infer = self.api.generate(
                    prompt_inputs_inference[idx2][1],
                    max_tokens = 1000)
                extracted_ans_from_infer = extract_label_persuasion(infer)

                predictions.append(extracted_ans_from_infer)

                accept_num += 1
            elif rel_rank_hyp is not None:
                prompt = self.prompt_class.inference({rel_rank_hyp: 0}, data, target_idx)
                infer = self.api._generate(messages = prompt, model = "gpt-4o-mini")

                predictions.append(infer)
            else:
                predictions.append(None)
        
            accept_den += 1

        # and once we get the actual labels
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]

        # we can return our predictions along with the labels
        return predictions, actual_labels, accept_num/max(accept_den, 1)

    def extract_relevance(self, response):
        logger = LoggerConfig.get_logger(logger_name)

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
            logger.info("Hypothesis is relevant")
            return True
        else:
            logger.info(f"Hypothesis is not relevant")
            return False

    def extract_relevance_ranking(self, text):
        pattern = r'1\.\s*(.*?)(?=\n2\.|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def filter_hypothesis(
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

        Parameters:
            data: the data to predict on
            hyp_bank: the hypotheses that we want to predict from
            k: the number of hypotheses to keep
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        logger = LoggerConfig.get_logger(logger_name)
        # iterate over the dataset and make predictions
        num_samples = len(data)

        prompt_inputs = [
            self.prompt_class.is_relevant({hypothesis: hyp_bank[hypothesis]}, data, i)
            for i in range(num_samples)
            for hypothesis in hyp_bank
        ]
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        # Returns a list of hypothesis banks, each containing only the relevant hypotheses
        relevant_hypotheses_banks = []
        for response in responses:
            relevant_hypotheses = {}
            for hypothesis in hyp_bank:
                is_relevant = self.extract_relevance(response)
                if is_relevant:
                    relevant_hypotheses[hypothesis] = hyp_bank[hypothesis]
                    logger.info("Hypothesis is relevant")
                else:
                    logger.info(f"Hypothsis is not relevant")
            relevant_hypotheses_banks.append(relevant_hypotheses)

        return relevant_hypotheses_banks
