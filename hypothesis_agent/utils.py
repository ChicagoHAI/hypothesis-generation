import os
import sys
import json
from typing import List, Dict, Tuple, Any

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.LLM_wrapper import GPTWrapper, LocalVllmWrapper
from hypogenic.algorithm.generation.utils import extract_hypotheses

from hypothesis_agent.data_analysis_agent.prompt import TestPrompt

class SpecificityBooster:
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        task: BaseTask,
    ):
        self.api = api
        self.prompt_class = prompt_class
        self.task = task
    
    def batched_boost_specificity(
        self,
        hyp_bank: Dict[str, Any],
        n_round: int = 1,
        max_concurrent: int = 32,
        cache_seed = None,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        hyp_list = list(hyp_bank.keys())
        tmp_hyp_bank = {}
        for round in range(n_round):
            prompt_inputs = []
            for i in range(0, len(hyp_list)):
                hyp = hyp_list[i]
                prompt_inputs.append(
                    self.prompt_class.boost_specificity([hyp_list[i]])
                )
            responses = self.api.batched_generate(
                prompt_inputs,
                max_concurrent,
                cache_seed,
                **generate_kwargs,
            )
            responses = responses[::-1]
            for i in range(0, len(hyp_list)):
                hyp = hyp_list[i]
                response = responses.pop(-1)
                new_hypothesis_list = extract_hypotheses(response, 1)
                new_hypothesis = new_hypothesis_list[0]
                hyp_list[i] = new_hypothesis
                if hyp in hyp_bank.keys():
                    tmp_hyp_bank[new_hypothesis] = hyp_bank[hyp]
                else:
                    tmp_hyp_bank[new_hypothesis] = tmp_hyp_bank[hyp]
        new_hyp_bank = {}
        for i in range(0, len(hyp_list)):
            new_hyp_bank[hyp_list[i]] = tmp_hyp_bank[hyp_list[i]]
        return new_hyp_bank

    def batched_balance_specificity(
        self,
        hyp_bank: Dict[str, Any],
        n_round: int = 1,
        max_concurrent: int = 32,
        cache_seed = None,
        **generate_kwargs,
    ):
        hyp_list = list(hyp_bank.keys())
        tmp_hyp_bank = {}
        for round in range(n_round):
            prompt_inputs = []
            for i in range(0, len(hyp_list)):
                hyp = hyp_list[i]
                prompt_inputs.append(
                    self.prompt_class.balance_specificity([hyp_list[i]])
                )
            responses = self.api.batched_generate(
                prompt_inputs,
                max_concurrent,
                cache_seed,
                **generate_kwargs,
            )
            responses = responses[::-1]
            for i in range(0, len(hyp_list)):
                hyp = hyp_list[i]
                response = responses.pop(-1)
                new_hypothesis_list = extract_hypotheses(response, 1)
                new_hypothesis = new_hypothesis_list[0]
                hyp_list[i] = new_hypothesis
                if hyp in hyp_bank.keys():
                    tmp_hyp_bank[new_hypothesis] = hyp_bank[hyp]
                else:
                    tmp_hyp_bank[new_hypothesis] = tmp_hyp_bank[hyp]
        new_hyp_bank = {}
        for i in range(0, len(hyp_list)):
            new_hyp_bank[hyp_list[i]] = tmp_hyp_bank[hyp_list[i]]
        return new_hyp_bank