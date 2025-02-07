from copy import deepcopy
import json
import logging
import math
from string import Template
from typing import Dict, List
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


from hypothesis_agent.literature_review_agent import LiteratureAgent
from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.inference import MultiHypInferenceWithRank
from hypothesis_agent.data_analysis_agent.summary_information import NewSummaryInformation
from hypothesis_agent.utils import SpecificityBooster
import matplotlib.pyplot as plt
import pandas as pd


logger = LoggerConfig.get_logger("Agent")


class OnlyPaperGeneration(DefaultGeneration):
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        inference_class: Inference,
        task: BaseTask,
        literature_agent: LiteratureAgent,
    ):
        super().__init__(api, prompt_class, inference_class, task)
        self.literature_agent = literature_agent

    def batched_hyp_list_generation(
        self,
        example_indices: List[int],
        num_hypotheses_generate: int,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ) -> List[str]:

        prompt_input = self.prompt_class.initialize_hypotheses_only_paper(
            num_hypotheses_generate,
            self.literature_agent.paper_infos,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        return extract_hypotheses(response, num_hypotheses_generate)
    
    def initialize_hypotheses_only_paper(
        self,
        num_hypotheses_generate,
        cache_seed=None,
        max_tokens=4000,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.initialize_hypotheses_only_paper(
            num_hypotheses_generate,
            self.literature_agent.paper_infos,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            **generate_kwargs,
        )

        return extract_hypotheses(response, num_hypotheses_generate)

    def initialize_hypotheses_only_paper_with_specificity_boost(
        self,
        num_hypotheses_generate,
        n_specificity_round=3,
        cache_seed=None,
        max_concurrent=32,
        max_tokens=4000,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.initialize_hypotheses_only_paper(
            num_hypotheses_generate,
            self.literature_agent.paper_infos,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            **generate_kwargs,
        )

        initial_hyp_list = extract_hypotheses(response, num_hypotheses_generate)
        specificity_booster = SpecificityBooster(
            self.api,
            self.prompt_class,
            self.task,
        )
        initial_hyp_bank = {}
        for hyp in initial_hyp_list:
            initial_hyp_bank[hyp] = {"hypothesis": hyp, "acc": 0.0}
        final_hyp_bank = specificity_booster.batched_boost_specificity(
            initial_hyp_bank,
            n_specificity_round,
            max_concurrent,
            cache_seed,
            **generate_kwargs,
        )
        return list(final_hyp_bank.keys())

class ZeroShotGeneration(DefaultGeneration):
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, inference_class, task)

    def batched_hyp_list_generation(
        self,
        example_indices: List[int],
        num_hypotheses_generate: int,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ) -> List[str]:

        prompt_input = self.prompt_class.initialize_hypotheses_0_shot(
            num_hypotheses_generate,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        return extract_hypotheses(response, num_hypotheses_generate)

    def initialize_hypotheses_0_shot(
        self,
        num_hypotheses_generate,
        cache_seed=None,
        max_concurrent=3,
        max_tokens=4000,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.initialize_hypotheses_0_shot(
            num_hypotheses_generate,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            **generate_kwargs,
        )

        return extract_hypotheses(response, num_hypotheses_generate)
    

class TestGeneration(DefaultGeneration):
    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        inference_class: Inference,
        task: BaseTask,
        literature_agent: LiteratureAgent,
        max_refine=6,
    ):
        super().__init__(api, prompt_class, inference_class, task)
        self.literature_agent = literature_agent
        self.max_refine = max_refine

    def set_max_refine(self, max_refine):
        self.max_refine = max_refine

    def batched_hypothesis_generation(
        self,
        example_ids,
        current_sample,
        num_hypotheses_generate: int,
        alpha: float,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        new_hypotheses = self.batched_hyp_list_generation(
            example_ids,
            num_hypotheses_generate,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        previous_hypotheses = new_hypotheses
        for rf_idx in range(self.max_refine):
            if rf_idx % 2 == 0:
                new_hypotheses = self.literature_agent.refine_hypotheses(
                    new_hypotheses,
                    cache_seed=cache_seed,
                    **generate_kwargs,
                )
            else:
                new_hypotheses = self.refine_hypotheses(
                    example_ids,
                    new_hypotheses,
                    cache_seed=cache_seed,
                    **generate_kwargs,
                )
            if new_hypotheses == previous_hypotheses:
                break
            previous_hypotheses = new_hypotheses

        return self.make_hypotheses_bank(
            example_ids,
            current_sample,
            alpha,
            new_hypotheses,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

    def refine_hypotheses(
        self,
        example_indices: List[int],
        hypotheses_list: List[str],
        cache_seed=None,
        **generate_kwargs,
    ):
        """
        Refine hypotheses using examples

        Parameters:
            example_ids: List of example ids
            hypotheses_list: List of hypotheses to refine
            cache_seed: Cache seed for caching the response. Default is None.

        Returns:
            List of refined hypotheses
        """
        example_bank = (
            self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
        )

        prompt_input = self.prompt_class.refine_with_data(example_bank, hypotheses_list)

        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        return extract_hypotheses(response, len(hypotheses_list))

    def initialize_hypotheses_0_shot(
        self,
        max_num_hypotheses,
        cache_seed=None,
        max_concurrent=32,
        max_tokens=4000,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.initialize_hypotheses_0_shot(
            max_num_hypotheses,
        )
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            **generate_kwargs,
        )

        return extract_hypotheses(response, max_num_hypotheses)

    def batched_initialize_hypotheses_with_paper(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        assert (
            num_init % init_batch_size == 0
        ), "Number of initial examples must be divisible by the batch size"
        num_batches = num_init // init_batch_size
        prompt_inputs = []

        for i in range(num_batches):
            example_indices = list(
                range(i * init_batch_size, (i + 1) * init_batch_size)
            )
            example_bank = (
                self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
            )
            prompt_inputs.append(
                self.prompt_class.batched_generation_with_paper(
                    example_bank,
                    init_hypotheses_per_batch,
                    self.literature_agent.paper_infos,
                )
            )
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        hypotheses_list = list(
            set(
                [
                    hyp
                    for response in responses
                    for hyp in extract_hypotheses(response, init_hypotheses_per_batch)
                ]
            )
        )

        return hypotheses_list

class MultiHypGenerationWithRank(TestGeneration):
    inference_class: MultiHypInferenceWithRank

    def __init__(
        self,
        api,
        prompt_class: TestPrompt,
        inference_class: MultiHypInferenceWithRank,
        task: BaseTask,
        literature_agent: LiteratureAgent,
        max_refine=6,
    ):
        super().__init__(
            api, prompt_class, inference_class, task, literature_agent, max_refine
        )

    def batched_hypothesis_generation(
        self,
        example_ids,
        current_sample,
        num_hypotheses_generate: int,
        alpha: float,
        beta: float,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        new_hypotheses = self.batched_hyp_list_generation(
            example_ids,
            num_hypotheses_generate,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        previous_hypotheses = new_hypotheses
        for rf_idx in range(self.max_refine):
            if rf_idx % 2 == 0:
                new_hypotheses = self.literature_agent.refine_hypotheses(
                    new_hypotheses,
                    cache_seed=cache_seed,
                    **generate_kwargs,
                )
            else:
                new_hypotheses = self.refine_hypotheses(
                    example_ids,
                    new_hypotheses,
                    cache_seed=cache_seed,
                    **generate_kwargs,
                )
            if new_hypotheses == previous_hypotheses:
                break
            previous_hypotheses = new_hypotheses

        return self.make_hypotheses_bank(
            example_ids,
            current_sample,
            alpha,
            beta,
            new_hypotheses,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

    def make_hypotheses_bank(
        self,
        example_indices,
        current_sample,
        alpha,
        beta,
        hypotheses_list: List[str],
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        idx_hyp_pair = []
        new_generated_hypotheses = {}

        for hyp in hypotheses_list:
            new_generated_hypotheses[hyp] = NewSummaryInformation(hypothesis=hyp)

        for index in example_indices:
            idx_hyp_pair.append((index, deepcopy(new_generated_hypotheses)))

        preds, labels, idx_relv_rank_pairs = self.inference_class.batched_predict(
            self.train_data,
            idx_hyp_pair,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        for index, pred, label, (_, relv_rank) in zip(
            example_indices, preds, labels, idx_relv_rank_pairs
        ):
            max_rank = len([hyp for hyp in relv_rank if relv_rank[hyp][0]])
            for hyp in relv_rank:
                new_generated_hypotheses[hyp].update_acc(
                    index,
                    label,
                    relv_rank[hyp][1],
                    max_rank,
                    pred == label,
                    current_sample,
                    alpha,
                    beta,
                )

        return new_generated_hypotheses