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
from hypogenic.algorithm.update import Update, DefaultUpdate
from hypogenic.extract_label import extract_label_register
from hypogenic.utils import set_seed, get_results
from hypogenic.logger_config import LoggerConfig
from hypogenic.algorithm.replace import DefaultReplace, Replace

import matplotlib.pyplot as plt
import pandas as pd

from .generation import TestGeneration
from .inference import MultiHypInferenceWithRank
from .summary_information import NewSummaryInformation
from ..literature_review_agent import LiteratureAgent


logger = LoggerConfig.get_logger("Agent")


class TestUpdate(DefaultUpdate):
    """
    DefaultUpdate uses ONE hypothesis to make a prediction on a new example.
    """

    def __init__(
        self,
        generation_class: TestGeneration,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        file_name_template: str = "hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=-1,
        alpha=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        super().__init__(
            generation_class,
            inference_class,
            replace_class,
            save_path,
            file_name_template,
            sample_num_to_restart_from,
            num_init,
            epoch_to_start_from,
            num_wrong_scale,
            k,
            alpha,
            update_batch_size,
            num_hypotheses_to_update,
            update_hypotheses_per_batch,
            only_best_hypothesis,
            save_every_n_examples,
        )

    def batched_initialize_hypotheses_with_paper(
        self,
        num_init=25,
        init_batch_size=5,
        init_hypotheses_per_batch=5,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ) -> Dict[str, SummaryInformation]:
        """
        Generates the initial hypotheses with paper summaries

        Parameters:
            num_init: Number of examples to use for initializing hypotheses. Default is 25
            init_batch_size: Batch size to generate hypotheses. Default is 5
            init_hypotheses_per_batch: Number of hypotheses to generate per batch. Default is 5

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        hypotheses_list = self.generation_class.batched_initialize_hypotheses_with_paper(
            num_init,
            init_batch_size,
            init_hypotheses_per_batch,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        return self.generation_class.make_hypotheses_bank(
            example_indices=list(range(num_init)),
            current_sample=num_init,
            alpha=self.alpha,
            hypotheses_list=hypotheses_list,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )


class MultiHypUpdate(TestUpdate):
    inference_class: MultiHypInferenceWithRank

    def __init__(
        self,
        generation_class: TestGeneration,
        inference_class: MultiHypInferenceWithRank,
        replace_class: Replace,
        save_path: str,
        file_name_template: str = "hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=-1,
        alpha=5e-1,
        beta=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        super().__init__(
            generation_class,
            inference_class,
            replace_class,
            save_path,
            file_name_template,
            sample_num_to_restart_from,
            num_init,
            epoch_to_start_from,
            num_wrong_scale,
            k,
            alpha,
            update_batch_size,
            num_hypotheses_to_update,
            update_hypotheses_per_batch,
            only_best_hypothesis,
            save_every_n_examples,
        )
        self.beta = beta

    def batched_initialize_hypotheses_with_paper(
        self,
        num_init=25,
        init_batch_size=5,
        init_hypotheses_per_batch=5,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ) -> Dict[str, SummaryInformation]:
        """
        Generates the initial hypotheses with paper summaries

        Parameters:
            num_init: Number of examples to use for initializing hypotheses. Default is 25
            init_batch_size: Batch size to generate hypotheses. Default is 5
            init_hypotheses_per_batch: Number of hypotheses to generate per batch. Default is 5

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        hypotheses_list = (
            self.generation_class.batched_initialize_hypotheses_with_paper(
                num_init,
                init_batch_size,
                init_hypotheses_per_batch,
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                **generate_kwargs,
            )
        )
        return self.generation_class.make_hypotheses_bank(
            example_indices=list(range(num_init)),
            current_sample=num_init,
            alpha=self.alpha,
            beta=self.beta,
            hypotheses_list=hypotheses_list,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

    def update(
        self,
        hypotheses_bank: Dict[str, NewSummaryInformation],
        current_epoch,
        current_seed,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        logger_name = "Agent"
        logger = LoggerConfig.get_logger(logger_name)

        num_train_examples = len(self.train_data)
        wrong_example_ids = set()

        if self.sample_num_to_restart_from >= 0:
            start_sample = self.sample_num_to_restart_from
        else:
            start_sample = self.num_init

        if current_epoch > self.epoch_to_start_from:
            start_sample = 0

        for i in range(start_sample, num_train_examples):

            current_sample = i + 1
            logger.info(f"Training on example {i}")

            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True
            )[: self.k]

            if self.num_wrong_scale > 0:
                num_wrong_to_add_bank = (
                    len(top_k_hypotheses) * i / num_train_examples
                ) * self.num_wrong_scale

            num_wrong_hypotheses = 0
            preds, labels, idx_relv_rank_pair = self.inference_class.batched_predict(
                self.train_data,
                [(i, {hyp: hypotheses_bank[hyp] for hyp in top_k_hypotheses})],
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                **generate_kwargs,
            )

            for pred, label, (_, relv_rank) in zip(preds, labels, idx_relv_rank_pair):
                max_rank = len([hyp for hyp in relv_rank if relv_rank[hyp][0]])
                for hyp in relv_rank:
                    hypotheses_bank[hyp].update_acc(
                        i,
                        label,
                        relv_rank[hyp][1],
                        max_rank,
                        pred == label,
                        current_sample,
                        self.alpha,
                        self.beta,
                    )

            if (
                num_wrong_hypotheses >= num_wrong_to_add_bank
                or len(top_k_hypotheses) == 0
            ):

                wrong_example_ids.add(i)
                if (
                    len(wrong_example_ids)
                    == self.update_batch_size * self.num_hypotheses_to_update
                ):
                    new_hyp_bank = {}

                    for j in range(self.num_hypotheses_to_update):
                        new_hypotheses = (
                            self.generation_class.batched_hypothesis_generation(
                                wrong_example_ids,
                                current_sample,
                                self.update_hypotheses_per_batch,
                                self.alpha,
                                self.beta,
                                cache_seed=cache_seed,
                                max_concurrent=max_concurrent,
                                **generate_kwargs,
                            )
                        )

                        if self.only_best_hypothesis:
                            best_hypothesis = max(
                                new_hypotheses, key=lambda x: new_hypotheses[x].reward
                            )
                            new_hyp_bank.update(
                                {best_hypothesis: new_hypotheses[best_hypothesis]}
                            )
                        else:
                            new_hyp_bank.update(new_hypotheses)
                    wrong_example_ids = set()

                    hypotheses_bank = self.replace_class.replace(
                        hypotheses_bank, new_hyp_bank
                    )

            if (i + 1) % self.save_every_n_examples == 0:
                self.save_to_json(
                    hypotheses_bank,
                    sample=i + 1,
                    seed=current_seed,
                    epoch=current_epoch,
                )

        return hypotheses_bank