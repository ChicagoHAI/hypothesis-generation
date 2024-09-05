from abc import ABC, abstractmethod
import os
import json
import math
from typing import Dict
from string import Template

from . import update_register
from .base import Update
from .default import DefaultUpdate
from ..generation import Generation
from ..inference import Inference
from ..replace import Replace
from ..summary_information import SummaryInformation
from ...logger_config import LoggerConfig

import pdb

logger_name = "HypoGenic - Relevance Update"


@update_register.register("relevance")
class RelevanceUpdate(Update):
    """
    DefaultUpdate uses ONE hypothesis to make a prediction on a new example.
    """

    def __init__(
        self,
        generation_class: Generation,
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
        """
        Initialize the update class

        Parameters:
            generation_class: The generation class that needs to be called in update for generating new hypotheses
            inference_class: The inference class that is called for inference in update for making predictions
            replace_class: The replace class that is called for replacing the old hypotheses with the new hypotheses
            save_path: Path to save the hypotheses.
            file_name_template: Template for the file name. Default is "hypotheses_training_sample\_${sample}\_seed\_${seed}\_epoch\_${epoch}.json"
            sample_num_to_restart_from: Sample number to resume from. Default is -1
            num_init: Number of examples to use for initializing hypotheses. Default is 25
            epoch_to_start_from: Epoch number to start from. When restarting, this should be > 1. Default is 0
            num_wrong_scale: Scale for dynamic num_wrong_to_add_bank. Default is 0.8
            k: The number of hypotheses checked per sample during training. Default is -1
            alpha: Exploration parameter. Default is 5e-1
            update_batch_size: Number of examples to use per prompt. Default is 5
            num_hypotheses_to_update: Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses. Default is 5
            update_hypotheses_per_batch: Number of hypotheses to generate per prompt. Default is 5
            only_best_hypothesis: If only the best hypothesis should be added in the newly generated hypotheses of the batch. Default is False
            save_every_n_examples: Save hypotheses every n examples. Default is 100
        """
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

    def update(
        self,
        hypotheses_bank: Dict[str, SummaryInformation],
        current_epoch,
        current_seed,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """
        We update the hypothesis bank once we reach a certain amount of regret

        Parameters:
            hypotheses_bank: The hypothesis bank
            current_epoch: The current epoch
            current_seed: The current seed
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: The maximum number of concurrent requests
        """
        logger = LoggerConfig.get_logger(logger_name)

        # initialize variables
        num_train_examples = len(self.train_data)
        wrong_example_ids = set()
        coverage_dictionary = { i: 0 for i in range(num_train_examples) }

        # ----------------------------------------------------------------------
        # Figuring out starting sample index
        # ----------------------------------------------------------------------
        if self.sample_num_to_restart_from >= 0:
            start_sample = self.sample_num_to_restart_from
        else:
            start_sample = self.num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
        # basically, if we've completed the starting epoch, we want to start the next one
        if current_epoch > self.epoch_to_start_from:
            start_sample = 0

        # ----------------------------------------------------------------------
        # Creating the new hypotheses
        # ----------------------------------------------------------------------
        culumative_acceptance = 0
        acceptance_den = 0
        for i in range(start_sample, num_train_examples):

            current_sample = i + 1
            logger.info(f"Training on example {i}")

            # We need to get the best k for testing the strength of our hypothesis bank
            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True
            )[: self.k]

            print(hypotheses_bank)

            # We are at the regret that we need in order to generate a new hypothesis
            if self.num_wrong_scale > 0:
                num_wrong_to_add_bank = (
                    len(top_k_hypotheses) * i / num_train_examples
                ) * self.num_wrong_scale

            # ------------------------------------------------------------------
            # We need to see how good our hypothesis is, which we do by way of the inference class
            # ------------------------------------------------------------------
            num_wrong_hypotheses = 0
            acceptance_rate = 0
            preds, labels, acceptance_rate = self.inference_class.batched_predict(
                self.train_data,
                [
                    (i, {hypothesis: hypotheses_bank[hypothesis]})
                    for hypothesis in top_k_hypotheses
                ],
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                target_idx = i,
                **generate_kwargs,
            )

            culumative_acceptance += acceptance_rate
            acceptance_den += 1

            # Comparison of the label and prediction
            # For the relevance update, if the hypothesis is irrelevant, the predicted output would be `None`
            for pred, label, hypothesis in zip(preds, labels, top_k_hypotheses):
                if pred is not None:
                    if pred != label:
                        num_wrong_hypotheses += 1
                        hypotheses_bank[hypothesis].update_info_if_not_useful(
                            current_sample, self.alpha
                        )  # let the bank know it got one wrong
                    else:
                        hypotheses_bank[hypothesis].update_info_if_useful(
                            current_sample, self.alpha
                        )  # let the bank know it got one right

                        # keeping track of good examples as we do in generation
                        hypotheses_bank[hypothesis].update_useful_examples(i, label)
                else:
                    num_wrong_hypotheses += 1

            # ------------------------------------------------------------------
            # Generating a new hypothesis
            # ------------------------------------------------------------------

            # if we get enough wrong examples as determined by num_wrong_to_add_bank,
            # we need to generate new hypotheses
            if (
                num_wrong_hypotheses >= num_wrong_to_add_bank
                or len(top_k_hypotheses) == 0
            ):

                # We note it as a bad sample
                wrong_example_ids.add(i)
                if (
                    len(wrong_example_ids)
                    == self.update_batch_size * self.num_hypotheses_to_update
                ):
                    new_hyp_bank = {}

                    # generate new hypotheses
                    for j in range(self.num_hypotheses_to_update):
                        # Go through poorly performing exmaples and generate hypotheses for them
                        # TODO: batched?
                        new_hypotheses = (
                            self.generation_class.batched_hypothesis_generation(
                                wrong_example_ids,
                                current_sample,
                                self.update_hypotheses_per_batch,
                                self.alpha,
                                cache_seed=cache_seed,
                                max_concurrent=max_concurrent,
                            )
                        )

                        # If we onlt take the best performing hypothesis from the batch
                        if self.only_best_hypothesis:
                            best_hypothesis = max(
                                new_hypotheses, key=lambda x: new_hypotheses[x].reward
                            )
                            new_hyp_bank.update(
                                {best_hypothesis: new_hypotheses[best_hypothesis]}
                            )
                        else:
                            new_hyp_bank.update(new_hypotheses)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class to update the bank
                    hypotheses_bank = self.replace_class.replace(
                        hypotheses_bank, new_hyp_bank
                    )

            # save hypotheses to json
            if (i + 1) % self.save_every_n_examples == 0:
                self.save_to_json(
                    hypotheses_bank,
                    sample=i + 1,
                    seed=current_seed,
                    epoch=current_epoch,
                )

        for _, summary in hypotheses_bank.items():
            cov = summary.correct_examples

            for ex in cov:
                coverage_dictionary[ex[0]] += 1

        # Our new bank
        return hypotheses_bank, coverage_dictionary, culumative_acceptance/max(acceptance_den, 1)
