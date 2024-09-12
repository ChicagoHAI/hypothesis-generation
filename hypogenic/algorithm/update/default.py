from abc import ABC, abstractmethod
import os
import json
import math
from typing import Dict
from string import Template

from . import update_register
from .base import Update
from ..generation import Generation
from ..inference import Inference
from ..replace import Replace
from ..summary_information import SummaryInformation
from ...logger_config import LoggerConfig

logger_name = "HypoGenic - Default Update"


@update_register.register("default")
class DefaultUpdate(Update):
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

        # ----------------------------------------------------------------------
        # Figuring out starting samples
        # ----------------------------------------------------------------------
        # go through training examples
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
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
        # from the start to the end
        for i in range(start_sample, num_train_examples):
            # the 'i' here is the sample we are testing each of the top hypotheses

            current_sample = i + 1
            logger.info(f"Training on example {i}")

            # We need to get the best k for testing the strength of our hypothesis bank
            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True
            )[: self.k]

            # We are at the regret that we need in order to generate a new hypothesis
            if self.num_wrong_scale > 0:
                num_wrong_to_add_bank = (
                    len(top_k_hypotheses) * i / num_train_examples
                ) * self.num_wrong_scale

            # ------------------------------------------------------------------
            # We need to see how good our hypothesis is, which we do by way of the inference class
            # ------------------------------------------------------------------
            num_wrong_hypotheses = 0
            preds, labels = self.inference_class.batched_predict(
                self.train_data,
                [
                    (i, {hypothesis: hypotheses_bank[hypothesis]})
                    for hypothesis in top_k_hypotheses
                ],
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                **generate_kwargs,
            )

            # Comparison of the label and prediction
            for pred, label, hypothesis in zip(preds, labels, top_k_hypotheses):
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
                                **generate_kwargs,
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

        # Our new bank
        return hypotheses_bank
