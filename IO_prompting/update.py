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

from .generation import IOGeneration


logger = LoggerConfig.get_logger("Agent")


class IOUpdate(DefaultUpdate):
    """
    DefaultUpdate uses ONE hypothesis to make a prediction on a new example.
    """

    def __init__(
        self,
        generation_class: IOGeneration,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        file_name_template: str = "hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
        sample_num_to_restart_from=-1,
        num_init=10,
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
        logger_name = "Agent"
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
            start_sample = 0

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
            logger.info(f"Evaluating hypotheses on example {i}")

            # For IO prompting, we just sort the hypotheses by accuracy without selecting k
            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].acc, reverse=True
            )


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
        # Evaluation done. Generate new hypotheses
        # ------------------------------------------------------------------

        new_hyp_bank = {}

        example_ids = list(range(start_sample, num_train_examples))
        # generate new hypotheses
        for j in range(self.num_hypotheses_to_update):
            # Go through poorly performing exmaples and generate hypotheses for them
            # TODO: batched?
            new_hypotheses = (
                self.generation_class.IO_batched_hypothesis_generation(
                    example_ids,
                    num_train_examples,
                    self.update_hypotheses_per_batch,
                    self.alpha,
                    cache_seed=cache_seed,
                    max_concurrent=max_concurrent,
                    hypotheses_dict=hypotheses_bank,
                    **generate_kwargs,
                )
            )
        new_hyp_bank.update(new_hypotheses)
        # Use the new hypothesis bank
        hypotheses_bank = new_hyp_bank

        # save hypotheses to json
        self.save_to_json(
            hypotheses_bank,
            sample=num_train_examples,
            seed=current_seed,
            epoch=current_epoch,
        )

        # Our new bank
        return hypotheses_bank
