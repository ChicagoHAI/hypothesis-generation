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

logger_name = "HypoGenic - Sampling Update"


@update_register.register("sampling")
class SamplingUpdate(Update):
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
        Update the hypotheses bank.

        Parameters:
            hypotheses_bank: The current hypotheses bank
            current_epoch: The current epoch
            current_seed: The current seed
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: The maximum number of concurrent requests
        """
        logger = LoggerConfig.get_logger(logger_name)

        num_train_examples = len(self.train_data)
        wrong_example_ids = set()

        # go through training examples
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
        if self.sample_num_to_restart_from >= 0:
            start_sample = self.sample_num_to_restart_from
        else:
            start_sample = self.num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
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

            # check if the hypothesis works for the generated hypotheses
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
            for pred, label, hypothesis in zip(preds, labels, top_k_hypotheses):
                if pred != label:
                    num_wrong_hypotheses += 1
                    hypotheses_bank[hypothesis].update_info_if_not_useful(
                        current_sample, self.alpha
                    )
                else:
                    hypotheses_bank[hypothesis].update_info_if_useful(
                        current_sample, self.alpha
                    )
                    hypotheses_bank[hypothesis].update_useful_examples(i, label)

            # if we get enough wrong examples
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

                    # generate new hypotheses
                    for _ in range(self.num_hypotheses_to_update):
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

                        max_visited = max(
                            hypotheses_bank, key=lambda x: hypotheses_bank[x].num_visits
                        )
                        new_hypotheses = self.balance_by_sample(
                            new_hypotheses,
                            current_sample,
                            hypotheses_bank[max_visited].num_visits,
                            self.num_init,
                            self.alpha,
                            cache_seed=cache_seed,
                            **generate_kwargs,
                        )
                        if self.only_best_hypothesis:
                            best_hypothesis = max(
                                new_hypotheses, key=lambda x: new_hypotheses[x].reward
                            )
                            new_hyp_bank.update(
                                {best_hypothesis: new_hypotheses[best_hypothesis]}
                            )
                        else:
                            new_hyp_bank = new_hypotheses
                            logger.info("Here is the new hypothesis bank:")
                            for hyp in new_hyp_bank:
                                logger.info(hyp)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class
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

        return hypotheses_bank

    def balance_by_sample(
        self,
        hypotheses_bank,
        current_sample,
        max_visits,
        num_init,
        alpha,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """
        Balance the number of samples for each hypothesis.

        Parameters:
            hypotheses_bank: The current hypotheses bank
            current_sample: The current sample number
            max_visits: The maximum number of visits
            num_init: The number of initial samples
            alpha: The alpha value
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: The maximum number of concurrent requests
        """
        if max_visits > 60:
            val = num_init
        elif max_visits > 30:
            val = 10
        else:
            val = 5
        preds, labels = self.inference_class.batched_predict(
            self.train_data,
            [
                (i, {hypothesis: hypotheses_bank[hypothesis]})
                for hypothesis in hypotheses_bank
                for i in range(val)
            ],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        preds, labels = preds[::-1], labels[::-1]
        for hypothesis in hypotheses_bank:
            num_right = 0
            ex = set(hypotheses_bank[hypothesis].correct_examples)
            for i in range(val):
                pred, label = preds.pop(-1), labels.pop(-1)
                if pred == label:
                    num_right += 1
                    ex.add((i, label))
            num_visits = hypotheses_bank[hypothesis].num_visits + val
            acc = (
                hypotheses_bank[hypothesis].acc * hypotheses_bank[hypothesis].num_visits
                + num_right
            ) / (num_visits)
            reward = acc + alpha * math.sqrt(math.log(current_sample) / num_visits)

            hypotheses_bank[hypothesis].set_example(list(ex))
            hypotheses_bank[hypothesis].set_reward(reward)
            hypotheses_bank[hypothesis].set_accuracy(acc)
            hypotheses_bank[hypothesis].set_num_visits(num_visits)

        return hypotheses_bank
