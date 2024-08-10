from abc import ABC, abstractmethod
import math
import os

from . import generation_register
from .base import Generation
from ..summary_information import SummaryInformation
from ..inference import Inference
from ...tasks import BaseTask
from ...prompt import BasePrompt


@generation_register.register("default")
class DefaultGeneration(Generation):
    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, inference_class, task)

    def initialize_hypotheses(
        self, num_init, init_batch_size, init_hypotheses_per_batch, alpha, **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
        ____________

        num_init:
        init_batch_size:
        init_hypotheses_per_batch:

        ____________

        Returns:
        ____________

        hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        assert (
            num_init % init_batch_size == 0
        ), "Number of initial examples must be divisible by the batch size"
        num_batches = num_init // init_batch_size
        hypotheses_bank = {}
        for i in range(num_batches):
            example_indices = list(
                range(i * init_batch_size, (i + 1) * init_batch_size)
            )
            new_hypotheses = self.batched_hypothesis_generation(
                example_indices,
                len(example_indices),
                init_hypotheses_per_batch,
                alpha,
            )
            hypotheses_bank.update(new_hypotheses)

        return hypotheses_bank

    def batched_initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        use_cache=1,
        **kwargs
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
            # TODO: need copy()?
            example_bank = (
                self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
            )
            prompt_inputs.append(
                self.prompt_class.batched_generation(
                    example_bank, init_hypotheses_per_batch
                )
            )
        responses = self.api.batched_generate(prompt_inputs, use_cache=use_cache)

        return self.batched_batched_hypothesis_generation(
            list(range(num_init)),
            num_init,
            init_hypotheses_per_batch,
            alpha,
            responses,
            use_cache=use_cache,
        )
