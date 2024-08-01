from abc import ABC, abstractmethod
import math
import os

from .base import Generation
from ..summary_information import SummaryInformation
from ..inference import Inference
from ...prompt import BasePrompt


class DefaultGeneration(Generation):
    def __init__(self, api, prompt_class, inference_class):
        super().__init__(api, prompt_class, inference_class)

    def initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        use_system_prompt,
        **kwargs
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
                use_system_prompt,
            )
            hypotheses_bank.update(new_hypotheses)

        return hypotheses_bank
