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
    """
    Add on extra functionality to the generation fucntion - we consider this
    the "default" task.
    """

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        """
        Parameters:
            api: the language model that you're using, which may or may not be local
            prompt_class: let's us know how the prompt is going to look
            inference_class: gives us a way to predict labels for accuracy sake
            task: determines the goal to accomplish
        """
        super().__init__(api, prompt_class, inference_class, task)

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # INITLALIZE HYPOTHESES                                                    #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        max_concurrent=3,
        **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
            num_init: the total amount of hypothesis you want to generate initalily
            init_batch_size: The amount of samples you want to include in each call to "batched_hypothesis_generation"
            init_hypotheses_per_batch: the amount of hypotheses that you're going to generate per batch
            alpha: the exploration constant in the hypogenic reward function
            max_concurrent: the maximum amount of concurrent calls to the API

        Returns:
            hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """

        # ----------------------------------------------------------------------
        # Finding batch size and confirming that it works
        # ----------------------------------------------------------------------
        assert (
            num_init % init_batch_size == 0
        ), "Number of initial examples must be divisible by the batch size"

        # To know how many batches we need to get to our desiered number of hypotheses
        num_batches = num_init // init_batch_size
        hypotheses_bank = {}

        # ----------------------------------------------------------------------
        # The hypothesis creation process
        # ----------------------------------------------------------------------
        for i in range(num_batches):

            # get the samples that you're going to use to generate the hypotheses
            example_indices = list(
                range(i * init_batch_size, (i + 1) * init_batch_size)
            )

            # the new_hypotheses
            new_hypotheses = self.batched_hypothesis_generation(
                example_indices,
                len(example_indices),
                init_hypotheses_per_batch,
                alpha,
                max_concurrent=max_concurrent,
            )

            hypotheses_bank.update(new_hypotheses)

        return hypotheses_bank

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCH INITLALIZE HYPOTHESES                                              #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def batched_initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        use_cache=1,
        max_concurrent=3,
        **kwargs
    ):
        """
        Batches the process of making new hypotheses

        Parameters:
            num_init: the number of hypotheses that you want to create out of this function
            init_batch size: the number of samples that will be used to generate these hypotheses
            init_hypotheses_per_batch: the amount of hypotheses that you want to generate per btach
            alpha: the exploration constant in the hypogenic reward function
            use_cache: whether or not you want to use the cache
            max_concurrent: the maximum amount of concurrent calls to the API
        """
        # ----------------------------------------------------------------------
        # Finding batch size and confirming that it works
        # ----------------------------------------------------------------------
        assert (
            num_init % init_batch_size == 0
        ), "Number of initial examples must be divisible by the batch size"
        num_batches = num_init // init_batch_size
        prompt_inputs = []

        # ----------------------------------------------------------------------
        # Making the batch of the responses that will be used in the batch generation
        # ----------------------------------------------------------------------
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
        responses = self.api.batched_generate(
            prompt_inputs, use_cache=use_cache, max_concurrent=max_concurrent
        )

        # ----------------------------------------------------------------------
        # Makes all the desired hypotheses
        # ----------------------------------------------------------------------
        return self.batched_batched_hypothesis_generation(
            list(range(num_init)),
            num_init,
            init_hypotheses_per_batch,
            alpha,
            responses,
            use_cache=use_cache,
            max_concurrent=max_concurrent,
        )
