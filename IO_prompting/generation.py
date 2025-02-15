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

from IO_prompting.prompt import IOPrompt
import matplotlib.pyplot as plt
import pandas as pd


logger = LoggerConfig.get_logger("Agent")


class IOGeneration(DefaultGeneration):
    def __init__(
        self,
        api,
        prompt_class: IOPrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, inference_class, task)

    def IO_hyp_list_generation_with_feedback(
        self,
        example_indices: List[int],
        num_hypotheses_generate: int,
        cache_seed=None,
        hypotheses_dict: Dict[str, SummaryInformation] = {},
        **generate_kwargs
    ) -> List[str]:
        """Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number

        Returns:
            hypotheses_list: A list containing all newly generated hypotheses.
        """

        # ----------------------------------------------------------------------
        # Gather the examples to use for generation
        # ----------------------------------------------------------------------
        # Gather examples based on example_indices
        # TODO: need copy()?
        example_bank = (
            self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
        )
        
        # ----------------------------------------------------------------------
        # Prompt LLM to generate hypotheses
        # ----------------------------------------------------------------------
        # Batch generate a bunch of prompts based on yaml file
        prompt_input = self.prompt_class.refine_with_feedback(
            example_bank, 
            num_hypotheses_generate,
            hypotheses_dict
        )

        # Batch generate responses based on the prompts that we just generated
        response = self.api.generate(
            prompt_input, cache_seed=cache_seed, **generate_kwargs
        )

        return extract_hypotheses(response, num_hypotheses_generate)

    def IO_batched_hypothesis_generation(
        self,
        example_ids,
        current_sample,
        num_hypotheses_generate: int,
        alpha: float,
        cache_seed=None,
        max_concurrent=3,
        hypotheses_dict: Dict[str, SummaryInformation] = {},
        **generate_kwargs,
    ):
        """
        Generates new hypotheses for the given examples

        Parameters:
            example_ids: The ids of the examples for which hypotheses need to be generated
            current_sample: the current sample in data which the algorithm is on
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            alpha: eploration constant in hypogenic reward funciton
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: The maximum number of concurrent requests

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        new_hypotheses = self.IO_hyp_list_generation_with_feedback(
            example_ids,
            num_hypotheses_generate,
            cache_seed=cache_seed,
            hypotheses_dict=hypotheses_dict,
            **generate_kwargs,
        )

        return self.make_hypotheses_bank(
            example_ids,
            current_sample,
            alpha,
            new_hypotheses,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )


    