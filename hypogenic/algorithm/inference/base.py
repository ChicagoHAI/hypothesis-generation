from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pulp
import random
import re

from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask


class Inference(ABC):
    """Inference abstract class. For each style of inference implement the inference function."""

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        """Initialize the inference class.

        Parameters:
            api: the LLM api wrapper
            prompt_class: the prompt class for the specified task
            task: the task that the prompt class is based off of
            train_data: self explanitorily - the training data
        """
        super().__init__()
        self.api = api
        self.prompt_class = prompt_class
        self.train_data = train_data
        self.task = task

    @abstractmethod
    def batched_predict(
        self,
        data,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        cache_seed=None,
        **generate_kwargs,
    ):
        """
        Generate responses for every pair of data and hypotheses.

        Parameters:
            data: the data to predict on
            idx_hyp_pair: a list of tuples of indices and hypothesis banks
        """
        pass

    @abstractmethod
    def run_inference_final(
        self,
        data,
        hyp_bank,
        cache_seed=None,
        max_concurrent=3,
        generate_kwargs={},
        **kwargs,
    ):
        """Implements a specific type of prediction

        Parameters:
            data: the specific dataset
            hyp_bank: a dictionary of hypotheses
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests

        Returns:
            accuracy: the accuracy over the dataset
        """
        pass
