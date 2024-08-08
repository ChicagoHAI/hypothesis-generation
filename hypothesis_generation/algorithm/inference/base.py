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

        Parameters
        _____________
        api: the LLM api wrapper
        prompt_class: the prompt class for the specified task
        _____________

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
        use_cache=1,
    ):
        """
        Generate responses for every pair of data and hypotheses.

        :param data: the data to predict on
        :param idx_hyp_pair: a list of tuples of indices and hypothesis banks
        """
        pass

    @abstractmethod
    def predict(self, data, index, hyp_bank, use_cache=1):
        """Implements a specific type of prediction

        Parameters
        __________
        args: the arguments of the algorithm
        data: the specific dataset
        index: the specific index to predict for
        hyp_bank: a dictionary of hypotheses

        Returns
        __________
        prediction: the predicted value
        actual_label: the actual label of the sample
        """
        pass

    @abstractmethod
    def run_inference_final(self, data, hyp_bank, **kwargs):
        """Implements a specific type of prediction

        Parameters
        __________
        args: the arguments of the algorithm
        data: the specific dataset
        hyp_bank: a dictionary of hypotheses
        k: the number of hypotheses to use

        Returns
        __________
        accuracy: the accuracy over the dataset
        """
        pass
