import json
import logging
import math
from string import Template
from typing import Callable, Dict, List, Union
import matplotlib.pyplot as plt
import pandas as pd
import yaml

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
from hypogenic.register import Register


class TestTask(BaseTask):
    """
    In this class, we set up and define the task along with prepping training data.

    All our information if from a yaml file that the user must set up.
    """

    def __init__(
        self,
        config_path: str,
        extract_label: Union[Callable[[str], str], None] = None,
        from_register: Union[Register, None] = None,
        extract_label_with_rank: Callable[[str], str] = None,
    ):
        super().__init__(config_path, extract_label, from_register)
        self.extract_label_with_rank = extract_label_with_rank