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

    
    