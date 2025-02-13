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

import matplotlib.pyplot as plt
import pandas as pd

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("Agent")

class IOPrompt(BasePrompt):
    def __init__(self, task: BaseTask):
        self.task = task

    def refine_with_feedback(self, 
                             train_data, 
                             num_hypotheses,
                             hypotheses_dict: Dict[str, SummaryInformation]):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """
        
        hypothesis_text = list(hypotheses_dict.keys())[0]
        top_hypothesis_correct_examples = [x[0] for x in hypotheses_dict[hypothesis_text].correct_examples]

        substitute_dict = {
            "num_hypotheses": num_hypotheses,
            "hypothesis_text": hypothesis_text}

        multi_sub_dicts = {"observations": []}

        for example_idx in range(len(train_data)):
            # Skip the correctly predicted examples
            if example_idx in top_hypothesis_correct_examples:
                continue
            # If the top hypothesis predicted wrong
            multi_sub_dicts["observations"].append(
                self._get_substitute_dict(train_data, example_idx)
            )
        
        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "IO_refine_with_feedback"
        )

        prompt = self._information_prompt(substitute_dict, "IO_refine_with_feedback")
        logger.debug(f"System prompt: {prompt[0]['content']}")
        logger.debug(f"User prompt: {prompt[1]['content']}")

        return prompt


