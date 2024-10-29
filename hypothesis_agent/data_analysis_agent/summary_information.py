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
from hypogenic.algorithm.update import Update, DefaultUpdate
from hypogenic.extract_label import extract_label_register
from hypogenic.utils import set_seed, get_results
from hypogenic.logger_config import LoggerConfig
from hypogenic.algorithm.replace import DefaultReplace, Replace

import matplotlib.pyplot as plt
import pandas as pd


class NewSummaryInformation(SummaryInformation):
    def __init__(
        self,
        hypothesis="",
        acc=0.0,
        reward=0,
        num_visits=0,
        num_select=0,
        correct_examples=None,
    ):
        """
        Initialize the SummaryInformation object

        Parameters:
            hypothesis: the hypothesis that the object is tracking
            acc: the accuracy of the hypothesis
            reward: the reward of the hypothesis
            num_visits: the number of times the hypothesis has been visited
            num_select: the number of times the hypothesis has been selected
            correct_examples: a list of tuples of the form (sample_index, label, rank, max_rank)
        """
        if correct_examples is None:
            correct_examples = []

        super().__init__(hypothesis, acc, reward, num_visits, correct_examples)
        self.num_select = num_select

    def update_reward(self, alpha, beta, num_examples):
        self.reward = (
            alpha * math.sqrt(math.log(num_examples) / self.num_visits)
            + beta * self.acc
            + self.num_select / self.num_visits
        )

    def update_not_select(self, alpha, beta, cur_exmpl):
        self.num_visits += 1
        self.update_reward(alpha, beta, cur_exmpl)

    def update_acc(
        self,
        exmpl_idx: int,
        label: str,
        rank: int,
        max_rank: int,
        correct: bool,
        cur_exmpl: int,
        alpha: float,
        beta: float,
    ):
        if rank < 0:
            self.update_not_select(alpha, beta, cur_exmpl)
            return

        if correct:
            self.acc = (self.acc * self.num_select + rank / max_rank) / (
                self.num_select + 1
            )
            self.correct_examples.append((exmpl_idx, label, rank, max_rank))
        else:
            self.acc = (self.acc * self.num_select) / (self.num_select + 1)
        self.num_select += 1
        self.num_visits += 1
        self.update_reward(alpha, beta, cur_exmpl)

    @staticmethod
    def from_dict(data: Dict[str, List[float]]) -> "NewSummaryInformation":
        return NewSummaryInformation(**data)