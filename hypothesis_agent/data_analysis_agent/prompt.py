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

class TestPrompt(BasePrompt):
    def __init__(self, task: BaseTask):
        self.task = task

    def batched_generation_with_paper(
        self, train_data: pd.DataFrame, num_hypotheses, paper_infos: List[Dict[str, str]]
    ):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        substitute_dict = {"num_hypotheses": num_hypotheses}

        multi_sub_dicts = {
            "observations": [],
            "relevant_papers": [],
        }
        for example_idx in range(len(train_data)):
            multi_sub_dicts["observations"].append(
                self._get_substitute_dict(train_data, example_idx)
            )
        for paper_info in paper_infos:
            multi_sub_dicts["relevant_papers"].append(paper_info)

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "batched_generation_with_paper"
        )
        prompt = self._information_prompt(
            substitute_dict, "batched_generation_with_paper"
        )

        return prompt


    def refine_with_data(self, train_data: pd.DataFrame, hypotheses_list: List[str]):
        """
        Refine hypotheses with data
        """

        substitute_dict = {
            "num_hypotheses": len(hypotheses_list),
            "hypotheses": "\n".join(
                [f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)]
            ),
        }

        multi_sub_dicts = {
            "observations": [],
        }
        for example_idx in range(len(train_data)):
            multi_sub_dicts["observations"].append(
                self._get_substitute_dict(train_data, example_idx)
            )

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "refine_with_data"
        )

        prompt = self._information_prompt(substitute_dict, "refine_with_data")
        return prompt


    def refine_with_literature(
        self, hypotheses_list: List[str], paper_infos: List[Dict[str, str]]
    ):
        """
        Refine hypotheses with literature
        """

        multi_sub_dicts = {
            "relevant_papers": [],
        }
        for paper_info in paper_infos:
            multi_sub_dicts["relevant_papers"].append(paper_info)

        substitute_dict = {
            "num_hypotheses": len(hypotheses_list),
            "hypotheses": "\n".join(
                [f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)]
            ),
        }

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "refine_with_literature"
        )

        prompt = self._information_prompt(substitute_dict, "refine_with_literature")

        return prompt

    def boost_specificity(
        self, hypotheses_list: List[str],
    ):
        if len(hypotheses_list) > 1:
            logger.warning("Warning: only one hypothesis refined each time!")
        substitute_dict = {
            "hypotheses": "\n".join(
                [f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)]
            ),
        }

        prompt = self._information_prompt(substitute_dict, "boost_specificity")

        return prompt

    def balance_specificity(
        self, hypotheses_list: List[str],
    ):
        if len(hypotheses_list) > 1:
            logger.warning("Warning: only one hypothesis refined each time!")
        substitute_dict = {
            "hypotheses": "\n".join(
                [f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)]
            ),
        }

        prompt = self._information_prompt(substitute_dict, "balance_specificity")

        return prompt

    def summarize_paper(self, paper_text: Dict[str, str]):
        """
        Summarize single paper
        """

        prompt = self._information_prompt(paper_text, info_key="summarize_paper")
        return prompt

    def initialize_hypotheses_only_paper(
        self, num_hypotheses, paper_infos: List[Dict[str, str]]
    ):
        """
        Generate initial hypotheses with paper summaries only
        """
        multi_sub_dicts = {
            "relevant_papers": [],
        }
        for paper_info in paper_infos:
            multi_sub_dicts["relevant_papers"].append(paper_info)

        substitute_dict = {
            "num_hypotheses": num_hypotheses,
        }

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "initialize_hypotheses_only_paper"
        )

        prompt = self._information_prompt(
            substitute_dict, info_key="initialize_hypotheses_only_paper"
        )
        return prompt

    def initialize_hypotheses_0_shot(self, num_hypotheses):
        substitute_dict = {"num_hypotheses": num_hypotheses}
        prompt = self._information_prompt(substitute_dict, "initialize_zero_shot")
        return prompt
    
    def multiple_hypotheses_inference(self, hypotheses_dict, test_data, test_idx):
        """
        Create multiple hypotheses inference prompt.
        """

        hypotheses_list = list(hypotheses_dict.keys())

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypotheses"] = "\n".join([f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)])

        prompt = self._information_prompt(substitute_dict, "multiple_hypotheses_inference")

        return prompt

    def test_autogen(
        self, train_data: pd.DataFrame, num_hypotheses, paper_infos: List[Dict[str, str]]
    ):
        substitute_dict = {"num_hypotheses": num_hypotheses}
        keys = self._get_substitute_key(self._get_prompt_template("test_autogen"))
        keys = [key for key in keys if key not in substitute_dict]

        df_paper_infos = pd.DataFrame(paper_infos).reset_index(drop=True)

        multi_sub_dicts_data = []
        for example_idx in range(len(train_data)):
            multi_sub_dicts_data.append(self._get_substitute_dict(train_data, example_idx))
        multi_sub_dicts_paper = []
        for paper_idx in range(len(paper_infos)):
            multi_sub_dicts_paper.append(self._get_substitute_dict(df_paper_infos, paper_idx))
        
        for key in keys:
            if self._is_multi_content(self._get_prompt_template(key)) and key == "observations":
                substitute_dict[key] = self._fill_multi_content(
                    ({}, multi_sub_dicts_data),
                    self._get_prompt_template(key),
                )
            elif self._is_multi_content(self._get_prompt_template(key)) and key == "relevant_papers":
                substitute_dict[key] = self._fill_multi_content(
                    ({}, multi_sub_dicts_paper),
                    self._get_prompt_template(key),
                )
            else:
                substitute_dict[key] = self._get_prompt_template(key)
        
        prompt = self._information_prompt(substitute_dict, "test_autogen")
        return prompt[1]["content"]

    def remove_hypothesis_repetition(
        self, hypotheses_list
    ):
        substitute_dict = {}
        substitute_dict["hypotheses"] = "\n".join([f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)])
        prompt = self._information_prompt(substitute_dict, "remove_hypothesis_repetition")
        return prompt

    def check_hypothesis_pair_repetition(
        self,
        hyp_list,
    ):
        substitute_dict = {}
        substitute_dict["hypotheses"] = "\n".join([f"{idx+1}. {hyp}" for idx, hyp in enumerate(hyp_list)])
        prompt = self._information_prompt(substitute_dict, "check_hypothesis_pair_repetition")
        return prompt

    def multi_hyp_inference_with_rank(self, hypotheses_dict, test_data, test_idx):
        hypotheses_list = list(hypotheses_dict.keys())
        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypotheses"] = "\n".join(
            [f"{idx+1}. {hyp}" for idx, hyp in enumerate(hypotheses_list)]
        )
        prompt = self._information_prompt(
            substitute_dict, "multi_hyp_inference_with_rank"
        )
        return prompt

    def headline_ood_few_shot_baseline(
        self, train_data: pd.DataFrame, num_few_shot, test_data, test_idx
    ):
        """
        Few shot prompt for baseline
        """

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        multi_sub_dicts = {"few_shot_observations": []}
        for j in range(num_few_shot):
            multi_sub_dicts["few_shot_observations"].append(
                self._get_substitute_dict(train_data, j)
            )
        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "few_shot_baseline"
        )

        prompt = self._information_prompt(substitute_dict, "few_shot_baseline")

        return prompt