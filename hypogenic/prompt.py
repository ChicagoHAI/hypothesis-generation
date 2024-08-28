from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from typing import List, Tuple, Union, Dict
from copy import deepcopy
import pandas as pd

from .tasks import BaseTask

import pdb


class BasePrompt(ABC):
    """
    This class gives us a way to conviniently generate prompts.
    """

    def __init__(self, task: BaseTask):
        self.task = task

    def _get_substitute_dict(
        self, data_dict: pd.DataFrame, example_idx 
    ) -> Dict[str, str]:
        substitute_dict = data_dict.loc[example_idx].to_dict()
        return substitute_dict

    def _substitute_obj(
        self, substitute_dict: Dict[str, str], obj: Union[str, List, Dict]
    ):
        if isinstance(obj, str):
            return Template(obj).substitute(substitute_dict)
        elif isinstance(obj, list):
            return [self._substitute_obj(substitute_dict, o) for o in obj]
        elif isinstance(obj, dict):
            return {k: self._substitute_obj(substitute_dict, v) for k, v in obj.items()}

    def _information_prompt(
        self, substitute_dict: Dict[str, str], info_key: str
    ) -> Dict[str, str]:
        prompt = deepcopy(self.task.prompt_template[info_key])
        return self._substitute_obj(substitute_dict, prompt)

    def _get_prompt_template(self, key: str) -> Union[str, List[Dict[str, str]]]:
        return deepcopy(self.task.prompt_template[key])

    def _convert_to_messages(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def few_shot_baseline(
        self, train_data: pd.DataFrame, num_few_shot, test_data, test_idx
    ):
        """
        Few shot prompt for baseline
        """

        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        observations = ""
        few_shot_prefix = ""
        if num_few_shot > 0:
            few_shot_prefix = self._information_prompt({}, "few_shot_prefix")
            for j in range(num_few_shot):
                observations += self._information_prompt(
                    self._get_substitute_dict(train_data, j), "observations"
                )

        substitute_dict["observations"] = observations
        substitute_dict["few_shot_prefix"] = few_shot_prefix

        prompt = self._information_prompt(substitute_dict, "few_shot_baseline")

        return prompt

    def batched_generation(self, train_data, num_hypotheses):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        observations = ""
        for example_idx in range(len(train_data)):
            observations += self._information_prompt(
                self._get_substitute_dict(train_data, example_idx), "observations"
            )

        substitute_dict = {
            "num_hypotheses": num_hypotheses,
            "observations": observations,
        }

        prompt = self._information_prompt(substitute_dict, "batched_generation")

        return prompt

    def inference(self, hypotheses_dict, test_data, test_idx):
        """
        Create inference prompt.
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        prompt = self._information_prompt(substitute_dict, "inference")

        return prompt

    def one_step_adaptive_inference(
        self, hypotheses_dict, train_data, test_data, test_idx
    ):
        """
        One step adaptive inference prompt
        """

        adaptive_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            adaptive_info_prompt += f"Pattern {hyp_idx + 1}: {hypothesis_text}\n"

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                adaptive_info_prompt += f"Example {ex_idx + 1}:\n"
                adaptive_info_prompt += self._information_prompt(
                    self._get_substitute_dict(train_data, example_info[0]),
                    "adaptive_info_prompt",
                )

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["adaptive_info_prompt"] = adaptive_info_prompt

        prompt = self._information_prompt(substitute_dict, "adaptive_inference")

        return prompt

    def adaptive_selection(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        Hypothesis selection prompt for two step adaptive inference
        """

        adaptive_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            adaptive_info_prompt += f"Pattern {hyp_idx + 1}: {hypothesis_text}\n"

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                adaptive_info_prompt += f"Example {ex_idx + 1}:\n"
                adaptive_info_prompt += self._information_prompt(
                    self._get_substitute_dict(train_data, example_info[0]),
                    "adaptive_info_prompt",
                )

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["adaptive_info_prompt"] = adaptive_info_prompt

        prompt = self._information_prompt(substitute_dict, "adaptive_selection")

        return prompt

    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        for k, v in substitute_dict.items(): 
            print(k, v)

        prompt = self._information_prompt(substitute_dict, "is_relevant")

        return prompt