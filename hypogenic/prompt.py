from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from typing import List, Tuple, Union, Dict
from copy import deepcopy
import pandas as pd

from .tasks import BaseTask


class BasePrompt(ABC):
    def __init__(self, task: BaseTask):
        self.task = task

    def _get_substitute_dict(
        self, data_dict: pd.DataFrame, example_idx
    ) -> Dict[str, str]:
        example = data_dict.loc[example_idx].to_dict()
        substitute_dict = {}

        for key, value in self.task.prompt_template.items():
            if not isinstance(value, str):
                continue
            # TODO: safe_substitute or substitute?
            substitute_dict[key] = Template(value).substitute(example)

        substitute_dict.update(example)

        return substitute_dict

    def _information_prompt(
        self, data_dict: pd.DataFrame, example_idx, info_key: str
    ) -> Dict[str, str]:
        example = data_dict.loc[example_idx].to_dict()
        return Template(self.task.prompt_template[info_key]).substitute(example)

    def _get_prompt_template(self, key: str) -> List[Dict[str, str]]:
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

        prompt = self._get_prompt_template("few_shot_baseline")
        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        observations = ""
        few_shot_prefix = ""
        if num_few_shot > 0:
            few_shot_prefix = substitute_dict["few_shot_prefix"]
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j, "observations")

        substitute_dict["observations"] = observations
        substitute_dict["few_shot_prefix"] = few_shot_prefix

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

        return prompt

    def batched_generation(self, train_data, num_hypotheses):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        prompt = self._get_prompt_template("batched_generation")

        observations = ""
        for example_idx in range(len(train_data)):
            observations += self._information_prompt(
                train_data, example_idx, "observations"
            )

        substitute_dict = {
            "num_hypotheses": num_hypotheses,
            "observations": observations,
        }

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

        return prompt

    def inference(self, hypotheses_dict, test_data, test_idx):
        """
        Create inference prompt.
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        prompt = self._get_prompt_template("inference")

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

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
                    train_data, example_info[0], "adaptive_info_prompt"
                )

        prompt = self._get_prompt_template("adaptive_inference")

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["adaptive_info_prompt"] = adaptive_info_prompt

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

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
                    train_data, example_info[0], "adaptive_info_prompt"
                )

        prompt = self._get_prompt_template("adaptive_selection")

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["adaptive_info_prompt"] = adaptive_info_prompt

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

        return prompt

    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        prompt = self._get_prompt_template("is_relevant")

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        for idx, p in enumerate(prompt):
            prompt[idx]["content"] = Template(p["content"]).substitute(substitute_dict)

        return prompt
