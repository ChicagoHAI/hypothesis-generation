from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from typing import Tuple, Union, Dict

from .tasks import BaseTask


def read_prompt(instruction_path, user_prompt_path):
    with open(instruction_path, 'r') as f:
        instruction_prompt = f.read()  # a string of the entire file

    with open(user_prompt_path, 'r') as f:
        user_prompt = f.read()  # a string of the entire file

    return instruction_prompt, user_prompt


class BasePrompt(ABC):
    def __init__(self, task: BaseTask):
        self.task = task

    def _get_substitute_dict(self, data_dict, example_idx) -> Dict[str, str]:
        example = {k: v[example_idx] for k, v in data_dict.items()}
        substitute_dict = {}

        for key, value in self.task.prompt_template.items():
            if not isinstance(value, str):
                continue
            # TODO: safe_substitute or substitute?
            substitute_dict[key] = Template(value).substitute(example)

        substitute_dict.update(example)

        return substitute_dict

    def _information_prompt(self, data_dict, example_idx, info_key: str) -> Dict[str, str]:
        example = {k: v[example_idx] for k, v in data_dict.items()}
        return Template(self.task.prompt_template[info_key]).substitute(example)

    def _get_prompt_template(self, key: str) -> Tuple[str, str]:
        instruction_prompt = self.task.prompt_template[key]["instructions"]
        user_prompt = self.task.prompt_template[key]["user"]
        return instruction_prompt, user_prompt

    def few_shot_baseline(self, train_data, num_few_shot, test_data, test_idx):
        """
        Few shot prompt for baseline
        """

        instruction_prompt, user_prompt = self._get_prompt_template('few_shot_baseline')
        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        observations = ""
        few_shot_prefix = ""
        if num_few_shot > 0:
            few_shot_prefix = substitute_dict['few_shot_prefix']
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j, 'observations')

        substitute_dict['observations'] = observations
        substitute_dict['few_shot_prefix'] = few_shot_prefix

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def batched_generation(self,
                           train_data,
                           num_hypotheses):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        instruction_prompt, user_prompt = self._get_prompt_template('batched_generation')

        observations = ""
        for example_idx in range(len(train_data['label'])):
            observations += self._information_prompt(train_data, example_idx, 'observations')

        substitute_dict = {"num_hypotheses": num_hypotheses, "observations": observations}

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def inference(self,
                  hypotheses_dict,
                  test_data,
                  test_idx):
        """
        Create inference prompt.
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        instruction_prompt, user_prompt = self._get_prompt_template('inference')

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict['hypothesis'] = hypothesis

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def knn_inference(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        KNN inference prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            knn_info_prompt += f'Pattern {hyp_idx + 1}: {hypothesis_text}\n'

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx + 1}:\n'
                knn_info_prompt += self._information_prompt(train_data, example_info[0], 'knn_info_prompt')

        instruction_prompt, user_prompt = self._get_prompt_template('knn')

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict['knn_info_prompt'] = knn_info_prompt

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def knn_selection(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        KNN hypothesis selection prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            knn_info_prompt += f'Pattern {hyp_idx + 1}: {hypothesis_text}\n'

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx + 1}:\n'
                knn_info_prompt += self._information_prompt(train_data, example_info[0], 'knn_info_prompt')

        instruction_prompt, user_prompt = self._get_prompt_template('knn_selection')

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict['knn_info_prompt'] = knn_info_prompt

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        instruction_prompt, user_prompt = self._get_prompt_template('is_relevant')

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict['hypothesis'] = hypothesis

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)
