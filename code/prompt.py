from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from tasks import BaseTask
from typing import Union, Dict

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

model_postfix = {
    'Mistral-7B': '_mixtral',
    'Mixtral-8x7B': '_mixtral',
    'Others': ''
}

MISTRAL_MODELS = ['Mistral-7B', 'Mixtral-8x7B']


def read_prompt(instruction_path, user_prompt_path):
    with open(instruction_path, 'r') as f:
        instruction_prompt = f.read()  # a string of the entire file

    with open(user_prompt_path, 'r') as f:
        user_prompt = f.read()  # a string of the entire file

    return instruction_prompt, user_prompt


class BasePrompt(ABC):
    def __init__(self, task: Union[BaseTask, None]):
        self.task = task

    def _get_substitute_dict(self, data_dict, example_idx, no_label_info=False) -> Dict[str, str]:
        # TODO: get specific entry from prompt_template
        example = {k: v[example_idx] for k, v in data_dict.items()}
        substitute_dict = {}
        for key, value in self.task.prompt_template.items():
            # TODO: safe_substitute or substitute?
            substitute_dict[key] = Template(value).substitute(example)
        substitute_dict.update(example)
        return substitute_dict

    # @abstractmethod
    def _information_prompt(self, data_dict, example_idx, no_label_info=False) -> Dict[str, str]:
        return self._get_substitute_dict(data_dict, example_idx)

    def few_shot_baseline(self, train_data, num_few_shot, test_data, test_idx):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/few_shot_baseline/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/few_shot_baseline/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)
        substitute_dict = self._information_prompt(test_data, test_idx, no_label_info=True)

        observations = ""
        few_shot_prefix = ""
        if num_few_shot > 0:
            few_shot_prefix = substitute_dict['few_shot_prefix']
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j)['observations']

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

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/batched_generation/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/batched_generation/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        observations = ""
        for example_idx in range(len(train_data['label'])):
            observations += self._information_prompt(train_data, example_idx)['observations']

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

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/inference/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/inference/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        substitute_dict = self._information_prompt(test_data, test_idx, no_label_info=True)
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
                knn_info_prompt += self._information_prompt(train_data, example_info[0])['knn_info_prompt']

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/knn/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/knn/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        substitute_dict = self._information_prompt(test_data, test_idx, no_label_info=True)
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
                knn_info_prompt += self._information_prompt(train_data, example_info[0])['knn_info_prompt']

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/knn_selection/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/knn_selection/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        substitute_dict = self._information_prompt(test_data, test_idx, no_label_info=True)
        substitute_dict['knn_info_prompt'] = knn_info_prompt

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)

    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/{self.task.task_name}/is_relevant/instructions.txt"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task.task_name}/is_relevant/user.txt"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        substitute_dict = self._information_prompt(test_data, test_idx, no_label_info=True)
        substitute_dict['hypothesis'] = hypothesis

        instruction_prompt = Template(instruction_prompt).substitute(substitute_dict)
        user_prompt = Template(user_prompt).substitute(substitute_dict)

        return (instruction_prompt, user_prompt)
