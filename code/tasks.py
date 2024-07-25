from abc import ABC, abstractmethod
import yaml
import json
import os
import random
import re

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")


class Task(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # @abstractmethod
    # def extract_label(self, text):
    #     pass

    @abstractmethod
    def get_data(self, num_train, num_test, num_val):
        pass


class BaseTask(Task):
    def __init__(self, task_name):
        self.task = task_name

        with open(f'{code_repo_path}/data/{task_name}/config.yaml', 'r') as f:
            data = yaml.safe_load(f)

        self.label_classes = data['label_classes']
        self.train_data_path = os.path.join(code_repo_path, data['train_data_path'])
        self.test_data_path = os.path.join(code_repo_path, data['test_data_path'])
        self.val_data_path = os.path.join(code_repo_path, data['val_data_path'])

        if "ood_test_data_path" in data:
            self.ood_test_data_path = os.path.join(code_repo_path, data['ood_test_data_path'])

        extract_label = {}
        exec(data["parser"], extract_label)
        self.extract_label = extract_label['extract_label']

    def get_data(self, num_train, num_test, num_val):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, 'r') as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(49)

            num_samples = min(num, len(data['label']))
            sampled_data = zip(*random.sample(list(zip(*list(data.values()))), num_samples))
            processed_data = {key: value for key, value in zip(data.keys(), sampled_data)}
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data
