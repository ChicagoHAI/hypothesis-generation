from abc import ABC, abstractmethod
import yaml
import json
import os
import random
import re
from typing import Callable

# TODO: Generate one single task object for every use


class BaseTask(ABC):
    def __init__(
        self,
        extract_label: Callable[[str], str],
        config_path: str,
    ):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.task_name = data["task_name"]

        self.label_classes = data["label_classes"]
        self.train_data_path = data["train_data_path"]
        self.test_data_path = data["test_data_path"]
        self.val_data_path = data["val_data_path"]

        if "ood_test_data_path" in data:
            self.ood_test_data_path = data["ood_test_data_path"]

        self.prompt_template = data["prompt_templates"]

        self.extract_label = extract_label

    def get_data(self, num_train, num_test, num_val, seed=49):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, "r") as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(seed)

            num_samples = min(num, len(data["label"]))
            sampled_data = zip(
                *random.sample(list(zip(*list(data.values()))), num_samples)
            )
            processed_data = {
                key: value for key, value in zip(data.keys(), sampled_data)
            }
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data
