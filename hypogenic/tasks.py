from abc import ABC, abstractmethod
import yaml
import json
import os
import random
import re
from typing import Callable, Tuple, Union
import pandas as pd
from .register import Register


class BaseTask(ABC):
    def __init__(
        self,
        config_path: str,
        extract_label: Union[Callable[[str], str], None] = None,
        from_register: Union[Register, None] = None,
    ):
        if from_register is None and extract_label is None:
            raise ValueError("Either from_register or extract_label should be provided")

        self.config_path = config_path
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.task_name = data["task_name"]

        self.train_data_path = data["train_data_path"]
        self.test_data_path = data["test_data_path"]
        self.val_data_path = data["val_data_path"]

        if "ood_test_data_path" in data:
            self.ood_test_data_path = data["ood_test_data_path"]

        self.prompt_template = {
            k: (
                [{"role": kk, "content": vv} for kk, vv in v.items()]
                if isinstance(v, dict)
                else v
            )
            for k, v in data["prompt_templates"].items()
        }

        self.extract_label = (
            extract_label if extract_label else from_register.build(self.task_name)
        )

    def get_data(self, num_train, num_test, num_val, seed=49) -> Tuple[pd.DataFrame]:
        def read_data(file_path, num, is_train=False):
            # Read from json
            file_path = os.path.join(os.path.dirname(self.config_path), file_path)
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
            return pd.DataFrame.from_dict(processed_data)

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data
