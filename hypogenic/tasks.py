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
    """
    In this class, we set up and define the task along with prepping training data.

    All our information if from a yaml file that the user must set up.
    """

    def __init__(
        self,
        config_path: str,
        extract_label: Union[Callable[[str], str], None] = None,
        from_register: Union[Register, None] = None,
    ):
        if from_register is None and extract_label is None:
            raise ValueError("Either from_register or extract_label should be provided")

        # ----------------------------------------------------------------------
        # get information from the yaml file
        # ----------------------------------------------------------------------
        self.config_path = config_path
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.task_name = data["task_name"]

        # data paths
        self.train_data_path = data["train_data_path"]
        self.test_data_path = data["test_data_path"]
        self.val_data_path = data["val_data_path"]

        if "ood_test_data_path" in data:
            self.ood_test_data_path = data["ood_test_data_path"]

        # getting omrpt templates from yaml file
        self.prompt_template = data["prompt_templates"]

        # task label
        self.extract_label = (
            extract_label
            if extract_label is not None
            else from_register.build(self.task_name)
        )

    def get_data(
        self, num_train, num_test, num_val, seed=49
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loading the data from the paths we collected in the yaml file
        """

        # ----------------------------------------------------------------------
        # define our function to read data
        # ----------------------------------------------------------------------
        def read_data(file_path, num, is_train=False):
            # Read from json
            file_path = os.path.join(os.path.dirname(self.config_path), file_path)
            with open(file_path, "r") as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(seed)

            if num is None:
                num_samples = len(data["label"])
            else:
                num_samples = min(num, len(data["label"]))

            sampled_data = zip(
                *random.sample(list(zip(*list(data.values()))), num_samples)
            )
            processed_data = {
                key: value for key, value in zip(data.keys(), sampled_data)
            }
            return pd.DataFrame.from_dict(processed_data)

        # ----------------------------------------------------------------------
        # use that function ot create our test sets
        # ----------------------------------------------------------------------

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data
