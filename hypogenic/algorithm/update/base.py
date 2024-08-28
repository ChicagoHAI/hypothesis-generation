from abc import ABC, abstractmethod
import os
import json
import math
from typing import Dict
from string import Template

import pandas as pd

from ..generation import Generation
from ..inference import Inference
from ..replace import Replace
from ..summary_information import SummaryInformation


class Update(ABC):
    """Update class. To use it implement the update function"""

    def __init__(
        self,
        generation_class: Generation,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        file_name_template: str = "hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=1,
        alpha=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        """
        Initialize the update class

        Parameters:
            generation_class: The generation class that needs to be called in update for generating new hypotheses
            inference_class: The inference class that is called for inference in update for making predictions
            replace_class: The replace class that is called for replacing the old hypotheses with the new hypotheses
            save_path: Path to save the hypotheses.
            file_name_template: Template for the file name. Default is "hypotheses_training_sample\_${sample}\_seed\_${seed}\_epoch\_${epoch}.json"
            sample_num_to_restart_from: Sample number to resume from. Default is -1
            num_init: Number of examples to use for initializing hypotheses. Default is 25
            epoch_to_start_from: Epoch number to start from. When restarting, this should be > 1. Default is 0
            num_wrong_scale: Scale for dynamic num_wrong_to_add_bank. Default is 0.8
            k: The number of hypotheses checked per sample during training. Default is -1
            alpha: Exploration parameter. Default is 5e-1
            update_batch_size: Number of examples to use per prompt. Default is 5
            num_hypotheses_to_update: Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses. Default is 5
            update_hypotheses_per_batch: Number of hypotheses to generate per prompt. Default is 5
            only_best_hypothesis: If only the best hypothesis should be added in the newly generated hypotheses of the batch. Default is False
            save_every_n_examples: Save hypotheses every n examples. Default is 100
        """
        self.generation_class = generation_class
        self.inference_class = inference_class
        self.replace_class = replace_class
        self.save_path = save_path
        self.file_name_template = file_name_template
        self.train_data: pd.Dataframe = self.inference_class.train_data
        self.sample_num_to_restart_from = sample_num_to_restart_from
        self.num_init = num_init
        self.epoch_to_start_from = epoch_to_start_from
        self.num_wrong_scale = num_wrong_scale
        self.k = k
        self.alpha = alpha
        self.update_batch_size = update_batch_size
        self.num_hypotheses_to_update = num_hypotheses_to_update
        self.update_hypotheses_per_batch = update_hypotheses_per_batch
        self.only_best_hypothesis = only_best_hypothesis
        self.save_every_n_examples = save_every_n_examples

    @abstractmethod
    def update(
        self,
        hypotheses_bank: Dict[str, SummaryInformation],
        current_epoch,
        current_seed,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """Implements how the algorithm runs through the samples. To run through the updated samples, start from args.num_init
        Call self.train_data for the train_data

        Parameters:
            args: the parsed arguments
            hypotheses_bank: a dictionary of hypotheses that is generated with the initial training data
            current_epoch: the current epoch number
            current_seed: the current seed number

        Returns
            final_hypotheses_bank: a dictionary of the final hypotheses as keys and the values being corresponding SummaryInformation of the hypotheses
        """
        pass

    def save_to_json(
        self,
        hypotheses_bank: Dict[str, SummaryInformation],
        file_name_template=None,
        **kwargs,
    ):
        """
        Saves hypotheses bank to a json file

        Prameters:
            hypotheses_bank: the hypotheses which are to be written
            file_name: the name of the file to save the hypotheses
            file_name_template: how to store the file if not specified
            kwargs: the arguments to be used in the file name template, for the default template we expect 'sample', 'seed', 'epoch'
        """
        # Use the default template if not specified
        if file_name_template is None:
            file_name_template = self.file_name_template

        # Write what we want to store
        temp_dict = {}
        for hypothesis in hypotheses_bank.keys():
            serialized_dict = hypotheses_bank[hypothesis].__dict__
            temp_dict[hypothesis] = serialized_dict

        json_string = json.dumps(temp_dict)

        # we expect 'sample', 'seed', 'epoch'
        kwargs = {k: str(v) for k, v in kwargs.items()}
        with open(
            os.path.join(
                self.save_path,
                Template(file_name_template).substitute(kwargs),
            ),
            "w",
        ) as f:
            f.write(json_string)

    def batched_initialize_hypotheses(
        self,
        num_init=25,
        init_batch_size=5,
        init_hypotheses_per_batch=5,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ) -> Dict[str, SummaryInformation]:
        """
        Generates the initial hypotheses

        Parameters:
            num_init: Number of examples to use for initializing hypotheses. Default is 25
            init_batch_size: Batch size to generate hypotheses. Default is 5
            init_hypotheses_per_batch: Number of hypotheses to generate per batch. Default is 5

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        hypotheses_list = self.generation_class.batched_initialize_hypotheses(
            num_init,
            init_batch_size,
            init_hypotheses_per_batch,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        return self.generation_class.make_hypotheses_bank(
            example_indices=list(range(num_init)),
            current_sample=num_init,
            alpha=self.alpha,
            hypotheses_list=hypotheses_list,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
