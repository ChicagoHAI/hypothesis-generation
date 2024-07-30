from abc import ABC, abstractmethod
import os
import json
import math
from typing import Dict

from .generation import Generation
from .inference import Inference
from .replace import Replace
from .summary_information import SummaryInformation
from ..utils import get_num_examples


class Update(ABC):
    """Update class. To use it implement the update function"""

    def __init__(
        self,
        generation_class: Generation,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=-1,
        use_system_prompt=True,
        alpha=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        """
        Initialize the update class

        :param generation_class: The generation class that needs to be called in update for generating new hypotheses
        :param inference_class: The inference class that is called for inference in update for making predictions
        :param replace_class: The replace class that is called for replacing the old hypotheses with the new hypotheses
        :param save_path: Path to save the hypotheses.
        :param sample_num_to_restart_from: Sample number to resume from. Default is -1
        :param num_init: Number of examples to use for initializing hypotheses. Default is 25
        :param epoch_to_start_from: Epoch number to start from. When restarting, this should be > 1. Default is 0
        :param num_wrong_scale: Scale for dynamic num_wrong_to_add_bank. Default is 0.8
        :param k: The number of hypotheses checked per sample during training. Default is -1
        :param use_system_prompt: Use instruction as system prompt. Default is True
        :param alpha: Exploration parameter. Default is 5e-1
        :param update_batch_size: Number of examples to use per prompt. Default is 5
        :param num_hypotheses_to_update: Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses. Default is 5
        :param update_hypotheses_per_batch: Number of hypotheses to generate per prompt. Default is 5
        :param only_best_hypothesis: If only the best hypothesis should be added in the newly generated hypotheses of the batch. Default is False
        :param save_every_n_examples: Save hypotheses every n examples. Default is 100
        """
        self.generation_class = generation_class
        self.inference_class = inference_class
        self.replace_class = replace_class
        self.save_path = save_path
        self.train_data = self.inference_class.train_data
        self.sample_num_to_restart_from = sample_num_to_restart_from
        self.num_init = num_init
        self.epoch_to_start_from = epoch_to_start_from
        self.num_wrong_scale = num_wrong_scale
        self.k = k
        self.use_system_prompt = use_system_prompt
        self.alpha = alpha
        self.update_batch_size = update_batch_size
        self.num_hypotheses_to_update = num_hypotheses_to_update
        self.update_hypotheses_per_batch = update_hypotheses_per_batch
        self.only_best_hypothesis = only_best_hypothesis
        self.save_every_n_examples = save_every_n_examples

    @abstractmethod
    def update(self, hypotheses_bank):
        """Implements how the algorithm runs through the samples. To run through the updated samples, start from args.num_init
        Call self.train_data for the train_data

        :param args: the parsed arguments
        :param hypotheses_bank: a dictionary of hypotheses that is generated with the initial training data

        :returns final_hypotheses_bank: a dictionary of the final hypotheses as keys and the values being corresponding SummaryInformation of the hypotheses

        """
        pass

    def save_to_json(self, file_name, hypotheses_bank: Dict[str, SummaryInformation]):
        """
        Saves hypotheses bank to a json file

        :param hypotheses_bank: the hypotheses which are to be written
        :param file_name: the name of the file to save the hypotheses

        """
        temp_dict = {}
        for hypothesis in hypotheses_bank.keys():
            serialized_dict = hypotheses_bank[hypothesis].__dict__
            temp_dict[hypothesis] = serialized_dict

        json_string = json.dumps(temp_dict)
        with open(os.path.join(self.save_path, file_name), "w") as f:
            f.write(json_string)

    def initialize_hypotheses(
        self, num_init=25, init_batch_size=5, init_hypotheses_per_batch=5
    ) -> Dict[str, SummaryInformation]:
        """
        Generates the initial hypotheses

        :param num_init: Number of examples to use for initializing hypotheses. Default is 25
        :param init_batch_size: Batch size to generate hypotheses. Default is 5
        :param init_hypotheses_per_batch: Number of hypotheses to generate per batch. Default is 5

        :returns hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        return self.generation_class.initialize_hypotheses(
            num_init,
            init_batch_size,
            init_hypotheses_per_batch,
            self.alpha,
            self.use_system_prompt,
        )


class DefaultUpdate(Update):
    """
    DefaultUpdate uses ONE hypothesis to make a prediction on a new example.
    """

    def __init__(
        self,
        generation_class: Generation,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=-1,
        use_system_prompt=True,
        alpha=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        super().__init__(
            generation_class,
            inference_class,
            replace_class,
            save_path,
            sample_num_to_restart_from,
            num_init,
            epoch_to_start_from,
            num_wrong_scale,
            k,
            use_system_prompt,
            alpha,
            update_batch_size,
            num_hypotheses_to_update,
            update_hypotheses_per_batch,
            only_best_hypothesis,
            save_every_n_examples,
        )

    def update(
        self,
        hypotheses_bank,
        current_epoch,
        current_seed,
    ):
        # initialize variables
        num_train_examples = get_num_examples(self.train_data)
        wrong_example_ids = set()

        # go through training examples
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
        if self.sample_num_to_restart_from >= 0:
            start_sample = self.sample_num_to_restart_from
        else:
            start_sample = self.num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
        if current_epoch > self.epoch_to_start_from:
            start_sample = 0
        for i in range(start_sample, num_train_examples):
            if self.num_wrong_scale > 0:
                num_wrong_to_add_bank = (
                    self.k * i / num_train_examples * self.num_wrong_scale
                )

            current_example = i + 1
            print(f"Training on example {i}")

            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True
            )[: self.k]

            # check if the hypothesis works for the generated hypotheses
            num_wrong_hypotheses = 0
            for hypothesis in top_k_hypotheses:
                pred, label = self.inference_class.predict(
                    self.train_data,
                    i,
                    {hypothesis: hypotheses_bank[hypothesis]},
                    self.use_system_prompt,
                )
                if pred != label:
                    num_wrong_hypotheses += 1
                    hypotheses_bank[hypothesis].update_info_if_not_useful(
                        current_example, self.alpha
                    )
                else:
                    hypotheses_bank[hypothesis].update_info_if_useful(
                        current_example, self.alpha
                    )
                    hypotheses_bank[hypothesis].update_useful_examples(i, label)

            # if we get enough wrong examples
            if (
                num_wrong_hypotheses >= num_wrong_to_add_bank
                or len(top_k_hypotheses) == 0
            ):
                wrong_example_ids.add(i)
                if (
                    len(wrong_example_ids)
                    == self.update_batch_size * self.num_hypotheses_to_update
                ):
                    new_hyp_bank = {}

                    # generate new hypotheses
                    for j in range(self.num_hypotheses_to_update):
                        new_hypotheses = (
                            self.generation_class.batched_hypothesis_generation(
                                wrong_example_ids,
                                current_example,
                                self.update_hypotheses_per_batch,
                            )
                        )
                        if self.only_best_hypothesis:
                            best_hypothesis = max(
                                new_hypotheses, key=lambda x: new_hypotheses[x].reward
                            )
                            new_hyp_bank.update(
                                {best_hypothesis: new_hypotheses[best_hypothesis]}
                            )
                        else:
                            new_hyp_bank.update(new_hypotheses)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class
                    hypotheses_bank = self.replace_class.replace(
                        hypotheses_bank, new_hyp_bank
                    )

            # save hypotheses to json
            if (i + 1) % self.save_every_n_examples == 0:
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)
            if ((i + 1) == 25) and (current_epoch == 0):
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)

        return hypotheses_bank


class SamplingUpdate(Update):
    def __init__(
        self,
        generation_class: Generation,
        inference_class: Inference,
        replace_class: Replace,
        save_path: str,
        sample_num_to_restart_from=-1,
        num_init=25,
        epoch_to_start_from=0,
        num_wrong_scale=0.8,
        k=-1,
        use_system_prompt=True,
        alpha=5e-1,
        update_batch_size=5,
        num_hypotheses_to_update=5,
        update_hypotheses_per_batch=5,
        only_best_hypothesis=False,
        save_every_n_examples=100,
    ):
        super().__init__(
            generation_class,
            inference_class,
            replace_class,
            save_path,
            sample_num_to_restart_from,
            num_init,
            epoch_to_start_from,
            num_wrong_scale,
            k,
            use_system_prompt,
            alpha,
            update_batch_size,
            num_hypotheses_to_update,
            update_hypotheses_per_batch,
            only_best_hypothesis,
            save_every_n_examples,
        )

    def update(
        self,
        hypotheses_bank: Dict[str, SummaryInformation],
        current_epoch,
        current_seed,
    ):
        num_train_examples = get_num_examples(self.train_data)
        wrong_example_ids = set()

        # go through training examples
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
        if self.sample_num_to_restart_from >= 0:
            start_sample = self.sample_num_to_restart_from
        else:
            start_sample = self.num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
        if current_epoch > self.epoch_to_start_from:
            start_sample = 0
        for i in range(start_sample, num_train_examples):
            if self.num_wrong_scale > 0:
                num_wrong_to_add_bank = (
                    self.k * i / num_train_examples * self.num_wrong_scale
                )

            current_example = i + 1
            print(f"Training on example {i}")

            top_k_hypotheses = sorted(
                hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True
            )[: self.k]

            # check if the hypothesis works for the generated hypotheses
            num_wrong_hypotheses = 0
            for hypothesis in top_k_hypotheses:
                pred, label = self.inference_class.predict(
                    self.train_data,
                    i,
                    {hypothesis: hypotheses_bank[hypothesis]},
                    self.use_system_prompt,
                )
                if pred != label:
                    num_wrong_hypotheses += 1
                    hypotheses_bank[hypothesis].update_info_if_not_useful(
                        current_example, self.alpha
                    )
                else:
                    hypotheses_bank[hypothesis].update_info_if_useful(
                        current_example, self.alpha
                    )
                    hypotheses_bank[hypothesis].update_useful_examples(i, label)

            # if we get enough wrong examples
            if (
                num_wrong_hypotheses >= num_wrong_to_add_bank
                or len(top_k_hypotheses) == 0
            ):
                wrong_example_ids.add(i)
                if (
                    len(wrong_example_ids)
                    == self.update_batch_size * self.num_hypotheses_to_update
                ):
                    new_hyp_bank = {}

                    # generate new hypotheses
                    for j in range(self.num_hypotheses_to_update):
                        new_hypotheses = (
                            self.generation_class.batched_hypothesis_generation(
                                wrong_example_ids,
                                current_example,
                                self.update_hypotheses_per_batch,
                                self.alpha,
                                self.use_system_prompt,
                            )
                        )
                        max_visited = max(
                            hypotheses_bank, key=lambda x: hypotheses_bank[x].num_visits
                        )
                        new_hypotheses = self.balance_by_sample(
                            new_hypotheses,
                            current_example,
                            hypotheses_bank[max_visited].num_visits,
                            self.num_init,
                            self.alpha,
                            self.use_system_prompt,
                        )
                        if self.only_best_hypothesis:
                            best_hypothesis = max(
                                new_hypotheses, key=lambda x: new_hypotheses[x].reward
                            )
                            new_hyp_bank.update(
                                {best_hypothesis: new_hypotheses[best_hypothesis]}
                            )
                        else:
                            new_hyp_bank = new_hypotheses
                            print("Here is the new hypothesis bank:")
                            for hyp in new_hyp_bank:
                                print(hyp)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class
                    hypotheses_bank = self.replace_class.replace(
                        hypotheses_bank, new_hyp_bank
                    )

            # save hypotheses to json
            if (i + 1) % self.save_every_n_examples == 0:
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)
            if ((i + 1) == 25) and (current_epoch == 0):
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)

        return hypotheses_bank

    def balance_by_sample(
        self,
        hypotheses_bank,
        current_sample,
        max_visits,
        num_init,
        alpha,
        use_system_prompt,
    ):
        if max_visits > 60:
            val = num_init
        elif max_visits > 30:
            val = 10
        else:
            val = 5
        for hypothesis in hypotheses_bank:
            num_right = 0
            ex = set(hypotheses_bank[hypothesis].correct_examples)
            for i in range(val):
                pred, label = self.inference_class.predict(
                    self.train_data,
                    i,
                    {hypothesis: hypotheses_bank[hypothesis]},
                    use_system_prompt,
                )
                if pred == label:
                    num_right += 1
                    ex.add((i, label))
            num_visits = hypotheses_bank[hypothesis].num_visits + val
            acc = (
                hypotheses_bank[hypothesis].acc * hypotheses_bank[hypothesis].num_visits
                + num_right
            ) / (num_visits)
            reward = acc + alpha * math.sqrt(math.log(current_sample) / num_visits)

            hypotheses_bank[hypothesis].set_example(list(ex))
            hypotheses_bank[hypothesis].set_reward(reward)
            hypotheses_bank[hypothesis].set_accuracy(acc)
            hypotheses_bank[hypothesis].set_num_visits(num_visits)

        return hypotheses_bank


UPDATE_DICT = {
    "default": DefaultUpdate,
    "sampling": SamplingUpdate,
}
