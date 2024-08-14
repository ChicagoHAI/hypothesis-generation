from abc import ABC, abstractmethod
import math
import os

from .utils import extract_hypotheses
from ..summary_information import SummaryInformation
from ..inference import Inference
from ...tasks import BaseTask
from ...prompt import BasePrompt


class Generation(ABC):
    """Generation class. Implement the initialize function"""

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        """Initialize the update class

        Parameters:
        ____________

        api: The LLM API to call for intialization and batched hypothesis generation
        prompt_class: the class containing specific prompts for the task
        inference_class: The Inference Class to call when checking for accuracy
        ____________

        """
        super().__init__()
        self.api = api
        self.prompt_class = prompt_class
        self.inference_class = inference_class
        self.task = task
        self.train_data = self.inference_class.train_data

    @abstractmethod
    def initialize_hypotheses(
        self, num_init, init_batch_size, init_hypotheses_per_batch, alpha, **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
        ____________

        args: the parsed arguments

        ____________

        Returns:
        ____________

        hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        pass

    @abstractmethod
    def batched_initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        use_cache=1,
        **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
        ____________

        args: the parsed arguments

        ____________

        Returns:
        ____________

        hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        pass

    def batched_batched_hypothesis_generation(
        self,
        example_indices,
        current_sample,
        num_hypotheses_generate,
        alpha,
        responses,
        use_cache=1,
    ):
        idx_hyp_pair = []
        # create Summary Information for each
        new_generated_hypotheses = {}
        extracted_hypotheses_list = []
        for response in responses:
            extracted_hypotheses = extract_hypotheses(response, num_hypotheses_generate)
            extracted_hypotheses_list.append(extracted_hypotheses)
            for hyp in extracted_hypotheses:
                new_generated_hypotheses[hyp] = SummaryInformation(
                    hypothesis=hyp, acc=0, num_visits=0, reward=0, correct_examples=[]
                )
                for index in example_indices:
                    idx_hyp_pair.append((index, {hyp: new_generated_hypotheses[hyp]}))
        preds, labels = self.inference_class.batched_predict(
            self.train_data, idx_hyp_pair, use_cache=use_cache
        )
        preds, labels = preds[::-1], labels[::-1]
        for extracted_hypotheses in extracted_hypotheses_list:
            for hyp in extracted_hypotheses:
                correct = 0
                ex = []
                for index in example_indices:
                    prediction, actual_label = preds.pop(-1), labels.pop(-1)
                    if prediction == actual_label:
                        correct += 1
                        ex.append((index, actual_label))
                acc = correct / len(example_indices)
                new_generated_hypotheses[hyp].set_accuracy(acc)
                new_generated_hypotheses[hyp].set_num_visits(len(example_indices))
                reward = acc + alpha * math.sqrt(
                    math.log(current_sample) / len(example_indices)
                )
                new_generated_hypotheses[hyp].set_reward(reward)
                new_generated_hypotheses[hyp].set_example(ex)

        return new_generated_hypotheses

    def batched_hypothesis_generation(
        self,
        example_indices,
        current_sample,
        num_hypotheses_generate,
        alpha,
        use_cache=1,
    ):
        """Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

        Parameters:
        ____________

        args: the parsed arguments
        example_indices: the indices of examples being used to generate hypotheses
        current_sample: the current sample in data which the algorithm is on
        num_hypotheses_generation: the number of hypotheses to generate
        ____________

        Returns:

        ____________

        new_generated_hypotheses: A dictionary containing all newly generated hypotheses. The keys are the hypotheses and the values are the Summary Information class

        """

        # gather the examples to use for generation
        # TODO: need copy()?
        example_bank = (
            self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
        )

        # Prompt LLM to generate hypotheses
        prompt_input = self.prompt_class.batched_generation(
            example_bank, num_hypotheses_generate
        )
        response = self.api.generate(prompt_input, use_cache=use_cache)
        extracted_hypotheses = extract_hypotheses(response, num_hypotheses_generate)

        # create Summary Information for each
        new_generated_hypotheses = {}

        idx_hyp_pair = []
        for hyp in extracted_hypotheses:
            new_generated_hypotheses[hyp] = SummaryInformation(
                hypothesis=hyp, acc=0, num_visits=0, reward=0, correct_examples=[]
            )
            for index in example_indices:
                idx_hyp_pair.append((index, {hyp: new_generated_hypotheses[hyp]}))

        preds, labels = self.inference_class.batched_predict(
            self.train_data, idx_hyp_pair, use_cache=use_cache
        )
        preds, labels = preds[::-1], labels[::-1]

        for hyp in extracted_hypotheses:
            correct = 0
            ex = []
            for index in example_indices:
                prediction, actual_label = preds.pop(-1), labels.pop(-1)
                if prediction == actual_label:
                    correct += 1
                    ex.append((index, actual_label))
            acc = correct / len(example_indices)
            new_generated_hypotheses[hyp].set_accuracy(acc)
            new_generated_hypotheses[hyp].set_num_visits(len(example_indices))
            reward = acc + alpha * math.sqrt(
                math.log(current_sample) / len(example_indices)
            )
            new_generated_hypotheses[hyp].set_reward(reward)
            new_generated_hypotheses[hyp].set_example(ex)

        return new_generated_hypotheses
