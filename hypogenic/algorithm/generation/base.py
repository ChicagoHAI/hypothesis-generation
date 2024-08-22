from abc import ABC, abstractmethod
import math
import os
from typing import List

from .utils import extract_hypotheses
from ..summary_information import SummaryInformation
from ..inference import Inference
from ...tasks import BaseTask
from ...prompt import BasePrompt


class Generation(ABC):
    """Generation class"""

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        """Initialize the update class

        Parameters:
            api: The LLM API to call for intialization and batched hypothesis generation
                It could also be a local LLM.
            prompt_class: the class containing specific prompts for the task
            inference_class: The Inference Class to call when checking for accuracy

        """
        super().__init__()
        self.api = api
        self.prompt_class = prompt_class
        self.inference_class = inference_class
        self.task = task
        self.train_data = self.inference_class.train_data

    @abstractmethod
    def batched_initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        alpha,
        cache_seed=None,
        max_concurrent=3,
        **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
            args: the parsed arguments

        Returns:
            hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        pass

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCHED_BATCHED_HYPOTHESIS GENERATION                                    #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #

    def batched_batched_hypothesis_generation(
        self,
        example_indices,
        current_sample,
        num_hypotheses_generate,
        alpha,
        responses: List[str],
        cache_seed=None,
        max_concurrent=3,
    ):
        """Based on multiple batched responses from the LM, create new hypotheses.

        You can thing of "responses" to be a list of such "response" objects:

            prompt_input = self.prompt_class.batched_generation( example_bank, num_hypotheses_generate)
            response = self.api.generate(prompt_input, cache_seed=cache_seed)

        See the beginning of function "batched_hypothesis_generation" below to understand how
        the reponses are created.

        From there, the behavior is similar to "batched_hypothesis_generation",
        expect that we're now dealing with a batch of the "response" variable rather
        than a single one.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            current_sample: the current sample in data which the algorithm is on
            num_hypotheses_generate: the number of hypotheses that we expect our repsonse to generate
            reponses: a batch of llm inferences
            alpha: eploration constant in hypogenic reward funciton
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests to make to the API

        Returns:
            new_generated_hypotheses: A dictionary containing all newly generated hypotheses. The keys are the hypotheses and the values are the Summary Information class
        """
        idx_hyp_pair = []
        new_generated_hypotheses = {}
        extracted_hypotheses_list = []

        # ----------------------------------------------------------------------
        # We now go through each of the responses and get the hypotheses
        # ----------------------------------------------------------------------
        for response in responses:

            # for each reponse, we extract the hypotheses
            extracted_hypotheses = extract_hypotheses(response, num_hypotheses_generate)
            extracted_hypotheses_list.append(extracted_hypotheses)

            # For each of these new hypotheses, we create an SummaryInformation which stores the stats we're interested in
            for hyp in extracted_hypotheses:
                new_generated_hypotheses[hyp] = SummaryInformation(
                    hypothesis=hyp, acc=0, num_visits=0, reward=0, correct_examples=[]
                )

                # And we keep track of the example's relation to the hypothesis
                for index in example_indices:
                    idx_hyp_pair.append((index, {hyp: new_generated_hypotheses[hyp]}))

        # ----------------------------------------------------------------------
        # We try to predict the ground truth labels
        # ----------------------------------------------------------------------
        preds, labels = self.inference_class.batched_predict(
            self.train_data,
            idx_hyp_pair,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
        )
        preds, labels = preds[::-1], labels[::-1]

        # ----------------------------------------------------------------------
        # Finding the accuracy and the correct examples for each hypothesis batch
        # ----------------------------------------------------------------------
        for extracted_hypotheses in extracted_hypotheses_list:
            for hyp in extracted_hypotheses:

                correct = 0
                ex = []

                # Finding accuracy
                for index in example_indices:
                    prediction, actual_label = preds.pop(-1), labels.pop(-1)
                    if prediction == actual_label:
                        correct += 1
                        ex.append((index, actual_label))

                # Record the accuracy, number of visits, reward, and correct examples
                acc = correct / len(example_indices)
                new_generated_hypotheses[hyp].set_accuracy(acc)
                new_generated_hypotheses[hyp].set_num_visits(len(example_indices))

                # hypogenic reward
                reward = acc + alpha * math.sqrt(
                    math.log(current_sample) / len(example_indices)
                )

                new_generated_hypotheses[hyp].set_reward(reward)
                new_generated_hypotheses[hyp].set_example(ex)

        return new_generated_hypotheses

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCHED_HYPOTHESIS GENERATION                                            #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def batched_hypothesis_generation(
        self,
        example_indices,
        current_sample,
        num_hypotheses_generate,
        alpha,
        cache_seed=None,
        max_concurrent=3,
    ):
        """Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            current_sample: the current sample in data which the algorithm is on
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            alpha: eploration constant in hypogenic reward funciton
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests to make to the API

        Returns:
            new_generated_hypotheses: A dictionary containing all newly generated hypotheses. The keys are the hypotheses and the values are the Summary Information class
        """

        # ----------------------------------------------------------------------
        # Gather the examples to use for generation
        # ----------------------------------------------------------------------
        # Gather examples based on example_indices
        # TODO: need copy()?
        example_bank = (
            self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
        )

        # ----------------------------------------------------------------------
        # Prompt LLM to generate hypotheses
        # ----------------------------------------------------------------------
        # Batch generate a bunch of prompts based on yaml file
        prompt_input = self.prompt_class.batched_generation(
            example_bank, num_hypotheses_generate
        )

        # Batch generate responses based on the prompts that we just generated
        response = self.api.generate(prompt_input, cache_seed=cache_seed)

        return self.batched_batched_hypothesis_generation(
            example_indices,
            current_sample,
            num_hypotheses_generate,
            alpha,
            [response],
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
        )
