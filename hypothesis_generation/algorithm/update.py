from abc import ABC, abstractmethod
import os
import json
import math

from .generation import Generation
from .inference import Inference
from .replace import Replace
from .summary_information import SummaryInformation
from ..utils import get_num_examples


class Update(ABC):
    """Update class. To use it implement the update function"""
    def __init__(self, generation_class: Generation, inference_class: Inference, replace_class: Replace):
        """ Initialize the update class

        Parameters:
        ____________
        generation_class: The generation class that needs to be called in update for generating new hypotheses
        inference_class: The inference class that is called for inference in update for making predictions
        ____________

        """
        super().__init__()
        self.generation_class = generation_class
        self.inference_class = inference_class
        self.replace_class = replace_class
        self.train_data = self.inference_class.train_data

    @abstractmethod
    def update(self, hypotheses_bank):
        """ Implements how the algorithm runs through the samples. To run through the updated samples, start from args.num_init
        Call self.train_data for the train_data

        Parameters:
        ____________

        args: the parsed arguments
        hypotheses_bank: a dictionary of hypotheses that is generated with the initial training data

        ____________

        Returns:
        ____________

        final_hypotheses_bank: a dictionary of the final hypotheses as keys and the values being corresponding SummaryInformation of the hypotheses

        """
        pass
    
    def save_to_json(self, index, hypotheses, output_folder, current_epoch):
        """ Saves hypotheses bank to a json file

        Parameters:
        ____________

        args: the parsed arguments
        index: the index of the training sample
        hypotheses_bank: the hypotheses which are to be written
        _____________

        """
        temp_dict = {}
        for hypothesis in hypotheses.keys():
            serialized_dict = hypotheses[hypothesis].__dict__
            temp_dict[hypothesis] = serialized_dict
        
        json_string = json.dumps(temp_dict)
        with open(f'{output_folder}/hypotheses_training_sample_{index}_epoch_{current_epoch}.json', 'w') as f:
            f.write(json_string)


class DefaultUpdate(Update):
    """
    DefaultUpdate uses ONE hypothesis to make a prediction on a new example.
    """
    def __init__(self, generation_class, inference_class, replace_class):
        super().__init__(generation_class, inference_class, replace_class)
        
    def update(
            self, 
            hypotheses_bank,
            sample_num_to_restart_from,
            num_init,
            current_epoch,
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
            current_seed
    ):
        # initialize variables
        num_train_examples = get_num_examples(self.train_data)
        wrong_example_ids = set()

        # go through training examples 
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
        if sample_num_to_restart_from >= 0:
            start_sample = sample_num_to_restart_from
        else:
            start_sample = num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
        if current_epoch > epoch_to_start_from:
            start_sample = 0
        for i in range(start_sample, num_train_examples):
            if num_wrong_scale > 0:
                num_wrong_to_add_bank = k * i / num_train_examples * num_wrong_scale

            current_example = i+1
            print(f"Training on example {i}")

            top_k_hypotheses = sorted(hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True)[:k]
            
            # check if the hypothesis works for the generated hypotheses
            num_wrong_hypotheses = 0
            for hypothesis in top_k_hypotheses:
                pred, label = self.inference_class.predict(self.train_data, i, {hypothesis: hypotheses_bank[hypothesis]}, use_system_prompt)
                if pred != label:
                    num_wrong_hypotheses += 1
                    hypotheses_bank[hypothesis].update_info_if_not_useful(current_example, alpha)
                else:
                    hypotheses_bank[hypothesis].update_info_if_useful(current_example, alpha)
                    hypotheses_bank[hypothesis].update_useful_examples(i, label)

            # if we get enough wrong examples
            if num_wrong_hypotheses >= num_wrong_to_add_bank or len(top_k_hypotheses) == 0:
                wrong_example_ids.add(i)
                if len(wrong_example_ids) == update_batch_size*num_hypotheses_to_update:
                    new_hyp_bank = {}
                    
                    # generate new hypotheses
                    for j in range(num_hypotheses_to_update):
                        new_hypotheses = self.generation_class.batched_hypothesis_generation(wrong_example_ids, current_example, update_hypotheses_per_batch)
                        if only_best_hypothesis:
                            best_hypothesis = max(new_hypotheses, key=lambda x: new_hypotheses[x].reward)
                            new_hyp_bank.update({best_hypothesis: new_hypotheses[best_hypothesis]})
                        else:
                            new_hyp_bank.update(new_hypotheses)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class
                    hypotheses_bank = self.replace_class.replace(hypotheses_bank, new_hyp_bank)

            # save hypotheses to json
            if (i+1) % save_every_n_examples == 0:
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)
            if ((i+1) == 25) and (current_epoch == 0):
                self.save_to_json( f"{i+1}_seed_{current_seed}", hypotheses_bank)

        return hypotheses_bank


class SamplingUpdate(Update):
    def __init__(self, generation_class, inference_class, replace_class):
        super().__init__(generation_class, inference_class, replace_class)
    
    def update(
            self, 
            hypotheses_bank,
            sample_num_to_restart_from,
            num_init,
            current_epoch,
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
            current_seed
    ):
        num_train_examples = get_num_examples(self.train_data)
        wrong_example_ids = set()

        # go through training examples 
        # When restarting from epoch > 0, no need to start at num_init
        # When not restarting, then default sample_num_to_restart_from = -1. start with num_init.
        # For multiple epochs restarts, there should always be a non-negative sample_num_to_restart_from
        if sample_num_to_restart_from >= 0:
            start_sample = sample_num_to_restart_from
        else:
            start_sample = num_init

        # This is to check if we are running more epochs than the starting epoch, if so, start at sample 0
        if current_epoch > epoch_to_start_from:
            start_sample = 0
        for i in range(start_sample, num_train_examples):
            if num_wrong_scale > 0:
                num_wrong_to_add_bank = k * i / num_train_examples * num_wrong_scale
            
            current_example = i+1
            print(f"Training on example {i}")

            top_k_hypotheses = sorted(hypotheses_bank, key=lambda x: hypotheses_bank[x].reward, reverse=True)[:k]
            
            # check if the hypothesis works for the generated hypotheses
            num_wrong_hypotheses = 0
            for hypothesis in top_k_hypotheses:
                pred, label = self.inference_class.predict(self.train_data, i, {hypothesis: hypotheses_bank[hypothesis]}, use_system_prompt)
                if pred != label:
                    num_wrong_hypotheses += 1
                    hypotheses_bank[hypothesis].update_info_if_not_useful(current_example, alpha)
                else:
                    hypotheses_bank[hypothesis].update_info_if_useful(current_example, alpha)
                    hypotheses_bank[hypothesis].update_useful_examples(i, label)

            # if we get enough wrong examples
            if num_wrong_hypotheses >= num_wrong_to_add_bank or len(top_k_hypotheses) == 0:
                wrong_example_ids.add(i)
                if len(wrong_example_ids) == update_batch_size*num_hypotheses_to_update:
                    new_hyp_bank = {}
                    
                    # generate new hypotheses
                    for j in range(num_hypotheses_to_update):
                        new_hypotheses = self.generation_class.batched_hypothesis_generation(wrong_example_ids, current_example, update_hypotheses_per_batch)
                        max_visited = max(hypotheses_bank, key=lambda x: hypotheses_bank[x].num_visits)
                        new_hypotheses = self.balance_by_sample(new_hypotheses, current_example, hypotheses_bank[max_visited].num_visits, num_init, alpha, use_system_prompt)
                        if only_best_hypothesis:
                            best_hypothesis = max(new_hypotheses, key=lambda x: new_hypotheses[x].reward)
                            new_hyp_bank.update({best_hypothesis: new_hypotheses[best_hypothesis]})
                        else:
                            new_hyp_bank = new_hypotheses
                            print("Here is the new hypothesis bank:")
                            for hyp in new_hyp_bank:
                                print(hyp)
                    # reset wrong examples to be empty
                    wrong_example_ids = set()

                    # call replace class
                    hypotheses_bank = self.replace_class.replace(hypotheses_bank, new_hyp_bank)

            # save hypotheses to json
            if (i+1) % save_every_n_examples == 0:
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)
            if ((i+1) == 25) and (current_epoch == 0):
                self.save_to_json(f"{i+1}_seed_{current_seed}", hypotheses_bank)

        return hypotheses_bank

    def balance_by_sample(self, hypotheses_bank, current_sample, max_visits, num_init, alpha, use_system_prompt):
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
                pred, label = self.inference_class.predict(self.train_data, i, {hypothesis: hypotheses_bank[hypothesis]}, use_system_prompt)
                if pred == label:
                    num_right += 1
                    ex.add((i, label))
            num_visits = hypotheses_bank[hypothesis].num_visits + val
            acc = (hypotheses_bank[hypothesis].acc * hypotheses_bank[hypothesis].num_visits + num_right)/(num_visits)
            reward = acc + alpha * math.sqrt(math.log(current_sample) / num_visits)

            hypotheses_bank[hypothesis].set_example(list(ex))
            hypotheses_bank[hypothesis].set_reward(reward)
            hypotheses_bank[hypothesis].set_accuracy(acc)
            hypotheses_bank[hypothesis].set_num_visits(num_visits)

        return hypotheses_bank


UPDATE_DICT = {
    'default': DefaultUpdate,
    'sampling' : SamplingUpdate
}


