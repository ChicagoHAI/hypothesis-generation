from typing import List

from . import generation_register, DefaultGeneration
from .utils import extract_hypotheses


@generation_register.register("augmented")
class AugmentedGeneration(DefaultGeneration):

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCHED_HYPOTHESIS GENERATION                                            #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def batched_hypothesis_generation(
        self,
        example_ids,
        current_sample,
        num_hypotheses_generate: int,
        alpha: float,
        cache_seed=None,
        max_concurrent=3,
        reference_info=None,# {hypo: {"correct": set(), "wrong": set()}}
        **generate_kwargs,
    ):
        """
        Generates new hypotheses for the given examples

        Parameters:
            example_ids: The ids of the examples for which hypotheses need to be generated
            current_sample: the current sample in data which the algorithm is on
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            alpha: exploration constant in hypogenic reward function
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: The maximum number of concurrent requests
            reference_info: A dictionary of reference hypotheses with their associated correct and wrong sets

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        new_hypotheses = self.batched_hyp_list_generation(
            example_ids,
            num_hypotheses_generate,
            cache_seed=cache_seed,
            reference_info=reference_info,
            **generate_kwargs,
        )

        return self.make_hypotheses_bank(
            example_ids,
            current_sample,
            alpha,
            new_hypotheses,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCHED HYPOTHESIS LIST GENERATION                                       #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def batched_hyp_list_generation(
        self,
        example_indices: List[int],
        num_hypotheses_generate: int,
        cache_seed=None,
        reference_info=None,# {hypo: {"correct": set(), "wrong": set()}}
        **generate_kwargs
    ) -> List[str]:
        all_new_hypos = []
        reference_items = list(reference_info.items())
        total = len(reference_items)
        
        prompt_inputs = []
        for i in range(0, total):
            prompt_input = self.prompt_class.error_augmented_generation(
                self.train_data, dict(reference_items[i])
            )
            prompt_inputs.append(prompt_input)
        
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed, 
            **generate_kwargs
        )
        
        for response in responses:
            all_new_hypos.extend(extract_hypotheses(response, 1))
        
        return all_new_hypos

    def clear_redundancy_update(
        self,
        example_ids,
        current_sample,
        current_hyp_bank,
        alpha: float,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.remove_redundancy(current_hyp_bank)
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        new_hyp_list = extract_hypotheses(response)
        return self.make_hypotheses_bank(
            example_ids,
            current_sample,
            alpha,
            new_hyp_list,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
    
    def clear_redundancy_final(
        self,
        hyp_bank,
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        prompt_input = self.prompt_class.remove_redundancy(hyp_bank)
        responses = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        new_hyp_list = extract_hypotheses(responses)
        return new_hyp_list