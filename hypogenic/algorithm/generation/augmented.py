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
        reference_hypotheses=None,# {hypo: {"correct": set(), "wrong": set()}}
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
            reference_hypotheses: A dictionary of reference hypotheses with their associated correct and wrong sets

        Returns:
            hypotheses_bank: A dictionary with keys as hypotheses and the values as the Summary Information class
        """
        new_hypotheses = self.batched_hyp_list_generation(
            example_ids,
            num_hypotheses_generate,
            cache_seed=cache_seed,
            reference_hypotheses=reference_hypotheses,
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
        reference_hypotheses=None,# {hypo: {"correct": set(), "wrong": set()}}
        **generate_kwargs
    ) -> List[str]:
        batch_size = 1
        all_new_hypos = []
        hypo_items = list(reference_hypotheses.items())
        total = len(hypo_items)
        for i in range(0, total, batch_size):
            batch = dict(hypo_items[i:i+batch_size])
            prompt_input = self.prompt_class.batched_error_augmented_generation(
                self.train_data, len(batch), batch
            )
            response = self.api.generate(
                prompt_input, cache_seed=cache_seed, **generate_kwargs
            )
            all_new_hypos.extend(extract_hypotheses(response, 1))
        return all_new_hypos

    def remove_redundancy(
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