from typing import List

from . import generation_register, DefaultGeneration
from .utils import extract_hypotheses


@generation_register.register("Augmented")
class AugmentedGeneration(DefaultGeneration):
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
