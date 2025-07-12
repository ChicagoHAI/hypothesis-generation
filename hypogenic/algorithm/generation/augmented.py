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
        reference_hypotheses=None,
        **generate_kwargs
    ) -> List[str]:
        """Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            reference_hypotheses: A dictionary that accumulates the set of wrong hypotheses for each sample

        Returns:
            hypotheses_list: A list containing all newly generated hypotheses.
        """
        # ----------------------------------------------------------------------
        # Prompt LLM to generate hypotheses
        # ----------------------------------------------------------------------
        # Batch generate a bunch of prompts based on yaml file
        prompt_input = self.prompt_class.batched_error_augmented_generation(
            self.train_data, num_hypotheses_generate, reference_hypotheses
        )

        # Batch generate responses based on the prompts that we just generated
        response = self.api.generate(
            prompt_input, cache_seed=cache_seed, **generate_kwargs
        )

        return extract_hypotheses(response, num_hypotheses_generate)
