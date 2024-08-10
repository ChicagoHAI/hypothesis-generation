import os
from abc import ABC, abstractmethod
from ..register import Register

replace_register = Register(name="replace")


class Replace(ABC):
    def __init__(self, max_num_hypotheses):
        self.max_num_hypotheses = max_num_hypotheses

    @abstractmethod
    def replace(self, hypotheses_bank, new_generated_hypotheses):
        pass


@replace_register.register("default")
class DefaultReplace(Replace):
    def __init__(self, max_num_hypotheses):
        super().__init__(max_num_hypotheses)

    def replace(self, hypotheses_bank, new_generated_hypotheses):
        """
        Add the new hypotheses to the hypotheses bank if they are not already present.
        Remove the lowest reward hypotheses from the merged bank if the number of hypotheses
        exceeds the maximum number of hypotheses.

        Parameters:
        ____________
        :param hypotheses_bank: the original dictionary of hypotheses
        :param new_generated_hypotheses: the newly generated dictionary of hypotheses

        ____________

        Returns:
        ____________

        updated_hyp_bank: the updated hypothesis bank

        """
        merged_hyp_bank = new_generated_hypotheses.copy()
        merged_hyp_bank.update(hypotheses_bank)
        sorted_hyp_bank = dict(
            sorted(
                merged_hyp_bank.items(), key=lambda item: item[1].reward, reverse=True
            )
        )
        updated_hyp_bank = dict(
            list(sorted_hyp_bank.items())[: self.max_num_hypotheses]
        )
        return updated_hyp_bank


REPLACE_CHOICES = ["default"]
