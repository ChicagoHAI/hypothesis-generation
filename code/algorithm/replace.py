import os
code_repo_path = os.environ.get("CODE_REPO_PATH")
from summary_information import SummaryInformation


class Replace():
    def __init__(self):
        pass

    def default_replace(self, args, hypotheses_bank, new_generated_hypotheses):
        """
        Add the new hypotheses to the hypotheses bank if they are not already present.
        Remove the lowest reward hypotheses from the merged bank if the number of hypotheses 
        exceeds the maximum number of hypotheses.

        Parameters:
        ____________
        args: the parsed arguments
        hypotheses_bank: the original dictionary of hypotheses
        new_generated_hypotheses: the newly generated dictionary of hypotheses

        ____________

        Returns:
        ____________

        updated_hyp_bank: the updated hypothesis bank

        """
        merged_hyp_bank = new_generated_hypotheses.copy()
        merged_hyp_bank.update(hypotheses_bank)
        sorted_hyp_bank = dict(sorted(merged_hyp_bank.items(), key=lambda item: item[1].reward, reverse=True))
        updated_hyp_bank = dict(list(sorted_hyp_bank.items())[:args.max_num_hypotheses])
        return updated_hyp_bank

    def replace(self, args, hypotheses_bank, new_generated_hypotheses):
        """ Performs the replace step of the algorithm. This should only replace the hypotheses bank.

        Parameters:
        ____________

        args: the parsed arguments
        hypotheses_bank: the original dictionary of hypotheses
        new_generated_hypotheses: the newly generated dictionary of hypotheses

        ____________

        Returns:
        ____________

        updated_hyp_bank: the updated hypothesis bank

        """
        # match args.replace_style:
        #     # to add new cases add "case 'fill_this_in':" above the 'case _' 
        #     # then write the code under it
        #     case _:
        #         # default is to use the old replace method
        #         updated_hyp_bank = self.default_replace(args, hypotheses_bank, new_generated_hypotheses)
        if args.replace_style == 'default':
            updated_hyp_bank = self.default_replace(args, hypotheses_bank, new_generated_hypotheses)

        return updated_hyp_bank
    
REPLACE_CHOICES = ['default']