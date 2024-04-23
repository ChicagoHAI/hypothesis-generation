from abc import ABC, abstractmethod
import math
import os
code_repo_path = os.environ.get("CODE_REPO_PATH")
from summary_information import SummaryInformation

def extract_hypotheses(args, text, num_hypotheses):
    import re
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    print("Text provided", text)
    hypotheses = pattern.findall(text)
    if len(hypotheses) == 0:
        print("No hypotheses are generated.")
        return []

    for i in range(len(hypotheses)):
        hypotheses[i] = hypotheses[i].strip()

    return hypotheses[:num_hypotheses]

class Generation(ABC):
    """Generation class. Implement the initialize function"""
    def __init__(self, api, prompt_class, inference_class):
        """ Initialize the update class

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
        self.train_data = self.inference_class.train_data

    @abstractmethod
    def initialize_hypotheses(self, args):
        """ Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
        ____________

        args: the parsed arguments

        ____________

        Returns:
        ____________

        hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        pass

    def batched_hypothesis_generation(self, args, example_indices, current_sample, num_hypotheses_generate):
        """ Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

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
        example_bank = {}
        for key in self.train_data:
            example_bank[key] = [self.train_data[key][idx] for idx in example_indices]

        # Prompt LLM to generate hypotheses
        prompt_input = self.prompt_class.batched_generation(example_bank, num_hypotheses_generate)
        response = self.api.generate(prompt_input)
        extracted_hypotheses = extract_hypotheses(args, response, num_hypotheses_generate)

        # create Summary Information for each
        new_generated_hypotheses = {}
        
        for hyp in extracted_hypotheses:
            correct = 0
            ex = []
            new_generated_hypotheses[hyp] = SummaryInformation(
                hypothesis=hyp,
                acc=0,
                num_visits=0,
                reward=0,
                correct_examples=ex
            )
            for index in example_indices:
                prediction, actual_label = self.inference_class.predict(args, self.train_data, index, {hyp: new_generated_hypotheses[hyp]}) 
                if prediction == actual_label:
                    correct += 1
                    ex.append((index, actual_label))
            acc = correct / len(example_indices)
            new_generated_hypotheses[hyp].set_accuracy(acc)
            new_generated_hypotheses[hyp].set_num_visits(len(example_indices))
            reward = acc + args.alpha * math.sqrt(math.log(current_sample) / len(example_indices))
            new_generated_hypotheses[hyp].set_reward(reward)
            new_generated_hypotheses[hyp].set_example(ex)
        
        return new_generated_hypotheses


class DefaultGeneration(Generation):
    def __init__(self, api, prompt_class, inference_class):
        super().__init__(api, prompt_class, inference_class)

    def initialize_hypotheses(self, args):
        """ Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
        ____________

        args: the parsed arguments

        ____________

        Returns:
        ____________

        hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        assert args.num_init % args.init_batch_size == 0, "Number of initial examples must be divisible by the batch size"
        num_batches = args.num_init // args.init_batch_size
        hypotheses_bank = {}
        for i in range(num_batches):
            example_indices = list(range(i*args.init_batch_size, (i+1)*args.init_batch_size))
            new_hypotheses = self.batched_hypothesis_generation(args, example_indices, len(example_indices), args.init_hypotheses_per_batch)
            hypotheses_bank.update(new_hypotheses)

        return hypotheses_bank


GENERATION_DICT = {
    'default': DefaultGeneration
}