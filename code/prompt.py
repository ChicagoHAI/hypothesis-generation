from abc import ABC, abstractmethod
import os
import textwrap

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")


model_postfix = {
    'Mistral-7B': '_mixtral',
    'Mixtral-8x7B': '_mixtral',
    'Others': ''
}

MISTRAL_MODELS = ['Mistral-7B', 'Mixtral-8x7B']


def read_prompt(instruction_path, user_prompt_path):
    with open(instruction_path, 'r') as f:
        instruction_prompt = f.read() # a string of the entire file

    with open(user_prompt_path, 'r') as f:
        user_prompt = f.read() # a string of the entire file
    
    return instruction_prompt, user_prompt


class BasePrompt(ABC):
    def __init__(self, task_name):
        self.task_name = task_name


    @abstractmethod 
    def _information_prompt(data_dict, example_idx, no_label_info=False):
        pass


    def few_shot_baseline(self, train_data, num_few_shot, test_data, test_idx):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/few_shot_baseline/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/few_shot_baseline/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)
        info = self._information_prompt(test_data, test_idx, no_label_info=True)

        observations = ""
        few_shot_flag = False
        if num_few_shot > 0:
            few_shot_flag = True
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j)
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    
    def batched_generation(self, 
                           train_data, 
                           num_hypotheses):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/batched_generation/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/batched_generation/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        observations = ""
        for example_idx in range(len(train_data['label'])):
            observations += self._information_prompt(train_data, example_idx)
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def inference(self, 
                  hypotheses_dict, 
                  test_data, 
                  test_idx):
        """
        Create inference prompt.
        """
        
        hypothesis = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/inference/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/inference/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        info = self._information_prompt(test_data, test_idx, no_label_info=True)
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def knn_inference(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        KNN inference prompt
        """
        
        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\n'
                knn_info_prompt += self._information_prompt(train_data, example_info[0])

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/knn/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/knn/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        info = self._information_prompt(test_data, test_idx, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def knn_selection(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        KNN hypothesis selection prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.correct_examples
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example_info in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\n'
                knn_info_prompt += self._information_prompt(train_data, example_info[0])

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/knn_selection/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/knn_selection/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        info = self._information_prompt(test_data, test_idx, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/{self.task_name}/is_relevant/instructions.py"
        user_prompt_path = f"{code_repo_path}/prompts/{self.task_name}/is_relevant/user.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        info = self._information_prompt(test_data, test_idx, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

class ShoePrompt(BasePrompt):
    def __init__(self):
        super().__init__(task_name='shoe')
    

    def _information_prompt(self, data_dict, example_idx, no_label_info=False):
        """
        Orginize information for a sample
        """
        
        appearance = data_dict['appearance'][example_idx]
        shoe = data_dict['shoe'][example_idx]

        color = shoe.split()[-1]
        shoe = shoe.strip()
        color = color.strip()
        prompt = f"A customer is {appearance}. This customer bought a pair of {color} shoes.\n"

        if no_label_info:
            return appearance
        else:
            return prompt



class RetweetPrompt(BasePrompt):
    def __init__(self):
        super().__init__(task_name='retweet') 
    

    def _information_prompt(self, data_dict, example_idx, no_label_info=False):
        """
        Orginize information for a sample
        """

        first_tweet = data_dict['tweets'][example_idx][0]
        second_tweet = data_dict['tweets'][example_idx][1]
        label = data_dict['label'][example_idx]

        prompt = f"The first tweet: {first_tweet}\n"
        prompt += f"The second tweet: {second_tweet}\n"

        if not no_label_info:
            prompt += f"Final answer: The {label} tweet got more retweets.\n"

        return prompt


class HotelReviewsPrompt(BasePrompt):
    def __init__(self):
        super().__init__(task_name='hotel_reviews') 


    def _information_prompt(self, data_dict, example_idx, no_label_info=False):
        """
        Orginize information for a sample
        """

        sentence = data_dict['review'][example_idx]
        # get rid of trailing whitespace and new line characters
        sentence = sentence.strip()
        prompt = f"A hotel review is the following: \"{sentence}\"\n"

        if not no_label_info:
            prompt += f"The review is: {data_dict['label'][example_idx]}.\n"
            prompt += "\n"

        return prompt
    

class HeadlineBinary(BasePrompt):
    def __init__(self):
        super().__init__(task_name='headline_binary')

    def _information_prompt(self, data_dict, example_idx, no_label_info=False):
        headlines = data_dict['headline'][example_idx]
        labels = data_dict['label'][example_idx]
        # E.g.: Headline 1: {headline} Clicks: {label}
        # Headline 2: {headline} Clicks: {label}
        prompt = f"Headline 1: {headlines[0]}\n"
        prompt += f"Headline 2: {headlines[1]}\n"

        if not no_label_info:
            if labels == "headline 1":
                prompt += "Observation: Headline 1 has more clicks than Headline 2."
            else:
                prompt += "Observation: Headline 2 has more clicks than Headline 1."
            prompt += "\n\n"
        return prompt


PROMPT_DICT = {
    'shoe': ShoePrompt,
    'hotel_reviews': HotelReviewsPrompt,
    'headline_binary': HeadlineBinary,
    'retweet': RetweetPrompt
}
