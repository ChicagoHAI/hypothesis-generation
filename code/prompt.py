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



class Prompt(ABC):
    @abstractmethod
    def __init__(self):
        pass


class ShoePrompt(Prompt):
    def __init__(self):
        return
    

    def _information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        """
        Orginize information for a sample
        """
        
        appearance = data_dict['appearance'][j]
        shoe = data_dict['shoe'][j]

        color = shoe.split()[-1]
        shoe = shoe.strip()
        color = color.strip()
        prompt = f"A customer is {appearance}. This customer bought a pair of {color} shoes.\n"

        return prompt


    def few_shot_baseline(self, train_data, num_few_shot, test_data, i, model_name='Mixtral-8x7B'):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/few_shot_baseline.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/few_shot_baseline.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)
        appearance = test_data['appearance'][i]

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

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/batched_generation.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/batched_generation.py"

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
                  i,
                  prob=False):
        """
        Create inference prompt.
        """
        
        hypothesis_high_reward = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/inference.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/inference.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        appearance = test_data['appearance'][i]
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def knn_inference(self, hypotheses_dict, train_data, test_data, i):
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

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/knn.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/knn.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        appearance = test_data['appearance'][i]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    def knn_selection(self, hypotheses_dict, train_data, test_data, i, args):
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

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/knn_selection.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/knn_selection.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        appearance = test_data['appearance'][i]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)

    def is_relevant(self, hypothesis, data, index):
        """
        Check if a hypothesis is relevant to a specific example
        """

        instruction_path = f"{code_repo_path}/prompts/shoes/instructions/is_relevant.py"
        user_prompt_path = f"{code_repo_path}/prompts/shoes/user/is_relevant.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        appearance = data['appearance'][index]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

class RetweetPrompt(Prompt):
    def __init__(self):
        return 
    

    def _information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        """
        Orginize information for a sample
        """

        first_tweet = data_dict['tweets'][j][0]
        second_tweet = data_dict['tweets'][j][1]
        label = data_dict['label'][j]

        prompt = f"The first tweet: {first_tweet}\n"
        prompt += f"The second tweet: {second_tweet}\n"

        if not no_label_info:
            prompt += f"Final answer: The {label} tweet got more retweets.\n"

        return prompt


    def few_shot_baseline(self, train_data, num_few_shot, test_data, i, model_name='Mixtral-8x7B'):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/few_shot_baseline.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/few_shot_baseline.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        observations = ""
        few_shot_flag = False
        if num_few_shot > 0:
            few_shot_flag = True
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j)
        
        test_info = self._information_prompt(test_data, i, no_label_info=True)
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def batched_generation(self, 
                           train_data, 
                           num_hypotheses):
        """
        Generate hypotheses that is useful for predicting which tweets get retweeted more.
        """

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/batched_generation.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/batched_generation.py"

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
                  i,
                  prob=False):
        """
        Create inference prompt.
        """
        
        hypothesis_high_reward = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/inference.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/inference.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        first_text = test_data['tweets'][i][0]
        second_text = test_data['tweets'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def knn_inference(self, hypotheses_dict, train_data, test_data, i):
        """
        KNN inference prompt
        """
        
        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'retweet')
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nThe first tweet: {example[0][0]}\n'
                knn_info_prompt += f'The second tweet: {example[0][1]}\n'
                knn_info_prompt += f'Label: {example[1]}\n'

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/knn.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/knn.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        first_text = test_data['tweets'][i][0]
        second_text = test_data['tweets'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def is_relevant(self, hypothesis, data, index):
        """
        Check if a hypothesis is relevant to a specific example
        """

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/is_relevant.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/is_relevant.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        first_text = data['tweets'][index][0]
        second_text = data['tweets'][index][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def knn_selection(self, hypotheses_dict, train_data, test_data, i, args):
        """
        KNN hypothesis selection prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'retweet')

            if args.example_only_selection:
                knn_info_prompt += f'Pattern {hyp_idx+1} holds for the following examples:\n'
            else:
                knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nThe first tweet: {example[0][0]}\n'
                knn_info_prompt += f'The second tweet: {example[0][1]}\n'
                knn_info_prompt += f'Label: {example[1]}\n'

        instruction_path = f"{code_repo_path}/prompts/retweet/instructions/knn_selection.py"
        user_prompt_path = f"{code_repo_path}/prompts/retweet/user/knn_selection.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        first_text = test_data['tweets'][i][0]
        second_text = test_data['tweets'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


class HotelReviewsPrompt(Prompt):
    def __init__(self):
        return

    def _information_prompt(self, data_dict, j, no_label_info=False):
        """
        Orginize information for a sample
        """

        sentence = data_dict['review'][j]
        # get rid of trailing whitespace and new line characters
        sentence = sentence.strip()
        prompt = f"A hotel review is the following: \"{sentence}\"\n"

        if not no_label_info:
            prompt += f"The review is: {data_dict['label'][j]}.\n"
            prompt += "\n"

        return prompt


    def few_shot_baseline(self, train_data, num_few_shot, test_data, i, model_name='Mixtral-8x7B'):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/few_shot_baseline.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/few_shot_baseline.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        observations = ""
        few_shot_flag = False
        if num_few_shot > 0:
            few_shot_flag = True
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j)
        
        test_info = self._information_prompt(test_data, i, no_label_info=True)
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def batched_generation(self, 
                           train_data, 
                           num_hypotheses):
        """
        Generate hypotheses that is useful for predicting if a hotel review is truthful or deceptive.
        """

        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/batched_generation.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/batched_generation.py"

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
                  i,
                  prob=False):
        """
        Create inference prompt.
        """
        assert len(hypotheses_dict.keys()) == 1, 'Only one hypothesis is supported for inference prompt'
        
        hypothesis_high_reward = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/inference.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/inference.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        review = self._information_prompt(test_data, i, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)


    def knn_inference(self, hypotheses_dict, train_data, test_data, i):
        """
        KNN inference prompt
        """
        
        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'hotel_reviews')
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nHotel review: {example[0]}'
                knn_info_prompt += f'Label: {example[1]}\n'
            knn_info_prompt += "\n"

        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/knn.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/knn.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        review = self._information_prompt(test_data, i, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    def is_relevant(self, hypothesis, data, index):
        """
        Check if a hypothesis is relevant to a specific example
        """
        
        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/is_relevant.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/is_relevant.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        review = self._information_prompt(data, index, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)

    def knn_selection(self, hypotheses_dict, train_data, test_data, i, args):
        """
        KNN hypothesis selection prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'hotel_reviews')
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nHotel review: {example[0]}'
                knn_info_prompt += f'Label: {example[1]}\n'
            knn_info_prompt += "\n"

        instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/knn_selection.py"
        user_prompt_path = f"{code_repo_path}/prompts/hotel_reviews/user/knn_selection.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        review = self._information_prompt(test_data, i, no_label_info=True)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

class HeadlineBinary(Prompt):
    def __init__(self):
        return 

    def _information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        headlines = data_dict['headline'][j]
        labels = data_dict['label'][j]
        # E.g.: Headline 1: {headline} Clicks: {label}
        # Headline 2: {headline} Clicks: {label}
        prompt = f"Headline 1: {headlines[0]}\n"
        prompt += f"Headline 2: {headlines[1]}\n"
        if labels == "headline 1":
            prompt += "Observation: Headline 1 has more clicks than Headline 2."
        else:
            prompt += "Observation: Headline 2 has more clicks than Headline 1."

        return prompt


    def few_shot_baseline(self, train_data, num_few_shot, test_data, i, model_name='Mixtral-8x7B'):
        """
        Few shot prompt for baseline
        """

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/few_shot_baseline.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/few_shot_baseline.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        observations = ""
        few_shot_flag = False
        if num_few_shot > 0:
            few_shot_flag = True
            for j in range(num_few_shot):
                observations += self._information_prompt(train_data, j)
        
        headlines_0 = test_data['headline'][i][0]
        headlines_1 = test_data['headline'][i][1]
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def batched_generation(self, 
                           train_data, 
                           num_hypotheses):
        """
        Generate hypotheses that is useful for predicting if a hotel review is truthful or deceptive.
        """

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/batched_generation.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/batched_generation.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        prev_observations = ""
        for example_idx in range(len(train_data['headline'])):
            prev_observations += f"Example {example_idx+1}:\n"
            prev_observations += self._information_prompt(train_data, example_idx)
            prev_observations += "\n"
        
        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    

    def inference(self, 
                  hypotheses_dict, 
                  test_data, 
                  i,
                  prob=False):
        """
        Create inference prompt.
        """
        
        hypothesis_high_reward = list(hypotheses_dict.keys())[0]

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/inference.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/inference.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        headlines_0 = test_data['headline'][i][0]
        headlines_1 = test_data['headline'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
        

    def knn_inference(self, hypotheses_dict, train_data, test_data, i):
        """
        KNN inference prompt
        """
        
        knn_info_prompt = ""
        assert len(hypotheses_dict.keys()) > 0, 'At least one hypothesis is required for KNN inference prompt'
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'headline_binary')
            knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nHeadline 1: {example[0][0]}\n'
                knn_info_prompt += f'Headline 2: {example[0][1]}\n'
                knn_info_prompt += f'Label: {example[1]}\n'

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/knn.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/knn.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        headlines_0 = test_data['headline'][i][0]
        headlines_1 = test_data['headline'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    def is_relevant(self, hypothesis, data, index):
        """
        Check if a hypothesis is relevant to a specific example
        """

        headlines_0 = data['headline'][index][0]
        headlines_1 = data['headline'][index][1]
        
        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/is_relevant.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/is_relevant.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    def knn_selection(self, hypotheses_dict, train_data, test_data, i, args):
        """
        KNN hypothesis selection prompt
        """

        knn_info_prompt = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'headline_binary')

            if args.example_only_selection:
                knn_info_prompt += f'Pattern {hyp_idx+1} holds for the following examples:\n'
            else:
                knn_info_prompt += f'Pattern {hyp_idx+1}: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                knn_info_prompt += f'Example {ex_idx+1}:\nHeadline 1: {example[0][0]}\n'
                knn_info_prompt += f'Headline 2: {example[0][1]}\n'
                knn_info_prompt += f'Label: {example[1]}\n'

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/knn_selection.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/knn_selection.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        headlines_0 = test_data['headline'][i][0]
        headlines_1 = test_data['headline'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
    
    def inference_with_examples(self, hypotheses_dict, train_data, test_data, i):
        hypothesis_with_examples = ""
        for hyp_idx, (_, hypothesis_class) in enumerate(hypotheses_dict.items()):
            hypothesis_text = hypothesis_class.hypothesis
            hypothesis_related_examples = hypothesis_class.get_examples(train_data,'headline_binary')
            hypothesis_with_examples += f'Pattern: {hypothesis_text}\n'

            for ex_idx, example in enumerate(hypothesis_related_examples):
                hypothesis_with_examples += f'Example {ex_idx+1}:\nHeadline 1: {example[0][0]}\n'
                hypothesis_with_examples += f'Headline 2: {example[0][1]}\n'
                hypothesis_with_examples += f'Label: {example[1]}\n'

        instruction_path = f"{code_repo_path}/prompts/headline_binary/instructions/inference_with_examples.py"
        user_prompt_path = f"{code_repo_path}/prompts/headline_binary/user/inference_with_examples.py"

        instruction_prompt, user_prompt = read_prompt(instruction_path, user_prompt_path)

        headlines_0 = test_data['headline'][i][0]
        headlines_1 = test_data['headline'][i][1]

        instruction_prompt = eval(instruction_prompt)
        user_prompt = eval(user_prompt)

        return (instruction_prompt,user_prompt)
        

PROMPT_DICT = {
    'shoe': ShoePrompt,
    'hotel_reviews': HotelReviewsPrompt,
    'headline_binary': HeadlineBinary,
    'retweet': RetweetPrompt
}
