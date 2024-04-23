from abc import ABC, abstractmethod
import os
import textwrap


code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")


class Prompt(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        pass

    @abstractmethod
    def check_usefulness_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def hypothesis_based_inference(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def hypothesis_based_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def can_generate_summary_prompt(self, summary, train_data, sample_examples_idx, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def generate_hypothesis(self, train_data, index_list, num_hypotheses, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def check_relevance_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        pass

    @abstractmethod
    def few_shot_baseline(self, train_data, k, test_data, i):
        pass

def read_from_template(template_path, replacements):
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()

    for placeholder, value in replacements.items():
        template_content = template_content.replace("{" + placeholder + "}", str(value))
    
    return template_content

class ShoePrompt(Prompt):
    def __init__(self):
        return
    
    def information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        appearance = data_dict['appearance'][j]
        shoe = data_dict['shoe'][j]
        # E.g.: The customer's appearance: a young and short man with black hat, orange shirt, and a large green bag. The customer bought a pair of orange shoes.
        color = shoe.split()[-1]
        shoe = shoe.strip()
        color = color.strip()
        prompt = f"A customer is {appearance}. This customer bought a pair of {color} shoes.\n"

        # print('******* information_prompt *******')
        # print(prompt)
        # print('**********************************')

        return prompt

    def check_usefulness_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        appearance = data['appearance'][i]

        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/check_usefulness.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/check_usefulness.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        
        prompt += f"We have the knowledge that: {hypothesis}\n"
        prompt += f"Now {appearance} bought a pair of shoes, the shoes should be which color?\n"

        print('******* check_usefulness_prompt *******')
        print(prompt)
        print('****************************************')

        return prompt
    
    def hypothesis_based_inference(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        """
        This inference requires agent to give answer and reasoning.
        """
        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/inference_with_reasoning.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/inference_with_reasoning.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'

        prompt += f"Pattern: {hypothesis_high_reward}\n"
        prompt += '\n'
        appearance = test_data['appearance'][i]
        prompt += f'New customer: {appearance} is buying a pair of shoes, the shoes should be which color?\n'
        prompt += '\n'
        prompt += 'Reasoning: '

        print('******* hypothesis_based_inference *******')
        print(prompt)
        print('************************************************************')

        return prompt 

    def hypothesis_based_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        """
        This inference only requires agent to give answer.
        """
        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/inference_without_reasoning.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt = prompt.strip()
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/inference_without_reasoning.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt = prompt.strip()
                prompt += '\n'

        hypothesis_high_reward = hypothesis_high_reward.strip()
        prompt += f"Pattern: {hypothesis_high_reward}\n"
        prompt += '\n'
        appearance = test_data['appearance'][i]
        prompt += f'New customer: {appearance} is buying a pair of shoes, the shoes should be which color?\n'
        prompt += '\n'
        prompt += 'Answer: '

        print('******* hypothesis_based_inference_without_reasoning *******')
        print(prompt)
        print('************************************************************')

        return prompt 
    
    def can_generate_summary_prompt(self, summary, train_data, sample_examples_idx, use_prev_messages=False, demonstration=False):
        """
        Use LLM to determine whether new observations can be used to update the summary.
        """
        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/can_generate_summary.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/can_generate_summary.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        
        prompt += f"We know the summary: {summary}\n"
        prompt += f"We made some new observations:\n"
        for example_idx in sample_examples_idx:
            prompt += self.information_prompt(train_data, example_idx)
        prompt += "\n"
        prompt += f"Can we update the summary to make it more useful to predict what color of shoes to recommend to customers based on their appearance? Type yes or no.\n"
        prompt += f"Your answer: "

        print('******* can_generate_summary_prompt *******')
        print(prompt)
        print('*******************************************')

        return prompt
        
    def generate_hypothesis(self, train_data, index_list, num_hypotheses, use_prev_messages=False, demonstration=False):
        """
        Generate a summary that is useful for predicting the color of the shoes given the appearance of the person.
        """
        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/generate_hypothesis.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/generate_hypothesis.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'

        prompt += f"We made some observations:\n"
        for example_idx in index_list:
            prompt += self.information_prompt(train_data, example_idx)
        prompt += f"Generate a summary that is useful for predicting the color of the shoes given the appearance of the person. Please be concise and keep the summary to be one-sentence long. Propose {num_hypotheses} possible summaries. Generate them in the format of 1. [summary], 2. [summary], ... 5. [summary]\n"

        print('******* generate_hypothesis *******')
        print(prompt)
        print('***********************************')

        return prompt
    
    def check_relevance_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        """
        Prompt for the model to check the relevance of the hypothesis.
        """        
        appearance = data['appearance'][i]

        if demonstration:
            demo_path = f"{code_repo_path}/prompts/shoes/demonstrations/check_relevance.txt"
            with open(demo_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'
        else:
            instruction_path = f"{code_repo_path}/prompts/shoes/instructions/check_relevance.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read() # a string of the entire file
                prompt += '\n'

        prompt += f"Previous hypothesis:\n{hypothesis}"
        prompt += '\n'
        prompt += f"New customer:\n{appearance} bought a pair of shoes.\n"
        prompt += '\n'
        prompt += 'We want to determine the color of the shoes.\n'
        prompt += '\n'
        prompt += f"Is the hypothesis relevant to the specific example we are considering?\n"
        prompt += f"Answer: "

        print('******* check_relevance_prompt *******')
        print(prompt)
        print('**************************************')

        return prompt

    def few_shot_baseline(self, train_data, k, test_data, i, model):
        # instruction 
        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction: 
                                 You are a shoe salesman and want to recommend shoes to customers. There are white, red, orange, green, blue, and black shoes. 
                                 Give your answer for the shoe color recommendation. The answer should be one color word. It has to be one of white, red, orange, green, blue, and black. If you do not have enough information to make a recommendation, you should give the answer "unknown". 
                                 Give your final answer in the format of "Final answer: [answer]." 
                                 ###
                                 
                                 ''')

        # add demo examples
        num_train = len(train_data)
        if k > 0 and k <= num_train and num_train > 0:
            prompt += "Here are some examples of customers with certain features buying certain products:\n"
            for j in range(k):
                p = self.information_prompt(train_data, j)
                prompt += p
            prompt += "\n"

        # add the test example
        prompt += textwrap.dedent(f'''\
                                 New customer: {test_data['appearance'][i].strip()} is buying a pair of shoes, the shoes should be which color?
                                 
                                 Answer: ''')

        print("****** few_shot_baseline ******")
        print(prompt)
        print("********************************************")

        return prompt

    def batched_learning_hypothesis_generation(self, train_data, num_hypotheses):
        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction:
                                 We are in a synthetic world.
                                 Given a set of observations, we want to generate hypotheses that are useful for recommending a color for shoes to a customer. Please be concise and keep each hypothesis to be one-sentence long.
                                 Propose {num_hypotheses} possible hypotheses.
                                 Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].
                                 Please make the hypotheses general enough to be applicable to new observations.
                                 ###
                                 ''')                               

        prompt += 'We made some observations:\n'
        for example_idx in range(len(train_data['appearance'])):
            prompt += self.information_prompt(train_data, example_idx)

        prompt += f"Generate hypotheses that are useful for recommending color of shoes. Please be concise and keep each hypothesis to be one-sentence long. Propose {num_hypotheses} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].\n\nProposed hypotheses:"

        return prompt

    def batched_learning_hypothesis_based_inference_without_reasoning(self, hypothesis, data, i):
        return self.new_inference_without_reasoning(hypothesis, data, i)

    def new_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, 
                                        few_shot=0, train_data=None):
        """
        This inference only requires agent to give answer.
        """
        assert few_shot >= 0

        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction:
                                 You are a shoe salesman and want to recommend shoes to customers. There are white, red, orange, green, blue, and black shoes.
                                 From past experiences, you learned a pattern.
                                 Now, at each time, you should apply the learned pattern, given below, to a new customer and recommend a shoe color.
                                 Give your answer for the shoe color recommendation. The answer should be one color word. It has to be one of white, red, orange, green, blue, and black. If you do not have enough information to make a recommendation, you should give the answer "unknown".
                                 Give your final answer in the format of "Final answer: [answer]."
                                 ###
                                 
                                 ''')

        if few_shot > 0:
            assert train_data is not None
            shot_index = list(range(few_shot))
            
            prompt += "Here are some cases of customers buying shoes from the past:\n"
            for j in shot_index:
                prompt += textwrap.dedent(f'''\
                                         Previous customer: {train_data['appearance'][j].strip()} is buying a pair of shoes, the shoes should be which color?
                                         Answer: {train_data['shoe'][j].strip()}

                                         ''')
    
        prompt += textwrap.dedent(f'''\
                                  Pattern: {hypothesis_high_reward.strip()}
                                  
                                  New customer: {test_data['appearance'][i].strip()} is buying a pair of shoes, the shoes should be which color?
                                  Answer: ''')
            

        # print("****** new_inference_without_reasoning ******")
        # print(prompt)
        # print("********************************************")

        return prompt


    def zero_shot_inference_without_hypothesis(self, data, i):
        appearance = data['appearance'][i]
        appearance = appearance.strip()
        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction: 
                                 You are a shoe salesman and want to recommend shoes to customers. There are white, red, orange, green, blue, and black shoes. 
                                 Give your answer for the shoe color recommendation. The answer should be one color word. It has to be one of white, red, orange, green, blue, and black. If you do not have enough information to make a recommendation, you should give the answer "unknown". 
                                 Give your final answer in the format of "Final answer: [answer]." 
                                 ###
                                 
                                 New customer: {appearance} is buying a pair of shoes, the shoes should be which color?
                                 
                                 Answer: ''')

        print("****** new_inference_without_reasoning ******")
        print(prompt)
        print("********************************************")

        return prompt


class HeadlineBinary(Prompt):
    def __init__(self):
        return 

    def information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
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

    def batched_learning_hypothesis_generation(self, train_data, num_hypotheses):
        instruction_path = f"{code_repo_path}/prompts/headline_binary/batched_learning.txt"
        
        prev_observations = ""
        for example_idx in range(len(train_data['headline'])):
            prev_observations += f"Example {example_idx+1}:\n"
            prev_observations += self.information_prompt(train_data, example_idx)
            prev_observations += "\n"
        
        variables = {
            "prev_observations" : prev_observations,
            "num_hypotheses" : num_hypotheses
        }

        prompt = read_from_template(instruction_path, variables)
        return prompt
    
    def new_inference_without_reasoning(self, hypothesis_high_reward, test_data, i):
        """
        This inference only requires agent to give answer.
        """
        headlines = test_data['headline'][i]
        hypothesis_high_reward = hypothesis_high_reward.strip()
        variables = {
            "hypothesis_high_reward": hypothesis_high_reward,
            "headlines0": headlines[0],
            "headlines1": headlines[1],
        }

        prompt = read_from_template(f"{code_repo_path}/prompts/headline_binary/inference_without_reasoning.txt", variables)
                                
        print("****** new_inference_without_reasoning ******")
        print(prompt)
        print("********************************************")

        return prompt

    def new_inference_with_reasoning_few_shot(self, hypothesis_high_reward, test_data, i, train_data, few_shot):
        headlines = test_data['headline'][i]
        hypothesis_high_reward = hypothesis_high_reward.strip()
        prior_examples = ""
        for index in range(few_shot):
            prior_examples += self.information_prompt(train_data, index)
            prior_examples += '\n'
        variables = {
            "hypothesis_high_reward": hypothesis_high_reward,
            "prior_examples": prior_examples,
            "headlines0": headlines[0],
            "headlines1": headlines[1],
        }
        prompt = read_from_template(f"{code_repo_path}/prompts/headline_binary/few_shot_inference_with_reasoning.txt", variables)       
        print("****** new_inference_with_reasoning ******")
        print(prompt)
        print("********************************************")
        return prompt
    
    def new_inference_with_reasoning(self, hypothesis_high_reward, test_data, i):
        headlines = test_data['headline'][i]
        hypothesis_high_reward = hypothesis_high_reward.strip()
        variables = {
            "hypothesis_high_reward": hypothesis_high_reward,
            "headlines0": headlines[0],
            "headlines1": headlines[1],
        }
        prompt = read_from_template(f"{code_repo_path}/prompts/headline_binary/inference_with_reasoning.txt", variables)       
        print("****** new_inference_with_reasoning ******")
        print(prompt)
        print("********************************************")

        return prompt
    
    def new_inference_with_zero_shot_reasoning(self, hypothesis_high_reward, test_data, i):
        """
        This inference only requires agent to give answer.
        """
        headlines = test_data['headline'][i]
        hypothesis_high_reward = hypothesis_high_reward.strip()
        variables = {
            "hypothesis_high_reward": hypothesis_high_reward,
            "headlines0": headlines[0],
            "headlines1": headlines[1],
        }
        prompt = read_from_template(f"{code_repo_path}/prompts/headline_binary/inference_with_zero_shot_reasoning.txt", variables)  
                                
        print("****** new_inference_with_zero_shot_reasoning ******")
        print(prompt)
        print("********************************************")

        return prompt

    def few_shot_baseline(self, train_data, k, test_data, i, model):
        prior_examples = ""
        if k:
            prior_examples += "Here are some previous examples to help you.\n"
            for j in range(k):
                prior_examples += self.information_prompt(train_data, j)
                prior_examples += "\n"

        headlines = test_data['headline'][i]

        variables = {
            "prior_examples" : prior_examples,
            "headlines0" : headlines[0],
            "headlines1" : headlines[1]
        }

        file_path = f"{code_repo_path}/prompts/headline_binary/few_shot_baseline.txt"
        prompt = read_from_template(file_path, variables)
        return prompt
    
    def knn_inference(self, hypotheses_dict, test_data, k):
        hyp_with_examples = ""
        for i, hypothesis in enumerate(hypotheses_dict.keys()):
            hyp_with_examples += f'Hypothesis {i+1}: {hypothesis}\n'
            for j in range(len(hypotheses_dict[hypothesis])):
                hyp_with_examples += f'Example {j+1}:\n Headline 1 - {hypotheses_dict[hypothesis][j][0]}\n'
                hyp_with_examples += f'Headline 2 -  {hypotheses_dict[hypothesis][j][1]}\n'
                hyp_with_examples += f'Label: {hypotheses_dict[hypothesis][j][-1]}\n'
            hyp_with_examples += f'\n'

        
        headlines = test_data['headline'][k]

        variables = {
            "hyp_with_examples" : hyp_with_examples,
            "headlines0" : headlines[0],
            "headlines1" : headlines[1]
        }
        file_path = f"{code_repo_path}/prompts/headline_binary/knn_inference.txt"
        prompt = read_from_template(file_path, variables)
        return prompt



    def generate_hypothesis(self, train_data, index_list, num_hypotheses, use_prev_messages=False, demonstration=False):
        raise NotImplementedError
    
    def check_usefulness_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        raise NotImplementedError
    
    def hypothesis_based_inference(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        """
        This inference requires agent to give answer and reasoning.
        """
        raise NotImplementedError

    def hypothesis_based_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        """
        This inference only requires agent to give answer.
        """
        raise NotImplementedError 
    
    def can_generate_summary_prompt(self, summary, train_data, sample_examples_idx, use_prev_messages=False, demonstration=False):
        """
        Use LLM to determine whether new observations can be used to update the summary.
        """
        raise NotImplementedError
    
    
    def check_relevance_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        """
        Prompt for the model to check the relevance of the hypothesis.
        """ 
        raise NotImplementedError
    
 


class HotelReviewsPrompt(Prompt):
    def __init__(self):
        return

    def information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        sentence = data_dict['review'][j]
        # get rid of trailing whitespace and new line characters
        sentence = sentence.strip()
        prompt = f"A hotel review is the following: \"{sentence}\"\n"
        prompt += "\n"

        if not no_label_info:
            prompt += f"The review is: {data_dict['label'][j]}.\n"

        prompt += "\n"

        print('******* information_prompt *******')
        print(prompt)
        print('**********************************')

        return prompt

    def check_usefulness_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        """
        Prompt for the model to check the usefulness of the hypothesis.
        """
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/check_usefulness.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'
        
        sentence = data['review'][i]

        prompt += f"We have the knowledge that: {hypothesis}\n"
        prompt += '\n'
        prompt += f"Now we have a hotel review: \"{sentence}\"\n"
        prompt += '\n'
        prompt += f"Reasoning: "

        print('******* check_usefulness_prompt *******')
        print(prompt)
        print('***************************************')

        return prompt
    
    def hypothesis_based_inference(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/inference_with_reasoning.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'
        
        prompt += f"Our learned pattern: {hypothesis_high_reward}\n"
        prompt += '\n'
        prompt += self.information_prompt(test_data, i, no_label_info=True)
        prompt += '\n'
        prompt += f"Reasoning: "

        print('******* hypothesis_based_inference *******')
        print(prompt)
        print('******************************************')

        return prompt
    
    def hypothesis_based_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/inference_without_reasoning.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'

        prompt += f"Our learned pattern: {hypothesis_high_reward}\n"
        prompt += '\n'
        prompt += self.information_prompt(test_data, i, no_label_info=True)
        prompt += '\n'
        prompt += f"Answer: "

        print('******* hypothesis_based_inference_without_reasoning *******')
        print(prompt)
        print('************************************************************')

        return prompt

    def can_generate_summary_prompt(self, summary, train_data, sample_examples_idx, use_prev_messages=False, demonstration=False):
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/can_generate_summary.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'
        
        prompt += f"We know the summary: {summary}\n"
        for example_idx in sample_examples_idx:
            prompt += f"We made some new observations:\n"
            prompt += self.information_prompt(train_data, example_idx)
        prompt += f"Can we update the summary to make it more useful? Type yes or no.\n"
        prompt += f"Your answer: "

        print('******* can_generate_summary_prompt *******')
        print(prompt)
        print('*******************************************')

        return prompt

    def generate_hypothesis(self, train_data, index_list, num_hypotheses, use_prev_messages=False, demonstration=False):
        """
        Generate a summary that is useful for predicting the truthfulness of the hotel review.
        """
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/generate_hypothesis.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'
        
        prompt += f"We have more hotel reviews:\n"
        for example_idx in index_list:
            prompt += self.information_prompt(train_data, example_idx)
        prompt += f"Generate a rule of thumb (i.e. summary) that is useful for predicting the truthfulness of the hotel reviews. Please be concise and keep the summary to be one-sentence long. Propose {num_hypotheses} possible summaries. Generate them in the format of 1. [summary], 2. [summary], ... 5. [summary]\n"

        print('******* generate_hypothesis *******')
        print(prompt)
        print('***********************************')

        return prompt
    
    def check_relevance_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        if demonstration:
            raise NotImplementedError
        else:
            instruction_path = f"{code_repo_path}/prompts/hotel_reviews/instructions/check_relevance.txt"
            with open(instruction_path, 'r') as f:
                prompt = f.read()
                prompt += '\n'
        
        sentence = data['review'][i]

        prompt += f"From previous examples, we learned a hypothesis: {hypothesis}\n"
        prompt += '\n'
        prompt += f"New hotel review: \"{sentence}\"\n"
        prompt += '\n'

        prompt += f"We want to determine whether this review is truthful or deceptive.\n"
        prompt += f"Is the hypothesis relevant to the specific example sentence we are considering?\n"
        prompt += f"Answer yes or no.\n"

        print('******* check_relevance_prompt *******')
        print(prompt)
        print('**************************************')

        return prompt
    
 

    def batched_learning_hypothesis_generation(self, train_data, num_hypotheses):
        prompt = f"###\nInstruction:\nGiven a set of hotel reviews, we want to generate hypotheses that are useful for predicting whether a review is truthful or deceptive. In other words, we want to know whether the review is written by a someone who actually lived in the hotel.\nPlease be concise and keep each hypothesis to be one-sentence long. Propose {num_hypotheses} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis]. The hypotheses should analyze what kind of reviews are likely to be truthful or deceptive.\n###\n"

        prompt += 'We have seen some hotel reviews:\n'
        print('len(train_data): ', len(train_data['review']))
        for example_idx in range(len(train_data['review'])):
            prompt += self.information_prompt(train_data, example_idx)

        prompt += f"Generate hypotheses that are useful for predicting whether a review is truthful or deceptive. Please be concise and keep each hypothesis to be one-sentence long. Propose {num_hypotheses} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].\n\nProposed hypotheses:"

        return prompt
    
    def new_inference_without_reasoning_version1(self, hypothesis_high_reward, test_data, i):
        prompt=textwrap.dedent("""\
                               ###
                               Instruction:
                               You are a deceptive detection agent and want to determine whether a hotel review is truthful or deceptive. 
                               In other words, we want to know whether the review is written by someone who actually lived in the hotel. 
                               From past experiences, you learned some patterns. 
                               You need to determine whether this pattern holds for the current hotel review, and also predict whether the current hotel review is truthful or deceptive. 
                               Give an answer. The answer should be one word (truthful or deceptive). 
                               Give your final answer in the format of "Final answer: [answer]."
                               ###
                               
                               Our learned pattern: {hypothesis_high_reward}
                               
                               {self.information_prompt(test_data, i, no_label_info=True)}
                               
                               Answer: """)
        
        print('******* new_inference_without_reasoning_version1 *******')
        print(prompt)
        print('********************************************************')

        return prompt

    def new_inference_without_reasoning_version2(self, hypothesis_high_reward, test_data, i):
        prompt=textwrap.dedent("""\
                               ###
                               Instruction:
                               You are a deceptive detection agent and want to determine whether a hotel review is truthful or deceptive. 
                               In other words, we want to know whether the review is written by someone who actually lived in the hotel. 
                               From past experiences, you learned some patterns. 
                               Below is the learned pattern and the new review that we received. You should apply the learned pattern to the hotel review and determine whether the review is truthful or deceptive. 
                               Your answer should be one word. It has to be either truthful or deceptive. 
                               Give your final answer in the format of "Final answer: [answer]."
                               ###
                               
                               Our learned pattern: {hypothesis_high_reward}
                               
                               {self.information_prompt(test_data, i, no_label_info=True)}
                               
                               Answer: """)
        
        print('******* new_inference_without_reasoning_version2 *******')
        print(prompt)
        print('********************************************************')

        return prompt


class RetweetPrompt(Prompt):
    def __init__(self):
        return

    def information_prompt(self, data_dict, j, no_label_info=False, use_prev_messages=False):
        first_tweet = data_dict['first_text'][j]
        second_tweet = data_dict['second_text'][j]
        label = data_dict['label'][j]

        prompt = f"The first tweet: {first_tweet}\n"
        prompt += f"The second tweet: {second_tweet}\n"

        if not no_label_info:
            prompt += f"Final answer: The {label} tweet got more retweets.\n"

        print('******* information_prompt *******')
        print(prompt)
        print('**********************************')

        return prompt
    
    def few_shot_baseline(self, train_data, k, test_data, i, model):
        # instruction 
        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction: 
                                 You are a social media expert.
                                 Given two tweets, you are asked to predict which tweet will attract more retweets.
                                 Give your final answer in the format of "Final answer: [the _ tweet got more retweet]." 
                                 ###
                                 
                                 ''')

        # add demo examples
        num_train = len(train_data)
        if k > 0 and k <= num_train and num_train > 0:
            prompt += "Here are some examples:\n"
            for j in range(k):
                p = self.information_prompt(train_data, j)
                prompt += p
                prompt += "\n"

        # add the test example
        prompt += self.information_prompt(test_data, i, no_label_info=True)
        prompt += 'Final answer:'

        print("****** few_shot_baseline ******")
        print(prompt)
        print("*******************************")

        return prompt

    def batched_learning_hypothesis_generation(self, train_data, num_hypotheses):
        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction:
                                 You are a social media expert.
                                 Given a set of observations, you want to generate hypotheses that are useful for predicting tweets that will attract more retweets. Please be concise and keep each hypothesis to be one-sentence long.
                                 Propose {num_hypotheses} possible hypotheses.
                                 Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].
                                 Please make the hypotheses general enough to be applicable to new observations.
                                 ###
                                 ''')                               

        prompt += 'We made some observations:\n'
        for example_idx in range(len(train_data['label'])):
            prompt += self.information_prompt(train_data, example_idx)

        prompt += f"Generate hypotheses that are useful for predicting tweets that will attract more retweets. Please be concise and keep each hypothesis to be one-sentence long. Propose {num_hypotheses} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].\n\nProposed hypotheses:"

        return prompt

    def batched_learning_hypothesis_based_inference_without_reasoning(self, hypothesis, data, i):
        return self.new_inference_without_reasoning(hypothesis, data, i)

    def new_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, 
                                        few_shot=0, train_data=None):
        assert few_shot >= 0

        prompt = textwrap.dedent(f'''\
                                 ###
                                 Instruction:
                                 You are a social media expert.
                                 Given two tweets, you are asked to predict which tweet will attract more retweets.
                                 From past experiences, you learned a pattern.
                                 You should apply the learned pattern when you predict which tweet will attract more retweets.
                                 Give your final answer in the format of "Final answer: [the _ tweet got more retweet]."
                                 ###

                                 ''')
        
        if few_shot > 0:
            assert train_data is not None
            shot_index = list(range(few_shot))

            prompt += "Here are some examples:\n"
            for j in range(few_shot):
                p = self.information_prompt(train_data, j)
                prompt += p
                prompt += "\n"

        prompt += textwrap.dedent(f'''\
                                  Pattern: {hypothesis_high_reward}
                                  
                                  The first tweet: {test_data['first_text'][i]}\n
                                  The second tweet: {test_data['second_text'][i]}\n
                                  Final answer: ''')

        print('******* new_inference_without_reasoning *******')
        print(prompt)
        print('***********************************************')

        return prompt

    def check_usefulness_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        raise NotImplementedError

    def hypothesis_based_inference(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        raise NotImplementedError

    def hypothesis_based_inference_without_reasoning(self, hypothesis_high_reward, test_data, i, use_prev_messages=False, demonstration=False):
        raise NotImplementedError

    def can_generate_summary_prompt(self, summary, train_data, sample_examples_idx, use_prev_messages=False, demonstration=False):
        raise NotImplementedError

    def generate_hypothesis(self, train_data, index_list, num_hypotheses, use_prev_messages=False, demonstration=False):
        raise NotImplementedError

    def check_relevance_prompt(self, data, i, hypothesis, use_prev_messages=False, demonstration=False):
        raise NotImplementedError


PROMPT_DICT = {
    'shoe': ShoePrompt,
    'hotel_reviews': HotelReviewsPrompt,
    'headline_binary': HeadlineBinary,
    'retweet': RetweetPrompt,
}
