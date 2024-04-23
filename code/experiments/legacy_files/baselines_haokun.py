import argparse
import time
import pickle
import sys
import os
import math

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')
from prompt_support_mixtral import PROMPT_DICT
from data_loader import get_train_data
from utils import LLMWrapper, set_seed, create_directory, get_num_examples, extract_label, GPT_MODELS, VALID_MODELS
from algorithm.summary_information import SummaryInformation
from tasks import TASKS


PROMPT_NAME_DICT = {
    'shoe': 'appearance',
    'sst': 'sentence',
    'original_sst': 'sentence',
    'binary_sst' : 'sentence',
    'binary_original_sst': 'sentence',
    'hotel_reviews': 'review',
    'headline_binary':'headline',
    'retweet':'tweets'
}

def subset_data_dict(data,subset_size):
    """
    Subset given dataset in two
    """

    if subset_size > len(data['label']):
        raise ValueError("Subset size larger than data size")
    subset_data = {}
    main_data = {}
    for key in data.keys():
        subset_data[key] = data[key][-subset_size:]
        main_data[key] = data[key][:-subset_size]
    return main_data, subset_data

class baseline_inference():
    """
    Baseline class for different type of inference
    """

    def __init__(self,
                 task_name,
                 api,
                 verbose='True'):
        self.api = api
        self.task_name = task_name
        self.prompt_class = PROMPT_DICT[task_name]()
        self.task_class = TASKS[task_name]()
        self.verbose = verbose
    

    def inference(self,
                  test_data,
                  i,
                  hypothesis=None,
                  few_shot_data=None):
        # Inference with or without hypothesis, with or without few-shot examples
        if hypothesis is not None:
            prompt = self.prompt_class.inference_without_reasoning(hypothesis_high_reward=hypothesis,
                                                                   test_data=test_data, 
                                                                   i=i,
                                                                   model=self.api.model,
                                                                   few_shot_data=few_shot_data)
        else:
            prompt = self.prompt_class.no_hypothesis_inference(test_data=test_data,
                                                               i=i,
                                                               model=self.api.model,
                                                               few_shot_data=few_shot_data)
        response = self.api.generate(prompt)

        response = response.lower()
        pred = self.task_class.extract_label(response)
                
        if self.verbose:
            print("\n***************************************************")
            print("Hotel review: ", test_data[PROMPT_NAME_DICT[self.task_name]][i])
            print("Model generate output:", response)
            print(f"True label: {test_data['label'][i]}, Predicted label: {pred}" )
            print("***************************************************\n")

        return pred
    
    
    def compute_acc(self,
                    test_data,
                    hypothesis=None,
                    few_shot_data=None):
        # Do inference for all test data and return accuracy score
        correct = 0
        for i in range(len(test_data['label'])):
            pred = self.inference(test_data,
                                  i,
                                  hypothesis=hypothesis,
                                  few_shot_data=few_shot_data)
            label = test_data['label'][i]
            if pred == label:
                correct += 1
        return correct / len(test_data['label'])
    
        
        



