# Assume we generated a list of hypotheses from the training data.
# Here, we want to get their training accuracy, and use the best one as the final hypothesis.
# And then we want to get the test accuracy of the final hypothesis.

import argparse
import re
import sys
import os

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code')
from tasks import BaseTask
from utils import LLMWrapper, VALID_MODELS, extract_label, extract_hypotheses,set_seed
from data_loader import get_data
from prompt import BasePrompt


def get_accuracy(args, api, hypothesis, data, prompt_class):
    """
        Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
    """
    correct = 0
    for i in range(len(data['label'])):
        hypothesis_dict = {hypothesis: None}
        prompt_input = prompt_class.inference(hypothesis_dict, data, i)
        response = api.generate(prompt_input)
        print("*** get_accuracy ***")
        print(response)
        pred = extract_label(args.task, response)
        print('pred:', pred)
        print('label:', data['label'][i])
        print('*********************')
        if pred == data['label'][i]:
            correct += 1
    accuracy = correct / len(data['label'])
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # TODO: config path instead of task name
    parser.add_argument('--task', type=str, choices=['binary_original_sst',
                                                     'shoe',
                                                     'retweet',
                                                     'hotel_reviews',
                                                     'headline_binary'
                                                     ], help='task to run')
    parser.add_argument('--generation_model', type=str, default='turbo_0613', choices=VALID_MODELS, help='Model used at generation time.')
    parser.add_argument('--inference_model', type=str, default='turbo_0613', choices=VALID_MODELS, help='Model to use at inference time.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--num_hypothesis', type=int, default=5, help='Number of hypotheses to generate.')
    parser.add_argument('--num_test', type=int, default=100, help='Number of test examples.')
    parser.add_argument('--num_val', type=int, default=0, help='Number of validation examples.')
    parser.add_argument('--hypothesis_file', type=str, default='', help='Place to load the generated hypothesis.')
    parser.add_argument('--use_ood_reviews', type=str, default="None", help="Use out-of-distribution hotel reviews.")
    parser.add_argument('--model_path', type=str, default=None, help="Path for loading models locally.")
    # argument for using api cache, default true (1)
    parser.add_argument('--use_cache', type=int, default=1, help='Use cache for API calls.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_seed(args)

    # load the output text file
    text = open(args.hypothesis_file).read()

    # Use regex to extract the hypotheses
    hypotheses = extract_hypotheses(args, text)
    print('Hypotheses: ', hypotheses)
    if len(hypotheses) == 0:
        print("No hypotheses found.")
        return

    # load training data
    train_data, test_data, _ = get_data(args)

    # initialization
    prompt_class = BasePrompt(BaseTask(args.task))
    api = LLMWrapper(args.inference_model, 
                     path_name=args.model_path,
                     use_cache=args.use_cache)

    # get the training accuracy of each hypothesis
    training_accuracies = []
    for hypothesis in hypotheses:
        # get the training accuracy of the hypothesis
        accuracy = get_accuracy(args, api, hypothesis, train_data, prompt_class)
        training_accuracies.append(accuracy)

    # get the test accuracy of the best hypothesis
    best_hypothesis = hypotheses[training_accuracies.index(max(training_accuracies))]
    test_accuracy = get_accuracy(args, api, best_hypothesis, test_data, prompt_class)

    print('Best hypothesis: ', best_hypothesis)
    print('Test accuracy of best hypothesis: ', test_accuracy)
    print('Training accuracy of best hypothesis: ', max(training_accuracies))
    print('Training accuracies: ', training_accuracies)
    

if __name__ == '__main__':
    main()