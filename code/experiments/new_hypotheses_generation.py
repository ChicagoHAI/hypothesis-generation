import argparse
import time
import pickle
import sys
import os
import math

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')
from prompt import PROMPT_DICT
from data_loader import get_train_data
from utils import LLMWrapper, SummaryInformation, set_seed, create_directory, get_num_examples, extract_label, GPT_MODELS, VALID_MODELS
from hypotheses_initialization import initialize_hypotheses, extract_hypotheses


def inference(args, hypothesis, train_data, api, prompt_class, example_index):
    """
    Run hypothesis-based inference. Returns the prediction and the label.
    """
    if args.task in ['shoe', 'retweet']:
        prompt_input = prompt_class.new_inference_without_reasoning(hypothesis, train_data, example_index)
    elif args.task == 'hotel_reviews':
        if args.hotel_inference_prompt == 'old':
            prompt_input = prompt_class.hypothesis_based_inference_without_reasoning(hypothesis, train_data, example_index) 
        elif args.hotel_inference_prompt == 'version1':
            prompt_input = prompt_class.new_inference_without_reasoning_version1(hypothesis, train_data, example_index)
        elif args.hotel_inference_prompt == 'version2':
            prompt_input = prompt_class.new_inference_without_reasoning_version2(hypothesis, train_data, example_index)
        else:
            raise ValueError(f'Invalid quick_dev argument. Your argument should be either `old`, `version1`, or `version2`, but received {args.hotel_inference_prompt}.')
    elif args.task == 'headline_binary':
        if args.headline_binary_inference == "no_reason":
            prompt_input = prompt_class.new_inference_without_reasoning(hypothesis, train_data, example_index)
        else:
            prompt_input = prompt_class.new_inference_with_reasoning(hypothesis, train_data, example_index)
    else:
        raise ValueError('Invalid task argument.')

    response = api.generate(prompt_input)
    response = response.lower()
    pred = extract_label(args.task, response.split('answer')[-1])
    label = train_data['label'][example_index]

    if args.task == 'headline_binary':
        if label[0] == 'high':
            label = "Headline 1"
        else:
            label = "Headline 2"

        pred = pred.lower()
        label = label.lower()

    # if args.verbose:
    #     print(f'In `inference` function: Example {example_index}: hypothesis "{hypothesis}" prediction "{pred}" label "{label}"')

    return pred, label


def useful_hypothesis(args, hypothesis, train_data, api, prompt_class, example_index):
    """
    Run hypothesis-based inference and check if the prediction is correct.
    """
    pred, label = inference(args, hypothesis, train_data, api, prompt_class, example_index)
    
    if pred == label:
        return True
    else:
        return False


def get_accuracy(args, hypothesis, train_data, api, prompt_class):
    """
    Get the accuracy of a hypothesis on the training data.
    """
    correct = 0
    for i in range(get_num_examples(train_data)):
        pred, label = inference(args, hypothesis, train_data, api, prompt_class, i)
        if pred == label:
            correct += 1
    return correct / get_num_examples(train_data)


def update_hypotheses_with_new_hypotheses(hypotheses, new_hypotheses, max_num_hypotheses, num_hypotheses_to_update):
    """
    Update the hypotheses with new hypotheses.
    We add new hypotheses to old hypotheses if they were not already present.
    If the number of hypotheses is more than the maximum number of hypotheses, we drop the lowest-ranking hypotheses.
    """
    # add new hypotheses to old hypotheses if they were not already present
    for hypothesis in new_hypotheses:
        if hypothesis not in hypotheses:
            hypotheses[hypothesis] = new_hypotheses[hypothesis]

    # sort the hypotheses by reward in descending order
    sorted_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward, reverse=True)
    while len(hypotheses) > max_num_hypotheses:
        # drop the lowest ranking hypotheses until we have the maximum number of hypotheses
        hypotheses.pop(sorted_hypotheses.pop())
    
    return hypotheses


def update_reward_for_top_k_hypotheses(args, top_k_hypotheses, hypotheses, num_example, train_data, api, prompt_class, i):
    """
    Update the reward for the top k hypotheses.
    """
    num_wrong = 0
    for hypothesis in top_k_hypotheses:
        hypothesis_info = hypotheses[hypothesis]
        if useful_hypothesis(args, hypothesis, train_data, api, prompt_class, i):
            num_wrong += 1
            hypothesis_info.update_info_if_useful()
        else:
            hypothesis_info.update_info_if_not_useful()
        hypothesis_info.update_reward(args.alpha, num_example+1)
    return num_wrong


def generate_new_hypothesis(args, api, train_data, prompt_class, wrong_example_ids, j):
    """
    Generate new hypotheses for the wrong examples
    Keep the hypothesis with the highest accuracy on the examples
    """
    ids_in_batch = wrong_example_ids[j*args.num_examples_per_generation_prompt: (j+1)*args.num_examples_per_generation_prompt]
    example_data = {}
    for key in train_data:
        example_data[key] = [train_data[key][id] for id in ids_in_batch]

    prompt_input = prompt_class.batched_learning_hypothesis_generation(example_data, args.num_hypotheses_per_generation_prompt)
    response = api.generate(prompt_input)
    new_hypotheses = extract_hypotheses(args, response)

    max_acc = 0
    best_hypothesis = None
    for hypothesis in new_hypotheses:
        acc = get_accuracy(args, hypothesis, example_data, api, prompt_class)
        if acc > max_acc:
            max_acc = acc
            best_hypothesis = hypothesis

    return best_hypothesis, max_acc

    
def update_hypotheses(args, api, train_data, prompt_class, hypotheses):
    wrong_example_ids = []
    num_example = 0
    for epoch in range(args.num_epochs):
        for i in range(get_num_examples(train_data)):
            
            # logging and saving information
            print(f'*** Epoch {epoch+1}, Example {i+1} ***')
            print(f'num wrong examples: {len(wrong_example_ids)}')
            if num_example % args.save_every_n_examples == 0:
                with open(f'{args.output_folder}/hypotheses_{num_example}.pkl', 'wb') as f:
                    pickle.dump(hypotheses, f)

            # update the reward for the top k hypotheses
            top_k_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward, reverse=True)[:args.k]
            num_wrong = update_reward_for_top_k_hypotheses(args, top_k_hypotheses, hypotheses, num_example, train_data, api, prompt_class, i)
            print(f'num wrong hypotheses: {num_wrong}')
            
            # if more than half of the top k hypotheses are wrong, we add the example to the list of wrong examples
            if num_wrong > 0 or len(top_k_hypotheses) == 0:
                if i not in wrong_example_ids:
                    wrong_example_ids.append(i)

                    # if we have enough wrong examples, we generate new hypotheses and update the current hypotheses
                    if len(wrong_example_ids) == args.num_examples_per_generation_prompt*args.num_hypotheses_to_update:
                        new_hypotheses = {}
                        for j in range(args.num_hypotheses_to_update):
                            # generate new hypothesis for the wrong examples
                            best_hypothesis, max_acc = generate_new_hypothesis(args, api, train_data, prompt_class, wrong_example_ids, j)
                            new_hypotheses[best_hypothesis] = SummaryInformation(
                                acc=max_acc,
                                num_visits=args.num_examples_per_generation_prompt,
                                reward=max_acc + args.alpha * math.sqrt(math.log(num_example+1) / args.num_examples_per_generation_prompt)
                            )
                    
                        # reset wrong examples to be empty
                        wrong_example_ids = []

                        # update the hypotheses
                        hypotheses = update_hypotheses_with_new_hypotheses(hypotheses, new_hypotheses, args.max_num_hypotheses, args.num_hypotheses_to_update)

                        # log info
                        print('use wrong examples to update hypotheses: ')
                        print_hypotheses_info(hypotheses)

            num_example += 1

    return hypotheses


def parse_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'hotel_reviews',
                                                     'headline_binary',
                                                     'retweet',
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--verbose', action='store_true', help='Print more information.')
    parser.add_argument('--output_folder', type=str, default=None, help="Specifies the output path")

    # initialization specific arguments
    parser.add_argument('--num_init', type=int, default=25, help='Number of examples to use for initializing hypotheses.')
    parser.add_argument('--num_hypotheses_per_init_prompt', type=int, default=5, help='Number of hypotheses to generate per prompt.')
    parser.add_argument('--num_examples_per_init_prompt', type=int, default=25, help='Number of examples to use per prompt.')
    
    # generation specific arguments
    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--save_every_n_examples', type=int, default=100, help='Save hypothese every n examples.')
    parser.add_argument('--k', type=int, default=-1, help='Number of summaries to check for each new example.')
    parser.add_argument('--alpha', type=float, default=5e-1, help='Exploration parameter.')
    parser.add_argument('--max_num_hypotheses', type=int, default=20, help='Maximum number of hypotheses to keep.')
    parser.add_argument('--num_hypotheses_to_update', type=int, default=5, help='Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses.')
    parser.add_argument('--num_hypotheses_per_generation_prompt', type=int, default=5, help='Number of hypotheses to generate per prompt. Note that we only keep the one with the highest accuracy on the examples.') # Currently: generate five hypotheses and just keep the one with the highest accuracy
    parser.add_argument('--num_examples_per_generation_prompt', type=int, default=5, help='Number of examples to use per prompt.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs.')

    # inference specific arguments
    parser.add_argument('--num_test', type=str, default=100, help='Number of test examples.')

    # quick dev arguments
    parser.add_argument('--hotel_inference_prompt', type=str, default=None, help='Prompt for inference.')

    #headline specific arguments
    parser.add_argument('--headline_binary_inference', type=str, default="no_reason", help="Type of inference used.")

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = f'{code_repo_path}/outputs_{args.message}/{args.task}/{args.model}_{args.seed}_{args.hotel_inference_prompt}/'
    assert args.num_init <= args.num_train, f'Number of initialization examples ({args.num_init}) should be less than or equal to the number of training examples ({args.num_train}).'

    assert args.num_init == args.num_examples_per_init_prompt, 'Number of initialization examples should be equal to the number of examples per prompt.'

    return args


def print_hypotheses_info(hypotheses):
    # rank the hypotheses by reward
    # print hypotheses, reward, accuracy, and number of visits
    sorted_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward, reverse=True)
    for hypothesis in sorted_hypotheses:
        print(f'hypothesis: {hypothesis}, reward: {hypotheses[hypothesis].reward}, accuracy: {hypotheses[hypothesis].acc}, num_visits: {hypotheses[hypothesis].num_visits}')
    print()


def main():
    # set up tools
    start_time = time.time()
    args = parse_args()
    set_seed(args)
    train_data = get_train_data(args)
    api = LLMWrapper(args.model)
    prompt_class = PROMPT_DICT[args.task]()
    create_directory(args.output_folder)

    # divide train data into init and update
    train_data_init = {}
    train_data_update = {}
    for key in train_data:
        train_data_init[key] = train_data[key][:args.num_init]
        train_data_update[key] = train_data[key][args.num_init:]

    # initialize the hypotheses
    hypotheses_without_reward_info = initialize_hypotheses(args, api, train_data_init, prompt_class)
    hypotheses = {}
    for hypothesis in hypotheses_without_reward_info:
        acc = get_accuracy(args, hypothesis, train_data_init, api, prompt_class)
        if hypothesis not in hypotheses:
            hypotheses[hypothesis] = SummaryInformation(
                acc=acc,
                num_visits=args.num_init,
                reward=acc + args.alpha * math.sqrt(math.log(args.num_init) / args.num_init)
            )
    if len(hypotheses) == 0:
        print('No hypotheses were generated.')
        return
    
    print("# of initial hypotheses: ", len(hypotheses))

    print('initial hypotheses: ')
    print_hypotheses_info(hypotheses)

    # online learning to update the hypotheses
    hypotheses = update_hypotheses(args, api, train_data_update, prompt_class, hypotheses)

    print('final hypotheses: ')
    print_hypotheses_info(hypotheses)

    # save the final hypotheses
    with open(f'{args.output_folder}/hypotheses_final.pkl', 'wb') as f:
        pickle.dump(hypotheses, f)

    # print experiment info
    print(f'Total time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()