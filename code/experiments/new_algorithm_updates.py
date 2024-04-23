import argparse
import time
import pickle
import sys
import os
import math
import json

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')
from prompt import PROMPT_DICT
from data_loader import get_train_data
from utils import LLMWrapper, SummaryInformation, set_seed, create_directory, get_num_examples, extract_label, GPT_MODELS, VALID_MODELS
from hypotheses_initialization import initialize_hypotheses, extract_hypotheses

PROMPT_NAME_DICT = {
    'shoe': 'appearance',
    'sst': 'sentence',
    'original_sst': 'sentence',
    'binary_sst' : 'sentence',
    'binary_original_sst': 'sentence',
    'hotel_reviews': 'review',
    'headline_binary': 'headline'
}


def inference(args, hypothesis, train_data, api, prompt_class, example_index):
    """
    Run hypothesis-based inference. Returns the prediction and the label.
    """
    if args.task == 'shoe':
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
        if args.with_reasoning:
            prompt_input = prompt_class.new_inference_with_reasoning(hypothesis, train_data, example_index)
        else:
            prompt_input = prompt_class.new_inference_without_reasoning(hypothesis, train_data, example_index)
            
    else:
        raise ValueError('Invalid task argument.')

    response = api.generate(prompt_input)
    response = response.lower()
    print(response)
    pred = extract_label(args.task, response.split('answer')[-1])
    label = train_data['label'][example_index]        
            
    if args.verbose:
        print(f'In `useful_hypothesis` function: Example {example_index}: hypothesis "{hypothesis}"\n prediction "{pred}" label "{label}"')

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
    ex = []
    for i in range(get_num_examples(train_data)):
        pred, label = inference(args, hypothesis, train_data, api, prompt_class, i)
        if pred == label:
            correct += 1
            x = train_data[PROMPT_NAME_DICT[args.task]][i]
            x.append(label)
            ex.append(x)
    return correct / get_num_examples(train_data), ex


def update_hypotheses_with_new_hypotheses(hypotheses, new_hypotheses, max_num_hypotheses, num_hypotheses_to_update):
    """
    Update the hypotheses with new hypotheses.
    If the number of hypotheses is less than the maximum number of hypotheses, we add the new hypotheses to the list of hypotheses.
    If the number of hypotheses is equal to the maximum number of hypotheses, we update the hypotheses by replacing the lowest-ranking hypotheses with the new ones. In particular we replace `num_hypotheses_to_update`` lowest-ranking hypotheses with the new ones.
    """
    # check number of non-overlapping hypotheses
    num_true_new_hypotheses = 0
    true_new_hypotheses = {}
    for hypothesis in new_hypotheses:
        if hypothesis not in hypotheses:
            num_true_new_hypotheses += 1
            true_new_hypotheses[hypothesis] = new_hypotheses[hypothesis]
    total_num_hypotheses = len(hypotheses) + num_true_new_hypotheses
    
    if len(hypotheses) < max_num_hypotheses:
        if total_num_hypotheses <= max_num_hypotheses:
            for hypothesis in true_new_hypotheses:
                hypotheses[hypothesis] = true_new_hypotheses[hypothesis]
        else:
            # remove the lowest-ranking hypotheses to make space for the new ones
            sorted_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward)
            for i in range(total_num_hypotheses - max_num_hypotheses):
                hypothesis_to_remove = sorted_hypotheses[i]
                del hypotheses[hypothesis_to_remove]
            # add the new hypotheses
            for hypothesis in true_new_hypotheses:
                hypotheses[hypothesis] = true_new_hypotheses[hypothesis]

    elif len(hypotheses) == max_num_hypotheses:
        # sort hypotheses by reward (in ascending order)
        sorted_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward)
        # remove the lowest-ranking hypotheses
        for i in range(num_hypotheses_to_update):
            hypothesis_to_remove = sorted_hypotheses[i]
            del hypotheses[hypothesis_to_remove]
        # add the new hypotheses
        for hypothesis in true_new_hypotheses:
            hypotheses[hypothesis] = true_new_hypotheses[hypothesis]
    else:
        raise ValueError('Number of hypotheses exceeds the maximum number of hypotheses.')
    
    return hypotheses

def save_to_json(args, training_sample, hypotheses):
    temp_dict = {}
    for hypothesis in hypotheses.keys():
        serialized_dict = hypotheses[hypothesis].__dict__
        temp_dict[hypothesis] = serialized_dict
    
    json_string = json.dumps(temp_dict)
    with open(f'{args.output_folder}/hypotheses_training_sample_{training_sample}.json', 'w') as f:
        f.write(json_string)



def update_hypotheses(args, api, train_data, prompt_class, hypotheses):
    wrong_example_ids = []
    num_train_examples = get_num_examples(train_data)
    for i in range(num_train_examples):
        if i % args.save_every_n_examples == 0:
            with open(f'{args.output_folder}/hypotheses_{i}.pkl', 'wb') as f:
                pickle.dump(hypotheses, f)
            save_to_json(args, i, hypotheses)
            
        # take the top k hypotheses
        top_k_hypotheses = sorted(hypotheses, key=lambda x: hypotheses[x].reward, reverse=True)[:args.k]

        # update the reward for the top k hypotheses
        is_wrong = True
        for hypothesis in top_k_hypotheses:
            hypothesis_info = hypotheses[hypothesis]
            if useful_hypothesis(args, hypothesis, train_data, api, prompt_class, i):
                is_wrong = False
                label = train_data['label'][i]
                hypothesis_info.update_info_if_useful(i)
                hypothesis_info.updated_useful_examples(train_data[PROMPT_NAME_DICT[args.task]][i], label)
            else:
                hypothesis_info.update_info_if_not_useful(i)
            hypothesis_info.update_reward(args.alpha, i+1)
            
        if is_wrong:
            wrong_example_ids.append(i)
            if len(wrong_example_ids) == args.num_examples_per_generation_prompt*args.num_hypotheses_to_update:
                new_hypotheses = {}
                for j in range(args.num_hypotheses_to_update):
                    # generate new hypotheses for the wrong examples
                    ids_in_batch = wrong_example_ids[j*args.num_examples_per_generation_prompt: (j+1)*args.num_examples_per_generation_prompt]
                    example_data = {}
                    for key in train_data:
                        example_data[key] = [train_data[key][id] for id in ids_in_batch]

                    prompt_input = prompt_class.batched_learning_hypothesis_generation(example_data, args.num_hypotheses_per_generation_prompt)
                    response = api.generate(prompt_input)
                    extracted_hypotheses = extract_hypotheses(args, response)
                    # keep the hypothesis with the highest accuracy on the examples
                    max_acc = 0
                    best_hypothesis = None
                    best_ex = []
                    for hypothesis in extracted_hypotheses:
                        acc, ex = get_accuracy(args, hypothesis, example_data, api, prompt_class)
                        if acc > max_acc:
                            max_acc = acc
                            best_hypothesis = hypothesis
                            best_ex = ex

                    # add the best hypothesis to the list of new hypotheses
                    new_hypotheses[best_hypothesis] = SummaryInformation(
                        acc=max_acc,
                        num_visits=args.num_examples_per_generation_prompt,
                        reward=max_acc + args.alpha * math.sqrt(math.log(i+1) / args.num_examples_per_generation_prompt),
                        correct_examples=best_ex
                    )
            
                # reset wrong examples to be empty
                wrong_example_ids = []

                # update the hypotheses
                hypotheses = update_hypotheses_with_new_hypotheses(hypotheses, new_hypotheses, args.max_num_hypotheses, args.num_hypotheses_to_update)

    return hypotheses


def parse_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'hotel_reviews',
                                                     'headline_binary'
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--verbose', action='store_true', help='Print more information.')

    # initialization specific arguments
    parser.add_argument('--num_init', type=int, default=25, help='Number of examples to use for initializing hypotheses.')
    parser.add_argument('--num_hypotheses_per_init_prompt', type=int, default=5, help='Number of hypotheses to generate per prompt.')
    parser.add_argument('--num_examples_per_init_prompt', type=int, default=5, help='Number of examples to use per prompt.')

    # generation specific arguments
    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--save_every_n_examples', type=int, default=100, help='Save hypothese every n examples.')
    parser.add_argument('--k', type=int, default=-1, help='Number of summaries to check for each new example.')
    parser.add_argument('--alpha', type=float, default=5e-1, help='Exploration parameter.')
    parser.add_argument('--max_num_hypotheses', type=int, default=20, help='Maximum number of hypotheses to keep.')
    parser.add_argument('--num_hypotheses_to_update', type=int, default=5, help='Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses.')
    parser.add_argument('--num_hypotheses_per_generation_prompt', type=int, default=5, help='Number of hypotheses to generate per prompt. Note that we only keep the one with the highest accuracy on the examples.') # Currently: generate five hypotheses and just keep the one with the highest accuracy
    parser.add_argument('--num_examples_per_generation_prompt', type=int, default=5, help='Number of examples to use per prompt.')

    # inference specific arguments
    parser.add_argument('--num_test', type=str, default=100, help='Number of test examples.')

    # quick dev arguments
    parser.add_argument('--hotel_inference_prompt', type=str, default=None, help='Prompt for inference.')

    #headline specific arguments
    parser.add_argument('--with_reasoning', type=bool, default=False, help="Type of inference used.")

    parser.add_argument('--output_folder', type=str, default=None, help="Specifies the output path")
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = f'{code_repo_path}/print_log_{args.message}/{args.task}/{args.model}_{args.seed}_{args.hotel_inference_prompt}/'
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Directory '{args.output_folder}' created.")

    assert args.num_init <= args.num_train, f'Number of initialization examples ({args.num_init}) should be less than or equal to the number of training examples ({args.num_train}).'

    return args


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
        acc, ex = get_accuracy(args, hypothesis, train_data_init, api, prompt_class)
        if hypothesis not in hypotheses:
            hypotheses[hypothesis] = SummaryInformation(
                acc=acc,
                num_visits=args.num_init,
                reward=acc + args.alpha * math.sqrt(math.log(args.num_init) / args.num_init),
                correct_examples=ex
            )
    print("# of initial hypotheses: ", len(hypotheses))

    # online learning to update the hypotheses
    hypotheses = update_hypotheses(args, api, train_data_update, prompt_class, hypotheses)

    # save the final hypotheses
    with open(f'{args.output_folder}/hypotheses_final.txt', 'wb') as f:
        pickle.dump(hypotheses, f)

    save_to_json(args, "final", hypotheses)

    # print experiment info
    print(f'Total time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()