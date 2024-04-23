from abc import ABC, abstractmethod
import os
code_repo_path = os.environ.get("CODE_REPO_PATH")
from summary_information import SummaryInformation
from tasks import TASKS
from utils import get_num_examples
from collections import OrderedDict
import numpy as np
import pulp
import random
import re

class Inference(ABC):
    """ Inference abstract class. For each style of inference implement the inference function. """
    def __init__(self, api, prompt_class, train_data):
        """ Initialize the inference class.

        Parameters
        _____________
        api: the LLM api wrapper
        prompt_class: the prompt class for the specified task
        _____________

        """
        super().__init__()
        self.api = api
        self.prompt_class = prompt_class
        self.train_data = train_data

    @abstractmethod
    def predict(self, args, data, index, hyp_bank):
        """Implements a specific type of prediction

        Parameters
        __________
        args: the arguments of the algorithm
        data: the specific dataset
        index: the specific index to predict for
        hyp_bank: a dictionary of hypotheses

        Returns
        __________
        prediction: the predicted value
        actual_label: the actual label of the sample
        """
        pass
    
    @abstractmethod
    def run_inference_final(self, args, data, hyp_bank):
        """Implements a specific type of prediction

        Parameters
        __________
        args: the arguments of the algorithm
        data: the specific dataset
        hyp_bank: a dictionary of hypotheses
        k: the number of hypotheses to use

        Returns
        __________
        accuracy: the accuracy over the dataset
        """
        pass

class DefaultInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def predict(self, args, data, index, hyp_bank):
        assert len(hyp_bank.keys()) == 1, 'default inference only supports one hypothesis at a time'
        prompt_input = self.prompt_class.inference(hyp_bank, data, index, prob=args.generate_prob)
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        response = self.api.generate(prompt_input, args.use_system_prompt)
        print(f"Response: {response}")
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        print(f"Prediction: {prediction}")
        actual_label = data['label'][index]
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label
    
    def run_inference_final(self, args, data, hyp_bank):
        top_hypothesis = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[0]
        num_samples = get_num_examples(data)

        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, {top_hypothesis : hyp_bank[top_hypothesis]})
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list


class MultiHypInference(DefaultInference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def concat_hypotheses(self, args, hyp_bank):
        k=args.k
        # get the top k hypotheses by reward (save as dictionary)
        if k > len(hyp_bank):
            k = len(hyp_bank)
        top_k_hypotheses = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[:k]
        multi_k_hypothesis = ""
        for hypothesis in top_k_hypotheses:
            multi_k_hypothesis += hypothesis + "\n"
        multi_k_hypothesis_class = SummaryInformation(hypothesis=multi_k_hypothesis)
        return {multi_k_hypothesis: multi_k_hypothesis_class}

    def run_inference_final(self, args, data, hyp_bank):
        top_hypothesis = self.concat_hypotheses(args,hyp_bank)
        num_samples = get_num_examples(data)

        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, top_hypothesis)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list
    

class ProbabilityInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def predict(self, args, data, index, hyp_bank):
        assert len(hyp_bank.keys()) == 1, 'default inference only supports one hypothesis at a time'
        prompt_input = self.prompt_class.inference(hyp_bank, data, index, prob=True)
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        response = self.api.generate(prompt_input, args.use_system_prompt)
        print(f"Response: {response}")
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        print(f"Prediction: {prediction}")
        import re
        pattern = r"Confidence: (\w+)}"
        match = re.search(pattern, response)
        if match:
            prob = match.group(1)
            print(f"Confidence: {prob}")
        else:
            print("Confidence output incorrect.")
            prob = None
        actual_label = data['label'][index]
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label, prob
    
    def run_inference_final(self, args, data, hyp_bank):
        top_hypothesis = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[0]
        num_samples = get_num_examples(data)

        pred_list = []
        label_list = []
        prob_list = []
        for i in range(num_samples):
            pred, label, prob = self.predict(args, data, i, {top_hypothesis : hyp_bank[top_hypothesis]})
            pred_list.append(pred)
            label_list.append(label)
            prob_list.append(prob)

        return pred_list, label_list, prob_list
    
    def run_two_side_inference(self, args, data, hyp_bank):
        pass

class KNNInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def predict(self, args, data, index, hyp_bank):
        prompt_input = self.prompt_class.knn_inference(hyp_bank, self.train_data, data, index)
        response = self.api.generate(prompt_input, args.use_system_prompt)
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        actual_label = data['label'][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def run_inference_final(self, args, data, hyp_bank):
        num_train_data_samples = get_num_examples(self.train_data)
        similarity_matrix, one_hot_encoded_dict = self.compute_similarity_matrix(hyp_bank, num_train_data_samples)
        assert list(one_hot_encoded_dict.keys()) == list(hyp_bank.keys()), "The keys of the one hot encoded dict and the hyp_bank should be the same"
        similarity_per_hypothesis = [np.sum(similarity_matrix[i]) for i, _ in enumerate(one_hot_encoded_dict.keys())]
        accuracy_per_hypothesis = [hyp_bank[hyp].acc for hyp in one_hot_encoded_dict] 
        print("Initial examples per hyp:")
        for hyp in hyp_bank:
            print(f"Hypothesis {hyp}, Examples: {hyp_bank[hyp].correct_examples}")
        
        print()
        print("One hot encoded dict:")
        for hyp in one_hot_encoded_dict:
            print(f"Hypothesis {hyp}, Encoded Examples: {one_hot_encoded_dict[hyp]}")
        print()
        print("Similarity matrix:\n", similarity_matrix, "\n")

        # choose hypotheses with the least similarities
        selected_indices = self.select_hypotheses_ilp(similarity_matrix, accuracy_per_hypothesis, similarity_per_hypothesis, args.knn_threshold)
        key_list = list(one_hot_encoded_dict.keys())
        selected_hypotheses = [key_list[idx] for idx in selected_indices]
        print("Selected hypotheses based upon non-similarity:", selected_hypotheses)

        top_k_hypotheses = sorted(selected_hypotheses, key=lambda x: hyp_bank[x].acc, reverse=True)[:args.knn_hypotheses]

        selected_hyp_bank = {}
        for hypothesis in top_k_hypotheses:
            selected_hyp_bank[hypothesis] = hyp_bank[hypothesis]
        for hyp in selected_hyp_bank:
            selected_hyp_bank[hyp].set_hypothesis(hyp)
            if len(selected_hyp_bank[hyp].correct_examples) > args.knn_num_examples:
                selected_hyp_bank[hyp].set_example(random.sample(selected_hyp_bank[hyp].correct_examples, args.knn_num_examples))

        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, selected_hyp_bank)
            pred_list.append(pred)
            label_list.append(label)
            
        return pred_list, label_list

    def compute_similarity_matrix(self, hyp_bank, num_train_data_samples):
        one_hot_encoded_dict = OrderedDict()

        for hypothesis in hyp_bank:
            indices = [ex[0] for ex in hyp_bank[hypothesis].correct_examples]
            result = [0] * num_train_data_samples  # Initialize array with zeros
            for idx in indices:
                result[idx] = 1  # Set elements at specified indices to 1
            one_hot_encoded_dict[hypothesis] = result
        
        similarity_matrix = np.zeros((len(hyp_bank), len(hyp_bank)))
        for i, hypothesis_one in enumerate(one_hot_encoded_dict.keys()):
            for j, hypothesis_two in enumerate(one_hot_encoded_dict.keys()):
                if hypothesis_one != hypothesis_two:
                    similarity_matrix[i][j] = np.dot(one_hot_encoded_dict[hypothesis_one], one_hot_encoded_dict[hypothesis_two])/(np.linalg.norm(one_hot_encoded_dict[hypothesis_one])*np.linalg.norm(one_hot_encoded_dict[hypothesis_two]))
        
        return similarity_matrix, one_hot_encoded_dict

    def select_hypotheses_ilp(self, similarity_matrix, accuracies, similarities, threshold):
        num_hypotheses = similarity_matrix.shape[0]
        problem = pulp.LpProblem("Hypothesis_Selection", pulp.LpMaximize)

        # Create a binary variable for each hypothesis, indicating whether it's selected
        selection_vars = [pulp.LpVariable(f'select_{i}', cat='Binary') for i in range(num_hypotheses)]

        # Objective: Maximize the number of training accuracy of selected hypotheses
        problem += pulp.lpSum([(selection_vars[i]*accuracies[i]) for i in range(num_hypotheses)])

        # Constraints: For each pair of hypotheses, if the similarity is above the threshold,
        # at least one hypothesis must not be selected.
        for i in range(num_hypotheses):
            for j in range(i+1, num_hypotheses):
                if similarity_matrix[i, j] >= threshold:
                    problem += selection_vars[i] + selection_vars[j] <= 1

        # Solve the problem
        problem.solve()

        # Get the indices of the selected hypotheses
        selected_indices = [i for i, var in enumerate(selection_vars) if var.value() == 1]

        return selected_indices


class BalancedExamplesKNNInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def predict(self, args, data, index, hyp_bank):
        prompt_input = self.prompt_class.knn_inference(hyp_bank, self.train_data, data, index)
        response = self.api.generate(prompt_input, args.use_system_prompt)
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        actual_label = data['label'][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label

    def run_inference_final(self, args, data, hyp_bank):
        knn_num_examples_per_class = args.knn_num_examples // 2 
        """
        # filter out hypotheses without enough examples
        for hyp in hyp_bank:
            if len(hyp_bank[hyp].correct_examples) < knn_num_examples:
                del hyp_bank[hyp]

        if len(hyp_bank) == 0:
            print("No hypotheses have enough examples")
            return 0
        """
        num_train_data_samples = get_num_examples(self.train_data)
        similarity_matrix, one_hot_encoded_dict = self.compute_similarity_matrix(hyp_bank, num_train_data_samples)
        assert list(one_hot_encoded_dict.keys()) == list(hyp_bank.keys()), "The keys of the one hot encoded dict and the hyp_bank should be the same"
        similarity_per_hypothesis = [np.sum(similarity_matrix[i]) for i, _ in enumerate(one_hot_encoded_dict.keys())]
        accuracy_per_hypothesis = [hyp_bank[hyp].acc for hyp in one_hot_encoded_dict] 
        print("Initial examples per hyp:")
        for hyp in hyp_bank:
            print(f"Hypothesis {hyp}, Examples: {hyp_bank[hyp].correct_examples}")
        
        print()
        print("One hot encoded dict:")
        for hyp in one_hot_encoded_dict:
            print(f"Hypothesis {hyp}, Encoded Examples: {one_hot_encoded_dict[hyp]}")
        print()
        print("Similarity matrix:\n", similarity_matrix, "\n")

        # choose hypotheses with the least similarities
        selected_indices = self.select_hypotheses_ilp(similarity_matrix, accuracy_per_hypothesis, similarity_per_hypothesis, args.knn_threshold)
        key_list = list(one_hot_encoded_dict.keys())
        selected_hypotheses = [key_list[idx] for idx in selected_indices]
        print("Selected hypotheses based upon non-similarity:", selected_hypotheses)
        
        top_k_hypotheses = sorted(selected_hypotheses, key=lambda x: hyp_bank[x].acc, reverse=True)[:args.knn_hypotheses]

        selected_hyp_bank = {}
        for hypothesis in top_k_hypotheses:
            selected_hyp_bank[hypothesis] = hyp_bank[hypothesis]
        to_del = []
        for hyp in selected_hyp_bank:
            selected_hyp_bank[hyp].set_hypothesis(hyp)
            # selected_hyp_bank[hyp].set_example(random.sample(selected_hyp_bank[hyp].correct_examples, args.knn_num_examples))

            # sample relatively balanced set of examples
            num_truthful = len([ex for ex in selected_hyp_bank[hyp].correct_examples if ex[1] == 'truthful'])
            num_deceptive = len([ex for ex in selected_hyp_bank[hyp].correct_examples if ex[1] == 'deceptive'])

            # filter out hypotheses without enough examples
            if num_truthful < knn_num_examples_per_class or num_deceptive < knn_num_examples_per_class:
                to_del.append(hyp)
                continue

            sampled_examples = random.sample([ex for ex in selected_hyp_bank[hyp].correct_examples if ex[1] == 'truthful'], knn_num_examples_per_class) + random.sample([ex for ex in selected_hyp_bank[hyp].correct_examples if ex[1] == 'deceptive'], knn_num_examples_per_class)
            random.shuffle(sampled_examples)
            assert len(sampled_examples) == knn_num_examples_per_class*2, "The number of examples should be equal to the number of examples per class"
            selected_hyp_bank[hyp].set_example(sampled_examples)

        # filter out hypotheses without enough examples
        for hyp in to_del:
            del selected_hyp_bank[hyp]

        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, selected_hyp_bank)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list


class FilterAndWeightInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
        
    def predict(self, args, data, index, hyp_bank):
        """
        Make prediction on one sample (index) of the dataset.
        Use the hypotheses in hyp_bank to make a weighted-vote prediction.

        Note this function may be called in generation as well. 
        Therefore, I only implement it to perform weighted-vote prediction (but not filtering).
        """
        assert len(hyp_bank.keys()) >= 1, 'Filter and weight inference requires at least one hypothesis'
        actual_label = data['label'][index]
        pred_dict = {}
        for hypothesis in hyp_bank:
            hypothesis_dict = {hypothesis: hyp_bank[hypothesis]}
            prompt_input = self.prompt_class.inference(hypothesis_dict, data, index)
            response = self.api.generate(prompt_input, args.use_system_prompt)
            task = TASKS[args.task]()
            pred = task.extract_label(response)
            weight = hyp_bank[hypothesis].acc
            if pred in pred_dict:
                pred_dict[pred] += weight
            else:
                pred_dict[pred] = weight
        prediction = max(pred_dict, key=pred_dict.get)

        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Predictions (weights): {pred_dict}")
        print(f"Prediction (final): {prediction}")
        print(f"Ground truth: {actual_label}")
        
        return prediction, actual_label

    def filter_hypotheses(self, args, data, index, hyp_bank):
        """
        Filter the hypotheses in hyp_bank to only include relevant hypotheses for the sample at index.

        Parameters
        __________
        data: the specific dataset
        index: the specific index to filter for
        hyp_bank: a dictionary of hypotheses

        Returns
        __________
        relevant_hypotheses: a dictionary of relevant hypotheses
        """
        relevant_hypotheses = {}
        for hypothesis in hyp_bank:
            prompt_input = self.prompt_class.is_relevant(hypothesis, data, index)
            response = self.api.generate(prompt_input, args.use_system_prompt)

            print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
            print(f"Response: {response}")
            
            # only keep the part after "Final answer:"
            if "Final answer:" in response:
                response = response[response.index("Final answer:") + len("Final answer:"):]
                response = response[:5]
                response = response.lower()

            print(f"Response (truncated): {response}")
            
            if 'yes' in response and 'no' in response:
                if 'yes or no' in response:
                    print(f"Hypothsis is not relevant")
                else:
                    raise ValueError(f'The response should not contain both "yes" and "no". Response: {response}')
            elif 'yes' in response:
                relevant_hypotheses[hypothesis] = hyp_bank[hypothesis]
                print('Hypothesis is relevant')
            else:
                print(f"Hypothsis is not relevant")
                
        return relevant_hypotheses

    def run_inference_final(self, args, data, hyp_bank):
        """
        Run over the entire dataset and make predictions.
        For each sample, prompt LLM to determine whether a hypothesis is relevant.
        Use the relevant hypotheses to make a weighted-vote prediction.
        """
        k = args.k
        # get the top k hypotheses by reward (save as dictionary)
        if k > len(hyp_bank):
            k = len(hyp_bank)
        top_hypotheses = {}
        for hypothesis in sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[:k]:
            top_hypotheses[hypothesis] = hyp_bank[hypothesis]

        # iterate over the dataset and make predictions
        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            filtered_hypotheses = self.filter_hypotheses(args, data, i, top_hypotheses)
            # if no hypothesis is relevant, use the hypothesis with the highest accuracy
            if len(filtered_hypotheses) == 0:
                best_hypothesis = max(top_hypotheses, key=lambda x: top_hypotheses[x].acc)
                filtered_hypotheses[best_hypothesis] = top_hypotheses[best_hypothesis]
            pred, label = self.predict(args, data, i, filtered_hypotheses)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list


class WeightedVoteInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def predict(self, args, data, index, hyp_bank):
        """
        Make prediction on one sample (index) of the dataset.
        Use the hypotheses in hyp_bank to make a weighted-vote prediction.

        Note this function may be called in generation as well. 
        Therefore, I only implement it to perform weighted-vote prediction (but not filtering).
        """
        assert len(hyp_bank.keys()) >= 1, 'Filter and weight inference requires at least one hypothesis'
        actual_label = data['label'][index]
        pred_dict = {}
        for hypothesis in hyp_bank:
            hypothesis_dict = {hypothesis: hyp_bank[hypothesis]}
            prompt_input = self.prompt_class.inference(hypothesis_dict, data, index)
            response = self.api.generate(prompt_input, args.use_system_prompt)
            task = TASKS[args.task]()
            pred = task.extract_label(response)
            weight = hyp_bank[hypothesis].acc
            if pred in pred_dict:
                pred_dict[pred] += weight
            else:
                pred_dict[pred] = weight
        prediction = max(pred_dict, key=pred_dict.get)

        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Predictions (weights): {pred_dict}")
        print(f"Prediction (final): {prediction}")
        print(f"Ground truth: {actual_label}")
        
        return prediction, actual_label
    
    def run_inference_final(self, args, data, hyp_bank):
        """
        Run over the entire dataset and make predictions.
        Use the top k hypotheses in hyp_bank to make a weighted-vote prediction.
        """
        k=args.k
        # get the top k hypotheses by reward (save as dictionary)
        if k > len(hyp_bank):
            k = len(hyp_bank)
        top_hypotheses = {}
        for hypothesis in sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[:k]:
            top_hypotheses[hypothesis] = hyp_bank[hypothesis]

        # iterate over the dataset and make predictions
        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, top_hypotheses)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list


class TestAllHypothesisInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def predict(self, args, data, index, hyp_bank):
        assert len(hyp_bank.keys()) == 1, 'default inference only supports one hypothesis at a time'
        prompt_input = self.prompt_class.inference(hyp_bank, data, index)
        response = self.api.generate(prompt_input, args.use_system_prompt)
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        actual_label = data['label'][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label
    
    def run_inference_final(self, args, data, hyp_bank):
        num_samples = get_num_examples(data)
        accuracy_list = []
        for hyp in hyp_bank:
            correct = 0
            for i in range(num_samples):
                pred, ground_truth = self.predict(args, data, i, {hyp : hyp_bank[hyp]})
                if pred == ground_truth:
                    correct += 1

            accuracy = correct / num_samples
            accuracy_list.append(accuracy)
            print(f"Hypothesis: {hyp}")
            print(f"Accuracy: {accuracy}")

        return sum(accuracy_list)/len(accuracy_list)

class TopKAndExamplesInference(Inference):
    """
    This class is essentially KNN inference without the hypotheses selection step
    """
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def predict(self, args, data, index, hyp_bank):
        prompt_input = self.prompt_class.knn_inference(hyp_bank, self.train_data, data, index)
        response = self.api.generate(prompt_input, args.use_system_prompt)
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        actual_label = data['label'][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label
    
    def run_inference_final(self, args, data, hyp_bank):
        k=args.k
        top_hypotheses = {}
        for hypothesis in sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)[:k]:
            top_hypotheses[hypothesis] = hyp_bank[hypothesis]

        for hyp in top_hypotheses:
            top_hypotheses[hyp].set_hypothesis(hyp)
            top_hypotheses[hyp].set_example(random.sample(top_hypotheses[hyp].correct_examples, args.knn_num_examples)) 

        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, top_hypotheses)
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list


class SeparateStepsKNNInference(KNNInference):
    """
    This class is essentially KNN inference with separate calls for
    selecting hypotheses and making predictions.
    """
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def default_predict(self, args, data, index, hyp_bank):
        assert len(hyp_bank.keys()) == 1, 'default inference only supports one hypothesis at a time'

        if args.add_examples:
            prompt_input = self.prompt_class.inference_with_examples(hyp_bank, self.train_data, data, index)
        else:
            prompt_input = self.prompt_class.inference(hyp_bank, data, index)
            
        response = self.api.generate(prompt_input, args.use_system_prompt)
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        actual_label = data['label'][index]
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        print(f"Response: {response}")
        print(f"Prediction: {prediction}")
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label
    
    def select_hypotheses(self, args, data, index, hyp_bank):
        prompt_input = self.prompt_class.knn_selection(hyp_bank, self.train_data, data, index, args)
        response = self.api.generate(prompt_input, args.use_system_prompt)

        print('Prompt:', prompt_input[0], prompt_input[1])
        print('Response:', response)
        
        hyp_idx = re.search(r'Chosen Pattern:\s*Pattern\s*(\d+)', response)

        if hyp_idx == None:
            print(f"Could not find chosen hypothesis in response: {response}\n\nHyp_bank: {hyp_bank.keys()}")
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            print(f'Use Hypothesis: {hyp}')
            return hyp

        hyp_idx = hyp_idx.group(1)
        hyp_idx = hyp_idx.strip()
        hyp_idx = int(hyp_idx)-1

        if hyp_idx >= len(list(hyp_bank.items())):
            print(f"No hypothesis chosen, return to default.")
            # return hyp with highest acc
            hyp = max(hyp_bank, key=lambda x: hyp_bank[x].acc)
            print(f'Use Hypothesis: {hyp}')
            return hyp
        
        print(f'Extracted Hypothesis Index: {hyp_idx}')
        items = list(hyp_bank.items())
        hyp = items[hyp_idx][0]
        print(f'Extracted Hypothesis: {hyp}')

        return hyp
    
    def predict(self, args, data, index, hyp_bank):
        # select one hypothesis that is most relevant to the sample
        hyp = self.select_hypotheses(args, data, index, hyp_bank)
 
        # make prediction using default_predict
        return self.default_predict(args, data, index, {hyp: hyp_bank[hyp]})


class DefaultInferenceFavorExploitation(DefaultInference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def run_inference_final(self, args, data, hyp_bank):
        # get top hypothesis in terms of accuracy - (reward - accuracy)
        tmp_hyp_bank = {}
        for hyp in hyp_bank:
            if hyp_bank[hyp].num_visits >= 200:
                tmp_hyp_bank[hyp] = hyp_bank[hyp].acc - (hyp_bank[hyp].reward - hyp_bank[hyp].acc)
        top_hypothesis = max(tmp_hyp_bank, key=tmp_hyp_bank.get)

        num_samples = get_num_examples(data)
        pred_list = []
        label_list = []
        for i in range(num_samples):
            pred, label = self.predict(args, data, i, {top_hypothesis : hyp_bank[top_hypothesis]})
            pred_list.append(pred)
            label_list.append(label)

        return pred_list, label_list

class CoverageBasedSelectionInference(Inference):
    
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)

    def predict(self, args, data, index, hyp_bank):
        '''
        TODO: decide how to implement this, could try one-step KNN, two-step KNN, or FAW
        '''
        pass

    def get_coverage(self, args, hyp):
        num_samples = get_num_examples(self.train_data)
        coverage = [0] * num_samples
        for i in range(num_samples):
            prompt_input = self.prompt_class.is_relevant(hyp, self.train_data, i)
            response = self.api.generate(prompt_input, args.use_system_prompt)

            print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
            print(f"Response: {response}")
            
            # only keep the part after "Final answer:"
            if "Final answer:" in response:
                response = response[response.index("Final answer:") + len("Final answer:"):]
                response = response[:5]
                response = response.lower()

            print(f"Response (truncated): {response}")
            
            if 'yes' in response and 'no' in response:
                if 'yes or no' in response:
                    print(f"Hypothsis is not relevant")
                else:
                    raise ValueError(f'The response should not contain both "yes" and "no". Response: {response}')
            elif 'yes' in response:
                coverage[i] = 1
                print('Hypothesis is relevant')
            else:
                print(f"Hypothsis is not relevant")
        return coverage

    def get_recall(self, args, hyp, hyp_bank):
        num_samples = get_num_examples(self.train_data)
        recall = [0] * num_samples
        for i in range(num_samples):
            hypothesis_dict = {hyp: hyp_bank[hyp]}
            prompt_input = self.prompt_class.inference(hypothesis_dict, self.train_data, i)
            response = self.api.generate(prompt_input, args.use_system_prompt)
            task = TASKS[args.task]()
            pred = task.extract_label(response)
            if pred == self.train_data['label'][i]:
                recall[i] = 1
        return recall


    def run_inference_final(self, args, data, hyp_bank):
        # get the coverage and recall of each hypothesis
        # key = hypothesis, value = (coverage, recall), where coverage and recall are one hot lists
        coverage_recall = {}
        for hyp in hyp_bank:
            coverage = self.get_coverage(args, hyp)
            recall = self.get_recall(args, hyp, hyp_bank)
            coverage_recall[hyp] = (coverage, recall)

        print(coverage_recall)

        # compute similarity of hypotheses

        # integer programming

        pass

class UpperboundInference(Inference):
    def __init__(self, api, prompt_class, train_data):
        super().__init__(api, prompt_class, train_data)
    
    def predict(self, args, data, index, hyp_bank):
        assert len(hyp_bank.keys()) == 1, 'default inference only supports one hypothesis at a time'
        prompt_input = self.prompt_class.inference(hyp_bank, data, index)
        print(f"Prompt: {prompt_input[0]}\n{prompt_input[1]}\n")
        response = self.api.generate(prompt_input, args.use_system_prompt)
        print(f"Response: {response}")
        task = TASKS[args.task]()
        prediction = task.extract_label(response)
        print(f"Prediction: {prediction}")
        actual_label = data['label'][index]
        print(f"Ground truth: {actual_label}")
        return prediction, actual_label
    
    def run_inference_final(self, args, data, hyp_bank):
        # sort hyp_bank by training accuracy from high to low
        hyp_bank = {k: v for k, v in sorted(hyp_bank.items(), key=lambda item: item[1].acc, reverse=True)}
        # keep top args.k hypotheses
        hyp_bank = {k: v for k, v in list(hyp_bank.items())[:args.k]}

        # run inference for each hypothesis
        num_samples = get_num_examples(data)
        pred_list = {hyp: [] for hyp in hyp_bank}
        label_list = []
        count = 1
        for hyp in hyp_bank:
            print(f'The {count}th hypothesis')
            for i in range(num_samples):
                pred, label = self.predict(args, data, i, {hyp : hyp_bank[hyp]})
                pred_list[hyp].append(pred)
                label_list.append(label)
            count += 1

        # compute accuracy for each hypothesis
        correct_list = {hyp: [] for hyp in hyp_bank}
        accuracy_list = {hyp: 0 for hyp in hyp_bank}
        for hyp in hyp_bank:
            for i in range(num_samples):
                if pred_list[hyp][i] == label_list[i]:
                    correct_list[hyp].append(1)
                else:
                    correct_list[hyp].append(0)
            accuracy_list[hyp] = sum(correct_list[hyp]) / num_samples

        # print the correctness of each hypothesis (in matrix form)
        print('Correctness:')
        for hyp in hyp_bank:
            print(f"{correct_list[hyp]}")

        # print accuracy for each hypothesis
        for hyp in hyp_bank:
            print(f"Hypothesis: {hyp}, Accuracy: {accuracy_list[hyp]}")

        # count as correct if one of the hypotheses is correct
        correct = 0
        for i in range(num_samples):
            for hyp in hyp_bank:
                if pred_list[hyp][i] == label_list[i]:
                    correct += 1
                    break
        accuracy = correct / num_samples
        print(f"Upperbound accuracy (if one hyp is correct): {accuracy}")

        return pred_list, label_list
    
INFERENCE_DICT = {
    'default': DefaultInference,
    'multi_k': MultiHypInference,
    'knn': KNNInference,
    'filter_and_weight': FilterAndWeightInference,
    'weighted_vote': WeightedVoteInference,
    'top_k_and_examples': TopKAndExamplesInference,
    'knn_balanced_examples': BalancedExamplesKNNInference,
    'knn_separate_steps': SeparateStepsKNNInference,
    'weighted_vote': WeightedVoteInference,
    'test_all' : TestAllHypothesisInference,
    'probability': ProbabilityInference,
    'default_favor_exploitation': DefaultInferenceFavorExploitation,
    'coverage_based_selection': CoverageBasedSelectionInference,
    'upperbound': UpperboundInference,
}