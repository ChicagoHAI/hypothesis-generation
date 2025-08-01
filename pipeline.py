import datetime
import json
import logging
import os
from hypogenic.utils import set_seed, get_results
from hypogenic.tasks import BaseTask
from hypogenic.extract_label import extract_label_register
from hypogenic.LLM_wrapper import (
    LLMWrapper,
    llm_wrapper_register,
)
from hypogenic.algorithm.update import DefaultUpdate
from hypogenic.algorithm.replace import DefaultReplace, Replace
from hypogenic.algorithm.inference import Inference, DefaultInference
from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.prompt import BasePrompt
from hypogenic.logger_config import LoggerConfig
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)
import pandas as pd
import yaml

from hypothesis_agent.data_analysis_agent.generation import (
    TestGeneration,
    OnlyPaperGeneration,
    ZeroShotGeneration,
)
from hypothesis_agent.data_analysis_agent.inference import MultiHypDefaultInference
from hypothesis_agent.data_analysis_agent.update import TestUpdate
from hypothesis_agent.literature_review_agent import LiteratureAgent
from hypothesis_agent.literature_review_agent.literature_processor.extract_info import (
    BaseExtractor,
    WholeExtractor,
)
from hypothesis_agent.literature_review_agent.literature_processor.summarize import (
    BaseSummarize,
    LLMSummarize,
)
from hypothesis_agent.data_analysis_agent.utils import multiple_hypotheses_remove_repetition
from hypothesis_agent.data_analysis_agent.prompt import TestPrompt


from IO_prompting.prompt import IOPrompt
from IO_prompting.update import IOUpdate
from IO_prompting.generation import IOGeneration

import argparse

parser = argparse.ArgumentParser()

# Required base arguments
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--literature_folder", type=str)

# This is needed for local models
parser.add_argument("--model_path", type=str)
parser.add_argument("--do_train", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging")

# Method toggle arguments
parser.add_argument("--run_zero_shot", action="store_true", help="Run zero-shot baseline")
parser.add_argument("--run_few_shot", action="store_true", help="Run few-shot baseline")
parser.add_argument("--run_zero_shot_gen", action="store_true", help="Run zero-shot generation")
parser.add_argument("--run_only_paper", action="store_true", help="Run literature-only")
parser.add_argument("--run_hyperwrite", action="store_true", help="Run HyperWrite")
parser.add_argument("--run_notebooklm", action="store_true", help="Run NotebookLM")
parser.add_argument("--run_hypogenic", action="store_true", help="Run original HypoGeniC")
parser.add_argument("--run_hyporefine", action="store_true", help="Run HypoRefine")
parser.add_argument("--run_union_hypo", action="store_true", help="Run Union HypoGeniC and Paper")
parser.add_argument("--run_union_refine", action="store_true", help="Run Union HypoRefine and Paper")
parser.add_argument("--run_cross_model", action="store_true", help="Run cross-model evaluation")
parser.add_argument("--run_io_refine", action="store_true", help="Run IO iterative refinement")
parser.add_argument("--cross_model_name", type=str, help="Name of the cross model")
parser.add_argument("--cross_hyp_folder", type=str, help="Folder containing cross model hypotheses")

# All algorithm-related arguments
parser.add_argument("--multihyp", action="store_true", default=True)
parser.add_argument("--use_val", action="store_true", default=False)
parser.add_argument("--max_num_hypotheses", type=int, default=10)
parser.add_argument("--num_init", type=int, default=10)
parser.add_argument("--num_train", type=int, default=20)
parser.add_argument("--num_test", type=int, default=20)
parser.add_argument("--num_val", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--alpha", type=float, default=5e-1)
parser.add_argument("--update_batch_size", type=int, default=10)
parser.add_argument("--num_hypotheses_to_update", type=int, default=1)
parser.add_argument("--update_hypotheses_per_batch", type=int, default=10)
parser.add_argument("--save_every_10_examples", type=int, default=10)
parser.add_argument("--init_batch_size", type=int, default=10)
parser.add_argument("--init_hypotheses_per_batch", type=int, default=10)
parser.add_argument("--cache_seed", type=int, default=None)
parser.add_argument("--temperature", type=float, default=1e-5)
parser.add_argument("--max_tokens", type=int, default=8000)
parser.add_argument("--use_refine", action="store_true", default=False)
parser.add_argument("--max_refine", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_ood", action="store_true", default=False, help="Use out-of-distribution data for testing")

args = parser.parse_args()

multihyp = args.multihyp
use_val = args.use_val
use_ood = args.use_ood
max_num_hypotheses = args.max_num_hypotheses
num_init = args.num_init
num_train = args.num_train
num_test = args.num_test
num_val = args.num_val
k = args.k
alpha = args.alpha
update_batch_size = args.update_batch_size
num_hypotheses_to_update = args.num_hypotheses_to_update
update_hypotheses_per_batch = args.update_hypotheses_per_batch
save_every_10_examples = args.save_every_10_examples
init_batch_size = args.init_batch_size
init_hypotheses_per_batch = args.init_hypotheses_per_batch
cache_seed = args.cache_seed
temperature = args.temperature
max_tokens = args.max_tokens
use_refine = args.use_refine
max_refine = args.max_refine
seed = args.seed

if args.literature_folder is None:
    literature_folder = args.task_name
else:
    literature_folder = args.literature_folder

def zero_shot_hyp(task_name, api, model_name):
    output_folder = (
        f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_zero_shot/"
    )
    os.makedirs(output_folder, exist_ok=True)
    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed=seed)

    prompt_class = TestPrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)

    generation_class = ZeroShotGeneration(
        api=api,
        prompt_class=prompt_class,
        inference_class=inference_class,
        task=task,
    )

    update_class = TestUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(max_num_hypotheses),
        save_path=output_folder,
        num_init=num_init,
        k=k,
        alpha=alpha,
        update_batch_size=update_batch_size,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        num_hypotheses_to_update=num_hypotheses_to_update,
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = generation_class.initialize_hypotheses_0_shot(
        num_hypotheses_generate=max_num_hypotheses,
        cache_seed=cache_seed,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=64,
    )
    hypotheses_bank = {hyp: SummaryInformation() for hyp in hypotheses_bank}
    update_class.save_to_json(hypotheses_bank, sample=0, seed=seed, epoch=0)


def only_paper(task_name, api, model_name):
    output_folder = (
        f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_only_paper/"
    )
    os.makedirs(output_folder, exist_ok=True)
    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed=seed)

    prompt_class = TestPrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    summarize_class = LLMSummarize(
        extractor=WholeExtractor(), api=api, prompt_class=prompt_class
    )

    literature_agent = LiteratureAgent(
        api=api,
        prompt_class=prompt_class,
        summizer=summarize_class,
    )
    literature_agent.summarize_papers(
        data_file=f"./literature/{literature_folder}/processed",
        cache_seed=cache_seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    literature_agent.save_paper_infos(os.path.join(output_folder, "paper_infos.json"))

    generation_class = OnlyPaperGeneration(
        api=api,
        prompt_class=prompt_class,
        inference_class=inference_class,
        task=task,
        literature_agent=literature_agent,
    )

    update_class = TestUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(max_num_hypotheses),
        save_path=output_folder,
        num_init=num_init,
        k=k,
        alpha=alpha,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = generation_class.initialize_hypotheses_only_paper(
        num_hypotheses_generate=init_hypotheses_per_batch,
        cache_seed=cache_seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    hypotheses_bank = {hyp: SummaryInformation() for hyp in hypotheses_bank}
    update_class.save_to_json(hypotheses_bank, sample=0, seed=seed, epoch=0)


def with_paper(task_name, api, model_name):
    output_folder = (
        f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/"
    )
    os.makedirs(output_folder, exist_ok=True)
    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed=seed)

    prompt_class = TestPrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    summarize_class = LLMSummarize(
        extractor=WholeExtractor(), api=api, prompt_class=prompt_class
    )

    literature_agent = LiteratureAgent(
        api=api,
        prompt_class=prompt_class,
        summizer=summarize_class,
    )
    literature_agent.summarize_papers(
        data_file=f"./literature/{literature_folder}/processed",
        cache_seed=cache_seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    literature_agent.save_paper_infos(os.path.join(output_folder, "paper_infos.json"))

    generation_class = TestGeneration(
        api=api,
        prompt_class=prompt_class,
        inference_class=inference_class,
        task=task,
        literature_agent=literature_agent,
        max_refine=max_refine,
    )

    update_class = TestUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(max_num_hypotheses),
        save_path=output_folder,
        num_init=num_init,
        k=k,
        alpha=alpha,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = update_class.batched_initialize_hypotheses_with_paper(
        num_init=num_init,
        init_batch_size=init_batch_size,
        init_hypotheses_per_batch=init_hypotheses_per_batch,
        cache_seed=cache_seed,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=64,
    )
    update_class.save_to_json(hypotheses_bank, sample=num_init, seed=seed, epoch=0)

    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=seed,
            cache_seed=cache_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=64,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=seed,
            epoch=epoch,
        )


def original_hypogenic(task_name, api, model_name):
    output_folder = f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/"

    os.makedirs(output_folder, exist_ok=True)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    set_seed(seed)
    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
    prompt_class = BasePrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = DefaultGeneration(api, prompt_class, inference_class, task)

    update_class = DefaultUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(max_num_hypotheses),
        save_path=output_folder,
        num_init=num_init,
        k=k,
        alpha=alpha,
        update_batch_size=update_batch_size,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        num_hypotheses_to_update=num_hypotheses_to_update,
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = {}
    hypotheses_bank = update_class.batched_initialize_hypotheses(
        num_init,
        init_batch_size=init_batch_size,
        init_hypotheses_per_batch=init_hypotheses_per_batch,
        cache_seed=cache_seed,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=64,
    )
    update_class.save_to_json(
        hypotheses_bank,
        sample=num_init,
        seed=seed,
        epoch=0,
    )
    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=seed,
            cache_seed=cache_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=64,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=seed,
            epoch=epoch,
        )

def IO_iterative_refinement(task_name, api, model_name):
    output_folder = f"./results/{task_name}/{model_name}/IO_refinement/"

    os.makedirs(output_folder, exist_ok=True)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    set_seed(seed)
    train_data, _, _ = task.get_data(10, num_test, num_val, seed)
    prompt_class = IOPrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = IOGeneration(api, prompt_class, inference_class, task)

    update_class = IOUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(max_num_hypotheses),
        save_path=output_folder,
        num_init=10,
        k=k,
        alpha=alpha,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = {}
    # Use the Qiu et al. 2024 paper hyperparameters
    hypotheses_bank = update_class.batched_initialize_hypotheses(
        num_init=10,
        init_batch_size=num_init,
        init_hypotheses_per_batch=5,
        cache_seed=cache_seed,
        temperature=0.7,
        max_tokens=max_tokens,
        max_concurrent=64,
    )
    # only keep the hypothesis with the highest accuracy
    # sorted_hypotheses = sorted(
    #     hypotheses_bank, key=lambda x: hypotheses_bank[x].acc, reverse=True
    # )
    # hypotheses_bank = {
    #     sorted_hypotheses[0]: hypotheses_bank[sorted_hypotheses[0]]
    # }
    update_class.save_to_json(
        hypotheses_bank,
        sample="init",
        seed=seed,
        epoch=0,
    )
    for epoch in range(3):
        # if there exist a hypothesis with accuracy 1.0, stop the training
        if any(hypotheses_bank[h].acc == 1.0 for h in hypotheses_bank):
            update_class.save_to_json(
                hypotheses_bank,
                sample="final",
                seed=seed,
                epoch=2,
            )
            break
        # Else, iteratively refine
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=seed,
            cache_seed=cache_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=64,
        )
        # sorted_hypotheses = sorted(
        #     hypotheses_bank, key=lambda x: hypotheses_bank[x].acc, reverse=True
        # )
        # hypotheses_bank = {
        #     sorted_hypotheses[0]: hypotheses_bank[sorted_hypotheses[0]]
        # }
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=seed,
            epoch=epoch,
        )

def union_hypotheses(task_name, api, model_name, use_refine=True, prioritize='balanced'):

    union_postfix = "refine_and_paper" if use_refine else "hypogenic_and_paper"
    output_folder = f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}"

    os.makedirs(output_folder, exist_ok=True)

    os.makedirs(f"./results/{task_name}/{model_name}/dedup_data_only", exist_ok=True)
    os.makedirs(f"./results/{task_name}/{model_name}/dedup_paper_only", exist_ok=True)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    set_seed(seed)
    prompt_class = TestPrompt(task)

    if use_refine:
        data_hyp_file =  f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_final_seed_{seed}_epoch_0.json"
    else:
        data_hyp_file =  f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json"

    with open(data_hyp_file) as f:
        hyp_dict = json.load(f)
    data_hyp_bank = {}
    for hypothesis in hyp_dict:
        data_hyp_bank[hypothesis] = SummaryInformation.from_dict(hyp_dict[hypothesis])

    paper_hyp_file = f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_only_paper/hypotheses_training_sample_0_seed_{seed}_epoch_0.json"
    
    with open(paper_hyp_file) as f:
        hyp_dict = json.load(f)
    paper_hyp_bank = {}
    for hypothesis in hyp_dict:
        paper_hyp_bank[hypothesis] = SummaryInformation.from_dict(hyp_dict[hypothesis])


    unique_data_hyp_bank = multiple_hypotheses_remove_repetition(
        prompt_class,
        api,
        data_hyp_bank,
        cache_seed,
        max_concurrent=64,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    unique_paper_hyp_bank = multiple_hypotheses_remove_repetition(
        prompt_class,
        api,
        paper_hyp_bank,
        cache_seed,
        max_concurrent=64,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    data_dump_dict = {}
    for hyp in unique_data_hyp_bank:
        data_dump_dict[hyp] = {"hypothesis": hyp, "acc": unique_data_hyp_bank[hyp].acc}

    paper_dump_dict = {}
    for hyp in unique_paper_hyp_bank:
        paper_dump_dict[hyp] = {"hypothesis": hyp, "acc": unique_paper_hyp_bank[hyp].acc}

    with open(f"./results/{task_name}/{model_name}/dedup_data_only/hypotheses_training_sample_final_seed_{seed}_epoch_0.json", 'w') as file:
        json.dump(data_dump_dict, file)

    with open(f"./results/{task_name}/{model_name}/dedup_paper_only/hypotheses_training_sample_final_seed_{seed}_epoch_0.json", 'w') as file:
        json.dump(paper_dump_dict, file)

    unique_data_hyp_list = sorted(unique_data_hyp_bank, key=lambda x: unique_data_hyp_bank[x].acc, reverse=True)
    unique_paper_hyp_list = sorted(unique_paper_hyp_bank, key=lambda x: unique_paper_hyp_bank[x].acc, reverse=True)
    union_hyp_bank = {}

    if prioritize == "data":
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
        print("num from data: ", len(union_hyp_bank))
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
    elif prioritize == "paper":
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
        print("num from paper: ", len(union_hyp_bank))
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
    else:
        while len(union_hyp_bank) < max_num_hypotheses // 2 and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
        print("num from data: ", len(union_hyp_bank))
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
    
    union_dump_dict = {}
    for hyp in union_hyp_bank:
        union_dump_dict[hyp] = {"hypothesis": hyp, "acc": union_hyp_bank[hyp].acc}

    with open(f'{output_folder}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json', 'w') as file:
        json.dump(union_dump_dict, file)

def get_res(filename: str, task_name, api, model_name, use_val=False, multihyp=False):
    logger = LoggerConfig.get_logger("Agent - get_res")

    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )

    train_data, test_data, val_data = task.get_data(num_train, num_test, num_val, seed)
    if use_val:
        test_data = val_data

    prompt_class = TestPrompt(task)
    

    with open(filename) as f:
        hyp_dict = json.load(f)
    hyp_bank = {}
    for hypothesis in hyp_dict:
        hyp_bank[hypothesis] = SummaryInformation.from_dict(hyp_dict[hypothesis])


    if multihyp:
        inference_class = MultiHypDefaultInference(api, prompt_class, train_data, task)
        pred_list, label_list = inference_class.run_inference_final(
                test_data,
                hyp_bank,
                cache_seed=cache_seed,
                max_concurrent=64,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        results_dict = get_results(pred_list, label_list)
        f1 = results_dict["f1"]
        acc = results_dict["accuracy"]
        logger_str = "Results:\n"
        logger_str += f"Accuracy: {acc}\n"
        logger_str += f"F1: {f1}\n\n"
        logger.info(logger_str)
        return results_dict  # Return results dictionary
    else:
        inference_class = DefaultInference(api, prompt_class, train_data, task)

        hyp_list = []

        for idx, hyp in enumerate(hyp_bank):
            pred_list, label_list = inference_class.run_inference_final(
                test_data,
                {hyp: hyp_bank[hyp]},
                cache_seed=cache_seed,
                max_concurrent=64,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results_dict = get_results(pred_list, label_list)
            hyp_list.append((hyp, results_dict["accuracy"], results_dict["f1"]))

        hyp_list = sorted(hyp_list, key=lambda x: x[2], reverse=True)
        logger_str = "Results:\n"
        for idx, (hyp, acc, f1) in enumerate(hyp_list):
            logger_str += f"{idx + 1}. {hyp}\n"
            logger_str += f"Train Accuracy: {hyp_bank[hyp].acc}\n"
            logger_str += f"Test Accuracy: {acc}\n"
            logger_str += f"Test F1: {f1}\n\n"
        logger.info(logger_str)

        # Format results in a more structured way
        formatted_results = {
            "hypotheses": [
                {
                    "hypothesis": hyp,
                    "train_accuracy": hyp_bank[hyp].acc,
                    "test_accuracy": acc,
                    "test_f1": f1
                } for hyp, acc, f1 in hyp_list
            ],
            "best": {
                "hypothesis": hyp_list[0][0],
                "test_accuracy": hyp_list[0][1],
                "test_f1": hyp_list[0][2]
            } if hyp_list else {}
        }
        return formatted_results

def baseline(few_shot_k, task_name, api, model_name, seed=42, use_val=False):
    def few_shot(
        api: LLMWrapper,
        train_data,
        test_data,
        prompt_class: BasePrompt,
        task,
        few_shot_k,
        cache_seed,
    ):
        """
        Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
        """
        results = []
        prompt_inputs = [
            prompt_class.few_shot_baseline(
                train_data.reset_index(drop=True), few_shot_k, test_data, i
            )
            for i in range(len(test_data))
        ]

        # print(
        #     yaml.dump(
        #         prompt_inputs[0],
        #         default_flow_style=False,
        #         sort_keys=False,
        #         allow_unicode=True,
        #         Dumper=yaml.SafeDumper,
        #     )
        # )

        responses = api.batched_generate(
            prompt_inputs, cache_seed=cache_seed, max_concurrent=64, max_tokens=4000
        )
        for i in range(len(test_data)):
            pred = task.extract_label(responses[i])
            label = test_data[task.label_name][i]

            results.append(
                {
                    "prompt": prompt_inputs[i],
                    "response": responses[i],
                    "label": label,
                    "pred": pred,
                }
            )

        return results

    def preprocess(train_data, k):
        data = []

        label_nunique = train_data[task.label_name].nunique()
        label_unique = train_data[task.label_name].unique()
        for i in range(k):
            data.append(
                train_data[train_data[task.label_name] == label_unique[i % label_nunique]].iloc[
                    i // label_nunique
                ]
            )

        return pd.DataFrame(data)

    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
        use_ood=use_ood
    )
    prompt_class = BasePrompt(task)
    results_list = []
    train_data, test_data, val_data = task.get_data(num_train, num_test, num_val, seed)
    if use_val:
        test_data = val_data

    if few_shot_k > 0:
        train_data = preprocess(train_data, few_shot_k)

    results = few_shot(
        api, train_data, test_data, prompt_class, task, few_shot_k, cache_seed
    )

    labels = [result["label"] for result in results]
    preds = [result["pred"] for result in results]

    results_dict = get_results(preds, labels)
    results_list.append((None, results_dict["accuracy"], results_dict["f1"]))
    
    # Format results in a structured way
    formatted_results = {
        "few_shot_k": few_shot_k,
        "accuracy": results_dict["accuracy"],
        "f1": results_dict["f1"],
        "precision": results_dict.get("precision", None),
        "recall": results_dict.get("recall", None)
    }
    
    return formatted_results

def save_method_results(method_name, results, task_name, model_name, seed=42, timestamp=None, use_ood=False):
    """Save results for a specific evaluation method to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = f"./results/{task_name}/{model_name}/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Add OOD suffix if using out-of-distribution data
    data_type = "OOD" if use_ood else "IND"
    filename = f"{results_dir}/{method_name}_{data_type}_seed_{seed}.json"
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    result_data = {
        "method": method_name,
        "task": task_name,
        "model": model_name,
        "data_type": data_type,
        "seed": seed,
        "timestamp": current_timestamp, 
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return filename

def combine_results(task_name, model_name, seed=42):
    """Combine all evaluation results into aggregated IND and OOD summary files."""
    results_dir = f"./results/{task_name}/{model_name}/evaluation_results"
    if not os.path.exists(results_dir):
        return None

    # Initialize combined results for IND and OOD
    combined_results = {
        "IND": {
            "task": task_name,
            "model": model_name,
            "data_type": "IND",
            "seed": seed,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "methods": {}
        },
        "OOD": {
            "task": task_name,
            "model": model_name,
            "data_type": "OOD",
            "seed": seed,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "methods": {}
        }
    }

    # Scan for all result files in the directory
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    for filename in result_files:
        with open(os.path.join(results_dir, filename), 'r') as f:
            data = json.load(f)
            method_name = data["method"]
            data_type = data["data_type"]  # Either "IND" or "OOD"
            combined_results[data_type]["methods"][method_name] = data["results"]

    # Save combined IND results
    ind_file = f"./results/{task_name}/{model_name}/combined_results_IND_seed_{seed}.json"
    with open(ind_file, 'w') as f:
        json.dump(combined_results["IND"], f, indent=2)

    # Save combined OOD results
    ood_file = f"./results/{task_name}/{model_name}/combined_results_OOD_seed_{seed}.json"
    with open(ood_file, 'w') as f:
        json.dump(combined_results["OOD"], f, indent=2)

    return ind_file, ood_file

def log_arguments(logger, args):
    """Log all configuration parameters in an organized way."""
    sections = {
        "Run Configuration": [
            ("Model Type", args.model_type),
            ("Model Name", args.model_name),
            ("Model Path", args.model_path),
            ("Task Name", args.task_name),
            ("Do Train", args.do_train),
            ("Use OOD", args.use_ood)
        ],
        "Method Configuration": [
            ("Run Zero Shot", args.run_zero_shot),
            ("Run Few Shot", args.run_few_shot),
            ("Run Zero Shot Gen", args.run_zero_shot_gen),
            ("Run Only Paper", args.run_only_paper),
            ("Run HyperWrite", args.run_hyperwrite),
            ("Run NotebookLM", args.run_notebooklm),
            ("Run HypoGeniC", args.run_hypogenic),
            ("Run HypoRefine", args.run_hyporefine),
            ("Run Union HypoGeniC", args.run_union_hypo),
            ("Run Union HypoRefine", args.run_union_refine),
            ("Run Cross Model", args.run_cross_model),
            ("Run IO Refine", args.run_io_refine)
        ],
        "Algorithm Configuration": [
            ("Multi Hypothesis", multihyp),
            ("Use Validation", use_val),
            ("Max Num Hypotheses", max_num_hypotheses),
            ("Num Init", num_init),
            ("Num Train", num_train),
            ("Num Test", num_test),
            ("Num Val", num_val),
            ("K", k),
            ("Alpha", alpha),
            ("Update Batch Size", update_batch_size),
            ("Num Hypotheses to Update", num_hypotheses_to_update),
            ("Update Hypotheses Per Batch", update_hypotheses_per_batch),
            ("Save Every N Examples", save_every_10_examples),
            ("Init Batch Size", init_batch_size),
            ("Init Hypotheses Per Batch", init_hypotheses_per_batch),
            ("Cache Seed", cache_seed),
            ("Temperature", temperature),
            ("Max Tokens", max_tokens),
            ("Use Refine", use_refine),
            ("Max Refine", max_refine),
            ("Seed", seed)
        ]
    }
    
    for section, params in sections.items():
        logger.info(f"\n=== {section} ===")
        for name, value in params:
            logger.info(f"{name}: {value}")
    logger.info("=====================\n")

if __name__ == "__main__":

    model_name = args.model_name
    model_path = args.model_path
    model_type = args.model_type
    task_name = args.task_name
    DO_TRAIN = args.do_train

    task_folder = (
        f"./results/{task_name}/"
    )
    model_folder = (
        f"./results/{task_name}/{model_name}/"
    )

    os.makedirs(task_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    
    # Set logger level based on debug flag
    logger_level = logging.DEBUG if args.debug else logging.INFO
    
    LoggerConfig.setup_logger(
        logger_level,
        f"results/{task_name}/{model_name}_seed_{seed}_{datetime.datetime.now().strftime('%Y-%m-%d,%H-%M-%S')}.log",
    )

    logger = LoggerConfig.get_logger("Agent")
    log_arguments(logger, args)
    
    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=model_path)
    methods_run = []  # Track which methods were executed
    
    logger.info(f"=-=-=-=-=-=-=-=-=-=-=-={model_name}=-=-=-=-=-=-=-=-=-=-=-=")
    logger.info(f"=-=-=-=-=-=-=-=-=-=-=-={task_name}=-=-=-=-=-=-=-=-=-=-=-=")

    # Modify the execution flow to use toggle arguments and save results
    if args.run_zero_shot:
        method_name = "zero_shot_baseline"
        methods_run.append(method_name)
        logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed {seed}=-=-=-=-=-=-=-=-=-=-=-=")
        results = baseline(
            0,
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            seed=seed,
        )
        logger.info(results)
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_few_shot:
        method_name = "few_shot_baseline"
        methods_run.append(method_name)
        logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed {seed}=-=-=-=-=-=-=-=-=-=-=-=")
        results = baseline(
            3,
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            seed=seed,
        )
        logger.info(results)
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_zero_shot_gen:
        method_name = "zero_shot_gen"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot generation=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            zero_shot_hyp(task_name=task_name, api=api, model_name=model_name)
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_zero_shot/hypotheses_training_sample_0_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_only_paper:
        method_name = "literature_only"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=Literature-only=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            only_paper(task_name=task_name, api=api, model_name=model_name)
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_only_paper/hypotheses_training_sample_0_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_hypogenic:
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=Original HypoGeniC=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            original_hypogenic(task_name=task_name, api=api, model_name=model_name)
        
        method_name = "hypogenic_no_update"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=No Update=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_10_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

        method_name = "hypogenic"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=With Update=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_hyporefine:
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=HypoRefine=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            with_paper(task_name=task_name, api=api, model_name=model_name)
        
        method_name = "hyporefine_no_update"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=No Update=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_10_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)
        
        method_name = "hyporefine"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=With Update=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_final_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_union_hypo:
        method_name = "union_hypogenic_paper"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=Union HypoGeniC and Paper=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            union_hypotheses(task_name=task_name, api=api, model_name=model_name, use_refine=False, prioritize='balanced')
        union_postfix = "hypogenic_and_paper"
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_union_refine:
        method_name = "union_hyporefine_paper"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=Union HypoRefine and Paper=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            union_hypotheses(task_name=task_name, api=api, model_name=model_name, use_refine=True, prioritize='balanced')
        union_postfix = "refine_and_paper"
        results = get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    if args.run_cross_model:
        cross_model_name = args.cross_model_name
        cross_hyp_folder = args.cross_hyp_folder
        cross_model_prefix = cross_model_name.split('/')[0]
        inf_model_prefix = model_name.split('/')[0]
        method_name = f"{cross_model_prefix}_gen_{inf_model_prefix}_inf"
        methods_run.append(method_name)
         
        if cross_model_name and cross_hyp_folder:
            logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Cross model {task_name}=-=-=-=-=-=-=-=-=-=-=-=")
            logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Generation model {cross_model_name}=-=-=-=-=-=-=-=-=-=-=-=")
            logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Inference model {model_name}=-=-=-=-=-=-=-=-=-=-=-=")
            results = get_res(
                f"results/{task_name}/{cross_model_name}/{cross_hyp_folder}/hypotheses_training_sample_final_seed_{seed}_epoch_0.json",
                task_name=task_name,
                api=api,
                model_name=model_name,
                use_val=use_val,
                multihyp=multihyp,
            )
            save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)
        else:
            logger.warning("Cross model name or hypothesis folder not provided. Skipping cross model evaluation.")

    if args.run_io_refine:
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=IO Iterative Refinement=-=-=-=-=-=-=-=-=-=-=-=")
        if DO_TRAIN:
            IO_iterative_refinement(task_name=task_name, api=api, model_name=model_name)

        method_name = "io_prompting"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=IO Prompting=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/IO_refinement/hypotheses_training_sample_init_seed_{seed}_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=False,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)
        
        method_name = "io_refinement"
        methods_run.append(method_name)
        logger.info("=-=-=-=-=-=-=-=-=-=-=-=IO Iterative refinement=-=-=-=-=-=-=-=-=-=-=-=")
        results = get_res(
            f"results/{task_name}/{model_name}/IO_refinement/hypotheses_training_sample_final_seed_{seed}_epoch_2.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=False,
        )
        save_method_results(method_name, results, task_name, model_name, seed, use_ood=use_ood)

    # Combine all results into a single summary file
    if methods_run:
        combined_files = combine_results(task_name, model_name, seed)
        if combined_files:
            logger.info(f"Combined IND results saved to: {combined_files[0]}")
            logger.info(f"Combined OOD results saved to: {combined_files[1]}")

    # Log total cost of the run
    if model_type == 'gpt':
        total_cost = api.get_cost()
        logger.info(f"Total cost: {total_cost}")
        
        # Save cost information to a separate file, using seed and data type
        data_type = "OOD" if use_ood else "IND"
        cost_file = f"./results/{task_name}/{model_name}/cost_{data_type}_seed_{seed}.json"
        with open(cost_file, 'w') as f:
            json.dump({
                "model": model_name,
                "task": task_name,
                "data_type": data_type,
                "seed": seed,
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "total_cost_usd": total_cost
            }, f, indent=2)

