import datetime
import json
import logging
import os
from hypogenic.utils import set_seed, get_results, adjust_label
from hypogenic.tasks import BaseTask
from hypogenic.extract_label import extract_label_register
from hypogenic.LLM_wrapper import (
    GPTWrapper,
    LLMWrapper,
    LocalVllmWrapper,
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--model_path", type=str)
parser.add_argument("--do_train", action="store_true", default=False)
args = parser.parse_args()

# task_name = "admission/size_5"
cross_model_postfix = "hypogenic_and_paper"
multihyp = True
use_val = False
max_num_hypotheses = 10
num_init = 10
num_train = 20
num_test = 20
num_val = 200
k = 10
alpha = 5e-1
update_batch_size = 10
num_hypotheses_to_update = 1
update_hypotheses_per_batch = 10
save_every_10_examples = 10
init_batch_size = 10
init_hypotheses_per_batch = 10
cache_seed = None
temperature = 1e-5
max_tokens = 8000
use_refine = False
max_refine = 1
seed = 42
# seed = 3407
# seed = 998244353
# seed = 11376
# seed = 8271

SEEDS = [42]
def zero_shot_hyp(task_name, api, model_name):
    output_folder = (
        f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_zero_shot/"
    )
    os.makedirs(output_folder, exist_ok=True)
    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
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
        num_hypotheses_generate=init_hypotheses_per_batch,
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
        data_file=f"./literature/{task_name}/processed",
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
        data_file=f"./literature/{task_name}/processed",
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

def union_hypotheses(task_name, api, model_name, use_refine=True, prioritize='balanced'):

    union_postfix = "refine_and_paper" if use_refine else "hypogenic_and_paper"
    output_folder = f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}"

    os.makedirs(output_folder, exist_ok=True)

    os.makedirs(f"./results/{task_name}/{model_name}/dedup_data_only", exist_ok=True)
    os.makedirs(f"./results/{task_name}/{model_name}/dedup_paper_only", exist_ok=True)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
    )

    set_seed(seed)
    prompt_class = TestPrompt(task)

    if use_refine:
        data_hyp_file =  f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_final_seed_42_epoch_0.json"
    else:
        data_hyp_file =  f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_final_seed_42_epoch_0.json"

    with open(data_hyp_file) as f:
        hyp_dict = json.load(f)
    data_hyp_bank = {}
    for hypothesis in hyp_dict:
        data_hyp_bank[hypothesis] = SummaryInformation.from_dict(hyp_dict[hypothesis])

    paper_hyp_file = f"./results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_only_paper/hypotheses_training_sample_0_seed_42_epoch_0.json"
    
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

    with open(f"./results/{task_name}/{model_name}/dedup_data_only/hypotheses_training_sample_final_seed_42_epoch_0.json", 'w') as file:
        json.dump(data_dump_dict, file)

    with open(f"./results/{task_name}/{model_name}/dedup_paper_only/hypotheses_training_sample_final_seed_42_epoch_0.json", 'w') as file:
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

    with open(f'{output_folder}/hypotheses_training_sample_final_seed_42_epoch_0.json', 'w') as file:
        json.dump(union_dump_dict, file)

def get_res(filename: str, task_name, api, model_name, use_val=False, multihyp=False):
    logger = LoggerConfig.get_logger("Agent - get_res")

    set_seed(seed)

    task = BaseTask(
        config_path=f"./data/{task_name}/config.yaml",
        from_register=extract_label_register,
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
        
        pred_list = adjust_label(pred_list, label_list)
        results_dict = get_results(pred_list, label_list)
        f1 = results_dict["f1"]
        acc = results_dict["accuracy"]
        logger_str = "Results:\n"
        logger_str += f"Accuracy: {acc}\n"
        logger_str += f"F1: {f1}\n\n"
        logger.info(logger_str)
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
            pred_list = adjust_label(pred_list, label_list)
            results_dict = get_results(pred_list, label_list)
            hyp_list.append((hyp, results_dict["accuracy"], results_dict["f1"]))

        hyp_list = sorted(hyp_list, key=lambda x: x[2], reverse=True)
        logger_str = "Results:\n"
        for idx, (hyp, acc, f1) in enumerate(hyp_list):
            logger_str += f"{idx + 1}. {hyp}\n"
            logger_str += f"Accuracy: {acc}\n"
            logger_str += f"F1: {f1}\n\n"
        logger.info(logger_str)

        return hyp_list

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

    preds = adjust_label(preds, labels)

    results_dict = get_results(preds, labels)
    results_list.append((None, results_dict["accuracy"], results_dict["f1"]))
    return results_list


if __name__ == "__main__":

    model_name = args.model_name
    model_path = args.model_path
    model_type = args.model_type
    task_name = args.task_name
    DO_TRAIN = args.do_train

    task_folder = (
        f"./results/{task_name}/"
    )
    os.makedirs(task_folder, exist_ok=True)

    LoggerConfig.setup_logger(
        logging.INFO,
        f"results/{task_name}/{model_name}_seed_ALL_{datetime.datetime.now().strftime('%Y-%m-%d,%H-%M-%S')}.log",
    )

    logger = LoggerConfig.get_logger("Agent")


    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=model_path)

    logger.info(f"=-=-=-=-=-=-=-=-=-=-=-={model_name}=-=-=-=-=-=-=-=-=-=-=-=")
    logger.info(f"=-=-=-=-=-=-=-=-=-=-=-={task_name}=-=-=-=-=-=-=-=-=-=-=-=")
    # BASELINE
    logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed 42=-=-=-=-=-=-=-=-=-=-=-=")
    logger.info(
        baseline(
            0,
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            seed=42,
        )
    )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed 3407=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         0,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=3407,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed 998244353=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         0,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=998244353,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed 11376=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         0,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=11376,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot baseline, seed 8271=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         0,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=8271,
    #     )
    # )

    logger.info("=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed 42=-=-=-=-=-=-=-=-=-=-=-=")
    logger.info(
        baseline(
            3,
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            seed=42,
        )
    )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed 3407=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         3,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=3407,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed 998244353=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         3,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=998244353,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed 11376=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         3,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=11376,
    #     )
    # )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Few-shot baseline, seed 8271=-=-=-=-=-=-=-=-=-=-=-=")
    # logger.info(
    #     baseline(
    #         3,
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         seed=8271,
    #     )
    # )

    # ZERO SHOT
    logger.info("=-=-=-=-=-=-=-=-=-=-=-=Zero-shot generation=-=-=-=-=-=-=-=-=-=-=-=")
    if DO_TRAIN:
        zero_shot_hyp(task_name=task_name, api=api, model_name=model_name)
    for s in SEEDS:
        seed = s
        get_res(
            f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_zero_shot/hypotheses_training_sample_0_seed_42_epoch_0.json",
            task_name=task_name,
            api=api,
            model_name=model_name,
            use_val=use_val,
            multihyp=multihyp,
        )

    # # LITERATURE REVIEW ONLY PAPER
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Literature-only=-=-=-=-=-=-=-=-=-=-=-=")
    # if DO_TRAIN:
    #     only_paper(task_name=task_name, api=api, model_name=model_name)
    # for s in SEEDS:
    #     seed = s
    #     get_res(
    #         f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_only_paper/hypotheses_training_sample_0_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # # HyperWrite 
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=HyperWrite=-=-=-=-=-=-=-=-=-=-=-=")
    # for s in SEEDS:
    #     seed = s
    #     get_res(
    #         f"results/{task_name}/HyperWrite/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_0_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # # NotebookLM 
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=NotebookLM=-=-=-=-=-=-=-=-=-=-=-=")
    # for s in SEEDS:
    #     seed = s
    #     get_res(
    #         f"results/{task_name}/NotebookLM/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_0_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # HYPOGENIC
    logger.info("=-=-=-=-=-=-=-=-=-=-=-=Original HypoGeniC=-=-=-=-=-=-=-=-=-=-=-=")
    if DO_TRAIN:
        original_hypogenic(task_name=task_name, api=api, model_name=model_name)

    logger.info("=-=-=-=-=-=-=-=-=-=-=-=No Update=-=-=-=-=-=-=-=-=-=-=-=")
    get_res(
        f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_10_seed_42_epoch_0.json",
        task_name=task_name,
        api=api,
        model_name=model_name,
        use_val=use_val,
        multihyp=multihyp,
    )

    logger.info("=-=-=-=-=-=-=-=-=-=-=-=With Update=-=-=-=-=-=-=-=-=-=-=-=")
    for s in SEEDS:
        seed = s
        get_res(
        f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}/hypotheses_training_sample_final_seed_42_epoch_0.json",
        task_name=task_name,
        api=api,
        model_name=model_name,
        use_val=use_val,
        multihyp=multihyp,
        )

    # # LITERATURE REVIEW
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=HypoRefine=-=-=-=-=-=-=-=-=-=-=-=")
    # if DO_TRAIN:
    #     with_paper(task_name=task_name, api=api, model_name=model_name)
    
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=No Update=-=-=-=-=-=-=-=-=-=-=-=")
    # get_res(
    #     f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_10_seed_42_epoch_0.json",
    #     task_name=task_name,
    #     api=api,
    #     model_name=model_name,
    #     use_val=use_val,
    #     multihyp=multihyp,
    # )
    
    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=With Update=-=-=-=-=-=-=-=-=-=-=-=")
    # for s in SEEDS:
    #     seed = s    
    #     get_res(
    #         f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_with_paper/hypotheses_training_sample_final_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Union HypoGeniC and Paper=-=-=-=-=-=-=-=-=-=-=-=")
    # if DO_TRAIN:
    #     union_hypotheses(task_name=task_name, api=api, model_name=model_name, use_refine=False, prioritize='balanced')
    # union_postfix = "hypogenic_and_paper"
    # for s in SEEDS:
    #     seed = s
    #     get_res(
    #         f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}/hypotheses_training_sample_final_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # logger.info("=-=-=-=-=-=-=-=-=-=-=-=Union HypoRefine and Paper=-=-=-=-=-=-=-=-=-=-=-=")
    # if DO_TRAIN:
    #     union_hypotheses(task_name=task_name, api=api, model_name=model_name, use_refine=True, prioritize='balanced')
    # union_postfix = "refine_and_paper"
    # for s in SEEDS:
    #     seed = s
    #     get_res(
    #         f"results/{task_name}/{model_name}/hyp_{max_num_hypotheses}_{union_postfix}/hypotheses_training_sample_final_seed_42_epoch_0.json",
    #         task_name=task_name,
    #         api=api,
    #         model_name=model_name,
    #         use_val=use_val,
    #         multihyp=multihyp,
    #     )

    # if "gpt" in model_type:
    #     cross_model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # else:
    #     cross_model_name = "gpt-4o-mini"

    # task_name = "llamagc_detect"
    # logger.info(f"=-=-=-=-=-=-=-=-=-=-=-=Cross model {task_name}=-=-=-=-=-=-=-=-=-=-=-=")
    # get_res(
    #     f"results/{task_name}/{cross_model_name}/hyp_{max_num_hypotheses}_{cross_model_postfix}/hypotheses_training_sample_final_seed_42_epoch_0.json",
    #     task_name=task_name,
    #     api=api,
    #     model_name=model_name,
    #     use_val=use_val,
    #     multihyp=multihyp,
    # )
