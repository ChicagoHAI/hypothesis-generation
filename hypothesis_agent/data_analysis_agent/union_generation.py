import argparse
import logging
import re
import time
import pickle
import sys

import os
import math
import json

import random
from typing import Callable, Tuple, Union, List, Dict, Any
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypogenic.extract_label import extract_label_register

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LocalVllmWrapper, LLMWrapper, GPTWrapper
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)

from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)
from hypogenic.algorithm.replace import DefaultReplace, Replace
from hypogenic.algorithm.update import SamplingUpdate, DefaultUpdate
from hypogenic.logger_config import LoggerConfig
from hypogenic.algorithm.summary_information import SummaryInformation

from hypothesis_agent.data_analysis_agent.generation import TestGeneration, OnlyPaperGeneration, MultiHypGenerationWithRank
from hypothesis_agent.data_analysis_agent.update import TestUpdate, MultiHypUpdate
from hypothesis_agent.data_analysis_agent.inference import MultiHypInferenceWithRank
from hypothesis_agent.literature_review_agent.literature_review import LiteratureAgent
from hypothesis_agent.literature_review_agent.literature_processor.extract_info import BaseExtractor
from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.utils import multiple_hypotheses_remove_repetition
from hypothesis_agent.data_analysis_agent.task import TestTask

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("Agent - Union Generation")

def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def generate_paper_only(
    task: BaseTask,
    prompt_class: TestPrompt,
    literature_agent: LiteratureAgent,
    api,
    train_data,
    n_specificity_round = 0,
    task_name = "deceptive_review",
    output_folder = "./tmp",
    output_filename = "paper_only.json",
    max_num_hypotheses = 20,
    seed = 42,
    cache_seed = None,
    max_concurrent = 32,
    temperature = 1e-5,
    max_tokens = 4000,
    **generate_kwargs,
) -> Dict[str, SummaryInformation]:
    set_seed(seed)
    if literature_agent.paper_infos == []:
        raise ValueError("Error when loading paper summaries")
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = OnlyPaperGeneration(api, prompt_class, inference_class, task, literature_agent)
    if n_specificity_round > 0:
        hypotheses_list = generation_class.initialize_hypotheses_only_paper_with_specificity_boost(
            max_num_hypotheses,
            n_specificity_round,
            cache_seed,
            max_concurrent,
            max_tokens,
            **generate_kwargs,
        )
    else:
        hypotheses_list = generation_class.initialize_hypotheses_only_paper(
            num_hypotheses_generate=max_num_hypotheses,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
        )
    hyp_dict = {}

    for hyp in hypotheses_list:
        hyp_dict[hyp] = SummaryInformation(hypothesis=hyp, acc=0.0)
    
    dump_dict = {}
    for hyp in hypotheses_list:
        dump_dict[hyp] = {"hypothesis": hyp, "acc": 0.0}
    
    with open(output_folder + output_filename, 'w') as file:
        json.dump(dump_dict, file)
    
    return hyp_dict

def generate_original_hypogenic(
    task: BaseTask,
    prompt_class: TestPrompt,
    literature_agent: LiteratureAgent,
    api,
    train_data,
    task_name = "deceptive_review",
    output_folder = "./tmp",
    max_num_hypotheses = 20,
    old_hypothesis_file = None,
    num_init = 10,
    seed = 42,
    k = 10,
    alpha= 5e-1,
    update_batch_size = 10,
    num_hypotheses_to_update = 1,
    save_every_10_examples = 10,
    init_batch_size = 10,
    init_hypotheses_per_batch = 10,
    cache_seed = None,
    max_concurrent = 32,
    temperature = 1e-5,
    max_tokens = 4000,
    **generate_kwargs,
) -> Dict[str, SummaryInformation]:
    set_seed(seed)
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
    if old_hypothesis_file is None:
        hypotheses_bank = update_class.batched_initialize_hypotheses(
            num_init,
            init_batch_size=init_batch_size,
            init_hypotheses_per_batch=init_hypotheses_per_batch,
            cache_seed=cache_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample=num_init,
            seed=seed,
            epoch=0,
        )
    else:
        dict = load_dict(old_hypothesis_file)
        for hypothesis in dict:
            hypotheses_bank[hypothesis] = SummaryInformation.from_dict(
                dict[hypothesis]
            )
    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=seed,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=seed,
            epoch=epoch,
        )
    return hypotheses_bank

def generate_init_both_multi_refine(
    task: BaseTask,
    inference_task: BaseTask,
    prompt_class: TestPrompt,
    literature_agent: LiteratureAgent,
    api,
    train_data,
    task_name = "deceptive_review",
    output_folder = "./tmp",
    max_num_hypotheses = 20,
    old_hypothesis_file = None,
    num_init = 10,
    seed = 42,
    k = 10,
    alpha = 5e-1,
    beta = 5e-1,
    update_batch_size = 10,
    num_hypotheses_to_update = 1,
    save_every_10_examples = 10,
    init_batch_size = 10,
    init_hypotheses_per_batch = 10,
    max_refine = 6,
    cache_seed = None,
    max_concurrent = 32,
    temperature = 1e-5,
    max_tokens = 4000,
    **generate_kwargs,
) -> Dict[str, SummaryInformation]:
    set_seed(seed)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = TestGeneration(api, prompt_class, inference_class, task, literature_agent)
    generation_class.set_max_refine(max_refine)
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
        save_every_n_examples=save_every_10_examples,
    )

    hypotheses_bank = {}
    if old_hypothesis_file is None:
        hypotheses_bank = update_class.batched_initialize_hypotheses_with_paper(
            num_init,
            init_batch_size=init_batch_size,
            init_hypotheses_per_batch=init_hypotheses_per_batch,
            cache_seed=cache_seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample=num_init,
            seed=seed,
            epoch=0,
        )
    else:
        dict = load_dict(old_hypothesis_file)
        for hypothesis in dict:
            hypotheses_bank[hypothesis] = SummaryInformation.from_dict(
                dict[hypothesis]
            )
    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=seed,
            cache_seed=cache_seed,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=seed,
            epoch=epoch,
        )
    return hypotheses_bank

def union_hypogenic_and_paper( # this is actually for hypogenic/refine + paper, set use_refine to True if want to use refine
    task: BaseTask,
    prompt_class: TestPrompt,
    literature_agent: LiteratureAgent,
    extractor: BaseExtractor,
    api,
    train_data,
    config_path = None,
    prioritize = "data", # choices = ["data", "paper", "balanced"]
    use_refine = True, # if True, union of refine + paper-only, otherwise hypogenic + paper-only
    old_data_based_hyp_bank: Dict[str, SummaryInformation] = None,
    n_paper_specificity_boost = 0, # n_round for specificity booster
    model_name = "gpt-4o-mini",
    papers_dir_path = None,
    task_name = "headline_binary",
    custom_dump_path = None,
    max_num_hypotheses = 20,
    old_hypothesis_file = None,
    num_init = 10,
    seed = 42,
    k = 10,
    alpha= 5e-1,
    update_batch_size = 10,
    num_hypotheses_to_update = 1,
    save_every_10_examples = 10,
    init_batch_size = 10,
    init_hypotheses_per_batch = 10,
    max_refine = 6,
    cache_seed = None,
    max_concurrent = 32,
    temperature = 1e-5,
    max_tokens = 4000,
    **generate_kwargs,
) -> Dict[str, SummaryInformation]:
    if literature_agent.paper_infos == [] or literature_agent.paper_infos == None:
        raise ValueError("Error when loading paper summaries")
    
    if n_paper_specificity_boost > 0:
        if use_refine:
            data_output_folder = f"./outputs/union/init_both_multi_refine/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/specificity_boost_paper/"
        else:
            data_output_folder = f"./outputs/union/original_hypogenic/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/specificity_boost_paper/"
        paper_output_folder = f"./outputs/union/init_paper_no_refine/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/specificity_boost_paper/"
        union_output_folder = f"./outputs/union/union/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/specificity_boost_paper/"
    else:
        if use_refine:
            data_output_folder = f"./outputs/union/init_both_multi_refine/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/"
        else:
            data_output_folder = f"./outputs/union/original_hypogenic/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/"
        paper_output_folder = f"./outputs/union/init_paper_no_refine/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/"
        union_output_folder = f"./outputs/union/union/{task_name}/{model_name}/hyp_{max_num_hypotheses}/refine_{max_refine}/"
    os.makedirs(data_output_folder, exist_ok=True)
    os.makedirs(paper_output_folder, exist_ok=True)
    os.makedirs(union_output_folder, exist_ok=True)
    if use_refine:
        logger.info(f"Output folder for hypotheses generated with HypoRefine: {data_output_folder}")
        logger.info(f"Output folder for hypotheses generated with literature information only: {paper_output_folder}")
        logger.info(f"Output folder for hypotheses generated with Literature U HypoRefine: {union_output_folder}")
    else:
        logger.info(f"Output folder for hypotheses generated with HypoGeniC: {data_output_folder}")
        logger.info(f"Output folder for hypotheses generated with literature information only: {paper_output_folder}")
        logger.info(f"Output folder for hypotheses generated with Literature U HypoGeniC: {union_output_folder}")
    if use_refine:
        # logger.info(f"union of hypothesis_agent (refine={max_refine}) and paper-only")
        union_filename = f"union_prioritize_{prioritize}_refine_{max_refine}.json"
    else:
        # logger.info(f"union of hypogenic and paper-only")
        union_filename = f"union_prioritize_{prioritize}.json"
    
    inference_task = task
    if custom_dump_path == None:
        union_dump_path = union_output_folder + union_filename
    else:
        union_dump_path = custom_dump_path
    data_max_num_hypotheses = max_num_hypotheses
    paper_max_num_hypotheses = max_num_hypotheses

    if old_data_based_hyp_bank is not None:
        data_hyp_bank = old_data_based_hyp_bank
    else:
        if use_refine:
            data_hyp_bank = generate_init_both_multi_refine(
                task=task,
                inference_task=inference_task,
                prompt_class=prompt_class,
                literature_agent=literature_agent,
                api=api,
                train_data=train_data,
                task_name=task_name,
                output_folder=data_output_folder,
                max_num_hypotheses=data_max_num_hypotheses,
                num_init=num_init,
                seed=seed,
                k=k,
                alpha=alpha,
                update_batch_size=update_batch_size,
                num_hypotheses_to_update=num_hypotheses_to_update,
                save_every_10_examples=save_every_10_examples,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                max_refine=max_refine,
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                temperature=temperature,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
        else:
            data_hyp_bank = generate_original_hypogenic(
                task=task,
                prompt_class=prompt_class,
                literature_agent=literature_agent,
                api=api,
                train_data=train_data,
                task_name=task_name,
                output_folder=data_output_folder,
                max_num_hypotheses=data_max_num_hypotheses,
                num_init=num_init,
                seed=seed,
                k=k,
                alpha=alpha,
                update_batch_size=update_batch_size,
                num_hypotheses_to_update=num_hypotheses_to_update,
                save_every_10_examples=save_every_10_examples,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                cache_seed=cache_seed,
                max_concurrent=max_concurrent,
                temperature=temperature,
                max_tokens=max_tokens,
                **generate_kwargs,
            )
    paper_hyp_bank = generate_paper_only(
        task=task,
        prompt_class=prompt_class,
        literature_agent=literature_agent,
        api=api,
        train_data=train_data,
        n_specificity_round=n_paper_specificity_boost,
        task_name=task_name,
        output_folder=paper_output_folder,
        max_num_hypotheses=paper_max_num_hypotheses,
        seed=seed,
        cache_seed=cache_seed,
        max_concurrent=max_concurrent,
        temperature=temperature,
        max_tokens=max_tokens,
        **generate_kwargs,
    )
    unique_data_hyp_bank = multiple_hypotheses_remove_repetition(
        prompt_class,
        api,
        data_hyp_bank,
        cache_seed,
        max_concurrent,
        **generate_kwargs,
    )
    unique_paper_hyp_bank = multiple_hypotheses_remove_repetition(
        prompt_class,
        api,
        paper_hyp_bank,
        cache_seed,
        max_concurrent,
        **generate_kwargs,
    )
    unique_data_hyp_list = sorted(unique_data_hyp_bank, key=lambda x: unique_data_hyp_bank[x].acc, reverse=True)
    unique_paper_hyp_list = sorted(unique_paper_hyp_bank, key=lambda x: unique_paper_hyp_bank[x].acc, reverse=True) # all paper-based hyps acc set to 0, so random
    union_hyp_bank = {}

    if prioritize == "data":
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
        n_hyp_data = len(union_hyp_bank)
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
        logger.info(f"number of hypotheses from data: {n_hyp_data}")
        logger.info(f"number of hypotheses from paper: {len(union_hyp_bank) - n_hyp_data}")
    elif prioritize == "paper":
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
        n_hyp_paper = len(union_hyp_bank)
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
        logger.info(f"number of hypotheses from data: {len(union_hyp_bank) - n_hyp_paper}")
        logger.info(f"number of hypotheses from paper: {n_hyp_paper}")
    else:
        while len(union_hyp_bank) < max_num_hypotheses // 2 and len(unique_data_hyp_list) > 0:
            hyp = unique_data_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_data_hyp_bank[hyp]
        n_hyp_data = len(union_hyp_bank)
        while len(union_hyp_bank) < max_num_hypotheses and len(unique_paper_hyp_list) > 0:
            hyp = unique_paper_hyp_list.pop(0)
            union_hyp_bank[hyp] = unique_paper_hyp_bank[hyp]
        logger.info(f"number of hypotheses from data: {n_hyp_data}")
        logger.info(f"number of hypotheses from paper: {len(union_hyp_bank) - n_hyp_data}")

    union_dump_dict = {}
    for hyp in union_hyp_bank:
        union_dump_dict[hyp] = {"hypothesis": hyp, "acc": union_hyp_bank[hyp].acc}

    with open(union_dump_path, 'w') as file:
        json.dump(union_dump_dict, file)
    
    return union_hyp_bank
    