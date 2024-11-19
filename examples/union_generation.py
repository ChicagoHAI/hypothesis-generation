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
from typing import Callable, Tuple, Union
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypogenic.extract_label import extract_label_register, persuasive_pairs_extract_label, dreaddit_extract_label

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
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import SamplingUpdate, DefaultUpdate
from hypogenic.logger_config import LoggerConfig
from hypogenic.utils import get_results

from hypothesis_agent.data_analysis_agent.generation import TestGeneration, MultiHypGenerationWithRank
from hypothesis_agent.data_analysis_agent.update import TestUpdate
from hypothesis_agent.literature_review_agent.literature_review import LiteratureAgent
from hypothesis_agent.literature_review_agent.literature_processor.extract_info import BaseExtractor, WholeExtractor
from hypothesis_agent.literature_review_agent.literature_processor.summarize import LLMSummarize
from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
from hypothesis_agent.data_analysis_agent.union_generation import union_hypogenic_and_paper
from hypothesis_agent.data_analysis_agent.inference import MultiHypDefaultInference

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("HypoRefine")

def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main():
    start_time = time.time()

    model_name = "gpt-4o-mini"
    # model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    test_ood = False # set to True if testing on OOD data
    use_valid = False # set to True if testing with validation set
    use_refine = True # set to True if using HypoRefine, or False for HypoGeniC

    if test_ood:
        config_version = "_ood"
    else:
        config_version = ""

    task_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), f"data/dreaddit/config{config_version}.yaml"
    )
    papers_dir_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "literature/dreaddit/processed"
    )

    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        model_path = "/net/projects/chai-lab/shared_models/Meta-Llama-3.1-70B-Instruct"
    max_num_hypotheses = 20
    max_refine = 6 # round of refinement for HypoRefine
    num_init = 10
    num_train = 200
    num_test = 500
    num_val = 300
    k = 10
    alpha = 5e-1
    update_batch_size = 10
    num_hypotheses_to_update = 1
    save_every_10_examples = 10
    init_batch_size = 10
    init_hypotheses_per_batch = 10
    cache_seed = None
    temperature = 1e-5
    max_tokens = 4000
    max_concurrent = 32
    seeds = [42]
    task_name = "dreaddit" # used for dirname for storing and loading paper summaries, any name is fine
    prioritize = "balanced" # merging strategy for Union methods
    n_paper_specificity_boost = 0 # n_round of specificity boost

    if "gpt" in model_name:
        api = GPTWrapper(model_name)
    else:
        api = LocalVllmWrapper(model_name, model_path, gpu_memory_utilization=0.95)
    task = BaseTask(
        task_config_path, extract_label=dreaddit_extract_label
    )

    accuracy_all = []
    f1_all = []

    for seed in seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = TestPrompt(task)
        extractor = WholeExtractor()
        summarizer = LLMSummarize(extractor, api, prompt_class)
        literature_agent = LiteratureAgent(api, prompt_class, summarizer)
        generate_kwargs = {}
        literature_agent.summarize_papers(
            data_file=papers_dir_path,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        union_hyp_bank = union_hypogenic_and_paper(
            task=task,
            prompt_class=prompt_class,
            literature_agent=literature_agent,
            extractor=extractor,
            api=api,
            train_data=train_data,
            config_path=task_config_path,
            prioritize=prioritize,
            model_name=model_name,
            papers_dir_path=papers_dir_path,
            task_name=task_name,
            custom_dump_path=None,
            max_num_hypotheses=max_num_hypotheses,
            use_refine=use_refine,
            n_paper_specificity_boost=n_paper_specificity_boost,
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
        
        if use_valid:
            logger.info("Using validation data")
            test_data = val_data
        else:
            logger.info("Using test data")

        inference_class = MultiHypDefaultInference(api, prompt_class, train_data, task)

        pred_list, label_list = inference_class.multiple_hypotheses_run_inference_final(
            test_data,
            union_hyp_bank,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            generate_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        results_dict = get_results(pred_list, label_list)

        if use_refine == False:
            logger.info(f"Multi-hypothesis Inference Results for seed {seed} and method Literature U HypoGeniC")
        else:
            logger.info(f"Multi-hypothesis Inference Results for seed {seed} and method Literature U HypoRefine")
        logger.info(f"Accuracy for seed {seed}: {results_dict['accuracy']}")
        logger.info(f"F1 for seed {seed}: {results_dict['f1']}")

        wrong_indices = [
            i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
        ]
        accuracy_all.append(results_dict["accuracy"])
        f1_all.append(results_dict["f1"])

    logger.info(f"Averaged accuracy: {sum(accuracy_all)/len(accuracy_all)}")
    logger.info(f"Averaged F1: {sum(f1_all)/len(f1_all)}")

    logger.info(f"Total time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
