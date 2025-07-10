from abc import ABC, abstractmethod
import pickle
import math
from typing import Dict, List
import torch
import re
import os
import numpy as np
import random
import openai
import json
import vllm
import asyncio
import tqdm
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from hypogenic.algorithm.summary_information import SummaryInformation

from pprint import pprint

from .LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from .tasks import BaseTask
from .logger_config import LoggerConfig

logger_name = "HypoGenic - utils"


def get_results(pred_list, label_list):
    """
    Compute accuracy and F1 score for multi-class classification
    
    Automatically ignores case differences when labels are strings.
    """
    # Check if labels are strings and convert to lowercase if they are
    if all(isinstance(label, str) for label in label_list + pred_list):
        pred_list = [pred.lower() for pred in pred_list]
        label_list = [label.lower() for label in label_list]
    
    valid_labels = set(label_list)
    accuracy = accuracy_score(label_list, pred_list)
    f1 = f1_score(label_list, pred_list, average="macro", labels=list(valid_labels))

    return {"accuracy": accuracy, "f1": f1}

def get_results_regression(pred_list, label_list):
    """
    Compute MSE for regression.
    """
    print('label_list: ', label_list)
    print('pred_list: ', pred_list)

    # Check if labels can be turned into floats
    try:
        label_list = [float(label) for label in label_list]
    except:
        raise ValueError("Labels cannot be turned into floats")
    
    # if a prediction cannot be turned into a float, then it is 'unknown' and then we use the mean of the predictions as the prediction
    mean_pred = np.mean([float(pred) for pred in pred_list if pred != 'unknown'])
    print('mean_pred: ', mean_pred)
    pred_list = [mean_pred if pred == 'unknown' else float(pred) for pred in pred_list]
    print('pred_list: ', pred_list)
    
    mse = np.mean((np.array(label_list) - np.array(pred_list))**2)

    return {"mse": mse}

def set_seed(seed):
    logger = LoggerConfig.get_logger(logger_name)
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_hypotheses(filename: str) -> Dict[str, SummaryInformation]:
    """Load hypotheses from a JSON file."""
    with open(filename) as f:
        hyp_dict = json.load(f)
    hyp_bank = {}
    for hypothesis in hyp_dict:
        hyp_bank[hypothesis] = SummaryInformation.from_dict(hyp_dict[hypothesis])
    return hyp_bank