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