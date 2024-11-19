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
from pprint import pprint

from .LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from .tasks import BaseTask
from .logger_config import LoggerConfig

logger_name = "HypoGenic - utils"


def get_results(pred_list, label_list):
    """
    Compute accuracy and F1 score for multi-class classification
    """
    accuracy = accuracy_score(label_list, pred_list)
    f1 = f1_score(label_list, pred_list, average="macro")

    return {"accuracy": accuracy, "f1": f1}


def set_seed(seed):
    logger = LoggerConfig.get_logger(logger_name)
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def adjust_label(preds, labels):
    preds_out = []
    existing_labels = set(labels)
    for i in range(len(labels)):
        if preds[i] not in existing_labels:
            for existing_label in existing_labels:
                if labels[i] != existing_label:
                    preds_out.append(existing_label)
                    break
        else:
            preds_out.append(preds[i])

    return preds_out