from abc import ABC, abstractmethod
import pickle
import math
from typing import Callable, Dict, List
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

from . import llm_wrapper_register
from .base import LLMWrapper
from ..LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from ..tasks import BaseTask


class LocalModelWrapper(LLMWrapper):
    def __init__(
        self,
        model,
        path_name=None,
        max_retry=30,
        port=6832,
        **kwargs,
    ):
        super().__init__(model)
        if path_name is None:
            path_name = model

        self.api_kwargs = {"model": path_name, **kwargs}
        self.api = None
        self.api_with_cache = LocalModelAPICache(port=port, max_retry=max_retry)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        raise NotImplementedError

    def _generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        return self._batched_generate(
            [messages],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )[0]


@llm_wrapper_register.register("huggingface")
class LocalHFWrapper(LocalModelWrapper):
    def __init__(self, model, path_name=None, max_retry=30, port=6832, **kwargs):
        super().__init__(
            model=model,
            path_name=path_name,
            max_retry=max_retry,
            port=port,
            # pipeline kwargs
            task="text-generation",
            device_map="auto",
            model_kwargs=kwargs,
        )

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        if len(messages) == 0:
            return []
        if self.api is None:
            self.api = pipeline(**self.api_kwargs)
        output = self.api(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return [o[0]["generated_text"][-1]["content"] for o in output]


@llm_wrapper_register.register("vllm")
class LocalVllmWrapper(LocalModelWrapper):
    def __init__(
        self,
        model,
        path_name=None,
        max_retry=30,
        port=6832,
        **kwargs,
    ):
        super(__class__, self).__init__(
            model=model,
            path_name=path_name,
            max_retry=max_retry,
            port=port,
            # VLLM kwargs
            tensor_parallel_size=torch.cuda.device_count(),
            **kwargs,
        )

    def _batched_generate(
        self,
        messages: List[List[Dict[str, str]]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        if len(messages) == 0:
            return []
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.api is None:
            self.api = vllm.LLM(**self.api_kwargs)
        tokenizer = self.api.get_tokenizer()
        formatted_prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        output = self.api.generate(formatted_prompts, sampling_params)
        return [o.outputs[0].text for o in output]
