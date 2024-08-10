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

@llm_wrapper_register.register("claude")
class ClaudeWrapper(LLMWrapper):
    def __init__(self, model, max_retry=30, port=6832, **kwargs):
        super().__init__(model)
        self.api = Anthropic()
        self.api_with_cache = ClaudeAPICache(port=port, max_retry=max_retry, **kwargs)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens=500,
        temperature=1e-5,
        max_concurrent=3,
        **kwargs,
    ):
        if len(messages) == 0:
            return []

        client = AsyncAnthropic()
        status_bar = tqdm.tqdm(total=len(messages))

        # TODO: retry on failure
        async def _async_generate(sem, messages, **kwargs):
            for idx, msg in enumerate(messages):
                if msg["role"] == "system":
                    system_prompt = messages.pop(idx)["content"]
                    break

            async with sem:
                resp = await client.messages.create(
                    system=system_prompt,
                    messages=messages,
                    **kwargs,
                )
                status_bar.update(1)
                return resp

        sem = asyncio.Semaphore(max_concurrent)
        tasks = [
            _async_generate(
                sem,
                messages=messages[i],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            for i in range(len(messages))
        ]
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(asyncio.gather(*tasks))
        return [r.content[0].text for r in resp]

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        for idx, msg in enumerate(messages):
            if msg["role"] == "system":
                system_prompt = messages.pop(idx)["content"]
                break

        response = self.api.messages.create(
            system=system_prompt,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text
