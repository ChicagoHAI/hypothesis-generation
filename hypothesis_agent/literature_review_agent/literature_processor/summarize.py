from abc import ABC, abstractmethod
import json
import os
import shutil
from typing import Dict, List, Union
from hypogenic.LLM_wrapper import LLMWrapper
from hypogenic.prompt import BasePrompt

from .extract_info import BaseExtractor


class BaseSummarize(ABC):
    def __init__(
        self,
        extractor: BaseExtractor,
    ):
        self.extractor = extractor

    def summarize(self, data_file: Union[List[str], str]) -> List[Dict[str, str]]:
        return self.extractor.extract_info(data_file)


class LLMSummarize(BaseSummarize):
    def __init__(
        self,
        extractor: BaseExtractor,
        api: LLMWrapper,
        prompt_class: BasePrompt,
    ):
        super().__init__(extractor)
        self.api = api
        self.prompt_class = prompt_class

    def summarize(
        self,
        data_file: Union[List[str], str],
        cache_seed=None,
        **generate_kwargs,
    ) -> List[Dict[str, str]]:
        paper_data_all = self.extractor.extract_info(data_file)

        paper_infos = []
        prompt_inputs = []
        for paper_data in paper_data_all:
            prompt = self.prompt_class.summarize_paper(paper_data)
            prompt_inputs.append(prompt)

        summaries = self.api.batched_generate(
            messages=prompt_inputs,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        for summary, paper_data in zip(summaries, paper_data_all):
            paper_info = {"title": paper_data["title"], "summary": summary}
            paper_infos.append(paper_info)

        return paper_infos
