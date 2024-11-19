import os
import json
import shutil
from string import Template
from typing import Dict, List, Union
from hypogenic.algorithm.generation.utils import extract_hypotheses
from hypogenic.LLM_wrapper import LLMWrapper

from .literature_processor.extract_info import BaseExtractor
from .literature_processor.summarize import BaseSummarize
from ..data_analysis_agent.prompt import TestPrompt


class LiteratureAgent:
    def __init__(
        self,
        api: LLMWrapper,
        prompt_class: TestPrompt,
        summizer: BaseSummarize,
        paper_infos: List[Dict[str, str]] = None,  # List of paper info
    ):
        self.api = api
        self.prompt_class = prompt_class
        self.summizer = summizer
        self.paper_infos = paper_infos if paper_infos is not None else []

    def summarize_papers(
        self,
        data_file: Union[List[str], str],
        cache_seed=None,
        **generate_kwargs,
    ):
        self.paper_infos = self.summizer.summarize(
            data_file, cache_seed, **generate_kwargs
        )
    
    def save_paper_infos(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.paper_infos, f)

    def refine_hypotheses(
        self,
        hypotheses_list: List[str],
        cache_seed=None,
        **generate_kwargs,
    ):
        prompt = self.prompt_class.refine_with_literature(
            hypotheses_list, paper_infos=self.paper_infos
        )

        response = self.api.generate(
            prompt,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        return extract_hypotheses(response, len(hypotheses_list))
