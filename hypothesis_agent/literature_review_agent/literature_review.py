import os
import json
import shutil
import subprocess
import time
import requests
from string import Template
from typing import Dict, List, Union
from hypogenic.algorithm.generation.utils import extract_hypotheses
from hypogenic.LLM_wrapper import LLMWrapper
from hypogenic.logger_config import LoggerConfig

from .literature_processor.extract_info import BaseExtractor
from .literature_processor.summarize import BaseSummarize
from ..data_analysis_agent.prompt import TestPrompt
from .literature_search import auto_literature_search
from doc2json.grobid2json.process_pdf import process_pdf_file

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

    def auto_process_literature(
        self,
        num_papers: int = 10,
        num_papers_per_trial: int = 10,
        max_search_trial: int = 5,
        cache_seed = None,
        **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("automated literature processing")
        task_name = self.prompt_class.task.task_name
        if "automated_literature_search_topic" in self.prompt_class.task.prompt_template:
            search_topic = self.prompt_class.task.prompt_template["automated_literature_search_topic"]
        else:
            search_topic = task_name
        save_dir = auto_literature_search(
            topic=search_topic,
            num_papers=num_papers,
            task_name=task_name,
            num_papers_per_trial=num_papers_per_trial,
            max_trial=max_search_trial,
        )
        if save_dir == "":
            return
        logger.info(f"automated literature search finished")
        
        run_grobid_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"modules/run_grobid.sh"
        )

        if not os.path.exists(run_grobid_path):
            logger.error(f"Need to set up grobid first. Please run bash modules/setup_grobid.sh")
            return 

        logger.info("running grobid")
        grobid_process = subprocess.Popen(
            ["bash", run_grobid_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(10)
        for i in range(30):
            try:
                response = requests.get("http://localhost:8070/api/isalive", timeout=2)
                if response.status_code == 200:
                    logger.info("Grobid is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        else:
            logger.error("Grobid did not become ready in time.")
            grobid_process.terminate()
            return

        try:
            raw_literature_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"literature/{task_name}/raw"
            )
            processed_literature_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"literature/{task_name}/processed"
            )
            os.makedirs(processed_literature_dir, exist_ok=True)

            for root, dirs, files in os.walk(raw_literature_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        process_pdf_file(file_path, "./tmp_dir", processed_literature_dir)
                    except ConnectionError:
                        logger.warning("Grobid service is not running, please run ./run_grobid.sh first.")
            shutil.rmtree("./tmp_dir")
            logger.info("PDF preprocessing completed!")

        except Exception as e:
            logger.warning(f"Error when processing paper PDFs with grobid: {e}")
            return

        finally:
            logger.info("stopping grobid process")
            grobid_process.terminate()
            grobid_process.wait()
        
        self.summarize_papers(
            data_file=processed_literature_dir,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        grobid_process.terminate()