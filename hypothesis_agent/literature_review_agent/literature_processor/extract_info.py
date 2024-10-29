from abc import ABC, abstractmethod
from typing import Dict, List, Union
import json
import os
import glob
from hypogenic.logger_config import LoggerConfig


class InfoExtractor(ABC):
    """
    Information extractor abstract class.
    This class is used to extract information from the json data, which are processed from academic papers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def process_single_file(self, filename: str) -> Union[Dict[str, str], None]:
        pass

    def extract_info(self, data_file: Union[List[str], str]) -> List[Dict[str, str]]:
        """
        Extract and process information from the json data.

        Parameters:
            data_file: folder path that contains json files or a list of json file paths
        """
        logger = LoggerConfig.get_logger("InfoExtractor")

        paper_data_all = []
        if isinstance(data_file, str):
            data_file = glob.glob(
                os.path.join(data_file, "**", "*.json"), recursive=True
            )

        for filename in data_file:
            try:
                paper_data = self.process_single_file(filename)
                if paper_data is not None:
                    paper_data_all.append(paper_data)
                else:
                    logger.warning(f"Empty data for file {filename}, skipping file.")
            except Exception as e:
                logger.warning(f"Error processing file {filename}: {e}, skipping file.")

        return paper_data_all


class BaseExtractor(InfoExtractor):
    """
    Base Extractor only extracts titles and abstracts from each paper.
    """

    def __init__(self):
        super().__init__()

    def process_single_file(self, filename: str) -> Union[Dict[str, str], None]:
        """
        Extracts information from the json data.
        Returns title and abstract of the paper.
        """
        with open(filename, "r") as file:
            paper_doc = json.load(file)
        paper_data_processed = {
            "title": paper_doc["title"],
            "abstract": paper_doc["abstract"],
        }

        for value in paper_data_processed.values():
            if value:
                return paper_data_processed
        return None


class WholeExtractor(InfoExtractor):
    def __init__(self):
        super().__init__()

    def process_single_file(self, filename: str) -> Union[Dict[str, str], None]:
        """
        Extracts information from the json data.
        Returns title and abstract of the paper.
        """
        with open(filename, "r") as file:
            paper_doc = json.load(file)
        paper_data_processed = {
            "title": paper_doc["title"],
            "paper_text": self.extract_whole_text_from_json(paper_doc),
        }

        for value in paper_data_processed.values():
            if value:
                return paper_data_processed
        return None

    def extract_whole_text_from_json(self, paper_doc):
        """
        Extracts information from json data, returns whole text of paper (abstract and body text), not including title
        """

        abstract = paper_doc["abstract"]
        body_text_list = paper_doc["pdf_parse"]["body_text"]
        body_text = "\n".join([body_text["text"] for body_text in body_text_list])
        whole_text = abstract + "\n" + body_text
        return whole_text
