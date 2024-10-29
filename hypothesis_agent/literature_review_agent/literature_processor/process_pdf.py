from abc import ABC, abstractmethod
import json
import subprocess
import os
from doc2json.grobid2json.process_pdf import process_pdf_file
from requests.exceptions import ConnectionError

class PDFProcessor(ABC):
    """
    PDF processor abstract class.
    This class is used to process the pdf file.
    """

    @abstractmethod
    def process_pdf(self, pdf_path):
        pass

class BaseProcessor(PDFProcessor):
    def __init__(self, tmp_dir="./tmp_dir", json_dir="./json_dir"):
        self.tmp_dir = tmp_dir
        self.json_dir = json_dir

    def process_pdf(self, pdf_path):
        """
        Processes the pdf file.
        For BaseProcessor, it just transforms the pdf file into a json file.
        """
        try:
            process_pdf_file(pdf_path, self.tmp_dir, self.json_dir)
        except ConnectionError:
            print("Grobid service is not running, please run ./run_grobid.sh first.")
        return