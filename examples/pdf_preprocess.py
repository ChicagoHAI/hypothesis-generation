import os
import json
import sys
import logging
import shutil
from doc2json.grobid2json.process_pdf import process_pdf_file
from requests.exceptions import ConnectionError
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hypogenic.logger_config import LoggerConfig

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("PDF Preprocessor")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task_name", type=str, required=True)
    
    args = parser.parse_args()
    
    directory = f"../literature/{args.task_name}/raw"
    json_dir = f"../literature/{args.task_name}/processed"

    os.makedirs(json_dir, exist_ok=True)

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                process_pdf_file(file_path, "./tmp_dir", json_dir)
            except ConnectionError:
                print("Grobid service is not running, please run ./run_grobid.sh first.")
    shutil.rmtree("./tmp_dir")
    logger.info("PDF preprocessing completed!")

if __name__ == "__main__":
    main()