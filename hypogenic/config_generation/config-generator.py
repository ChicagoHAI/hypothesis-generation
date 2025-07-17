import os
import argparse
import zipfile
import json
from ..logger_config import LoggerConfig

from ..LLM_wrapper import llm_wrapper_register


BASE_DIR = os.path.join(os.curdir, "hypogenic", "config_generation")
logger = LoggerConfig.get_logger("Config logger")

def read_dataset(dataset_zip_path):
    data_dir = os.path.join(BASE_DIR, "data")
    zip_dataset = os.path.expanduser(os.path.join(data_dir, dataset_zip_path))

    extract_dir = os.path.join(data_dir, "dataset")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_dataset, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    with open(os.path.join(extract_dir, "metadata.json")) as metadata_file:
        metadata = json.load(metadata_file)

    # TODO: read in names of train, val, and test files. 
    # TODO: extract task name and label name

    return metadata

def generate(mod, mod_name, rq, instr, dataset_zip_path):
    api = llm_wrapper_register.build(mod)(mod_name)
    logger.info("API call created.")

    metadata = read_dataset(dataset_zip_path)
    metadata_json = json.dumps(metadata)

    example_file = os.path.join(BASE_DIR, "config_examples", "config.yaml")

    with open(example_file) as f:
        example_text = f.read()

        if example_text is None:
            logger.error("No example file provided.")
            return ""

    # TODO: add format for final answer in prompt

    prompt = f"""You are an AI that generates configuration files based on instructionsl, a research question, and the metadata of the dataset.

Research Question:
{rq}


Instructions:
{instr}


This is the metadata for the dataset:
{metadata_json}


Below are is an example configuration file for reference:
{example_text}


Please generate a new configuration file based on the research question and instructions.
Only return the content of the configuration file with NO headers or footers.
    """

    try:
        messages = [{"role": "user", "content": prompt}]

        logger.info(f"Cost before generation: {api.get_cost()} USD")

        response = api.api_with_cache.api_call(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.15,
            max_tokens=5000
        )

        logger.info(f"Cost after generation: {api.get_cost()} USD")

        return response.strip()

    except Exception as e:
        logger.error("Error generating config:", e)
        return ""
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--research_question", type=str, required=True)
    parser.add_argument("--instructions", type=str, required=True)

    parser.add_argument("--dataset_zip_path", type=str, required=True)

    parser.add_argument("--config_file", type=str, required=False, default = "~/Downloads/config.yaml")

    parser.add_argument("--model_type", type=str, required=False, default="gpt")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-4o-mini")

    args = parser.parse_args()

    config = generate(
        args.model_type,
        args.model_name,
        args.research_question,
        args.instructions,
        args.dataset_zip_path
    )

    file = os.path.expanduser(args.config_file)

    with open(file, 'w') as f:
        f.write(config)