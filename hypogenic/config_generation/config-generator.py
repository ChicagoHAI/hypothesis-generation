import os
import argparse
from ..logger_config import LoggerConfig

from ..LLM_wrapper import llm_wrapper_register


BASE_DIR = os.path.join(os.curdir, "hypogenic", "config_generation")
logger = LoggerConfig.get_logger("Config logger")

def generate(mod, mod_name, rq, instr, task, train_path, val_path, test_path, input_variable_name, label_name):
    api = llm_wrapper_register.build(mod)(mod_name)
    logger.info("API call created.")

    example_file = os.path.join(BASE_DIR, "config_examples", "config.yaml")

    with open(example_file) as f:
        example_text = f.read()

        if example_text is None:
            logger.info("ERROR: No example file provided.")
            return ""

    prompt = f"""You are an AI that generates configuration files based on instructionsl, a research question, and some file metadata.

Research Question:
{rq}

Instructions:
{instr}

Here is some metadata from the corresponding dataset:
 - Task: {task}
 - Training path: {train_path}
 - Validate path: {val_path}
 - Test path: {test_path}
 - Input variable name: {input_variable_name}
 - Label name: {label_name}

Below are is an example configuration file for reference:
{example_text}

Please generate a new configuration file based on the research question and instructions.
Only return the content of the configuration file. Do not include any extra explanation.
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
        logger.info("Error generating config:", e)
        return ""
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--research_question", type=str, required=True)
    parser.add_argument("--instructions", type=str, required=True)

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--input_variable_name", type=str, required=True)
    parser.add_argument("--label_name", type=str, required=True)


    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--config_file", type=str, required=False, default = "~/Downloads/config.yaml")

    parser.add_argument("--model_type", type=str, required=False, default="gpt")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-4o-mini")

    args = parser.parse_args()

    config = generate(
        args.model_type,
        args.model_name,
        args.research_question,
        args.instructions,
        args.task,
        args.train_path,
        args.val_path,
        args.test_path,
        args.input_variable_name,
        args.label_name
    )

    file = os.path.expanduser(args.config_file)

    with open(file, 'w') as f:
        f.write(config)