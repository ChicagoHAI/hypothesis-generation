def load_dict(file_path):
    import json

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_config_path",
        type=str,
        default="./data/hotel_reviews/config.yaml",
        help="Path to the task config.yaml file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the local model. If None, will use the model from the HuggingFace model hub.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vllm",
        choices=["gpt", "claude", "vllm", "huggingface"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--max_num_hypotheses",
        type=int,
        default=20,
        help="Maximum number of hypotheses to keep in the hypothesis bank.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to the output folder for saving hypotheses.",
    )
    parser.add_argument(
        "--old_hypothesis_file",
        type=str,
        default=None,
        help="Path to the old hypothesis file to restart from.",
    )
    parser.add_argument(
        "--num_init",
        type=int,
        default=10,
        help="Number of examples to use for initializing hypotheses.",
    )
    parser.add_argument(
        "--num_train", type=int, default=200, help="Number of training examples."
    )
    parser.add_argument(
        "--num_test", type=int, default=100, help="Number of testing examples."
    )
    parser.add_argument(
        "--num_val", type=int, default=100, help="Number of validation examples."
    )

    parser.add_argument("--seed", type=int, default=49, help="Random seed.")

    parser.add_argument(
        "--file_name_template",
        type=str,
        default="hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
        help="Template for the file name to save hypotheses.",
    )
    parser.add_argument(
        "--sample_num_to_restart_from",
        type=int,
        default=-1,
        help="Sample number to restart from.",
    )
    parser.add_argument(
        "--epoch_to_start_from", type=int, default=0, help="Epoch to start from."
    )
    parser.add_argument(
        "--num_wrong_scale",
        type=float,
        default=0.8,
        help="Set the scale for dynamically changing w_\{hyp\}. For more details, please see Appendix B.1 in the paper.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="The number of top hypotheses checked per example during training.",
    )
    parser.add_argument(
        "--alpha", type=float, default=5e-1, help="Exploration parameter."
    )
    parser.add_argument(
        "--update_batch_size",
        type=int,
        default=10,
        help="Number of examples to use per hypothesis-generation prompt.",
    )
    parser.add_argument(
        "--num_hypotheses_to_update",
        type=int,
        default=1,
        help="Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses.",
    )
    parser.add_argument(
        "--update_hypotheses_per_batch",
        type=int,
        default=5,
        help="Number of hypotheses to generate per prompt.",
    )
    parser.add_argument(
        "--only_best_hypothesis",
        action="store_true",
        default=False,
        help="If only the best hypothesis should be added in the newly generated hypotheses of the batch.",
    )
    parser.add_argument(
        "--save_every_n_examples",
        type=int,
        default=10,
        help="Save hypotheses every n examples visited.",
    )

    parser.add_argument(
        "--init_batch_size",
        type=int,
        default=10,
        help="Batch size to generate the initial hypotheses.",
    )
    parser.add_argument(
        "--init_hypotheses_per_batch",
        type=int,
        default=10,
        help="Number of hypotheses to generate per batch during initialization.",
    )

    parser.add_argument(
        "--cache_seed",
        type=int,
        default=None,
        help="If `None`, will not use cache, otherwise will use cache with corresponding seed number",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6832,
        help="Port for the redis server for LLM caching.",
    )
    parser.add_argument(
        "--generation_style",
        type=str,
        default="default",
        help="Type of generation method.",
    )
    parser.add_argument(
        "--inference_style",
        type=str,
        default="default",
        help="Type of inference method.",
    )
    parser.add_argument(
        "--replace_style", type=str, default="default", help="Type of replace method."
    )
    parser.add_argument(
        "--update_style", type=str, default="default", help="Type of update method."
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=3,
        help="The maximum number of concurrent calls to the API.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file. If None, will only log to stdout.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "max_tokens",
        type=int,
        default=4096,
        help="The maximum number of tokens that can be generated by the model.",
    )
    parser.add_argument(
        "temperature",
        type=float,
        default=1e-5,
        help="The temperature for the model.",
    )

    args = parser.parse_args()

    return args


def main():
    import time

    # set up tools
    start_time = time.time()

    args = parse_args()

    import os

    from hypogenic.extract_label import extract_label_register

    from hypogenic.tasks import BaseTask
    from hypogenic.prompt import BasePrompt
    from hypogenic.utils import set_seed
    from hypogenic.LLM_wrapper import llm_wrapper_register
    from hypogenic.algorithm.summary_information import (
        SummaryInformation
    )

    from hypogenic.algorithm.generation import generation_register
    from hypogenic.algorithm.inference import inference_register
    from hypogenic.algorithm.replace import replace_register
    from hypogenic.algorithm.update import update_register, Update
    from hypogenic.logger_config import LoggerConfig

    LoggerConfig.setup_logger(args.log_file, args.log_level)

    logger = LoggerConfig.get_logger("HypoGenic")

    task = BaseTask(args.task_config_path, from_register=extract_label_register)

    if args.output_folder is None:
        args.output_folder = f"./outputs/{task.task_name}/{args.model_name}/hyp_{args.max_num_hypotheses}/"

    os.makedirs(args.output_folder, exist_ok=True)
    api = llm_wrapper_register.build(args.model_type)(
        args.model_name, args.model_path, port=args.port
    )

    set_seed(args.seed)
    train_data, _, _ = task.get_data(
        args.num_train, args.num_test, args.num_val, args.seed
    )
    prompt_class = BasePrompt(task)
    inference_class = inference_register.build(args.inference_style)(
        api, prompt_class, train_data, task
    )
    generation_class = generation_register.build(args.generation_style)(
        api, prompt_class, inference_class, task
    )
    replace_class = replace_register.build(args.replace_style)(args.max_num_hypotheses)

    update_class: Update = update_register.build(args.update_style)(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=replace_class,
        save_path=args.output_folder,
        file_name_template=args.file_name_template,
        sample_num_to_restart_from=args.sample_num_to_restart_from,
        num_init=args.num_init,
        epoch_to_start_from=args.epoch_to_start_from,
        num_wrong_scale=args.num_wrong_scale,
        k=args.k,
        alpha=args.alpha,
        update_batch_size=args.update_batch_size,
        num_hypotheses_to_update=args.num_hypotheses_to_update,
        update_hypotheses_per_batch=args.update_hypotheses_per_batch,
        only_best_hypothesis=args.only_best_hypothesis,
        save_every_n_examples=args.save_every_n_examples,
    )

    hypotheses_bank = {}
    if args.old_hypothesis_file is None:
        hypotheses_bank = update_class.batched_initialize_hypotheses(
            num_init=args.num_init,
            init_batch_size=args.init_batch_size,
            init_hypotheses_per_batch=args.init_hypotheses_per_batch,
            cache_seed=args.cache_seed,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample=args.num_init,
            seed=args.seed,
            epoch=0,
        )
    else:
        dict = load_dict(args.old_hypothesis_file)
        for hypothesis in dict:
            hypotheses_bank[hypothesis] = SummaryInformation.from_dict(dict[hypothesis])
    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=args.seed,
            cache_seed=args.cache_seed,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=args.seed,
            epoch=epoch,
        )

    # print experiment info
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
