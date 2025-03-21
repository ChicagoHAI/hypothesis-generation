import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import wandb
import json
import re

from hypogenic.LLM_wrapper import LocalVllmWrapper
from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.extract_label import extract_label_register
from hypogenic.register import Register
from hypogenic.utils import get_results
from hypogenic.logger_config import LoggerConfig


response_map_register = Register("response_map")


@response_map_register.register("deceptive_reviews")
def response_map_hotel_reviews(response):
    return f"Final answer: {response['label']}"

@response_map_register.register("headline_binary")
def response_map_headline_binary(response):
    if response["label"] == "Headline 2 has more clicks than Headline 1.":
        return f"Answer: Headline 2"
    else:
        return f"Answer: Headline 1"


@response_map_register.register("gptgc_detect")
@response_map_register.register("llamagc_detect")
def response_map_aigc_detect(response):
    return f"Final answer: {response['label']}"


@response_map_register.register("retweet")
def response_map_retweet(response):
    return f"Final answer: the {response['label']} tweet"


@response_map_register.register("persuasive_pairs")
def response_map_persuasive_pairs(response):
    return (
        f"Final answer: the {response['label']} argument uses more persuasive language."
    )

@response_map_register.register("dreaddit")
def response_map_dreaddit(response):
    return f"Final answer: {response['label']}"


@extract_label_register.register("persuasive_pairs")
def persuasive_pairs_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")

    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    pattern = r"answer: the (\w+) argument"
    patterns = [
        r"answer: the (\w+) argument",
        r"answer: \[the (\w+) argument",
        r"answer: (\w+) argument",
    ]

    prev_answer = ""
    # check
    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if prev_answer == "":
                prev_answer = answer
            elif prev_answer != "" and answer != prev_answer:
                return "conflict"

    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if answer == "first":
                return "first"
            elif answer == "second":
                return "second"
            else:
                return "other"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"


@extract_label_register.register("dreaddit")
def dreaddit_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")

    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    # pattern = r"answer: the (\w+) argument"
    patterns = [
        r"answer: (\w+) stress",
        r"answer: \[(\w+) stress",
    ]

    prev_answer = ""
    # check
    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if prev_answer == "":
                prev_answer = answer
            elif prev_answer != "" and answer != prev_answer:
                return "conflict"

    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if answer == "has":
                return "has stress"
            elif answer == "no":
                return "no stress"
            else:
                return "other"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"


def extract_label_headline_binary_new(text):
    return extract_label_register.build("headline_binary")(text)


@extract_label_register.register("retweet_new")
def extract_label_retweet_new(text):
    return extract_label_register.build("retweet")(text)


@extract_label_register.register("aigc_detect")
@extract_label_register.register("gptgc_detect")
@extract_label_register.register("llamagc_detect")
def extract_label_aigc_detect(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    text = text.lower()
    pattern = r"final answer:\s+(ai|human)"

    match = re.findall(pattern, text)
    if len(match) > 0:
        return match[-1].upper()
    else:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"


def get_model(model_path="/net/projects/chai-lab/shared_models/Meta-Llama-3.1-8B-Instruct/"):

    # model_path = "/net/projects/chai-lab/shared_models/Meta-Llama-3.1-8B-Instruct/"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)

    return model, tokenizer


def get_dataset(task, tokenizer, prompt_class, response_map):
    def process_func(example):
        MAX_LENGTH = 4000
        input_ids, attention_mask, labels = [], [], []

        instruction = tokenizer.apply_chat_template(
            prompt_class.few_shot_baseline(None, 0, pd.DataFrame([example]), 0),
            tokenize=False,
            add_generation_prompt=True,
        )
        response = f"{response_map(example)}{tokenizer.eos_token}"

        instruction = tokenizer(instruction, add_special_tokens=False)
        response = tokenizer(response, add_special_tokens=False)

        input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        )
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
        )

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_data, test_data, val_data = task.get_data(200, None, None, 42)
    train_dataset, test_dataset, val_dataset = (
        Dataset.from_pandas(train_data),
        Dataset.from_pandas(test_data),
        Dataset.from_pandas(val_data),
    )
    train_dataset, test_dataset, val_dataset = (
        train_dataset.map(process_func, remove_columns=train_dataset.column_names),
        test_dataset.map(process_func, remove_columns=test_dataset.column_names),
        val_dataset.map(process_func, remove_columns=val_dataset.column_names),
    )

    return train_dataset, test_dataset, val_dataset


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    return parser.parse_args()


def main():
    args = get_args()
    LoggerConfig.setup_logger(level="INFO")

    model, tokenizer = get_model(model_path=args.model_path)

    config_path = f"../data/{args.task}/config.yaml"
    task = BaseTask(
        config_path=config_path,
        from_register=extract_label_register,
    )
    prompt_class = BasePrompt(task)
    response_map = response_map_register.build(task.task_name)

    train_dataset, _, val_dataset = get_dataset(
        task, tokenizer, prompt_class, response_map
    )

    wandb.init(
        project=f"hypogenic-llama31-8b-{task.task_name}",
        entity="", # Add your entity here
    )

    training_args = TrainingArguments(
        output_dir=f"./output/llama31-8b/{task.task_name}/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=12,
        save_steps=50,
        eval_steps=10,
        logging_steps=10,
        warmup_ratio=0.02,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    def check_labels(labels):
        start_idx = 0
        while start_idx < len(labels) and labels[start_idx] == -100:
            start_idx += 1

        end_idx = len(labels) - 1
        while end_idx >= 0 and labels[end_idx] == -100:
            end_idx -= 1

        return start_idx, end_idx

    def classification_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        pred_list, label_list = [], []

        for pred, label in zip(preds, labels):
            bg_idx, ed_idx = check_labels(label)
            label = tokenizer.decode(
                label[bg_idx : ed_idx - 1], skip_special_tokens=True
            )
            label = task.extract_label(label)
            label_list.append(label)

            bg_idx, ed_idx = check_labels(pred)
            pred = tokenizer.decode(pred[bg_idx : ed_idx + 1], skip_special_tokens=True)
            pred_list.append(pred)

        return get_results(pred_list, label_list)

    def process_logits_for_metrics(predictions, label_ids):
        res = []

        for pred, label in zip(predictions, label_ids):
            bg_idx, ed_idx = check_labels(label)
            pred = pred.argmax(axis=-1)
            pred = tokenizer.decode(
                pred[bg_idx - 1 : ed_idx - 1], skip_special_tokens=True
            )
            pred = task.extract_label(pred)

            res.append(
                torch.tensor(tokenizer(pred, add_special_tokens=False)["input_ids"])
            )

        return torch.nn.utils.rnn.pad_sequence(
            res, batch_first=True, padding_value=-100
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=process_logits_for_metrics,
        compute_metrics=classification_metrics,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    trainer.save_model()
    with open(os.path.join(training_args.output_dir, "results.txt"), "w") as f:
        f.write(f"Best model is {trainer.state.best_model_checkpoint}\n")


def test_model():
    args = get_args()
    LoggerConfig.setup_logger(level="INFO")

    logger = LoggerConfig.get_logger("test_model")

    config_path = f"../data/{args.task}/config.yaml"
    task = BaseTask(
        config_path=config_path,
        from_register=extract_label_register,
    )
    prompt_class = BasePrompt(task)

    model_path = args.model_path
    api = LocalVllmWrapper(None, model_path, enable_lora=True, max_lora_rank=16)

    if args.steps is None:
        with open(f"./output/llama31-8b/{task.task_name}/results.txt") as f:
            path_names = [f.read().replace("Best model is ", "").strip()]
            logger.info(f"Using model from {path_names}")
    else:
        path_names = [
            f"./output/llama31-8b/{task.task_name}/checkpoint-{step}/"
            for step in args.steps
        ]
    _, test_data, _ = task.get_data(200, None, None, 42)

    prompt_inputs = [
        prompt_class.few_shot_baseline(None, 0, test_data, i) for i in test_data.index
    ]
    for path_name in path_names:
        responses = api.batched_generate(
            prompt_inputs, max_tokens=4000, temperature=0, lora_path=path_name
        )

        pred_list, label_list = [], []
        for i in range(len(test_data)):
            pred_list.append(task.extract_label(responses[i]))
            label_list.append(test_data["label"][i])

        results = get_results(pred_list, label_list)
        logger.info(f"Results for {path_name}: {results}")


if __name__ == "__main__":
    main()
    test_model()
