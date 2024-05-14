"""
CLI to run training on a model
"""
import logging
from pathlib import Path
from typing import Tuple, Union

import fire
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.prompt_strategies.sharegpt import register_chatml_template
from axolotl.train import train

import wandb 
import json 
import os

LOG = logging.getLogger("axolotl.cli.train")

def load_jsonl_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def construct_prompt(cfg, record):
    system_prompt = cfg["type"]["system_prompt"]
    system_prefix = cfg["type"]["field_system"]
    instruct_format = cfg["type"]["format"].format(instruction=record["instruction"])
    output = record["output"]

    return f"{system_prefix} {system_prompt} {instruct_format} {output}"

def construct_prompt_template_table(cfg):
    system_prompt = cfg["type"]["system_prompt"]
    system_prefix = cfg["type"]["field_system"]
    instruct_format = cfg["type"]["format"]
    no_input_instruct_format = cfg["type"]["no_input_format"]

    prompt_template = f"{system_prefix} {system_prompt} {instruct_format}"

    prompt_template_table = wandb.Table(columns=["system_prompt", "field_system", "format", "no_input_format", "prompt_template"],
                                data=[[system_prompt, system_prefix, instruct_format, no_input_instruct_format, prompt_template]])

    return prompt_template_table

def log_dataset(cfg):
    datasets = cfg["datasets"]
    wandb_project = cfg["wandb_project"]
    wandb_run_id = cfg["wandb_run_id"]

    base_model = cfg["base_model"]
    base_model_name = base_model.split("/")[1]

    # 1. Start a W&B Run
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_id,
        tags=["training", base_model],
        group=base_model,
        job_type="training"
    )

    dataset_table = wandb.Table(columns=["dataset_name", "instruction", "input", "output", "prompt"])
    artifact = wandb.Artifact(name=f"prompt_{base_model_name}", type="prompt_template")

    for dataset in datasets:
        dataset_name = os.path.basename(dataset["path"]).split(".")[0]
        # log prompt artifact
        prompt_template_table = construct_prompt_template_table(dataset)
        artifact.add(prompt_template_table, f"{dataset_name}")

        data = load_jsonl_dataset(dataset["path"])
        for record in data:
            prompt = construct_prompt(dataset, record)
            dataset_table.add_data(dataset_name,
                                   record["instruction"],
                                   record["input"] if "input" in record else "", 
                                   record["output"], 
                                   prompt)

    LOG.info("Save prompt template to Wandb run as an artifact")
    run.log_artifact(artifact)
    LOG.info("Save dataset to Wandb run")
    run.log({"dataset": dataset_table})


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    
    # log_dataset(parsed_cfg)
    
    return do_train(parsed_cfg, parsed_cli_args)


def do_train(cfg, cli_args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    if cfg.chat_template == "chatml" and cfg.default_system_message:
        LOG.info(
            f"ChatML set. Adding default system message: {cfg.default_system_message}"
        )
        register_chatml_template(cfg.default_system_message)
    else:
        register_chatml_template()

    if cfg.rl:
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    return train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
