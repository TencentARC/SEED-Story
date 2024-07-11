# flake8: noqa
import hydra

import pyrootutils
import os
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
from flask import Flask, request
from typing import List, Union
import json
from typing import Optional
import transformers
from dataclasses import dataclass, field, asdict, is_dataclass
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, \
    SequentialReadingService
import logging

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.schedular import get_scheduler
from src.train.dist_utils import all_gather

# logger = get_logger(__name__, log_level='info')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)
os.environ["WANDB_MODE"] = "offline"


@dataclass
class ConfigPathArguments:
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None,
                                     metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    # model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    visual_encoder: Optional[str] = field(default=None, metadata={"help": "config path of visual encoder"})
    text_encoder: Optional[str] = field(default=None, metadata={"help": "config path of visual encoder"})
    discrete_model: Optional[str] = field(default=None, metadata={"help": "config path of discrete model"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}, )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."})
    resume_steps: Optional[int] = field(default=None, metadata={"help": "The training sterps of saved checkpoint"})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    mixed_precision: Optional[str] = field(
        default='no',
        metadata={
            "help":
                "Whether to use mixed precision. \
                    Choose between fp16 and bf16 (bfloat16). \
                        Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU."
        })
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "Total number of training steps to perform. "})
    save_steps: int = field(default=10000, metadata={"help": "Number of updates steps before two checkpoint saves."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    min_lr_ratio: float = field(default=0.01, metadata={"help": "Minimal learning rate ratio."})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    project_name: str = field(default="DiscreteLearning", metadata={"help": "The name of experiment"})
    expr_name: str = field(default="", metadata={"help": "The name of experiment"})


def build_dataloader(dataset_cfg, image_transform, tokenizer, dataloader_num_workers=4):
    dataset = hydra.utils.instantiate(dataset_cfg, image_transform=image_transform, tokenizer=tokenizer)
    mp_service = MultiProcessingReadingService(num_workers=dataloader_num_workers)
    dist_service = DistributedReadingService()
    reading_service = SequentialReadingService(dist_service, mp_service)
    dataloader = DataLoader2(dataset, reading_service=reading_service)
    return dataloader


def get_metric(output):
    metric = {}
    for key, value in output.items():
        if 'loss' in key:
            metric[key] = value.item()
    return metric


def get_code_usage(indices):
    indices_list = all_gather(indices)
    indices = torch.cat(indices_list, dim=0)
    code_usage = indices.unique().numel()
    return code_usage


def merge_config(**kwargs):
    config = {}
    for key, value in kwargs.items():
        if isinstance(value, argparse.Namespace):
            config[key] = vars(value)
        elif isinstance(value, DictConfig):
            config[key] = OmegaConf.to_object(value)
        elif is_dataclass(value):
            config[key] = asdict(value)
        elif isinstance(value, dict):
            config[key] = value
        else:
            logger.error(f'key: {key}, value: {value} will not be merged.')
    return config


def trainable_params(model):
    count = 0
    for name, param in model.named_parameters():
        count += param.numel()
    return count


def train():
    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, args = parser.parse_args_into_dataclasses()

    project_config = ProjectConfiguration(project_dir=args.output_dir,
                                          logging_dir=os.path.join(args.output_dir, 'logs'))

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=['tensorboard', 'wandb'],
        project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
    )
    logger.info('Init accelerator done.')

    os.makedirs(args.output_dir, exist_ok=True)

    visual_encoder_cfg = OmegaConf.load(cfg_path.visual_encoder)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    logger.info('Load visual encoder done.')

    discrete_model_cfg = OmegaConf.load(cfg_path.discrete_model)
    discrete_model = hydra.utils.instantiate(discrete_model_cfg)
    logger.info('Load discrete model done.')

    train_dataset_cfg = OmegaConf.load(cfg_path.train_dataset)

    if cfg_path.text_encoder is not None:
        text_encoder_cfg = OmegaConf.load(cfg_path.text_encoder)
        text_encoder = hydra.utils.instantiate(text_encoder_cfg)
    else:
        text_encoder_cfg = None
        text_encoder = None

    if cfg_path.image_transform is not None:
        image_transform_cfg = OmegaConf.load(cfg_path.image_transform)
        image_transform = hydra.utils.instantiate(image_transform_cfg)
    else:
        image_transform_cfg = None
        image_transform = None

    if cfg_path.tokenizer is not None:
        tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
        tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    else:
        tokenizer_cfg = None
        tokenizer = None

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    visual_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info('Freeze visual encoder...')
    visual_encoder.requires_grad_(False)
    if text_encoder is not None:
        logger.info('Freeze text encoder...')
        text_encoder.requires_grad_(False)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    discrete_model.to(accelerator.device, dtype=weight_dtype)

    discrete_model = accelerator.prepare(discrete_model)
    optimizer = torch.optim.AdamW(discrete_model.parameters(),
                                  lr=args.learning_rate,
                                  betas=[args.adam_beta1, args.adam_beta2],
                                  eps=args.adam_epsilon,
                                  weight_decay=args.weight_decay)
    logger.info('Init optimizer done.')
    scheduler = get_scheduler(name=args.lr_scheduler_type,
                              optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.max_steps,
                              min_lr_ratio=args.min_lr_ratio)
    # accelerator.register_for_checkpointing(scheduler)

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    logger.info('Prepare accelerator done.')

    config_record = merge_config(discrete_model=discrete_model_cfg,
                                 visual_encoder=visual_encoder_cfg,
                                 text_encoder=text_encoder_cfg,
                                 image_transform=image_transform_cfg,
                                 tokenizer=tokenizer_cfg,
                                 train_dataset=train_dataset_cfg,
                                 train_args=args)
    accelerator.init_trackers(project_name=args.project_name,
                              init_kwargs={"wandb": {
                                  "config": config_record,
                                  "name": args.expr_name,
                                  "dir": args.output_dir
                              }})
    if args.resume_from_checkpoint is not None:
        logger.info(f'Load checkpoint from {args.resume_from_checkpoint}')
        accelerator.load_state(args.resume_from_checkpoint)

    num_params = trainable_params(discrete_model)
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    logger.info(f"  Total trainable params = {num_params}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    if args.resume_steps is not None:
        global_step = args.resume_steps
        progress_bar.update(args.resume_steps)

    train_dataloader = build_dataloader(dataset_cfg=train_dataset_cfg,
                                        image_transform=image_transform,
                                        tokenizer=tokenizer,
                                        dataloader_num_workers=args.dataloader_num_workers)
    for epoch in range(args.num_train_epochs):
        discrete_model.train()
        logger.info('Start new epoch')

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(discrete_model):
                with torch.no_grad():
                    image_embeds = visual_encoder(batch['images'].to(accelerator.device, dtype=weight_dtype))
                    if text_encoder is not None:
                        text_embeds = text_encoder(batch['text_input_ids'].to(accelerator.device))
                    else:
                        text_embeds = None

                output = discrete_model(image_embeds=image_embeds, text_embeds=text_embeds)

                loss = output['total_loss']
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discrete_model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            metric = get_metric(output)
            metric['lr'] = optimizer.param_groups[0]['lr']
            metric['code_usage'] = get_code_usage(output['indices'])
            metric = {key: (format(value, ".6f") if isinstance(value, float) else value) for key, value in
                      metric.items()}
            accelerator.log(metric, step=global_step)
            if accelerator.is_main_process:
                tqdm.write(str(metric))
            # print(metric)
            if global_step >= args.max_steps:
                break

    accelerator.end_training()


if __name__ == '__main__':
    train()
