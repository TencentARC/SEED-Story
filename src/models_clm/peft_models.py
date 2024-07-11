from peft import (
    LoraConfig,
    PeftModel,
    LoraModel,
    PeftModelForCausalLM,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import _set_trainable, PromptLearningConfig
from peft.utils import PeftConfig

import torch
from transformers import LlamaForCausalLM
from omegaconf import DictConfig
import hydra


def get_peft_model_with_resize_embedding(
        model,
        peft_config=None,
        model_id=None,
        vocab_size=None,
        torch_dtype='bf16'
):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    # model.gradient_checkpointing_enable()

    assert (peft_config is None) + (model_id is None) == 1

    # print(type(peft_config.target_modules))
    if vocab_size is not None:
        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)

    if peft_config is not None:
        print('peft config: ', peft_config)
        peft_model = get_peft_model(model=model, peft_config=peft_config)
        peft_model.get_input_embeddings().requires_grad_(True)
        peft_model.get_output_embeddings().requires_grad_(True)

        peft_model.print_trainable_parameters()

        # param_count = 0
        # if peft_model.modules_to_save is not None:
        #     for name, param in peft_model.named_parameters():
        #         if any(module_name in name for module_name in peft_model.modules_to_save):
        #             param_count += param.numel()
        #             print(name, param.numel())

    else:
        peft_model = PeftModel.from_pretrained(model=model, model_id=model_id)

    return peft_model


def get_model_with_resize_embedding(model, vocab_size=None, torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    model.requires_grad_(False)
    if vocab_size is not None:
        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)

    return model


def get_full_model_with_resize_embedding(model, vocab_size=None, torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    if vocab_size is not None:
        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)

    return model
