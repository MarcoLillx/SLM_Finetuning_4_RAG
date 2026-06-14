"""
LoRA and quantization configuration.

Provides factory functions for creating PEFT and BitsAndBytes configs
from the project configuration.
"""
import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig

from src.config import Config, get_config


def get_lora_config(config: Config = None) -> LoraConfig:
    """Create LoRA configuration from project config."""
    if config is None:
        config = get_config()

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def get_bnb_config(config: Config = None) -> BitsAndBytesConfig:
    """Create BitsAndBytes 4-bit quantization config."""
    if config is None:
        config = get_config()

    compute_dtype = (
        torch.float16 if config.quantization.bnb_4bit_compute_dtype == "float16"
        else torch.bfloat16
    )

    return BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )
