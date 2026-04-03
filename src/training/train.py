"""
QLoRA fine-tuning script.

Fine-tunes Qwen3-1.7B using QLoRA (4-bit quantization + LoRA adapters)
with the SFTTrainer from TRL on the synthetic QA dataset.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import Config, get_config
from src.training.lora_config import get_lora_config, get_bnb_config

logger = logging.getLogger(__name__)


def train_model(config: Optional[Config] = None, num_epochs: Optional[int] = None):
    """
    Fine-tune Qwen3-1.7B with QLoRA.
    
    Args:
        config: Project configuration
        num_epochs: Override number of training epochs
    """
    if config is None:
        config = get_config()

    if num_epochs is not None:
        config.training.num_train_epochs = num_epochs

    logger.info("=" * 60)
    logger.info("Starting QLoRA Fine-Tuning")
    logger.info("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    dataset_path = config.paths.qa_dataset_dir / "sft_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. "
            "Run dataset_builder first."
        )

    dataset = load_from_disk(str(dataset_path))
    logger.info(
        f"Dataset loaded — Train: {len(dataset['train'])}, "
        f"Validation: {len(dataset['validation'])}"
    )

    # ── 2. Load tokenizer ────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 3. Load model with 4-bit quantization ────────────────────────────
    logger.info(f"Loading model: {config.model.model_id} (4-bit quantized)")
    bnb_config = get_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # ── 4. Apply LoRA adapters ───────────────────────────────────────────
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # ── 5. Training arguments ────────────────────────────────────────────
    output_dir = config.training.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        max_grad_norm=config.training.max_grad_norm,
        max_seq_length=config.training.max_seq_length,
        report_to=config.training.report_to,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        gradient_checkpointing=True,  # Save VRAM
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        seed=42,
    )

    # ── 6. Initialize SFTTrainer ─────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────
    logger.info("Starting training...")
    train_result = trainer.train()

    # ── 8. Save ──────────────────────────────────────────────────────────
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training log history
    log_history = trainer.state.log_history
    log_path = Path(output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Total steps: {trainer.state.global_step}")
    logger.info(f"  Training loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Training runtime: {metrics.get('train_runtime', 'N/A'):.1f}s")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info("=" * 60)

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = train_model()
    print(f"\nTraining metrics: {json.dumps(metrics, indent=2)}")
