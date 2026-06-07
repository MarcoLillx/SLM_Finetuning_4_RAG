"""
LLM generation module.

Loads Qwen3-1.7B (base or fine-tuned) and generates answers
given a context and query.
"""
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.config import Config, get_config
#from src.data.dataset_builder import get_inference_prompt

logger = logging.getLogger(__name__)


class Generator:
    """LLM-based answer generator for the RAG pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        use_finetuned: bool = False,
    ):
        """
        Initialize the generator.
        
        Args:
            config: Project configuration
            use_finetuned: If True, load the fine-tuned LoRA adapter
        """
        if config is None:
            config = get_config()
        self.config = config
        self.use_finetuned = use_finetuned

        self.model, self.tokenizer = self._load_model()
        logger.info(
            f"Generator initialized ({'fine-tuned' if use_finetuned else 'base'} model)"
        )

    def _load_model(self):
        """Load the LLM with optional LoRA adapter."""
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load fine-tuned adapter if requested
        if self.use_finetuned:
            adapter_path = self.config.training.output_dir
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            logger.info("LoRA adapter merged successfully")

        model.eval()
        return model, tokenizer

    def generate(
        self,
        context: str,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate an answer given context and question.
        
        Args:
            context: Retrieved document context
            question: User question
            max_new_tokens: Override max tokens
            
        Returns:
            Generated answer string
        """
        prompt = get_inference_prompt(context=context, question=question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.training.max_seq_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        max_tokens = max_new_tokens or self.config.model.max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                do_sample=self.config.model.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

        # Decode only generated tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated.strip()

    def cleanup(self):
        """Free GPU memory."""
        del self.model, self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Generator cleanup complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gen = Generator(use_finetuned=False)
    answer = gen.generate(
        context="L'articolo 1 stabilisce le norme per gli appalti pubblici.",
        question="Cosa stabilisce l'articolo 1?",
    )
    print(f"\nGenerated answer:\n{answer}")
    gen.cleanup()
