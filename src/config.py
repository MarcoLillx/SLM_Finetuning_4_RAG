"""
Central configuration for the SLM Fine-Tuning for RAG project.
All paths, model IDs, and hyperparameters are defined here.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


# ── Project root (two levels up from this file) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    """All project paths."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Output subdirectories
    chunks_dir: Path = PROJECT_ROOT / "outputs" / "chunks"
    qa_dataset_dir: Path = PROJECT_ROOT / "outputs" / "qa_dataset"
    vectorstore_dir: Path = PROJECT_ROOT / "outputs" / "vectorstore"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    eval_dir: Path = PROJECT_ROOT / "outputs" / "evaluation"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"

    def create_all(self):
        """Create all output directories."""
        for p in [
            self.output_dir, self.chunks_dir, self.qa_dataset_dir,
            self.vectorstore_dir, self.models_dir, self.eval_dir,
            self.plots_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model identifiers and generation settings."""
    # Base LLM
    model_id: str = "Qwen/Qwen3-1.7B"
    # Embedding model (multilingual for Italian docs)
    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class ChunkingConfig:
    """Document chunking parameters."""
    chunk_size: int = 512       # characters
    chunk_overlap: int = 64     # characters
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class QAGenerationConfig:
    """Synthetic QA pair generation settings."""
    num_qa_per_chunk: int = 2          # QA pairs to generate per chunk
    max_chunks_to_process: int = None  # None = all chunks
    batch_size: int = 1                # generation batch size (low for 4GB VRAM)


@dataclass
class LoRAConfig:
    """LoRA / QLoRA adapter configuration."""
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization settings."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters for SFTTrainer."""
    output_dir: str = str(PROJECT_ROOT / "outputs" / "models" / "qwen3-1.7b-qlora-rag")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    eval_strategy: str = "steps"
    eval_steps: int = 50
    fp16: bool = True
    bf16: bool = False
    max_grad_norm: float = 0.3
    max_seq_length: int = 1024
    report_to: str = "none"


@dataclass
class RAGConfig:
    """RAG pipeline settings."""
    top_k: int = 3                  # Number of chunks to retrieve
    similarity_threshold: float = 0.0  # Minimum similarity score


@dataclass
class EvalConfig:
    """Evaluation settings."""
    num_eval_samples: int = 50       # Number of QA samples for evaluation
    ragas_metrics: list = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )


# ── Master config ─────────────────────────────────────────────────────────
@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    qa_gen: QAGenerationConfig = field(default_factory=QAGenerationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        self.paths.create_all()


def get_config() -> Config:
    """Factory function to get the project config."""
    return Config()
