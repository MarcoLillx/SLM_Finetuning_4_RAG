"""
Main pipeline orchestrator.

CLI entry point for running all stages of the SLM Fine-Tuning for RAG pipeline.
Usage: python -m src.main <command> [options]
"""
import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_preprocess(args):
    """Run the full data preprocessing pipeline."""
    from src.config import get_config
    from src.data.pdf_extractor import extract_all_pdfs
    from src.data.chunker import chunk_documents
    from src.data.qa_generator import generate_qa_dataset
    from src.data.dataset_builder import build_dataset

    config = get_config()

    logger.info("=" * 60)
    logger.info("PHASE 1: Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # Step 1: Extract PDFs
    logger.info("Step 1/4: Extracting text from PDFs...")
    documents = extract_all_pdfs(config)

    # Step 2: Chunk documents
    logger.info("Step 2/4: Chunking documents...")
    chunks = chunk_documents(config)

    # Step 3: Generate QA pairs
    logger.info("Step 3/4: Generating synthetic QA pairs...")
    qa_pairs = generate_qa_dataset(config)

    # Step 4: Build SFT dataset
    logger.info("Step 4/4: Building training dataset...")
    dataset = build_dataset(config)

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  QA pairs: {len(qa_pairs)}")
    logger.info(f"  Train samples: {len(dataset['train'])}")
    logger.info(f"  Validation samples: {len(dataset['validation'])}")
    logger.info("=" * 60)


def cmd_build_index(args):
    """Build the FAISS vector index."""
    from src.config import get_config
    from src.rag.vectorstore import build_vectorstore

    config = get_config()

    logger.info("=" * 60)
    logger.info("PHASE 2: Building Vector Index")
    logger.info("=" * 60)

    vectorstore = build_vectorstore(config)

    # Quick test
    results = vectorstore.similarity_search("appalti pubblici", k=3)
    logger.info(f"Test query returned {len(results)} results")
    logger.info("Vector index built successfully!")


def cmd_train(args):
    """Run QLoRA fine-tuning."""
    from src.config import get_config
    from src.training.train import train_model

    config = get_config()
    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else None

    logger.info("=" * 60)
    logger.info("PHASE 3: QLoRA Fine-Tuning")
    logger.info("=" * 60)

    metrics = train_model(config, num_epochs=epochs)
    logger.info("Training complete!")


def cmd_evaluate(args):
    """Evaluate a RAG pipeline."""
    from src.config import get_config
    from src.evaluation.evaluate import evaluate_rag

    config = get_config()
    use_finetuned = args.model == "finetuned"

    logger.info("=" * 60)
    logger.info(f"PHASE 4: Evaluating {'Fine-tuned' if use_finetuned else 'Base'} Model")
    logger.info("=" * 60)

    scores = evaluate_rag(config, use_finetuned=use_finetuned)

    logger.info("Evaluation complete!")
    if "aggregate_scores" in scores:
        for metric, value in scores["aggregate_scores"].items():
            logger.info(f"  {metric}: {value}")


def cmd_compare(args):
    """Compare base vs fine-tuned models."""
    from src.config import get_config
    from src.evaluation.compare import run_comparison

    config = get_config()

    logger.info("=" * 60)
    logger.info("PHASE 5: Comparative Analysis")
    logger.info("=" * 60)

    summary = run_comparison(config)
    logger.info("Comparison complete!")


def cmd_run_all(args):
    """Run the entire pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL PIPELINE")
    logger.info("=" * 60)

    start = time.time()

    # Preprocess
    cmd_preprocess(args)

    # Build index
    cmd_build_index(args)

    # Train
    cmd_train(args)

    # Evaluate base
    args.model = "base"
    cmd_evaluate(args)

    # Evaluate fine-tuned
    args.model = "finetuned"
    cmd_evaluate(args)

    # Compare
    cmd_compare(args)

    elapsed = time.time() - start
    logger.info(f"\nFull pipeline completed in {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="SLM Fine-Tuning for RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main preprocess          # Extract, chunk, generate QA, build dataset
  python -m src.main build-index         # Build FAISS vector index
  python -m src.main train               # Fine-tune with QLoRA
  python -m src.main train --epochs 1    # Quick training run
  python -m src.main evaluate --model base       # Evaluate base model
  python -m src.main evaluate --model finetuned  # Evaluate fine-tuned model
  python -m src.main compare             # Compare base vs fine-tuned
  python -m src.main run-all             # Run everything
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # Preprocess
    subparsers.add_parser("preprocess", help="Run data preprocessing pipeline")

    # Build index
    subparsers.add_parser("build-index", help="Build FAISS vector index")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune model with QLoRA")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate RAG pipeline")
    eval_parser.add_argument("--model", choices=["base", "finetuned"], default="base",
                            help="Which model to evaluate")

    # Compare
    subparsers.add_parser("compare", help="Compare base vs fine-tuned")

    # Run all
    all_parser = subparsers.add_parser("run-all", help="Run entire pipeline")
    all_parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "preprocess": cmd_preprocess,
        "build-index": cmd_build_index,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "compare": cmd_compare,
        "run-all": cmd_run_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
