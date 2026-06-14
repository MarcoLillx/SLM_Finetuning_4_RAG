"""
RAG evaluation using RAGAS framework.

Evaluates the RAG pipeline using standard metrics:
Faithfulness, Answer Relevancy, Context Precision, Context Recall.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

from src.config import Config, get_config
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def load_eval_questions(config: Config) -> list[dict]:
    """
    Load evaluation QA pairs from the generated dataset.
    Uses the validation split as evaluation set.
    """
    qa_path = config.paths.qa_dataset_dir / "qa_pairs.json"
    if not qa_path.exists():
        raise FileNotFoundError(f"QA pairs not found at {qa_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        all_qa = json.load(f)

    # Use a subset for evaluation
    n = min(config.evaluation.num_eval_samples, len(all_qa))
    # Take evenly spaced samples for diversity
    step = max(1, len(all_qa) // n)
    eval_qa = all_qa[::step][:n]

    logger.info(f"Loaded {len(eval_qa)} evaluation samples")
    return eval_qa


def evaluate_rag(
    config: Optional[Config] = None,
    use_finetuned: bool = False,
) -> dict:
    """
    Evaluate a RAG pipeline using RAGAS metrics.
    
    Args:
        config: Project configuration
        use_finetuned: Whether to evaluate the fine-tuned model
        
    Returns:
        Dict with metric scores and per-sample results
    """
    if config is None:
        config = get_config()

    model_type = "finetuned" if use_finetuned else "base"
    logger.info(f"Evaluating RAG pipeline ({model_type} model)")

    # Load evaluation questions
    eval_qa = load_eval_questions(config)

    # Initialize RAG pipeline
    pipeline = RAGPipeline(config=config, use_finetuned=use_finetuned)

    # Run RAG on evaluation questions
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, qa in enumerate(eval_qa):
        logger.info(f"Evaluating {i+1}/{len(eval_qa)}")
        try:
            result = pipeline.query(qa["question"])
            questions.append(qa["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(qa["answer"])
        except Exception as e:
            logger.error(f"Error on question {i+1}: {e}")
            continue

    pipeline.cleanup()

    # Build RAGAS evaluation dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    logger.info("Running RAGAS evaluation...")
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        ragas_result = ragas_evaluate(eval_dataset, metrics=metrics)

        scores = {
            "model_type": model_type,
            "num_samples": len(questions),
            "aggregate_scores": dict(ragas_result),
        }

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        logger.info("Falling back to manual evaluation metrics...")
        scores = _manual_evaluation(questions, answers, contexts, ground_truths, model_type)

    # Save results
    output_path = config.paths.eval_dir / f"ragas_results_{model_type}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Save detailed per-sample results
    details = []
    for i in range(len(questions)):
        details.append({
            "question": questions[i],
            "generated_answer": answers[i],
            "ground_truth": ground_truths[i],
            "num_contexts": len(contexts[i]),
        })

    details_path = config.paths.eval_dir / f"eval_details_{model_type}.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    return scores


def _manual_evaluation(
    questions: list,
    answers: list,
    contexts: list,
    ground_truths: list,
    model_type: str,
) -> dict:
    """
    Fallback manual evaluation when RAGAS is unavailable.
    Computes simple text-based metrics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Answer Relevancy: cosine similarity between question and answer
    answer_relevancy_scores = []
    # Faithfulness proxy: overlap between answer and context
    faithfulness_scores = []
    # Context relevancy: overlap between question and context
    context_scores = []

    for i in range(len(questions)):
        # Answer relevancy via TF-IDF cosine similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([questions[i], answers[i]])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            answer_relevancy_scores.append(float(sim))
        except Exception:
            answer_relevancy_scores.append(0.0)

        # Faithfulness: word overlap between answer and context
        answer_words = set(answers[i].lower().split())
        context_text = " ".join(contexts[i]).lower()
        context_words = set(context_text.split())
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            faithfulness_scores.append(overlap)
        else:
            faithfulness_scores.append(0.0)

        # Context relevancy
        question_words = set(questions[i].lower().split())
        if question_words:
            ctx_overlap = len(question_words & context_words) / len(question_words)
            context_scores.append(ctx_overlap)
        else:
            context_scores.append(0.0)

    # Ground truth similarity
    gt_scores = []
    for i in range(len(answers)):
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([ground_truths[i], answers[i]])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            gt_scores.append(float(sim))
        except Exception:
            gt_scores.append(0.0)

    scores = {
        "model_type": model_type,
        "num_samples": len(questions),
        "aggregate_scores": {
            "answer_relevancy": float(np.mean(answer_relevancy_scores)),
            "faithfulness": float(np.mean(faithfulness_scores)),
            "context_relevancy": float(np.mean(context_scores)),
            "ground_truth_similarity": float(np.mean(gt_scores)),
        },
        "per_sample": {
            "answer_relevancy": answer_relevancy_scores,
            "faithfulness": faithfulness_scores,
            "context_relevancy": context_scores,
            "ground_truth_similarity": gt_scores,
        },
    }

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Evaluate base model
    print("Evaluating base model...")
    base_scores = evaluate_rag(use_finetuned=False)
    print(f"\nBase model scores: {json.dumps(base_scores['aggregate_scores'], indent=2)}")
