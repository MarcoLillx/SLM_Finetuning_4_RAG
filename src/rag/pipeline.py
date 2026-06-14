"""
End-to-end RAG pipeline.

Orchestrates retrieval and generation into a complete
question-answering pipeline.
"""
import logging
import time
from typing import Optional

from src.config import Config, get_config
from src.rag.retriever import Retriever
from src.rag.generator import Generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: query → retrieve → generate."""

    def __init__(
        self,
        config: Optional[Config] = None,
        use_finetuned: bool = False,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Project configuration
            use_finetuned: Use fine-tuned model instead of base
        """
        if config is None:
            config = get_config()
        self.config = config
        self.use_finetuned = use_finetuned

        logger.info("Initializing RAG pipeline...")
        self.retriever = Retriever(config)
        self.generator = Generator(config, use_finetuned=use_finetuned)
        logger.info(
            f"RAG pipeline ready ({'fine-tuned' if use_finetuned else 'base'} model)"
        )

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Run the full RAG pipeline for a question.
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dict with keys: question, answer, contexts, retrieval_time, generation_time
        """
        start_time = time.time()

        # Retrieve relevant context
        retrieved_chunks = self.retriever.retrieve(question, top_k)
        retrieval_time = time.time() - start_time

        # Build context string from the already-retrieved chunks
        
        context = "\n\n---\n\n".join(
            f"[Fonte: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
        )

        # Generate answer
        gen_start = time.time()
        answer = self.generator.generate(context=context, question=question)
        generation_time = time.time() - gen_start

        total_time = time.time() - start_time

        result = {
            "question": question,
            "answer": answer,
            "contexts": [chunk["text"] for chunk in retrieved_chunks],
            "source_documents": [chunk["source"] for chunk in retrieved_chunks],
            "retrieval_scores": [chunk["score"] for chunk in retrieved_chunks],
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
            "total_time_s": round(total_time, 3),
            "model_type": "finetuned" if self.use_finetuned else "base",
        }

        logger.info(
            f"Query completed in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"
        )
        return result

    def batch_query(self, questions: list[str], top_k: Optional[int] = None) -> list[dict]:
        """
        Run the RAG pipeline on a batch of questions.
        
        Args:
            questions: List of questions
            top_k: Number of context chunks per question
            
        Returns:
            List of result dicts
        """
        results = []
        for i, q in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.query(q, top_k)
            results.append(result)
        return results

    def cleanup(self):
        """Free resources."""
        self.generator.cleanup()
        logger.info("RAG pipeline cleanup complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test with base model
    pipeline = RAGPipeline(use_finetuned=False)

    test_questions = [
        "Quali sono le procedure per gli appalti pubblici?",
        "Cosa prevede la normativa sulle firme elettroniche?",
        "Quali sono gli obblighi della stazione appaltante?",
    ]

    for q in test_questions:
        result = pipeline.query(q)
        print(f"\n{'='*60}")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Time: {result['total_time_s']}s | Sources: {result['source_documents']}")

    pipeline.cleanup()
