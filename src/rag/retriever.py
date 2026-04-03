"""
Document retriever.

Wraps the FAISS vectorstore with top-k similarity search
and returns ranked document chunks.
"""
import logging
from typing import Optional

from langchain.schema import Document

from src.config import Config, get_config
from src.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks from the FAISS vector store."""

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        self.vectorstore = load_vectorstore(config)
        logger.info("Retriever initialized")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (default from config)
            
        Returns:
            List of dicts with 'text', 'source', 'chunk_id', 'score'
        """
        k = top_k or self.config.rag.top_k

        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        retrieved = []
        for doc, score in results_with_scores:
            # FAISS returns L2 distance; lower = more similar
            # Convert to similarity score (higher = better)
            similarity = 1.0 / (1.0 + score)

            retrieved.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": float(similarity),
            })

        logger.debug(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
        return retrieved

    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve chunks and concatenate them into a single context string.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            Concatenated context string
        """
        chunks = self.retrieve(query, top_k)
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Fonte: {chunk['source']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    retriever = Retriever()
    results = retriever.retrieve("Quali sono le procedure per gli appalti pubblici?")
    print(f"\nRetrieved {len(results)} chunks:")
    for r in results:
        print(f"  Score: {r['score']:.4f} | {r['source']} | {r['text'][:80]}...")
