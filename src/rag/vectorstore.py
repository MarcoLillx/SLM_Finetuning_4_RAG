"""
FAISS vector store builder.

Creates and manages a FAISS index from document chunks using
a multilingual sentence transformer for embeddings.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import Config, get_config

logger = logging.getLogger(__name__)


def get_embedding_model(config: Optional[Config] = None) -> HuggingFaceEmbeddings:
    """Load the multilingual embedding model."""
    if config is None:
        config = get_config()

    logger.info(f"Loading embedding model: {config.model.embedding_model_id}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.model.embedding_model_id,
        model_kwargs={"device": "cpu"},  # Keep embeddings on CPU to save GPU for LLM
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def build_vectorstore(config: Optional[Config] = None) -> FAISS:
    """
    Build FAISS vector store from document chunks.
    
    Returns:
        FAISS vectorstore instance
    """
    if config is None:
        config = get_config()

    # Load chunks
    chunks_path = config.paths.chunks_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found at {chunks_path}. Run chunker first."
        )

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks for indexing")

    # Convert to LangChain Documents
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk in chunks
    ]

    # Create embeddings
    embeddings = get_embedding_model(config)

    # Build FAISS index
    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    logger.info(f"FAISS index built with {len(documents)} vectors")

    # Save index
    save_path = str(config.paths.vectorstore_dir)
    vectorstore.save_local(save_path)
    logger.info(f"FAISS index saved to {save_path}")

    return vectorstore


def load_vectorstore(config: Optional[Config] = None) -> FAISS:
    """Load a previously saved FAISS vector store."""
    if config is None:
        config = get_config()

    embeddings = get_embedding_model(config)
    save_path = str(config.paths.vectorstore_dir)

    logger.info(f"Loading FAISS index from {save_path}")
    vectorstore = FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    vs = build_vectorstore()
    # Test query
    results = vs.similarity_search("appalti pubblici", k=3)
    print(f"\nTest query 'appalti pubblici' → {len(results)} results:")
    for r in results:
        print(f"  [{r.metadata['source']}] {r.page_content[:100]}...")
