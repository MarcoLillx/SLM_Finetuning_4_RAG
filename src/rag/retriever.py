"""
Advanced Document retriever.

Implements:
1. DenseRetriever (FAISS + embeddings)
2. KGRetriever (Knowledge Graph 1-hop traversal)
3. HybridRetriever (Reciprocal Rank Fusion)
4. CrossEncoderReranker (Re-ranking)
"""
import json
import logging
from collections import defaultdict
from typing import Optional

import faiss
import numpy as np

from src.config import Config, get_config

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Simple entity extraction using regex (matches Kaggle implementation)."""
    import re
    patterns = {
        'articolo': r'(?i)art\.\s*\d+|articolo\s*\d+',
        'legge/decreto': r'(?i)d\.lgs\.\s*\d+/\d+|legge\s*\d+/\d+',
        'ente': r'(?i)stazione appaltante|pubblica amministrazione|anac|consiglio di stato',
        'concetto': r'(?i)appalto|subappalto|concessione|gara|bando|offerta|aggiudicazione'
    }
    entities = []
    for etype, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append((match.group().lower().strip(), etype))
    return entities


class DenseRetriever:
    """Retrieves relevant document chunks from the raw FAISS vector store."""
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        
        faiss_path = config.paths.data_dir / "faiss_index.bin"
        chunks_path = config.paths.data_dir / "chunks.json"
        
        logger.info(f"Loading raw FAISS index from {faiss_path}")
        self.index = faiss.read_index(str(faiss_path))
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(config.model.embedding_model_id)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.config.rag.top_k
        
        emb = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, I = self.index.search(np.array(emb, dtype=np.float32), k)

        retrieved = []
        for idx, score in zip(I[0], scores[0]):
            if idx != -1:
                chunk = self.chunks[idx].copy()
                # Use absolute score mapping or keep dot product
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": float(score),
                })
        return retrieved


class KGRetriever:
    """Retrieves chunks based on Knowledge Graph entity matching."""
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
            
        kg_path = config.paths.data_dir / "knowledge_graph.json"
        chunks_path = config.paths.data_dir / "chunks.json"
        
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        self.chunk_map = {c['chunk_id']: c for c in chunks}
        self.entity_chunks = kg.get('entity_chunks', {})
        self.entity_relations = defaultdict(list)
        
        for t in kg.get('triples', []):
            self.entity_relations[t['subject'].lower()].append(t)
            self.entity_relations[t['object'].lower()].append(t)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qents = extract_entities(query)
        if not qents:
            return []
            
        chunk_scores = defaultdict(float)
        chunk_matches = defaultdict(set)
        
        for entity, _ in qents:
            key = entity.lower()
            # Direct matches
            for cid in self.entity_chunks.get(key, []):
                chunk_scores[cid] += 1.0
                chunk_matches[cid].add(entity)
            # 1-hop related entities
            for triple in self.entity_relations.get(key, []):
                related = (triple['object'].lower()
                           if triple['subject'].lower() == key
                           else triple['subject'].lower())
                for cid in self.entity_chunks.get(related, []):
                    chunk_scores[cid] += 0.5
                    chunk_matches[cid].add(entity)
                    
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [{
            'text': self.chunk_map[cid]['text'],
            'source': self.chunk_map[cid]['source'],
            'chunk_id': cid,
            'score': float(sc),
            'matched_entities': list(chunk_matches[cid])
        } for cid, sc in ranked if cid in self.chunk_map]


class HybridRetriever:
    """Reciprocal Rank Fusion of Dense + KG retrievers."""
    def __init__(self, dense: DenseRetriever, kg: KGRetriever, rrf_k: int = 60):
        self.dense = dense
        self.kg = kg
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        d_res = self.dense.retrieve(query, top_k=top_k * 2)
        k_res = self.kg.retrieve(query, top_k=top_k * 2)
        
        rrf = defaultdict(float)
        data = {}
        
        for rank, r in enumerate(d_res):
            rrf[r['chunk_id']] += 1.0 / (self.rrf_k + rank + 1)
            data[r['chunk_id']] = r
            
        for rank, r in enumerate(k_res):
            rrf[r['chunk_id']] += 1.0 / (self.rrf_k + rank + 1)
            if r['chunk_id'] not in data:
                data[r['chunk_id']] = r
                
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for cid, score in ranked:
            d = data[cid].copy()
            d['score'] = float(score) # overwrite score with rrf score
            results.append(d)
        return results


class CrossEncoderReranker:
    """Reranks retrieved chunks using a Cross-Encoder model."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder model: {model_name}")
        # Note: model will be downloaded automatically if not present
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []
            
        # Prepare pairs of (query, chunk_text)
        pairs = [[query, c['text']] for c in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Add scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
            
        ranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return ranked_chunks[:top_k]


class AdvancedRetriever:
    """Facade class combining Hybrid Retrieval + CrossEncoder Reranking."""
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        self.dense = DenseRetriever(config)
        self.kg = KGRetriever(config)
        self.hybrid = HybridRetriever(self.dense, self.kg)
        self.reranker = CrossEncoderReranker()
        logger.info("Advanced Retriever initialized")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.config.rag.top_k
        
        # Retrieve twice as many chunks using Hybrid
        hybrid_chunks = self.hybrid.retrieve(query, top_k=k * 2)
        
        # Rerank to get final top-k
        final_chunks = self.reranker.rerank(query, hybrid_chunks, top_k=k)
        
        return final_chunks
        
    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        chunks = self.retrieve(query, top_k)
        context_parts = []
        for i, chunk in enumerate(chunks):
            score_str = f"Rerank Score: {chunk.get('rerank_score', 0):.2f}"
            context_parts.append(
                f"[Fonte: {chunk['source']} | {score_str}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    retriever = AdvancedRetriever()
    results = retriever.retrieve("Quali sono le procedure per gli appalti pubblici?")
    print(f"\nRetrieved {len(results)} chunks:")
    for r in results:
        print(f"  Rerank: {r['rerank_score']:.4f} | {r['source']} | {r['text'][:80]}...")
