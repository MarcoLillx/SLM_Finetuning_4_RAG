# SLM Fine-Tuning for RAG

**Fine-tuning Qwen3-1.7B with QLoRA for Retrieval-Augmented Generation on Italian Public Administration Documents**

---

## Architecture

```
User Query → Retriever (FAISS + Multilingual Embeddings) → Top-k Chunks → Generator (Qwen3-1.7B / QLoRA Fine-tuned) → Answer
```

### Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| **Preprocessing** | `python -m src.main preprocess` | PDF extraction → Chunking → Synthetic QA generation → SFT dataset |
| **Indexing** | `python -m src.main build-index` | Build FAISS vector index from chunks |
| **Training** | `python -m src.main train` | QLoRA fine-tuning with SFTTrainer |
| **Evaluation** | `python -m src.main evaluate --model base/finetuned` | RAGAS evaluation metrics |
| **Comparison** | `python -m src.main compare` | Base vs fine-tuned analysis + plots |
| **Full Pipeline** | `python -m src.main run-all` | Run everything end-to-end |

---

## Technical Stack

- **Base Model**: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Fine-tuning**: QLoRA (4-bit NF4 quantization + LoRA r=8)
- **Training**: SFTTrainer from TRL with paged AdamW 8-bit
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Store**: FAISS
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
- **Framework**: HuggingFace Transformers + PEFT + LangChain

---

## 📁 Project Structure

```
├── src/
│   ├── config.py                 # Central configuration
│   ├── data/
│   │   ├── pdf_extractor.py      # PDF → raw text via PyMuPDF
│   │   ├── chunker.py            # Recursive character text splitting
│   │   ├── qa_generator.py       # Synthetic QA via Qwen3-1.7B
│   │   └── dataset_builder.py    # SFT instruction-tuning format
│   ├── rag/
│   │   ├── vectorstore.py        # FAISS index management
│   │   ├── retriever.py          # Top-k similarity retrieval
│   │   ├── generator.py          # LLM answer generation
│   │   └── pipeline.py           # End-to-end RAG orchestrator
│   ├── training/
│   │   ├── lora_config.py        # QLoRA/BnB configuration
│   │   └── train.py              # QLoRA training with SFTTrainer
│   ├── evaluation/
│   │   ├── evaluate.py           # RAGAS evaluation
│   │   └── compare.py            # Comparative analysis + plots
│   └── main.py                   # CLI pipeline orchestrator
├── notebooks/
│   └── colab_training.ipynb      # Google Colab training notebook
├── data/                         # Italian PA regulation PDFs
├── outputs/                      # Generated artifacts
│   ├── chunks/                   # Extracted text & chunks
│   ├── qa_dataset/               # Synthetic QA pairs & SFT dataset
│   ├── vectorstore/              # FAISS index
│   ├── models/                   # Fine-tuned model weights
│   ├── evaluation/               # Metric scores (JSON)
│   └── plots/                    # Comparison charts
└── requirements.txt
```

---

## Quick Start

### Local (GTX 1650 / any GPU)

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Run pipeline
python -m src.main preprocess   # ~30 min (QA generation is slow on small GPU)
python -m src.main build-index  # ~2 min
python -m src.main train        # ~1-2 hours on GTX 1650
python -m src.main evaluate --model base
python -m src.main evaluate --model finetuned
python -m src.main compare
```

### Google Colab (T4 GPU — Recommended for Training)

1. Upload project to Google Drive
2. Open `notebooks/slm-finetuning-rag-pa.ipynb`
3. Set Runtime → T4 GPU
4. Follow the notebook cells

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Are the claims in the answer supported by the retrieved context? |
| **Answer Relevancy** | Does the answer address the original question? |
| **Context Precision** | Are relevant documents ranked higher? |
| **Context Recall** | Does the context contain all needed information? |

---

## Data

16 Italian Public Administration regulatory documents from Empulia, including:
- EU Directives (2014/23/UE, 2014/24/UE, 2014/25/UE)
- Italian Legislative Decrees (D.Lgs. 50/2016, D.Lgs. 163/2006)
- Regional laws and deliberations
- Financial laws (Finanziaria 2000, 2001, 2007)

---

## License

MIT License
