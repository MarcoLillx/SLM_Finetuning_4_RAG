import streamlit as st
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.rag.retriever import AdvancedRetriever
from src.config import get_config

warnings.filterwarnings("ignore")

# Force full width for better UI
st.set_page_config(page_title="RAG Legal Assistant", layout="wide")

config = get_config()

@st.cache_resource
def load_rag_components():
    """Load the Retriever, Tokenizer, and QLoRA Model into GPU memory."""
    retriever = AdvancedRetriever(config)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    import torch_directml
    dml_device = torch_directml.device()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    adapter_path = config.paths.project_root / "notebooks" / "qlora_adapter"
    if not adapter_path.exists():
        st.error(f"Adapter directory not found at {adapter_path}. Did you download it from Kaggle?")
        st.stop()
        
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(dml_device)
    model.eval()
    
    return retriever, tokenizer, model

def generate_answer(query, contexts, model, tokenizer):
    context_text = "\n\n---\n\n".join(
        [f"[Fonte: {c['source']} | Rerank: {c.get('rerank_score', 0):.2f}]\n{c['text']}" for c in contexts]
    )
    prompt = (
        "### Istruzione:\nSei un assistente esperto di normativa italiana per la Pubblica Amministrazione. "
        "Rispondi alla domanda basandoti esclusivamente sul contesto fornito. "
        "Se il contesto non contiene informazioni sufficienti, dichiaralo esplicitamente.\n\n"
        f"### Contesto:\n{context_text}\n\n"
        f"### Domanda:\n{query}\n\n"
        "### Risposta:\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.1, 
            top_p=0.95, 
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return answer, context_text

def generate_answer_from_knowledge(query, model, tokenizer):
    """Generate an answer using only the model's internal knowledge (no RAG context)."""
    prompt = (
        "### Istruzione:\nSei un assistente esperto di normativa italiana e diritto pubblico. "
        "Rispondi alla domanda utilizzando la tua conoscenza generale del diritto italiano e "
        "della normativa europea. Fornisci una risposta accurata e utile.\n\n"
        f"### Domanda:\n{query}\n\n"
        "### Risposta:\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.3, 
            top_p=0.95, 
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return answer, ""


st.title("⚖️ Assistente Legale PA")
st.markdown("Finetuned **Qwen3-1.7B** + **Hybrid Retrieval** (Dense & KG) + **CrossEncoder Reranking**")

with st.spinner("Caricamento Modelli nella GPU locale in corso..."):
    retriever, tokenizer, model = load_rag_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg:
            with st.expander("Mostra Documenti Recuperati e Punteggi"):
                st.markdown(msg["context"])

if query := st.chat_input("Fai una domanda sulla normativa..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Ricerca e Reranking dei documenti..."):
            chunks = retriever.retrieve(query)
            
        # Check if retrieved documents are relevant enough
        top_score = chunks[0].get("rerank_score", -10.0) if chunks else -10.0
        
        # Format retrieved docs for display (always shown)
        context_display = ""
        for chunk in chunks[:3]:
            score = chunk.get('rerank_score', -10.0)
            source = chunk.get('source', 'Sconosciuta')
            text = chunk.get('text', '')
            context_display += f"**[Fonte: {source} | Rerank: {score:.2f}]** {text[:500]}...\n\n"
        
        if top_score < 0.5:
            # Low relevance: let the model answer from its own knowledge
            with st.spinner("Documenti non sufficienti. Generazione risposta dal modello..."):
                answer, _ = generate_answer_from_knowledge(query, model, tokenizer)
                st.info("⚠️ I documenti recuperati non contengono informazioni sufficienti. "
                        "La risposta è generata dalla conoscenza interna del modello.")
                st.markdown(answer)
        else:
            # High relevance: standard RAG generation
            with st.spinner("Generazione della risposta in corso..."):
                answer, _ = generate_answer(query, chunks, model, tokenizer)
                st.markdown(answer)
        
        with st.expander("Mostra Documenti Recuperati e Punteggi"):
            st.markdown(context_display)
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "context": context_display})

