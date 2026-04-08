# 🎯 Project 1 — Advanced RAG System (Production-Style)

1- User upload documents
2- Documents are chunked 
3- Embeddings stored in vector DB
4- Retriever fetches relevant chunks
5- LLM generates answer
6- Return source

## System Architecture

User Uploads PDF/TXT
        ↓
Document Loader
        ↓
Text Splitter
        ↓
Embeddings Model
        ↓
Vector Database (Chroma or Qdrant)
        ↓
Retriever
        ↓
RAG Chain
        ↓
Answer + Sources

Then later:

LangGraph Agent
        ↓
Retry if retrieval fails
Ask human if low confidence
Memory across questions


## 📦 Tech Stack We Will Use

LangChain
LangGraph
Chroma (vector DB)
OpenAI or local model
RecursiveTextSplitter
Structured Output

#### Later

FastAPI (API)
Streamlit (UI)
Docker (deployment)

## 🧩 Phase 1 — Basic RAG Pipeline (Foundation)

Load documents
Split text
Create embeddings
Store in Chroma
Ask questions
Return answers

## 🧩 Phase 2 — Return Sources (Very Important)

Return answer + source chunks

## 🧩 Phase 3 — Multi-Document Upload

pdf,txt,docx

## 🧩 Phase 4 — Retrieval Optimization

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

Top-k tuning
MMR retrieval = Maximal Marginal Relevance
        Not just similar chunks
        But diverse useful chunks
        Better answers
        Less repetition
        More context coverage

Metadata filtering

## 🧩 Phase 5 — LangGraph Integration

If retrieval fails → retry
If confidence low → ask human
If user asks follow-up → memory


