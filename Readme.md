# 🚀 Advanced RAG System with LangChain & LangGraph

This project implements an **Advanced Retrieval-Augmented Generation
(RAG) system** using **LangChain**, **LangGraph**, and **ChromaDB**,
enhanced with **document reranking** and **stateful workflows**.

It supports **multi-document ingestion**, **source tracking**,
**reranking**, and a **LangGraph-based retry workflow**, making it
closer to **real-world production RAG systems**.

------------------------------------------------------------------------

# 🧠 Features

✅ Multi-file document ingestion\
- Supports `.pdf`, `.txt`, `.docx`\
- Automatically extracts metadata\
- Tracks document sources

✅ Advanced Retrieval Pipeline\
- Recursive text chunking\
- Vector embeddings\
- Chroma vector database\
- MMR-based retrieval

✅ Document Reranking\
- Uses CrossEncoder reranker\
- Improves answer quality\
- Reduces irrelevant context

✅ LangGraph Workflow\
- Stateful RAG execution\
- Retry mechanism\
- Confidence-based routing

✅ Source Attribution\
- Displays document source\
- Shows page references

------------------------------------------------------------------------

# 🏗️ System Architecture

User Query\
↓\
Retriever (Chroma)\
↓\
Reranker (CrossEncoder)\
↓\
LLM Answer Generation\
↓\
Confidence Check\
↓\
Retry (if needed)\
↓\
Final Answer

------------------------------------------------------------------------

# 📦 Tech Stack

-   Python\
-   LangChain\
-   LangGraph\
-   ChromaDB\
-   OpenAI / LLM API\
-   SentenceTransformers\
-   CrossEncoder Reranker

------------------------------------------------------------------------

# 📁 Project Structure

advanced_rag/\
│\
├── main.py\
├── load_documents.py\
├── reranker.py\
├── graph.py\
│\
├── data/\
│ sample.pdf\
│\
├── vector_store/\
│\
├── .env\
├── requirements.txt\
└── README.md

------------------------------------------------------------------------

# ⚙️ Installation

``` bash
git clone https://github.com/YOUR_USERNAME/advanced-rag-langgraph.git

cd advanced-rag-langgraph
```

Create virtual environment:

``` bash
python -m venv venv

source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔑 Environment Setup

Create:

.env

Add your API key:

OPENAI_API_KEY=your_api_key_here

------------------------------------------------------------------------

# ▶️ Usage

Add documents to:

data/

Supported formats:

.pdf\
.txt\
.docx

Run:

``` bash
python main.py
```

Example query:

What is this document about?

------------------------------------------------------------------------

# 🧪 Example Output

Answer:\
This document explains the insurance claim process...

Sources:\
- policy.pdf (page 3)\
- report.docx (page 2)

Confidence:\
0.9

------------------------------------------------------------------------

# 🚀 Future Improvements

-   Streamlit UI\
-   FastAPI deployment\
-   Redis caching\
-   Evaluation metrics\
-   Hybrid search

------------------------------------------------------------------------

# 📊 Skills Demonstrated

-   Retrieval-Augmented Generation (RAG)\
-   LangChain pipelines\
-   LangGraph workflows\
-   Vector databases\
-   Document reranking\
-   AI system design

------------------------------------------------------------------------

# 👨‍💻 Author

Your Name

GitHub:\
https://github.com/YOUR_USERNAME

------------------------------------------------------------------------

# 📜 License

MIT License
