
import os
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import (PyPDFLoader,TextLoader,Docx2txtLoader)
from langgraph.graph import StateGraph, END

'''
LangGraph Architecture

Nodes:

retrieve_node
rerank_node
generate_node
check_confidence_node

Edges:

retrieve → rerank
rerank → generate
generate → check
check → retry OR end


User Query
      ↓
Retrieve Documents
      ↓
Rerank Documents
      ↓
Generate Answer
      ↓
Check Confidence
      ↓
Low → Retry Retrieval
Good → Return Answer

'''

#Load environment variables
load_dotenv()


# Load reranker model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

#This improves answer quality
def rerank_documents(query, docs, top_k=4):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:top_k]]
    return reranked_docs


def load_documents(directory):

    all_documents = []

    for filename in os.listdir(directory):

        filepath = os.path.join(directory, filename)

        if filename.endswith(".pdf"):

            loader = PyPDFLoader(filepath)
            documents = loader.load()

        elif filename.endswith(".txt"):

            loader = TextLoader(filepath)
            documents = loader.load()

        elif filename.endswith(".docx"):

            loader = Docx2txtLoader(filepath)
            documents = loader.load()

        else:
            continue

        # Add metadata
        for doc in documents:
            doc.metadata["source"] = filename

        all_documents.extend(documents)

    # 🔥 Split after loading everything
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    return chunks




directory = "data"
chunks = load_documents(directory)

#Create embeddings and store in vector database
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    model="text-embedding-3-small"
)
# vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="vector_store")

# Delete old database
if os.path.exists("vector_store"):
    shutil.rmtree("vector_store")

# Initialize Chroma
vectorstore = Chroma(persist_directory="vector_store", embedding_function=embeddings)

# Add documents in batches
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)

#Create retriever from vector database
# retriever = vectorstore.as_retriever()
#Retrieval tunning: retrieve more docs than you think you need, and let the LLM decide which are relevant
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

# 2. Define your prompt
template = """Answer the question based only on the following context:{context}
            Question: {input}"""

rag_prompt = ChatPromptTemplate.from_template(template)


#llM
llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0.7,
                 api_key=os.getenv("GITHUB_TOKEN"),
                 base_url="https://models.inference.ai.azure.com")

doc_chain = create_stuff_documents_chain(llm, rag_prompt)



class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    confidence: float


def retrieve_node(state: RAGState):
    query = state["query"]
    docs = retriever.invoke(query)
    
    return {
        "retrieved_docs": docs
    }



def rerank_node(state: RAGState):

    query = state["query"]
    docs = state["retrieved_docs"]
    reranked = rerank_documents(
        query,
        docs,
        top_k=4
    )

    return {
        "reranked_docs": reranked
    }

def generate_node(state: RAGState):

    query = state["query"]

    docs = state["reranked_docs"]

    answer = doc_chain.invoke({
        "input": query,
        "context": docs
    })

    return {
        "answer": answer
    }

def check_confidence_node(state: RAGState):

    docs = state["reranked_docs"]

    if len(docs) >= 2:
        confidence = 0.9
    else:
        confidence = 0.3

    return {
        "confidence": confidence
    }


def should_retry(state: RAGState):

    if state["confidence"] < 0.5:
        return "retry"

    return "end"



workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("check", check_confidence_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "check")

workflow.add_conditional_edges(
    "check",
    should_retry,
    {
        "retry": "retrieve",
        "end": END
    }
)

graph = workflow.compile()

result = graph.invoke({
    "query": "What is graphic design"
})

print("\nAnswer:")
print(result["answer"])

print("\nConfidence:")
print(result["confidence"])
