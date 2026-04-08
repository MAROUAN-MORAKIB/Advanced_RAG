import os
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
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


#Create document chain

doc_chain = create_stuff_documents_chain(llm, rag_prompt)

#Create RAG chain
# qa_chain = create_retrieval_chain(retriever, doc_chain)


# Custom retrieval chain with reranking
def custom_retrieval_chain(query):
    # Step 1: Retrieve more documents
    docs = retriever.invoke(query)

    # Step 2: Rerank them
    reranked_docs = rerank_documents(query, docs, top_k=4)

    # Step 3: Send best docs to LLM
    answer = doc_chain.invoke({
        "input": query,
        "context": reranked_docs
    })

    return {
        "answer": answer,
        "context": reranked_docs
    }


# Invoke the chain with a question
question = "What is prompte ?"
response = custom_retrieval_chain(question)
print("Answer:", response["answer"])
print("Sources:")
for doc in response["context"]:
    print(
        f"- Page {doc.metadata.get('page', 'N/A')} from {doc.metadata.get('source', 'N/A')}"
    )



