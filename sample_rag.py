import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#Load environment variables
load_dotenv()

#Load PDF document
loader = PyPDFLoader("data/test.pdf")
documents = loader.load_and_split()

# Split the text into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

#Create embeddings and store in vector database
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    model="text-embedding-3-small"
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="vector_store")

#Create retriever from vector database
retriever = vectorstore.as_retriever()

#llM
llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0.7,
                 api_key=os.getenv("GITHUB_TOKEN"),
                 base_url="https://models.inference.ai.azure.com")

format_docs = lambda docs: "\n".join(doc.page_content for doc in docs)

#Create RAG chain
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 2. Define your prompt
template = """Answer the question based only on the following context:{context}
            Question: {question}"""

rag_prompt = ChatPromptTemplate.from_template(template)

qa_chain = (
    {
        "context": retriever | format_docs,  # retrieve + format
        "question": RunnablePassthrough()    # pass question through untouched
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

#Ask a question
question = "What is the main topic of the document?"
answer = qa_chain.invoke(question)
print("Answer:", answer)