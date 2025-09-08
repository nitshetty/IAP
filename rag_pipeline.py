# rag_pipeline.py
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY is not set in the .env file.")

def build_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [
        Document(page_content="Albert Einstein developed the theory of relativity."),
        Document(page_content="The capital of France is Paris."),
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Groq LLM
    llm = ChatGroq(api_key=API_KEY, model="llama-3.1-8b-instant")

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Return both
    return qa_chain, llm

if __name__ == "__main__":
    rag, _ = build_rag_pipeline()
    query = "Who developed the theory of relativity?"
    result = rag.invoke(query)
    print("Query:", query)
    print("Answer:", result["result"])