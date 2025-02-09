import os
import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import cassio
from PyPDF2 import PdfReader

# Initialize environment variables
ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
ASTRA_DB_TOKEN = st.secrets["ASTRA_DB_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ASTRA_DB_KEYSPACE = st.secrets.get("ASTRA_DB_KEYSPACE", "default_keyspace")

# Initialize Cassandra/Astra DB
cassio.init(
    token=ASTRA_DB_TOKEN,
    database_id=ASTRA_DB_ID,
    keyspace=ASTRA_DB_KEYSPACE,
)

def process_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                documents.append(page_text)
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    return texts

def main():
    st.title("PDF RAG Application with Astra DB")
    
    # Initialize OpenAI components
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Initialize Astra DB vector store
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="pdf_rag_demo",
        keyspace=ASTRA_DB_KEYSPACE,
    )
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            texts = process_pdfs(uploaded_files)
            astra_vector_store.add_documents(texts)
            st.success(f"Processed {len(texts)} document chunks")
    
    # Question answering section
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Searching for answers..."):
            astra_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
            answer = astra_index.query(question, llm=llm).strip()
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
