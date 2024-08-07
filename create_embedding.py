import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils import huggingface_embedding


def vector_embedding(start_page,end_page,chunk_size,chunk_overlap):

    '''
    Function to create embedding for required documents 
    '''

    FAISS_INDEX_PATH = "faiss_index"
    embeddings = huggingface_embedding()
    loader = PyPDFDirectoryLoader(r"PDF_DOC")  # Data Ingestion from given directory
    docs = loader.load()  # Document Loading
    print(f"Loaded {len(docs)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs[start_page:end_page])
    print(f"Splited into {len(final_documents)} chunks.")
    
    if not final_documents:
        print("No final documents available for embeddings.")
        
    print("Vector embeddings Started.")

    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
    print("Vector embeddings created.")
    
    vectors.save_local(FAISS_INDEX_PATH)
    print("Vector embeddings Saved.")

#start page and end page is selected for chap 1-2
vector_embedding(start_page = 5 ,end_page = 54 ,chunk_size = 1000 ,chunk_overlap = 100)
print("Vector Store DB Is Ready")