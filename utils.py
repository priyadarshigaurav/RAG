from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import time
import streamlit as st


groq_api_key = st.secrets["GROQ_API_KEY"]


def groq_llm():
    '''
    Function to setup groq LLM
    '''
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    return llm


def huggingface_embedding():
    '''
    Function for setup huggingface_embedding
    '''

    embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
    )

    return embeddings


def load_vector_embeddings():
    '''
    Function to load vector embeddings from a file
    '''
    FAISS_INDEX_PATH = r"faiss_index"
    vectors = FAISS.load_local(folder_path=FAISS_INDEX_PATH,embeddings = huggingface_embedding(),allow_dangerous_deserialization=True)
    return vectors        
