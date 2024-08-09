import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import utils


#UI Title
st.title("Concepts of Biology(Chap-1,2)")

#LLM Initilization
llm = utils.groq_llm()

#LLM Input prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
"""
)

def initialize_session_state():

    '''
    Initialize session state attributes if they do not exist
    '''
    if "vectors" not in st.session_state:
        st.session_state.vectors = None


ques = st.text_input("Enter Your Question From Documents")

if ques:
    initialize_session_state()  # Initializing session state attributes
    #Initializing Embedding vector
    st.session_state.vectors = utils.load_vector_embeddings() if utils.load_vector_embeddings()!=None else st.write("No Vectors found")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever() if st.session_state.vectors else None
    
    if retriever:
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': ques.lower()})
        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response['answer'])
    
        #Streamlit expander
        with st.expander("Reference"):
            # Finding the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Vector store is not initialized. Please create the embeddings first by clicking the 'Documents Embedding' button.")
