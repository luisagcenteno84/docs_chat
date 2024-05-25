import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from google.cloud import storage

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_google_community import GCSFileLoader

def load_pdf(file_path):
    return PyPDFLoader(file_path)

# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

# Access the uploaded ref via a key.
doc_file = st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

if ss.pdf:
    ss.pdf_ref = ss.pdf  # backup

# Now you can access "pdf_ref" anywhere in your app.
binary_data = None
vector_store = None
rag_chain = None

question = "Create a summary of this document"

if ss.pdf_ref:

    question = st.chat_input(placeholder="Question:")



    binary_data = ss.pdf_ref.getvalue()

    file_name = ''
    if doc_file is not None:
        file_name = doc_file.name

    #print("FileName:["+file_name+"]")

    storage_client = storage.Client()
    bucket = storage_client.bucket('docs-chat-context')

    blob = bucket.blob("context_documents/"+file_name)
    blob.upload_from_string(binary_data, content_type="application/pdf")

    blob_path = "https://storage.cloud.google.com/docs-chat-context/context_documents/"+file_name




    #print(blob_path)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    loader = GCSFileLoader(
        project_name="docs-chat", bucket="docs-chat-context", blob="context_documents/"+file_name, loader_func=load_pdf
    )

    #print("Loader:["+str(loader)+"]")

    #loader = PyPDFLoader(file_path=blob_path)
    data = loader.load()

    text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)

    #print("data:["+str(data)+"]")


    all_splits = text_splitters.split_documents(data)

    print("all_splits:["+str(all_splits)+"]")
    print("all_splits len:["+str(len(all_splits))+"]")

    vector_store = FAISS.from_documents(all_splits, embeddings)

    print("FAISS vector store created:"+str(vector_store))

    retriever = vector_store.as_retriever()

    model = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0.2)

    ### Contextualize question ###
    template = """
                    You are an chatbot that answers people questions about their documents in natural language. 
                    The relevant documents will be provided in the context below.
                    Be polite and provide answers based on the provided context only and do not make up any data. 
                    
                    Use only the provided data and not prior knowledge.

                    Follow exactly these 3 steps:
                    1. Read the context below 
                    2. Answer the question using ONLY the provided context below. If the question cannot be answered based on the context below, politely decline to answer and say that you are only allowed to answer questions about the contextual data passed in the document. 
                    3. Make sure to nicely format the output so it is easy to read on a small screen.

                    If you don't know the answer, just say you don't know. 
                    Do NOT try to make up an answer.
                    If the question is not related to the information about the context below, 
                    politely respond that you are tuned to only answer questions about the context provided. 
                    Use as much detail as possible when responding but keep your answer to up to 100 words.
                    At the end ask if the user would like to have more information or what else they would like to know about this document.
                    
                    Context:
                    {context}

                    Question:
                    {question}
                """

    context_prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever,"question": RunnablePassthrough()}
        | context_prompt
        | model
        | StrOutputParser()
    )



    pdf_viewer(input=binary_data, width=700)

st.write("Question: "+str(question))
st.write("Answer: "+rag_chain.invoke(question))



        


