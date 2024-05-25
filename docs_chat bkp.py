import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from google.cloud import storage

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer


# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

# Access the uploaded ref via a key.
doc_file = st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

if ss.pdf:
    ss.pdf_ref = ss.pdf  # backup

# Now you can access "pdf_ref" anywhere in your app.
binary_data = None

if ss.pdf_ref:
    binary_data = ss.pdf_ref.getvalue()

    file_name = ''
    if doc_file is not None:
        file_name = doc_file.name

    print("FileName:["+file_name+"]")

    storage_client = storage.Client()
    bucket = storage_client.bucket('docs-chat-context')

    blob = bucket.blob("context_documents/"+file_name)
    blob.upload_from_string(binary_data, content_type="application/pdf")

    blob_path = "https://storage.cloud.google.com/docs-chat-context/context_documents/"+file_name


    pdf_viewer(input=binary_data, width=700)





#loader = PyPDFLoader(file_path=file_name)
#data = loader.load()