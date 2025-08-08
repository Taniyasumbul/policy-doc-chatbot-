import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

st.title(" Document Q&A Chatbot using Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding_model)

custom_prompt_template = """
You are a smart insurance document assistant. Your job is to answer user queries based only on the content in the document.

Given the following user query:
"{query}"

Return your answer in this exact JSON format:

{{
  "decision": "<Approved or Rejected>",
  "amount": "<If any, else null>",
  "justification": "<The specific clause or sentence from the document that supports your decision>"
}}

If the answer is not found in the document, then:
"decision" = "Rejected",
"amount" = null,
"justification" = "No relevant clause found in the document."
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=custom_prompt_template,
)

qa_chain = RetrievalQA.from_chain_type(
     llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3),
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)


llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)


user_query = st.text_input("Ask a question from the document:")

if user_query:
        response = qa_chain.run(user_query)
        st.subheader("📄 JSON Output:")
        st.json(response)  # It will show the response in proper JSON format
