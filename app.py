import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_store", embeddings=embedding_model, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)


st.set_page_config(page_title=" Gemini Document Chatbot", layout="centered")

st.markdown("""
    <style>
    body {
        
    }
    .big-font {
        font-size:25px !important;
        font-weight: bold;
    }
    .chat-bubble {
      
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .bot-response {
    
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'> Ask Your Document!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with Gemini + FAISS + LangChain</p>", unsafe_allow_html=True)

query = st.text_input(" Ask a question from your document:", placeholder="e.g., What is the policy start date?")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
            st.markdown(f"<div class='chat-bubble'> You: {query}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-response'> Bot: {response}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><small>Made with  using Gemini, LangChain, FAISS and Streamlit</small></center>", unsafe_allow_html=True)

