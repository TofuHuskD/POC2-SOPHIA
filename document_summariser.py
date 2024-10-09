import streamlit as st
import PyPDF2
import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set up OpenAI API Key (You need to add your OpenAI API key here)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()
    return extracted_text

# Function to build vector store
def build_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    doc_store = FAISS.from_texts(texts, embeddings)
    return doc_store

# Function to create a RetrievalQA chain (RAG-style)
def create_rag_chain(vector_store):
    llm = OpenAI(temperature=0.2)
    qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain

# Streamlit Web App
st.title("RAG Chatbot with PDF Ingestion")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    with st.spinner('Extracting text from PDF...'):
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extraction complete!")

    # Display extracted text
    st.subheader("Extracted Text")
    st.write(extracted_text[:500] + "...")  # Show first 500 characters as a preview

    # Option to build knowledge base
    if st.button("Build Knowledge Base"):
        with st.spinner("Building vector store..."):
            vector_store = build_vector_store([extracted_text])
            st.success("Knowledge base built!")

        # Chatbot interaction
        if st.button("Start Chatting"):
            st.subheader("Ask a question based on the document")

            query = st.text_input("Enter your question")
            if query:
                # Create RAG Chain
                qa_chain = create_rag_chain(vector_store)

                with st.spinner("Generating answer..."):
                    response = qa_chain.run(query)
                    st.write(f"**Answer:** {response}")