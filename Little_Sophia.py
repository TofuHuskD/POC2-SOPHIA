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

# Function to extract text from a single PDF and capture sections
def extract_text_with_sections(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    sections = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        
        # Simple section detection (assuming sections have titles like "Section 1.0", etc.)
        section_headers = [line for line in page_text.split('\n') if "section" in line.lower() or "step" in line.lower()]
        
        for header in section_headers:
            sections.append((header, page_num))  # Store section title and page number
        extracted_text += page_text
    return extracted_text, sections

# Function to process multiple PDFs and extract text and sections
def extract_texts_and_sections_from_multiple_pdfs(uploaded_files):
    all_texts = []
    all_sections = []
    
    for pdf_file in uploaded_files:
        extracted_text, sections = extract_text_with_sections(pdf_file)
        all_texts.append(extracted_text)
        all_sections.append(sections)
    
    return all_texts, all_sections

# Function to build vector store from multiple texts
def build_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    doc_store = FAISS.from_texts(texts, embeddings)
    return doc_store

# Function to create a RetrievalQA chain (RAG-style) with formal and deterministic behavior
def create_formal_rag_chain(vector_store):
    # Set temperature to 0 for deterministic and formal responses
    llm = OpenAI(temperature=0, model_kwargs={"max_tokens": 1500})
    
    # Create a custom chain with formal and structured prompts
    qa_chain = RetrievalQA(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        prompt_template=(
            "You are an expert assistant that provides formal and structured step-by-step instructions "
            "based on the following documents. Always maintain a professional tone and cite the sections "
            "from which the steps are derived. Answer only based on the content of the SOPs provided. "
            "If the query is not related to the SOPs or you do not know the answer, respond with 'I do not know.' "
            "Then ask a follow-up question to help the user clarify their request.\n\n"
            "Query: {query}\n\nProvide a step-by-step plan based on the documents."
        )
    )
    return qa_chain

# Function to check if the response is relevant to the SOP
def is_query_relevant_to_sop(response, query):
    # If response is empty or not specific, consider it irrelevant
    if len(response.strip()) == 0 or "I don't know" in response or "no information" in response:
        return False
    return True

# Function to generate step-by-step plan with section references, asking follow-up questions if answer is not found
def generate_step_by_step_plan(query, qa_chain, sections):
    response = qa_chain.run(query)
    
    # Check if the response is relevant to the SOP content
    if not is_query_relevant_to_sop(response, query):
        follow_up_question = "I do not know. Can you please provide more details or clarify your question?"
        return follow_up_question

    # Here, we assume the response might include steps. We simulate by breaking response into steps.
    steps = response.split("\n")  # Simple split into steps (can be improved with prompt engineering)
    
    plan_with_references = []
    
    for step in steps:
        if step.strip():  # Skip empty lines
            referenced_section = "Referenced section not found"
            # Match step to a section in the document
            for section in sections:
                if section[0].lower() in step.lower():  # Check if step matches a section title
                    referenced_section = f"Referenced from: {section[0]}, Page {section[1] + 1}"
                    break
            
            # Combine step with reference
            plan_with_references.append(f"{step.strip()} ({referenced_section})")
    
    return "\n".join(plan_with_references)

# Streamlit Web App
st.title("SOP Formal Step-by-Step Generator with Section References")

uploaded_files = st.file_uploader("Upload one or more PDF files (SOPs)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Extract text and sections from all uploaded PDFs
    with st.spinner('Extracting text and sections from uploaded SOPs...'):
        extracted_texts, all_sections = extract_texts_and_sections_from_multiple_pdfs(uploaded_files)
        st.success("Text and section extraction complete!")

    # Display a summary of extracted text and sections
    st.subheader("Extracted Text Summary")
    for idx, (text, sections) in enumerate(zip(extracted_texts, all_sections)):
        st.write(f"Document {idx+1}: {text[:500]}...")  # Show first 500 characters as a preview
        st.write(f"Sections: {[section[0] for section in sections]}")  # List section headers

    # Option to build knowledge base
    if st.button("Build Knowledge Base"):
        with st.spinner("Building vector store from SOPs..."):
            vector_store = build_vector_store(extracted_texts)
            st.success("Knowledge base built!")

        # Chatbot interaction
        if st.button("Start Chatting"):
            st.subheader("Ask a question to generate a formal step-by-step plan")

            query = st.text_input("Enter your question")
            if query:
                # Create formal RAG Chain
                qa_chain = create_formal_rag_chain(vector_store)

                with st.spinner("Generating formal step-by-step plan..."):
                    plan_with_references = generate_step_by_step_plan(query, qa_chain, all_sections)
                    st.write(f"**Response:**\n\n{plan_with_references}")