import streamlit as st
import PyPDF2
import openai
import os
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains import RetrievalQA  # Import for RetrievalQA
from langchain_openai import OpenAI  # Updated import for OpenAI
from langchain.prompts import PromptTemplate

# Define your OpenAI API key directly in the script (for testing purposes only)
OPENAI_API_KEY = "sk-proj-yDd5hFvujTQGYnv0JsYaT3BlbkFJxMNDm3k3uhuudz2i7k19"

# Set OpenAI API key for use with OpenAIEmbeddings and other OpenAI operations
openai.api_key = OPENAI_API_KEY

def extract_text_with_sections(pdf_file):
    """Extract text and section headers from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    sections = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text() or ""  # Handle potential None
        
        section_headers = [line for line in page_text.split('\n') if "section" in line.lower() or "step" in line.lower()]
        
        for header in section_headers:
            sections.append((header, page_num))  # Store section title and page number
        extracted_text += page_text
    return extracted_text, sections

def extract_texts_and_sections_from_multiple_pdfs(uploaded_files):
    """Extract text and sections from multiple PDFs."""
    all_texts = []
    all_sections = []
    
    for pdf_file in uploaded_files:
        extracted_text, sections = extract_text_with_sections(pdf_file)
        all_texts.append(extracted_text)
        all_sections.append(sections)
    
    return all_texts, all_sections

def build_vector_store(texts):
    """Build a vector store from texts."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Pass the API key directly here
    doc_store = FAISS.from_texts(texts, embeddings)
    return doc_store

def create_formal_rag_chain(vector_store):
    """Create a formal RAG chain for querying."""
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=1000)  # Adjust max tokens for brevity

    prompt_template = PromptTemplate(
        template=(
            "You are an expert assistant that provides highly structured and detailed step-by-step instructions "
            "based only on the provided documents (SOPs). Each step you provide must be clearly derived from the content "
            "and must cite the section from where it was taken. If the query cannot be answered from the documents, reply with: "
            "'I do not know based on the provided SOPs.' and ask for clarification.\n\n"
            "Documents: {context}\n\n"
            "Query: {query}\n\n"
            "Provide your answer as a step-by-step guide, ensuring that every step cites a specific section of the SOP."
        ),
        input_variables=["query", "context"],  # Ensure both 'query' and 'context' are included
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' method for combining documents
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}  # Pass context as doc variable
    )
    
    return qa_chain

def is_query_relevant_to_sop(response, query):
    """Check if the response is relevant to the SOP."""
    if len(response.strip()) == 0 or "I don't know" in response or "no information" in response:
        return False

    relevant_keywords = ["step", "section", "procedure", "SOP"]
    return any(keyword in response.lower() for keyword in relevant_keywords)

def generate_step_by_step_plan(query, qa_chain, sections):
    """Generate a step-by-step plan with references from sections."""
    context = "\n\n".join([f"{section[0]} (Page {section[1] + 1})" if isinstance(section[1], int) else f"{section[0]}" for section in sections])

    inputs = {
        'query': query,
        'context': context
    }
    
    response = qa_chain.invoke(inputs)  # Use invoke() to handle multiple output keys
    
    # Check if the necessary output keys exist
    if 'result' not in response or 'source_documents' not in response:
        return "An error occurred while processing your request."

    # Extract the result and source documents
    result = response['result']
    source_documents = response['source_documents']
    
    # Check if the response is relevant to the SOP content
    if not is_query_relevant_to_sop(result, query):
        return "I do not know. Can you please provide more details or clarify your question?"

    # Here, we assume the response might include steps; break into steps
    steps = result.split("\n")
    plan_with_references = []
    
    for step in steps:
        if step.strip():  # Skip empty lines
            referenced_section = "Referenced section not found"
            for doc in source_documents:
                if step.lower() in doc.page_content.lower():  # Match step with document content
                    referenced_section = f"Referenced from: {doc.metadata['source']} (Page {doc.metadata['page']})"
                    break
            
            plan_with_references.append(f"{step.strip()} ({referenced_section})")
    
    return "\n".join(plan_with_references)

# Streamlit Web App
st.title("SOP Formal Step-by-Step Generator with Section References")

# Sidebar for page selection
page = st.sidebar.selectbox("Select a page:", ["Document Ingestion", "Query Knowledge Base"])

if page == "Document Ingestion":
    st.header("Document Ingestion")
    
    uploaded_files = st.file_uploader("Upload one or more PDF files (SOPs)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner('Extracting text and sections from uploaded SOPs...'):
            extracted_texts, all_sections = extract_texts_and_sections_from_multiple_pdfs(uploaded_files)
            st.success("Text and section extraction complete!")

        if st.button("Build Knowledge Base"):
            with st.spinner("Building vector store from SOPs..."):
                vector_store = build_vector_store(extracted_texts)
                st.success("Knowledge base built!")
                st.session_state['vector_store'] = vector_store
                st.session_state['all_sections'] = all_sections

elif page == "Query Knowledge Base":
    st.header("Query Knowledge Base")

    if 'vector_store' in st.session_state:
        query = st.text_input("Enter your question:")  # Input for user query

        if query:  # Make sure there is a valid query
            qa_chain = create_formal_rag_chain(st.session_state['vector_store'])

            with st.spinner("Generating formal step-by-step plan..."):
                plan_with_references = generate_step_by_step_plan(query, qa_chain, st.session_state['all_sections'])
                st.write(f"**Response:**\n\n{plan_with_references}")

            # Keep the query input box active
            query = st.text_input("Enter your next question:", key="next_query")
    else:
        st.warning("Please build the knowledge base first on the Document Ingestion page.")