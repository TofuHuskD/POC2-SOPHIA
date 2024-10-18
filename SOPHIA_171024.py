import os
import re
import streamlit as st
import openai
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="Document Q&A", page_icon="📄")

st.title("📄 Document-Based Q&A with LangChain and OpenAI")

st.session_state

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Step 1: Enter API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    st.markdown("[Get your OpenAI API Key here](https://platform.openai.com/account/api-keys)")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
    else:
        st.warning("Please enter your OpenAI API Key.")
    
    # Reset Knowledge Base and Query History Button
    if st.session_state['vectorstore'] or st.session_state['query_history']:
        if st.button("Reset Knowledge Base"):
            st.session_state['vectorstore'] = None
            st.session_state['query_history'] = []
            st.success("Knowledge base and query history have been reset.")

# Define the preprocess_text function
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Define the extract_text_from_pdf function with source metadata
def extract_text_from_pdf(file):
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                # Tag by page number; modify as needed for actual sections
                doc = Document(page_content=page_text, metadata={"source": file.name, "section": f"Page {page_num + 1}"})
                text += page_text + "\n"
            else:
                st.warning(f"Warning: No text found on page {page_num + 1} of {file.name}.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from {file.name}: {e}")
        return ""

# Define the extract_text_from_docx function
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error extracting text from {file.name}: {e}")
        return ""

# Function to validate file types
def is_safe_file(file):
    allowed_extensions = ["txt", "pdf", "docx"]
    return file.name.split('.')[-1].lower() in allowed_extensions

# Step 2: Upload Documents
uploaded_files = st.file_uploader(
    "Upload documents (TXT, PDF, DOCX):",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True  # Allow multiple file uploads
)

if st.session_state['vectorstore'] is None:

    if uploaded_files and api_key:
        # Filter out unsafe files
        safe_files = [file for file in uploaded_files if is_safe_file(file)]
        if not safe_files:
            st.error("No valid files uploaded. Please upload TXT, PDF documents.")
        else:
            all_text = ""
            documents = []
            st.subheader("Uploaded Documents:")
            for uploaded_file in safe_files:
                st.write(f"- **{uploaded_file.name}**")
                if uploaded_file.type == "application/pdf":
                    # Extract text from the PDF
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # Extract text from DOCX
                    text = extract_text_from_docx(uploaded_file)
                else:
                    # Read TXT file
                    try:
                        text = uploaded_file.getvalue().decode("utf-8")
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")
                        text = ""
                
                if text:
                    # Preprocess the extracted text
                    text = preprocess_text(text)
                    
                    # Create a Document object with metadata
                    # Optionally, parse sections from text if available
                    doc = Document(page_content=text, metadata={"source": uploaded_file.name, "section": "Full Document"})
                    documents.append(doc)
            
            if documents:
                # Split the documents into chunks using LangChain's TextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                # Create embeddings using OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                
                # Build vectorstore (FAISS) with metadata
                with st.spinner("Building vector store..."):
                    vectorstore = FAISS.from_documents(texts, embeddings)
                    st.session_state['vectorstore'] = vectorstore
                
                st.success("Documents processed and knowledge base built successfully!")

# Step 3: Input Query
if st.session_state.get('vectorstore') and api_key:
    # Define the custom prompt
    custom_prompt = """
    You are an expert assistant that provides highly structured and detailed step-by-step instructions based only on the provided documents (SOPs). Each step you provide must be clearly derived from the content and must cite the section and the source document from where it was taken. If the query cannot be answered from the documents, reply with: 'I do not know based on the provided SOPs.' and ask for clarification.
    
    Documents: {context}
    
    Question: {question}
    
    Provide your answer as a step-by-step guide, ensuring that every step cites a specific section and the corresponding source document of the SOP.
    1. You must directly perform all instructions with reference to the appropriate sections of the knowledge base
    2. You must only refer to sections of the knowledge base which is relevant to your task.
    3. You must always review your output to determine if the facts are consistent with the knowledge base
    4. Do not do math calculations and just cite the data as it is.
    5. Cite text in verbatim as far as possible
    6. In your output, retain the keywords and tone from the documents.
    7. If the output to the instructions cannot be derived from the knowledge base, strictly only reply “There is no relevant information, please only query about SOP related information”.
    """
    
    # Create a PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt
    )
    
    # Initialize the language model
    llm = OpenAI(temperature=0, openai_api_key=api_key, max_tokens=1000)
    
    # Create a QA chain using load_qa_chain with the custom prompt
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",  # You can choose "stuff", "map_reduce", "refine" based on your needs
        prompt=prompt_template
    )
    
    # Create a retriever from the vectorstore
    retriever = st.session_state['vectorstore'].as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Initialize the RetrievalQA chain with the custom QA chain
    qa = RetrievalQA(
        combine_documents_chain=qa_chain,
        retriever=retriever
    )
    
    # Create tabs for querying and history
    query_tab, history_tab = st.tabs(["Ask a Question", "Query History"])
    
    with query_tab:
        st.subheader("Ask a question about the documents:")
        query = st.text_input("Enter your query:", placeholder="What information are you looking for?")
        if query:
            # Optional: Implement query limits
            MAX_QUERIES = 20
            if len(st.session_state.query_history) >= MAX_QUERIES:
                st.warning("You have reached the maximum number of queries for this session.")
            else:
                with st.spinner("Generating answer..."):
                    # Retrieve relevant documents based on the query
                    relevant_docs = retriever.get_relevant_documents(query)
                    
                    if not relevant_docs:
                        response = "I do not know based on the provided SOPs. Could you please clarify or provide more details?"
                        st.write(response)
                        # Append to history
                        st.session_state.query_history.append({"query": query, "answer": response})
                    else:
                        # Combine the content of relevant documents to form the context
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Run the QA chain with the provided context and query
                        answer = qa.run(query)
                        
                        st.subheader("Answer:")
                        st.write(answer)
                        
                        # Append to history
                        st.session_state.query_history.append({"query": query, "answer": answer})
                        
                        # Display source sections and documents if available
                        if relevant_docs:
                            st.subheader("Source Documents and Sections:")
                            for doc in relevant_docs:
                                source = doc.metadata.get('source', 'Unknown Document')
                                section = doc.metadata.get('section', 'Unknown Section')
                                st.write(f"- **Document:** {source} | **Section:** {section}")

    with history_tab:
        if st.session_state.query_history:
            st.subheader("Query History:")
            for idx, entry in enumerate(st.session_state.query_history, 1):
                st.markdown(f"**{idx}. Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}\n")
        else:
            st.info("No queries yet. Ask a question to see the history here.")
else:
    if not api_key:
        st.warning("Please enter your OpenAI API Key.")
    elif not st.session_state.get('vectorstore'):
        st.info("Please upload documents to build the knowledge base.")