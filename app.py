import streamlit as st
import subprocess
import sys

# Function to install required packages
def install_packages():
    packages = [
        "sentence-transformers",
        "langchain-groq",
        "langchain-community",
        "transformers",
        "torch"
    ]
    
    for package in packages:
        try:
            st.write(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success(f"✅ {package} installed successfully")
        except Exception as e:
            st.error(f"Failed to install {package}: {str(e)}")
            return False
    return True

# Install packages if not already installed
try:
    import sentence_transformers
except ImportError:
    st.warning("Installing required packages...")
    if not install_packages():
        st.error("Failed to install required packages")
        st.stop()
from langchain_groq import ChatGroq
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_models():
    """Initialize LLM models and tools"""
    models = {}
    
    # Initialize Groq LLM
    try:
        models['groq_llm'] = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            temperature=0.3
        )
        st.success("✅ Groq LLM initialized successfully")
    except Exception as e:
        st.error(f"Groq initialization error: {str(e)}")
    
    # Initialize HuggingFace Embeddings
    try:
        models['embeddings'] = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        st.success("✅ Embeddings model initialized successfully")
    except Exception as e:
        st.error(f"Embeddings initialization error: {str(e)}")
    
    models['search'] = DuckDuckGoSearchRun()
    return models

def initialize_retriever():
    """Process uploaded PDF and create retriever"""
    try:
        pdf_path = "Cricket Rules.pdf"
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.write(f"Number of pages loaded: {len(documents)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        # Get models
        models = initialize_models()
        
        if 'embeddings' not in models:
            st.error("Embeddings model not available")
            return None
            
        try:
            with st.spinner("Creating vector store..."):
                vectorstore = FAISS.from_documents(splits, models['embeddings'])
                st.success("✅ Vector store created successfully")
                return vectorstore.as_retriever()
        except Exception as e:
            st.error(f"Failed to create vector store: {str(e)}")
            return None
            
    except FileNotFoundError:
        st.error(f"PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_answer(question):
    """Generate answer based on context and question"""
    models = initialize_models()
    
    try:
        # If no retriever available, use DuckDuckGo
        if st.session_state.retriever is None:
            search_result = models['search'].run(question)
            return f"DuckDuckGo Search Result:\n{search_result}"
        
        # Get context from retriever
        try:
            context = st.session_state.retriever.invoke(question)
            context_text = "\n".join([doc.page_content for doc in context])
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            search_result = models['search'].run(question)
            return f"DuckDuckGo Search Result (Retrieval Error):\n{search_result}"
        
        # Use Groq to generate answer
        if 'groq_llm' in models:
            try:
                groq_prompt = f"""
                Use the following context from cricket rules to answer the question.
                If the context doesn't contain relevant information, say so clearly.
                
                Context: {context_text}
                Question: {question}
                """
                
                groq_answer = models['groq_llm'].invoke(groq_prompt)
                
                # Check if answer indicates no relevant information
                if any(phrase in groq_answer.content.lower() for phrase in 
                       ["does not contain", "no information", "cannot answer", 
                        "don't have", "doesn't mention", "not mentioned"]):
                    search_result = models['search'].run(question)
                    return f"DuckDuckGo Search Result (No relevant context):\n{search_result}"
                
                return f"Answer (Based on PDF):\n{groq_answer.content}"
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                search_result = models['search'].run(question)
                return f"DuckDuckGo Search Result (LLM Error):\n{search_result}"
        
        # If no Groq available, fall back to DuckDuckGo
        search_result = models['search'].run(question)
        return f"DuckDuckGo Search Result:\n{search_result}"
        
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        try:
            search_result = models['search'].run(question)
            return f"DuckDuckGo Search Result (Error Fallback):\n{search_result}"
        except Exception as search_error:
            return f"Error: Unable to generate answer. Please try again later. ({str(e)})"

def main():
    st.title("🏏 Cricket Rules Chatbot")
    
    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize retriever if not already done
    if 'retriever' not in st.session_state or st.session_state.retriever is None:
        with st.spinner("Loading PDF..."):
            st.session_state.retriever = initialize_retriever()
            if st.session_state.retriever:
                st.success("✅ PDF processed successfully!")
    
    # Sidebar
    with st.sidebar:
        st.title("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                st.session_state.retriever = initialize_retriever()
                if st.session_state.retriever:
                    st.success("PDF processed successfully!")
        
        st.title("💬 Chat History")
        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()
        
        # Show question history
        if st.session_state.messages:
            for i, msg in enumerate([m for m in st.session_state.messages if m["role"] == "user"], 1):
                st.write(f"{i}. {msg['content']}")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask about cricket rules..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(question)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
