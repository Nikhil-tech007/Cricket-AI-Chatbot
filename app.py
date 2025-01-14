import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Cricket Rules Chatbot",
    page_icon="🏏",
    layout="wide"
)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_models():
    """Initialize LLM models and tools"""
    return {
        'llama_llm': OllamaLLM(model="llama3"),
        'groq_llm': ChatGroq(api_key=os.getenv("GROQ_API_KEY"), temperature=0.3),
        'search': DuckDuckGoSearchRun(),
        'embeddings': OllamaEmbeddings(model="llama3")
    }

def initialize_retriever():
    """Process uploaded PDF and create retriever"""
    try:
        # Specify the path to your PDF file
        pdf_path = "Cricket Rules.pdf"  # Adjust this path to your PDF location
        
        # Add debug information
        st.write(f"Processing PDF from: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add debug information
        st.write(f"Number of pages loaded: {len(documents)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        models = initialize_models()
        vectorstore = FAISS.from_documents(splits, models['embeddings'])
        
        # Add debug information
        st.write("Vector store created successfully")
        
        return vectorstore.as_retriever()
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def get_answer(question):
    """Generate answer based on context and question"""
    try:
        if st.session_state.retriever is None:
            return "Please upload a PDF file first."
            
        models = initialize_models()
        context = st.session_state.retriever.invoke(question)
        
        # Check context relevance
        question_words = set(question.lower().split())
        context_text = " ".join([doc.page_content.lower() for doc in context])
        matching_words = sum(1 for word in question_words if word in context_text)
        
        if matching_words < 2:
            search_result = models['search'].run(question)
            return f"DuckDuckGo Search Result:\n{search_result}"
        
        context_text = "\n".join([doc.page_content for doc in context])
        groq_prompt = f"""
        Use the following context from cricket rules to answer the question:
        
        Context: {context_text}
        
        Question: {question}
        
        If the context doesn't contain relevant information to answer the question, 
        please explicitly state that.
        """
        
        groq_answer = models['groq_llm'].invoke(groq_prompt)
        
        if any(phrase in groq_answer.content.lower() for phrase in 
              ["does not contain", "no information", "cannot answer", "don't have"]):
            search_result = models['search'].run(question)
            return f"DuckDuckGo Search Result (Fallback):\n{search_result}"
        
        return f"Answer (Based on PDF):\n{groq_answer.content}"
            
    except Exception as e:
        search_result = models['search'].run(question)
        return f"DuckDuckGo Search Result (Error Fallback):\n{search_result}"

def main():
    st.title("🏏 Cricket Rules Chatbot")
    # Initialize retriever on startup if not already done
    if 'retriever' not in st.session_state or st.session_state.retriever is None:
        with st.spinner("Loading PDF..."):
            st.session_state.retriever = initialize_retriever()
            if st.session_state.retriever:
                st.success("PDF processed successfully!")
    
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