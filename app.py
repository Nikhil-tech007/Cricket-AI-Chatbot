import streamlit as st
import numpy as np
from typing import List
import time

class SimpleSearch:
    """Simple search implementation without external dependencies"""
    def __init__(self):
        self.documents = []
        
    def add_document(self, text: str):
        """Add a document to the search index"""
        self.documents.append(text.lower())
        
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant documents"""
        query = query.lower()
        scores = []
        
        # Calculate similarity scores
        for doc in self.documents:
            # Simple word overlap score
            query_words = set(query.split())
            doc_words = set(doc.split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / (len(query_words) + len(doc_words) - overlap + 1e-6)
            scores.append(score)
        
        # Get top results
        if not scores:
            return []
            
        indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in indices if scores[i] > 0]

def get_answer(question: str) -> str:
    """Generate answer based on question"""
    try:
        # Simple response generation
        search_results = st.session_state.search_engine.search(question)
        if search_results:
            return "\n\n".join([
                "Based on the available information:",
                *search_results
            ])
        return "I'm sorry, I don't have enough information to answer that question specifically. Please try asking about basic cricket rules."
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, there was an error processing your question. Please try again."

def main():
    st.title("🏏 Cricket Rules Chatbot")
    
    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "search_engine" not in st.session_state:
        with st.spinner("Initializing cricket knowledge..."):
            st.session_state.search_engine = SimpleSearch()
            # Add comprehensive cricket rules
            cricket_rules = [
                "Cricket is played between two teams of eleven players each on a field.",
                "The game is played with a bat and ball on a field with a 22-yard pitch in the center.",
                "A match is divided into innings where one team bats while the other team bowls and fields.",
                "The objective is to score more runs than the opposing team.",
                "Runs are scored by running between the wickets after hitting the ball.",
                "A batsman can be dismissed in several ways including being bowled, caught, LBW, or run out.",
                "The team with the most runs at the end of the match wins.",
                "Each team gets to bat and bowl in turns called innings.",
                "A bowler must bowl the ball with a straight arm, not throw or jerk it.",
                "The batting team tries to score runs while the bowling team tries to dismiss the batsmen.",
                "A boundary is worth 4 runs if the ball touches the ground before crossing, and 6 runs if it crosses without touching.",
                "LBW (Leg Before Wicket) is when the ball would have hit the wickets but hits the batsman's leg instead.",
                "The wicketkeeper stands behind the wickets to catch balls and attempt dismissals.",
                "Fielders try to catch the ball or stop runs by returning it to the wickets.",
                "A wide ball or no-ball results in a penalty run being awarded to the batting team.",
            ]
            for rule in cricket_rules:
                st.session_state.search_engine.add_document(rule)
            time.sleep(1)  # Brief pause for visual feedback
            st.success("✅ Cricket knowledge base initialized!")
    
    # Sidebar with information
    with st.sidebar:
        st.title("ℹ️ About")
        st.write("""
        This is a simple cricket rules chatbot that can answer basic questions about cricket.
        
        Try asking questions like:
        - How many players in a cricket team?
        - What is LBW?
        - How do you score runs?
        - What is a boundary worth?
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
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
            with st.spinner("Finding relevant information..."):
                response = get_answer(question)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
