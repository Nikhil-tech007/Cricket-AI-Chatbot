# Cricket-AI-Chatbot
This project involves the development of an AI-driven chatbot that answers cricket-related questions using Llama 3.1 and Groq. The chatbot is designed to provide accurate responses based on a cricket PDF dataset and utilizes additional tools like FAISS, DuckDuckGo Search, and Streamlit for enhanced functionality and user interaction.

Table of Contents
Overview
Features
Technologies Used
Installation
Usage
Contributing
License
Overview
The Cricket AI Chatbot is designed to answer queries about cricket rules, history, and more. It uses advanced language models and search tools to provide accurate and contextual responses. When the context from the PDF dataset is insufficient, the chatbot uses the DuckDuckGo search tool to fetch additional information.

Features
AI-Powered Responses: Utilizes Llama 3.1 and Groq for natural language processing and response generation.
Contextual Understanding: Searches a cricket rules PDF for relevant information using FAISS for vector-based document retrieval.
Fallback Search: Uses DuckDuckGo Search for answering questions outside the dataset's scope.
User Interface: A simple and interactive UI built with Streamlit for easy user interaction.
Technologies Used
Llama 3.1: Language model for understanding and generating responses.
Groq: Enhances processing speed and efficiency.
FAISS: Vector database for efficient information retrieval.
DuckDuckGo Search: Provides additional information when the dataset lacks context.
LangChain: Framework for chaining various AI components.
Streamlit: Framework for building the user interface.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/cricket-ai-chatbot.git
Navigate to the project directory:
bash
Copy code
cd cricket-ai-chatbot
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Set up the environment variables:
Create a .env file in the project root directory.
Add your API keys and other configurations:
env
Copy code
GROQ_API_KEY=your_groq_api_key
Usage
Run the Streamlit application:
bash
Copy code
streamlit run app.py
Interact with the chatbot through the provided UI.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add a feature').
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for more information.
