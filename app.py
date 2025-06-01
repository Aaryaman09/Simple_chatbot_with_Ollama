import json, os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load configuration from key.json
with open("key.json", "r") as f:
    config = json.load(f)

# Ensure the Ollama server is running and accessible on Langsmith server
os.environ["LANGCHAIN_API_KEY"] = config["langchain_api_key"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = config["langchain_project"]


# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond concisely."),
        ("user", "Question : {question}"),
    ]
)

# Streamlit app to interact with Ollama LLM
st.title("Ollama Chatbot")
st.write("Ask me anything!")
user_input = st.text_input("Your question:")

# Initialize the Ollama LLM with the specified model and parameters
llm = Ollama(
    model=config["model"],
    temperature=0.1,
    # max_tokens=1000,
    # top_p=0.9,
    # top_k=40,
)

# Create a chain that combines the prompt, LLM, and output parser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# If user input is provided, invoke the chain and display the response
if user_input:
    st.write(chain.invoke({"question": user_input}))
