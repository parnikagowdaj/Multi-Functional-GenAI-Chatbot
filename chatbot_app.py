
import streamlit as st
from transformers import pipeline
import torch

# Page Configuration
st.set_page_config(
    page_title="Multi-Functional GenAI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Multi-Functional GenAI Chatbot")

st.write(
    "A simple Generative AI chatbot using Hugging Face and Streamlit."
)

# Load Model
@st.cache_resource
def load_model():

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    model = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    return model

pipe = load_model()

# Select Task
task = st.selectbox(
    "Select a Task",
    [
        "Text Generation",
        "Code Generation",
        "Question & Answer",
        "Summarization",
        "Translation"
    ]
)

# Input
if task == "Text Generation":
    user_input = st.text_area(
        "Enter your prompt",
        placeholder="Write a story about artificial intelligence..."
    )

elif task == "Code Generation":
    user_input = st.text_area(
        "Describe the code you want",
        placeholder="Create a Python program to calculate factorial..."
    )

elif task == "Question & Answer":
    user_input = st.text_area(
        "Enter your question",
        placeholder="What is machine learning?"
    )

elif task == "Summarization":
    user_input = st.text_area(
        "Enter the text to summarize",
        placeholder="Paste a long paragraph here..."
    )

elif task == "Translation":
    user_input = st.text_area(
        "Enter the text to translate",
        placeholder="Translate this English sentence to Hindi..."
    )

# Generate Button
if st.button("🧪 TEST"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        if task == "Text Generation":
            prompt = f"""
You are a helpful text generation assistant.

Generate high-quality content based on the following request:

{user_input}
"""

        elif task == "Code Generation":
            prompt = f"""
You are an expert programming assistant.

Generate correct and simple code for the following requirement:

{user_input}

Explain the code briefly after providing it.
"""

        elif task == "Question & Answer":
            prompt = f"""
You are a helpful question-answering assistant.

Answer the following question clearly and accurately:

{user_input}
"""

        elif task == "Summarization":
            prompt = f"""
You are a summarization assistant.

Summarize the following text into clear and important key points:

{user_input}
"""

        elif task == "Translation":
            prompt = f"""
You are a translation assistant.

Translate the following text according to the language requested by the user:

{user_input}
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        with st.spinner("Generating response..."):

            result = pipe(
                messages,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )

        response = result[0]["generated_text"][-1]["content"]

        st.subheader("✨ Result")
        st.write(response)
