import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# Streamlit UI
st.title("📄 Text Summarization using LangChain")
st.write("Upload a text or PDF file to generate a concise summary.")

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

if uploaded_file is not None:
    text = ""

    # Process PDF files
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content

    # Process Text files
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    # Display extracted text
    st.subheader("Extracted Text:")
    st.text_area("Content", text, height=200)

    if st.button("Summarize"):
        if text:
            # Splitting text for better summarization
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            chunks = text_splitter.create_documents([text])

            # Prompt templates
            map_prompt = PromptTemplate(
                input_variables=['text'],
                template="Please summarize the below speech:\nSpeech: `{text}`\nSummary:"
            )

            final_combine_prompt = PromptTemplate(
                input_variables=['text'],
                template="Provide a final summary with key points.\nSpeech: `{text}`"
            )

            # Load and run summarization chain
            summary_chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=final_combine_prompt
            )

            summary = summary_chain.run(chunks)

            # Display Summary
            st.subheader("📌 Generated Summary:")
            st.write(summary)
        else:
            st.warning("No text found in the file.")
