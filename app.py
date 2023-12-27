import streamlit as st 
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import torch
import base64

# Model and tokenizer loading
def load_model():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
    return tokenizer, base_model

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(input_text):
    tokenizer, base_model = load_model()
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# Function to display PDF
def display_pdf(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI code
def main():
    st.title("Document Summarization App")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            with col1:
                st.info("Uploaded File")
                display_pdf(uploaded_file)

            with col2:
                processed_text = file_preprocessing(uploaded_file)
                summary = llm_pipeline(processed_text)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
