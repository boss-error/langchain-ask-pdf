import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import g4f

g4f.debug.logging = False # enable logging
g4f.check_version = False # Disable automatic version checking

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Using g4f.client for LLM
            client = Client()
            messages = [{"role": "system", "content": "You are a question answer assistant based on content."}]
            for doc in docs:
                messages.append({"role": "user", "content": doc.page_content})
            messages.append({"role": "user", "content": user_question})

            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=messages
            )
            
               
            st.write(response)

if __name__ == '__main__':
    main()
