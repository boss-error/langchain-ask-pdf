import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from g4f.client import Client  # Importing Client from g4f.client

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

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            response_text = response.choices[0].message['content']
               
            st.write(response_text)

if __name__ == '__main__':
    main()
