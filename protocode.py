import os
import json
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM 

def load_data(json_path="law_office_data.json"):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    text_data = [
    f"Law Firm Name: {data['business_info']['name']}",
    f"Business Hours: {data['business_info']['hours']}",
    f"Legal Services: {', '.join(data['business_info']['services'])}",
    f"Consultation Fee: {data['consultations']['initial_fee']}, Hourly Rate: {data['consultations']['hourly_rate']}",
    f"Court Fees: {', '.join([f'{k}: {v}' for k, v in data['court_fees'].items()])}",
    "\nClient Appointments:"
    ]

    for client in data["clients"]:
        text_data.append(
            f"Client: {client['name']}, "
            f"Appointment: {client['appointment_time']}, "
            f"Case Type: {client['case_type']}, "
            f"Status: {client['status']}, "
            f"Contact: {client['contact']}"
        )

    return text_data

def preprocess_data(data):
    documents = [Document(page_content=text) for text in data]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def ask_question(vector_store, query):
    llm = OllamaLLM(model="mistral")  # Fixed LLM Import
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return qa.invoke(query)  # Fix for LangChain Deprecation Warning

if __name__ == "__main__":
    data = load_data()
    documents = preprocess_data(data)
    vector_store = create_vector_store(documents)

    while True:
        query = input("\nAsk the AI legal assistant a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = ask_question(vector_store, query)
        print("\nAI Legal Assistant:", response)

