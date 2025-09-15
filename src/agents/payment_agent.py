
# Payment Agent (Production Ready)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_payments_from_api():
    url = "https://api.example.com/payments"
    try:
        response = requests.get(url)
        response.raise_for_status()
        payments = response.json()
        payment_text = "\n".join([
            f"{p['type']}: {p['info']}" for p in payments
        ])
        return payment_text
    except Exception as e:
        return f"Error fetching payment info: {e}"

def answer_payment(query):
    payment_text = fetch_payments_from_api()
    if payment_text.startswith("Error"):
        return payment_text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(payment_text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
