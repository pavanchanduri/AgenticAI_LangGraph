

# Order Agent (RAG-based, dynamic API)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_orders_from_api(user_id=None):
    # Replace with your actual API endpoint and parameters
    url = "https://api.example.com/orders"
    params = {"user_id": user_id} if user_id else {}
    response = requests.get(url, params=params)
    response.raise_for_status()
    orders = response.json()
    # Convert orders to text format for chunking
    order_text = "\n".join([
        f"Order ID: {o['order_id']}, Customer: {o['customer']}, Status: {o['status']}, Items: {', '.join(o['items'])}, Expected Delivery: {o['expected_delivery']}"
        for o in orders
    ])
    return order_text

def answer_order(query, user_id=None):
    # 1. Fetch data from API
    order_text = fetch_orders_from_api(user_id)
    # 2. Chunking
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(order_text)
    # 3. Embedding generation
    embeddings = OpenAIEmbeddings()
    # 4. Vector DB (FAISS)
    db = FAISS.from_texts(docs, embeddings)
    # 5. RetrievalQA chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
