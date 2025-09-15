
# Order Agent (Production Ready)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_orders_from_api(user_id=None):
    url = "https://api.example.com/orders"
    params = {"user_id": user_id} if user_id else {}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        orders = response.json()
        order_text = "\n".join([
            f"Order ID: {o['order_id']}, Customer: {o['customer']}, Status: {o['status']}, Items: {', '.join(o['items'])}, Expected Delivery: {o['expected_delivery']}"
            for o in orders
        ])
        return order_text
    except Exception as e:
        return f"Error fetching orders: {e}"

def answer_order(query, user_id=None):
    order_text = fetch_orders_from_api(user_id)
    if order_text.startswith("Error"):
        return order_text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(order_text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
