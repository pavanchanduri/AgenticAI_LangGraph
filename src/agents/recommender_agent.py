
# Recommender Agent (Production Ready)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_products_from_api():
    url = "https://api.example.com/products"
    try:
        response = requests.get(url)
        response.raise_for_status()
        products = response.json()
        product_text = "\n".join([
            f"Name: {p['name']}, Category: {p['category']}, Price: ${p['price']}, Description: {p['description']}" for p in products
        ])
        return product_text
    except Exception as e:
        return f"Error fetching products: {e}"

def answer_recommend(query):
    product_text = fetch_products_from_api()
    if product_text.startswith("Error"):
        return product_text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(product_text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
