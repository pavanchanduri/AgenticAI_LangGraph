
# Analytics Agent (Production Ready)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_analytics_from_api():
    url = "https://api.example.com/analytics"
    try:
        response = requests.get(url)
        response.raise_for_status()
        analytics = response.json()
        analytics_text = "\n".join([
            f"{a['metric']}: {a['value']}" for a in analytics
        ])
        return analytics_text
    except Exception as e:
        return f"Error fetching analytics: {e}"

def answer_analytics(query):
    analytics_text = fetch_analytics_from_api()
    if analytics_text.startswith("Error"):
        return analytics_text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(analytics_text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
