# Recommender Agent (RAG-based)
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. Load products document
products_path = "data/products.txt"
loader = TextLoader(products_path)
documents = loader.load()

# 2. Chunking
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embedding generation
embeddings = OpenAIEmbeddings()

# 4. Vector DB (FAISS)
db = FAISS.from_documents(docs, embeddings)

# 5. RetrievalQA chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
)

def answer_recommend(query):
    return qa_chain.run(query)
