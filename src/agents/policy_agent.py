# """
# Policy Agent
#
# Description:
# This agent provides answers to user queries about company policies (returns, privacy, shipping, refunds, etc.) in an ecommerce setting. It is designed to be production-ready and fetches policy data dynamically from a backend API, ensuring up-to-date responses.
#
# Detailed Working:
# 1. When a user asks a policy-related question, the agent calls a REST API endpoint to fetch the latest policy data (e.g., return policy, privacy policy).
# 2. The API response (JSON) is converted into a text format suitable for chunking and semantic search.
# 3. The text is split into manageable chunks using LangChain's CharacterTextSplitter.
# 4. Each chunk is embedded using OpenAIEmbeddings, and stored in a FAISS vector database for fast similarity search.
# 5. When a query is received, it is embedded and used to search the vector DB for the most relevant policy chunks.
# 6. The retrieved context is passed to an LLM (OpenAI) via the RetrievalQA chain, which generates a natural language answer for the user.
# 7. The agent includes robust error handling for API failures and provides clear feedback to the user.
#
# This approach ensures that policy answers are accurate, current, and context-aware, making the agent suitable for real-world ecommerce applications.
# """
# Policy Agent (Production Ready)
import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# os.environ["OPENAI_API_KEY"] = "your-api-key"

def fetch_policies_from_api():
    # Replace with your actual API endpoint
    url = "https://api.example.com/policies"
    try:
        response = requests.get(url)
        response.raise_for_status()
        policies = response.json()
        # Convert policies to text format for chunking
        policy_text = "\n".join([
            f"{p['type']}: {p['description']}" for p in policies
        ])
        return policy_text
    except Exception as e:
        return f"Error fetching policies: {e}"

def answer_policy(query):
    policy_text = fetch_policies_from_api()
    if policy_text.startswith("Error"):
        return policy_text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(policy_text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    llm = OpenAI(temperature=0)

    # Create the QA chain
    # llm - The large language model (OpenAI) used to generate answers.
    # chain_type - Specifies the type of chain. "stuff" means the retrieved documents (chunks) are simply 
    #              concatenated and passed as context to the LLM. Other chain types may use different ways to combine or process retrieved documents.
    # retriever - Converts the FAISS vector database into a retriever object. This retriever takes a query, embeds it, and finds the most similar chunks in the database.
    #
    # How it works:
    # When a query is received, the retriever searches the vector DB for relevant policy chunks.
    # The retrieved chunks are concatenated ("stuffed") together.
    # The concatenated context and the user’s query are sent to the LLM.
    # The LLM generates a natural language answer using the context.
    #
    # Summary:
    # This line sets up a pipeline that performs semantic search over your policy data and uses an LLM to answer questions, making your agent capable of accurate, context-aware responses.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    return qa_chain.run(query)
