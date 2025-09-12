import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key in the environment
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. Load FAQ document
faq_path = "faq.txt"  # Place your FAQ text file here
loader = TextLoader(faq_path)
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

if __name__ == "__main__":
    print("FAQ/Support Bot is ready. Type your question!")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print("Bot:", answer)
