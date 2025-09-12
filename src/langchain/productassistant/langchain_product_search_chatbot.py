from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.productassistant.products import products
import os

# Set your OpenAI API key in the environment
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize the LLM (using OpenAI as example, you can swap for other providers)
llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    input_variables=["query", "products"],
    template="""
You are a helpful product search assistant for an ecommerce store. Given the following products:
{products}

Answer the user's query: {query}
If the product is not found, say so politely.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def search_products(query):
    product_info = "\n".join([
        f"Name: {p['name']}, Category: {p['category']}, Price: ${p['price']}, Description: {p['description']}"
        for p in products
    ])
    response = chain.run({"query": query, "products": product_info})
    return response

if __name__ == "__main__":
    user_query = input("Ask about a product: ")
    answer = search_products(user_query)
    print("Chatbot:", answer)
