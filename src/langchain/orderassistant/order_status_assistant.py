from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.orderassistant.orders import orders
import os

# Set your OpenAI API key in the environment
# os.environ["OPENAI_API_KEY"] = "your-api-key"

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    input_variables=["query", "orders"],
    template="""
You are a helpful order status assistant for an ecommerce store. Given the following orders:
{orders}

Answer the user's query: {query}
If the order is not found, say so politely.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def search_orders(query):
    order_info = "\n".join([
        f"Order ID: {o['order_id']}, Customer: {o['customer']}, Status: {o['status']}, Items: {', '.join(o['items'])}, Expected Delivery: {o['expected_delivery']}"
        for o in orders
    ])
    response = chain.run({"query": query, "orders": order_info})
    return response

if __name__ == "__main__":
    user_query = input("Ask about your order status: ")
    answer = search_orders(user_query)
    print("Order Assistant:", answer)
