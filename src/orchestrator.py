
# Orchestrator for Amazon Chatbot using LangGraph
from agents.policy_agent import answer_policy
from agents.order_agent import answer_order
from agents.return_refund_agent import answer_return_refund
from agents.payment_agent import answer_payment
from agents.recommender_agent import answer_recommend
from agents.analytics_agent import answer_analytics
from langgraph.graph import StateGraph, Message, Tool

# Define tools for each agent
class PolicyTool(Tool):
    def run(self, message: Message):
        return answer_policy(message.text)

class OrderTool(Tool):
    def run(self, message: Message):
        return answer_order(message.text)

class ReturnRefundTool(Tool):
    def run(self, message: Message):
        return answer_return_refund(message.text)

class PaymentTool(Tool):
    def run(self, message: Message):
        return answer_payment(message.text)

class RecommenderTool(Tool):
    def run(self, message: Message):
        return answer_recommend(message.text)

class AnalyticsTool(Tool):
    def run(self, message: Message):
        return answer_analytics(message.text)

# Build the graph
graph = StateGraph()
graph.add_tool("policy", PolicyTool())
graph.add_tool("order", OrderTool())
graph.add_tool("return_refund", ReturnRefundTool())
graph.add_tool("payment", PaymentTool())
graph.add_tool("recommender", RecommenderTool())
graph.add_tool("analytics", AnalyticsTool())

# Intent detection function
def detect_intent(query):
    query_lower = query.lower()
    if "policy" in query_lower:
        return "policy"
    elif "order" in query_lower:
        return "order"
    elif "return" in query_lower or "refund" in query_lower:
        return "return_refund"
    elif "payment" in query_lower:
        return "payment"
    elif "recommend" in query_lower:
        return "recommender"
    elif "analytics" in query_lower:
        return "analytics"
    else:
        return None

if __name__ == "__main__":
    print("Amazon Chatbot (LangGraph) is ready. Type your question!")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        intent = detect_intent(user_query)
        if intent:
            message = Message(text=user_query)
            response = graph.run_tool(intent, message)
        else:
            response = "Sorry, I couldn't understand your request. Please specify if it's about policy, order, return, refund, payment, recommendation, or analytics."
        print("Bot:", response)
