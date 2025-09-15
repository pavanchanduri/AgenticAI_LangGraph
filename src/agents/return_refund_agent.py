

# Return and Refund Agent (Extended Dynamic Logic)
import os
import re
from datetime import datetime, timedelta
from agents.order_agent import answer_order

def parse_order_info(order_details):
    # Extract order ID and delivery date using regex
    order_id_match = re.search(r"Order ID: (\w+)", order_details)
    delivery_date_match = re.search(r"Expected Delivery: (\d{4}-\d{2}-\d{2})", order_details)
    status_match = re.search(r"Status: (\w+)", order_details)
    order_id = order_id_match.group(1) if order_id_match else None
    delivery_date = delivery_date_match.group(1) if delivery_date_match else None
    status = status_match.group(1) if status_match else None
    return order_id, delivery_date, status

def is_eligible_for_refund(delivery_date, status):
    # Eligible if delivered and within 30 days of delivery
    if status != "Delivered" or not delivery_date:
        return False
    try:
        delivery_dt = datetime.strptime(delivery_date, "%Y-%m-%d")
        days_since_delivery = (datetime.now() - delivery_dt).days
        return days_since_delivery <= 30
    except Exception:
        return False

def create_refund(order_id):
    # Simulate API call
    return f"Refund initiated for Order ID: {order_id}. You will receive confirmation soon."

def create_replacement(order_id):
    # Simulate API call
    return f"Replacement initiated for Order ID: {order_id}. You will receive confirmation soon."

def answer_return_refund(query, user_id=None):
    # 1. Get order details using Order Agent
    order_details = answer_order(query, user_id)
    order_id, delivery_date, status = parse_order_info(order_details)
    # 2. Check eligibility
    if is_eligible_for_refund(delivery_date, status):
        print(f"Order {order_id} delivered on {delivery_date} is eligible for refund or replacement.")
        choice = input("Would you like a 'refund' or 'replacement'? ").strip().lower()
        if choice == "refund":
            return create_refund(order_id)
        elif choice == "replacement":
            return create_replacement(order_id)
        else:
            return "Invalid choice. Please type 'refund' or 'replacement'."
    elif status == "Delivered":
        return f"Order {order_id} delivered on {delivery_date} is not eligible for refund/replacement (over 30 days)."
    elif order_id:
        return f"Order {order_id} is not delivered yet. Refund/replacement can be requested after delivery."
    else:
        return "Sorry, could not find your order details. Please check your query or order ID."
