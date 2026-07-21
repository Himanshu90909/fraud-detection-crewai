"""
Transaction Tools — Custom CrewAI tools for querying and profiling transactions.
"""

import json
from pathlib import Path
from crewai.tools import tool


# Load transaction dataset once at module level
DATA_PATH = Path(__file__).parent / ".." / "data" / "transactions.json"


def _load_transactions():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


@tool("Get All Transactions")
def get_all_transactions() -> str:
    """Retrieve all transactions from the dataset for analysis.
    Returns a JSON string of all transaction records."""
    transactions = _load_transactions()
    return json.dumps(transactions, indent=2)


@tool("Get Transactions by Customer")
def get_transactions_by_customer(customer_id: str) -> str:
    """Retrieve all transactions for a specific customer by their customer_id (e.g. 'CUST-1001').
    Returns a JSON string of matching transactions."""
    transactions = _load_transactions()
    filtered = [t for t in transactions if t["customer_id"] == customer_id]
    return json.dumps(filtered, indent=2)


@tool("Get Transaction Summary")
def get_transaction_summary() -> str:
    """Generate a summary of all transactions — total count, total amount, unique customers,
    unique merchants, and international transaction count. Returns a formatted string."""
    transactions = _load_transactions()
    total_amount = sum(t["amount"] for t in transactions)
    unique_customers = len(set(t["customer_id"] for t in transactions))
    unique_merchants = len(set(t["merchant"] for t in transactions))
    intl_count = sum(1 for t in transactions if t["is_international"])

    summary = (
        f"Transaction Summary:\n"
        f"  Total Transactions: {len(transactions)}\n"
        f"  Total Amount: ${total_amount:,.2f}\n"
        f"  Unique Customers: {unique_customers}\n"
        f"  Unique Merchants: {unique_merchants}\n"
        f"  International Transactions: {intl_count}\n"
    )
    return summary


@tool("Get High Value Transactions")
def get_high_value_transactions(threshold: float = 1000.0) -> str:
    """Retrieve all transactions above a given amount threshold (default $1000).
    Returns a JSON string of transactions exceeding the threshold."""
    transactions = _load_transactions()
    high_value = [t for t in transactions if t["amount"] >= threshold]
    return json.dumps(high_value, indent=2)


@tool("Get International Transactions")
def get_international_transactions() -> str:
    """Retrieve all international transactions (where is_international is true).
    Returns a JSON string of international transactions."""
    transactions = _load_transactions()
    intl = [t for t in transactions if t["is_international"]]
    return json.dumps(intl, indent=2)


@tool("Get Late Night Transactions")
def get_late_night_transactions() -> str:
    """Retrieve all transactions that occurred between 11 PM and 6 AM (suspicious hours).
    Returns a JSON string of late-night transactions."""
    transactions = _load_transactions()
    late_night = []
    for t in transactions:
        hour = int(t["timestamp"].split("T")[1].split(":")[0])
        if hour >= 23 or hour < 6:
            late_night.append(t)
    return json.dumps(late_night, indent=2)
