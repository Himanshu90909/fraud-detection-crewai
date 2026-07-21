"""
Anomaly Detection Tools — Statistical methods to flag suspicious transaction patterns.
"""

import json
import statistics
from pathlib import Path
from crewai.tools import tool


DATA_PATH = Path(__file__).parent / ".." / "data" / "transactions.json"


def _load_transactions():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


@tool("Detect Amount Anomalies")
def detect_amount_anomalies() -> str:
    """Detect transactions with anomalous amounts using Z-score analysis.
    Any transaction with a Z-score > 2 (amount is more than 2 standard deviations
    above the mean) is flagged. Returns a JSON string of flagged transactions with their Z-scores."""
    transactions = _load_transactions()
    amounts = [t["amount"] for t in transactions]
    mean = statistics.mean(amounts)
    stdev = statistics.stdev(amounts) if len(amounts) > 1 else 0

    flagged = []
    for t in transactions:
        if stdev > 0:
            z_score = (t["amount"] - mean) / stdev
            if z_score > 2:
                flagged.append({
                    "transaction_id": t["transaction_id"],
                    "customer_id": t["customer_id"],
                    "amount": t["amount"],
                    "z_score": round(z_score, 2),
                    "reason": f"Amount ${t['amount']:.2f} is {round(z_score, 1)} std deviations above the mean (${mean:.2f})"
                })
    return json.dumps(flagged, indent=2) if flagged else "No amount anomalies detected."


@tool("Detect Velocity Fraud")
def detect_velocity_fraud() -> str:
    """Detect velocity fraud — multiple transactions by the same customer within a short
    time window (within 5 minutes). Returns a JSON string of flagged transaction groups."""
    transactions = _load_transactions()

    # Group by customer
    by_customer = {}
    for t in transactions:
        by_customer.setdefault(t["customer_id"], []).append(t)

    flagged = []
    for customer_id, txns in by_customer.items():
        if len(txns) < 2:
            continue
        # Sort by timestamp
        txns.sort(key=lambda x: x["timestamp"])
        for i in range(len(txns) - 1):
            t1 = txns[i]["timestamp"]
            t2 = txns[i + 1]["timestamp"]
            # Parse timestamps (simplified — compare as strings, then compute minute diff)
            from datetime import datetime
            dt1 = datetime.fromisoformat(t1)
            dt2 = datetime.fromisoformat(t2)
            diff_minutes = abs((dt2 - dt1).total_seconds()) / 60
            if diff_minutes < 5:
                flagged.append({
                    "customer_id": customer_id,
                    "transaction_ids": [txns[i]["transaction_id"], txns[i + 1]["transaction_id"]],
                    "time_diff_minutes": round(diff_minutes, 2),
                    "reason": f"Two transactions within {round(diff_minutes, 1)} minutes — possible velocity fraud"
                })
    return json.dumps(flagged, indent=2) if flagged else "No velocity fraud detected."


@tool("Detect Geographic Anomalies")
def detect_geographic_anomalies() -> str:
    """Detect geographic anomalies — international transactions or transactions from
    unusual locations. Flags all international transactions as potentially suspicious.
    Returns a JSON string of flagged transactions."""
    transactions = _load_transactions()
    flagged = []
    for t in transactions:
        if t["is_international"]:
            flagged.append({
                "transaction_id": t["transaction_id"],
                "customer_id": t["customer_id"],
                "amount": t["amount"],
                "location": t["location"],
                "reason": f"International transaction from {t['location']} — requires verification"
            })
    return json.dumps(flagged, indent=2) if flagged else "No geographic anomalies detected."


@tool("Detect Unusual Merchant Categories")
def detect_unusual_merchants() -> str:
    """Detect transactions involving high-risk merchant categories such as Cryptocurrency
    and Wire Transfers. Returns a JSON string of flagged transactions."""
    transactions = _load_transactions()
    high_risk_categories = ["Cryptocurrency", "Wire Transfer", "Luxury Goods"]
    flagged = []
    for t in transactions:
        if t["merchant_category"] in high_risk_categories:
            flagged.append({
                "transaction_id": t["transaction_id"],
                "customer_id": t["customer_id"],
                "amount": t["amount"],
                "merchant": t["merchant"],
                "merchant_category": t["merchant_category"],
                "reason": f"High-risk merchant category: {t['merchant_category']}"
            })
    return json.dumps(flagged, indent=2) if flagged else "No unusual merchant categories detected."


@tool("Run Full Anomaly Scan")
def run_full_anomaly_scan() -> str:
    """Run all anomaly detection checks — amount anomalies, velocity fraud, geographic anomalies,
    and unusual merchant categories. Returns a consolidated JSON report of all flagged transactions."""
    # Collect results from all detectors
    results = {}

    results["amount_anomalies"] = json.loads(detect_amount_anomalies())
    results["velocity_fraud"] = json.loads(detect_velocity_fraud())
    results["geographic_anomalies"] = json.loads(detect_geographic_anomalies())
    results["unusual_merchants"] = json.loads(detect_unusual_merchants())

    total_flags = (
        len(results["amount_anomalies"]) if isinstance(results["amount_anomalies"], list) else 0
    ) + (
        len(results["velocity_fraud"]) if isinstance(results["velocity_fraud"], list) else 0
    ) + (
        len(results["geographic_anomalies"]) if isinstance(results["geographic_anomalies"], list) else 0
    ) + (
        len(results["unusual_merchants"]) if isinstance(results["unusual_merchants"], list) else 0
    )

    results["total_flags"] = total_flags
    return json.dumps(results, indent=2)
