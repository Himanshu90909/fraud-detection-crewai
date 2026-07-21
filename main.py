"""
Fraud Detection Using CrewAI
================================

A multi-agent fraud detection system that analyzes financial transactions,
identifies suspicious patterns, and generates a structured fraud report.

Agents:
  1. Transaction Analyst — Profiles and summarizes transactions
  2. Anomaly Detector — Runs statistical anomaly detection checks
  3. Risk Assessor — Scores flagged transactions and assigns risk levels
  4. Report Generator — Compiles a structured fraud detection report

Usage:
    export OPENAI_API_KEY="your-key"
    python main.py
"""

import os
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from tools.transaction_tools import (
    get_all_transactions,
    get_transaction_summary,
    get_high_value_transactions,
    get_international_transactions,
    get_late_night_transactions,
)
from tools.anomaly_tools import (
    detect_amount_anomalies,
    detect_velocity_fraud,
    detect_geographic_anomalies,
    detect_unusual_merchants,
    run_full_anomaly_scan,
)

load_dotenv()

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
)

# ─────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────

transaction_analyst = Agent(
    role="Senior Transaction Analyst",
    goal="Analyze all financial transactions in the dataset and create a detailed profile "
         "of transaction patterns — amounts, frequencies, merchant categories, geographic "
         "distribution, and time-of-day patterns.",
    backstory="""You are a senior transaction analyst at a major financial institution with
    10+ years of experience in payment systems. You have a keen eye for transaction patterns
    and can quickly identify what 'normal' looks like vs. what stands out. You are methodical
    and thorough, always starting with a complete picture before jumping to conclusions.""",
    tools=[
        get_all_transactions,
        get_transaction_summary,
        get_high_value_transactions,
        get_international_transactions,
        get_late_night_transactions,
    ],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

anomaly_detector = Agent(
    role="Fraud Anomaly Detection Specialist",
    goal="Run statistical anomaly detection checks on all transactions to identify suspicious
         patterns — amount outliers (Z-score), velocity fraud (rapid successive transactions),
         geographic anomalies (international transactions), and high-risk merchant categories.",
    backstory="""You are a fraud detection specialist who uses statistical methods to catch
    anomalies that humans might miss. You are experienced with Z-score analysis, velocity
    checks, and behavioral pattern analysis. You don't make assumptions — you run the data
    through rigorous checks and let the numbers speak. You flag everything suspicious and
    let the Risk Assessor determine severity.""",
    tools=[
        detect_amount_anomalies,
        detect_velocity_fraud,
        detect_geographic_anomalies,
        detect_unusual_merchants,
        run_full_anomaly_scan,
    ],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

risk_assessor = Agent(
    role="Risk Assessment Officer",
    goal="Evaluate every flagged transaction from the anomaly detector and assign a risk
         score from 0 to 100, classify it as Low / Medium / High / Critical risk, and
         provide a clear justification for the score.",
    backstory="""You are a risk assessment officer at a financial institution. You evaluate
    flagged transactions and assign risk scores based on multiple factors: transaction amount,
    time of day, merchant risk level, geographic anomaly, velocity pattern, and customer
    history. You are decisive but fair — you don't want to block legitimate transactions,
    but you also can't let fraud slip through. You always provide clear reasoning for your
    risk scores.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

report_generator = Agent(
    role="Fraud Report Generator",
    goal="Compile all findings from the transaction analysis, anomaly detection, and risk
         assessment into a structured, professional fraud detection report with an executive
         summary, detailed findings, risk scores, and actionable recommendations.",
    backstory="""You are a technical writer and fraud analyst who specializes in creating
    clear, actionable reports for executives and fraud investigation teams. You take complex
    findings from multiple analysts and synthesize them into a single, well-organized report.
    Your reports are known for being concise yet comprehensive, with clear recommendations
    that investigators can act on immediately.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────

analysis_task = Task(
    description="""Analyze the complete transaction dataset using the available tools.

    1. First, get a summary of all transactions (count, total amount, unique customers, merchants).
    2. Retrieve all transactions and profile them:
       - What is the average, median, and max transaction amount?
       - Which customers have the most transactions?
       - What are the most common merchant categories?
       - Are there any international transactions?
       - Are there transactions at unusual hours (11 PM – 6 AM)?
    3. Provide a structured transaction profile summary.

    Be thorough — this profile will be used by the anomaly detector to identify deviations
    from normal patterns.""",
    expected_output="A structured transaction profile with statistics, patterns, and notable observations.",
    agent=transaction_analyst,
)

anomaly_task = Task(
    description="""Based on the transaction profile from the analyst, run comprehensive anomaly
    detection on the dataset.

    1. Run the full anomaly scan to check ALL anomaly types.
    2. For each anomaly type, review the specific results:
       - Amount anomalies (Z-score > 2)
       - Velocity fraud (multiple transactions within 5 minutes)
       - Geographic anomalies (international transactions)
       - Unusual merchant categories (Cryptocurrency, Wire Transfer, Luxury Goods)
    3. Compile a list of ALL flagged transactions with:
       - Transaction ID
       - Customer ID
       - Amount
       - Specific anomaly type(s) detected
       - Brief reason for flagging

    Do NOT filter or dismiss any flagged transaction — pass ALL flags to the Risk Assessor
    for evaluation.""",
    expected_output="A comprehensive list of all flagged transactions with anomaly types and reasons.",
    agent=anomaly_detector,
    context=[analysis_task],
)

risk_task = Task(
    description="""Evaluate each flagged transaction from the anomaly detector and assign a
    risk score.

    For EACH flagged transaction, provide:
    1. Transaction ID
    2. Risk Score (0-100, where 100 = highest risk)
    3. Risk Level: Low (0-25) | Medium (26-50) | High (51-75) | Critical (76-100)
    4. Justification: 1-2 sentences explaining the score
    5. Recommended Action:
       - Low: Monitor
       - Medium: Hold for verification (send OTP)
       - High: Block and contact customer
       - Critical: Block, freeze account, escalate to fraud team

    Consider multiple factors when scoring:
    - Transaction amount relative to normal patterns
    - Time of day (late night = higher risk)
    - Merchant category risk level
    - Geographic anomaly
    - Velocity (multiple rapid transactions = higher risk)
    - Combination of multiple anomaly types (compounding risk)

    Present the results as a structured table.""",
    expected_output="A risk assessment table with scores, levels, justifications, and recommendations for each flagged transaction.",
    agent=risk_assessor,
    context=[anomaly_task],
)

report_task = Task(
    description="""Compile all findings into a final structured Fraud Detection Report.

    The report MUST include:

    1. EXECUTIVE SUMMARY
       - Date of analysis
       - Total transactions analyzed
       - Total flagged as suspicious
       - Breakdown by risk level (Critical / High / Medium / Low)

    2. TRANSACTION PROFILE
       - Key statistics from the transaction analysis
       - Normal transaction patterns identified

    3. ANOMALIES DETECTED
       - Summary of each anomaly type
       - Count of transactions flagged per type

    4. FLAGGED TRANSACTIONS — DETAILED TABLE
       For each flagged transaction:
       - Transaction ID | Amount | Timestamp | Merchant | Location
       - Risk Score | Risk Level
       - Reason for Flagging
       - Recommended Action

    5. RECOMMENDATIONS
       - Immediate actions for Critical/High risk transactions
       - Monitoring recommendations for Medium/Low risk
       - General fraud prevention suggestions

    Format the report clearly with headers and separators. This is a professional document
    that will be reviewed by the fraud investigation team.""",
    expected_output="A complete, formatted Fraud Detection Report with executive summary, detailed findings, risk scores, and recommendations.",
    agent=report_generator,
    context=[analysis_task, anomaly_task, risk_task],
)

# ─────────────────────────────────────────────
# CREW
# ─────────────────────────────────────────────

fraud_detection_crew = Crew(
    agents=[transaction_analyst, anomaly_detector, risk_assessor, report_generator],
    tasks=[analysis_task, anomaly_task, risk_task, report_task],
    process=Process.sequential,
    verbose=True,
)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   FRAUD DETECTION SYSTEM — CrewAI Multi-Agent Pipeline")
    print("=" * 60)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    result = fraud_detection_crew.kickoff()

    print()
    print("=" * 60)
    print("   FRAUD DETECTION COMPLETE")
    print("=" * 60)
    print()

    # Save report to file
    report_path = os.path.join(os.path.dirname(__file__), "fraud_report_output.txt")
    with open(report_path, "w") as f:
        f.write(str(result))
    print(f"Report saved to: {report_path}")
    print()
    print(result)
