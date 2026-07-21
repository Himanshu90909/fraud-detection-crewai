# Fraud Detection Using CrewAI

A multi-agent fraud detection system built with [CrewAI](https://github.com/crewAIInc/crewAI) that analyzes financial transactions, identifies suspicious patterns, and generates a structured fraud report.

## Overview

This project uses a **multi-agent system** (CrewAI) with four specialized AI agents working together:

| Agent | Role |
|---|---|
| 🔍 **Transaction Analyst** | Parses and profiles transactions — amounts, frequency, merchant categories, geographic patterns |
| 🚨 **Anomaly Detector** | Flags outliers using statistical rules (Z-score, velocity checks, unusual merchant/location) |
| 📊 **Risk Assessor** | Scores each flagged transaction 0–100 and classifies risk level (Low / Medium / High / Critical) |
| 📝 **Report Generator** | Compiles findings into a structured fraud detection report with recommendations |

## Architecture

```
Transaction Data
       │
       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Transaction    │────▶│  Anomaly         │────▶│  Risk           │────▶│  Report          │
│  Analyst        │     │  Detector        │     │  Assessor       │     │  Generator       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
       │                        │                        │                        │
       ▼                        ▼                        ▼                        ▼
  Transaction Profile     Suspicious Flags         Risk Scores              Final Report
```

## Project Structure

```
fraud-detection-crewai/
├── main.py                 # Entry point — defines agents, tasks, and runs the crew
├── tools/
│   ├── __init__.py
│   ├── transaction_tools.py   # Mock transaction data + query tools
│   └── anomaly_tools.py       # Statistical anomaly detection functions
├── data/
│   └── transactions.json      # Sample transaction dataset (20 transactions)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Himanshu90909/fraud-detection-crewai.git
cd fraud-detection-crewai

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the fraud detection crew
python main.py
```

## Sample Output

```
============================================================
              FRAUD DETECTION REPORT
============================================================
Analysis Period: 2026-07-20
Total Transactions Analyzed: 20
Flagged as Suspicious: 5
Critical Risk: 1 | High Risk: 2 | Medium Risk: 1 | Low Risk: 1

FLAGGED TRANSACTIONS:
─────────────────────
1. [CRITICAL] TXN-015 | $14,500 | 03:42 AM | Merchant: "CryptoExchange-X"
   Reason: Unusual large amount at odd hours, first-time merchant
   Recommendation: Block transaction, freeze account, contact customer

2. [HIGH] TXN-007 | $3,200 | 11:58 PM | Location: Lagos, Nigeria
   Reason: Geographic anomaly — user typically transacts in India
   Recommendation: Hold for verification, send OTP to customer
   ...
============================================================
```

## Key Features

- **Multi-Agent Orchestration** — Four agents with distinct roles collaborate via CrewAI's task delegation
- **Statistical Anomaly Detection** — Z-score analysis, velocity checks, amount thresholding
- **Tool-Calling** — Agents use custom tools to query transactions and run anomaly detection
- **Structured Reporting** — Final agent generates a formatted fraud report with risk scores and recommendations
- **Extensible** — Add new agents, tools, or data sources easily

## Tech Stack

- **CrewAI** — Multi-agent orchestration framework
- **Python** — Core language
- **OpenAI GPT-4** — LLM backend for agent reasoning
- **Statistical Methods** — Z-score, IQR, velocity checks (no external ML model needed)

## Requirements

- Python 3.10+
- OpenAI API key (or any CrewAI-supported LLM)

## License

MIT
