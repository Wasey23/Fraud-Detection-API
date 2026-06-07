# Real-Time Fraud Detection API

An end-to-end MLOps pipeline and production-grade machine learning system designed to detect fraudulent credit card transactions in real-time. This project bridges the gap between static data science research and live production systems by implementing a stateful inference architecture.

## System Architecture

Instead of a standard stateless model that evaluates transactions in a vacuum, this API utilizes an **XGBoost Classifier** backed by an ultra-fast **Redis** in-memory database to track user behavior over time.

- **Model:** XGBoost (Trained on the highly imbalanced IEEE-CIS dataset)
- **Inference Server:** FastAPI / Uvicorn (Asynchronous framework for low-latency web requests)
- **State Store:** Redis (Maintains real-time transaction history for feature engineering)
- **Deployment:** Docker & Docker Compose

## Key Engineering Achievements

### 1. Stateful Feature Engineering (The Sliding Window)
Fraud is dynamic, so the model must be dynamic. The API calculates "Card Velocity" (the number of transactions per card in the last 10 minutes) on the fly. Every incoming transaction is logged to a Redis list, allowing the model to detect high-frequency bot attacks that a stateless model would miss.

### 2. Eliminating Training-Serving Skew
A critical failure point in ML systems occurs when training logic differs from inference logic. This project utilizes an Object-Oriented `FeatureEnricher` class to guarantee that the vectorized Pandas math used during offline training perfectly mirrors the Redis-based math used during real-time online inference.

### 3. Business Logic & Imbalanced Data
Fraud detection is finding a "needle in a haystack." The model was evaluated using **AUPRC** (Area Under the Precision-Recall Curve) rather than standard accuracy. The API implements a strict business guardrail (`CUSTOMER_PRECISION_THRESHOLD = 0.10`), reflecting the asymmetric cost matrix of real-world banking where catching organized fraud justifies a lower probability threshold.

### 4. Production Guardrails
- **Memory Management:** Redis lists are strictly capped using `LTRIM` (max 250 records per card) to prevent Out-Of-Memory (OOM) crashes during aggressive botnet attacks.
- **Atomic Model Updates:** The offline training pipeline (`train.py`) uses temporary files and atomic `os.replace` operations to ensure the live API never attempts to load a corrupted or partially saved `.pkl` artifact.
- **Asynchronous Lifespan:** Machine learning artifacts are loaded into RAM once during the FastAPI startup lifespan, enabling millisecond-latency predictions.

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (if running locally without Docker)

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/Wasey23/Fraud-Detection-API.git](https://github.com/Wasey23/Fraud-Detection-API.git)
   cd Fraud-Detection-API