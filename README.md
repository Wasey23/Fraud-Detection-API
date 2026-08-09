# Real-Time Fraud Detection API

An end-to-end Machine Learning API built to instantly evaluate credit card transactions and block fraudulent activity in milliseconds.

## Purpose
Credit card fraud costs businesses billions of dollars every year. The challenge isn't just catching the fraud; it's catching it **instantly** before the transaction goes through, without falsely declining legitimate customers. 

This project takes historical transaction data, trains intelligent Machine Learning models to spot the hidden patterns of thieves, and packages those models into a lightning-fast web server (API) that can handle high volumes of real-time traffic.

---

## How It Works 

When a customer swipes their credit card, here is exactly what happens under the hood:

1. **Data Ingestion:** The API receives a digital receipt (a JSON payload) containing raw transaction details (e.g., card type, amount, time).
2. **Feature Enrichment (The "Translator"):** The raw data is passed through a custom `FeatureEnricher`. This pipeline instantly translates text (like "Visa" or "Debit") into math, and calculates historical metrics on the fly (e.g., "How many times has this card been used in the last 24 hours?").
3. **Inference (The "Brain"):** The enriched math is handed to a pre-trained Machine Learning model.
4. **The Decision:** In milliseconds, the model returns a risk probability, returning either an **`APPROVED`** or **`BLOCKED`** verdict to the merchant.

---

## Key Engineering Achievements

### 1. Multi-Model Architecture (A/B Testing)
To find the perfect balance between catching bad guys and keeping the system fast, this API hosts multiple algorithms simultaneously. Front-end systems can easily send identical data to different endpoints to race the models against each other:
* **`/predict/xgboost`**: Routes the transaction to a highly precise, cautious **XGBoost** model. 
* **`/predict/lightgbm`**: Routes the transaction to a blazingly fast, aggressive **LightGBM** model that casts a wider net for suspicious activity.

### 2. Eliminating "Training-Serving Skew"
A common issue in Machine Learning is that a model performs great in a lab but fails in the real world because the live data looks slightly different than the training data. 
I built a unified, Object-Oriented `FeatureEnricher` class. This guarantees that the exact same data-cleaning steps used to train the models offline are perfectly mirrored when processing live, single-transaction web requests.

### 3. Load Testing & Traffic Optimization
To ensure the API doesn't crash during a Black Friday shopping rush, the system was stress-tested using **Locust** to simulate 50 users hitting the server at the exact same millisecond.
* The models originally tried to use all available CPU power for every single request, causing traffic jams and massive delays (up to 13-second wait times).
* The models were re-engineered to process linearly, and the API server was scaled horizontally using **Dockerized Worker Processes**, I added 4 workers. 
* Response times dropped by over 70%, with 99% of all transactions successfully processed in under 2.3 seconds.

---

## Quick Start 

Because this project is fully containerized with Docker, launching the server and its dependencies on your machine is seamless.

**1. Clone the repository:**
```bash
git clone [https://github.com/wasey23/fraud-detection-api.git](https://github.com/wasey23/fraud-detection-api.git)
cd fraud-detection-api
