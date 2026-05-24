import requests
import time
import random
from src.utils.logger import get_logger

# Initialize our professional logger
logger = get_logger(__name__)

# The endpoint where your Docker container is listening
API_URL = "http://localhost:8000/predict"

# A curated list of transactions to simulate real-world traffic
TRANSACTIONS = [
    {"TransactionDT": 86400, "card1": 1001, "TransactionAmt": 25.50, "P_emaildomain": "gmail.com"},
    {"TransactionDT": 86405, "card1": 1002, "TransactionAmt": 150.00, "P_emaildomain": "yahoo.com"},
    # The Velocity Attack Simulation: Same card, huge amounts, rapid succession
    {"TransactionDT": 86410, "card1": 9999, "TransactionAmt": 950.00, "P_emaildomain": "anonymous.com"},
    {"TransactionDT": 86411, "card1": 9999, "TransactionAmt": 950.00, "P_emaildomain": "anonymous.com"},
    {"TransactionDT": 86412, "card1": 9999, "TransactionAmt": 950.00, "P_emaildomain": "anonymous.com"},
    {"TransactionDT": 86420, "card1": 1003, "TransactionAmt": 12.99, "P_emaildomain": "aol.com"}
]

def run_simulation():
    """
    Iterates through the transaction list, sending each payload to the API
    and logging the model's real-time risk assessment.
    """
    logger.info("Initializing Live Transaction Simulation...")
    logger.info(f"Target API: {API_URL}")
    print("-" * 50)

    for i, txn in enumerate(TRANSACTIONS, start=1):
        logger.info(f"Sending Transaction #{i} -> Card: {txn['card1']}, Amount: ${txn['TransactionAmt']:.2f}")

        try:
            # Send the JSON payload via an HTTP POST request
            response = requests.post(API_URL, json=txn)

            # Ensure we catch any HTTP errors (like 404 or 500)
            response.raise_for_status()

            # Parse the API's JSON response
            result = response.json()

            # Audit the result using our business logic thresholds
            if result.get("risk_level") == "High":
                logger.warning(f"API ALERT | Score: {result.get('fraud_probability')} | Status: BLOCKED")
            else:
                logger.info(f"API CLEAR | Score: {result.get('fraud_probability')} | Status: APPROVED")

        except requests.exceptions.ConnectionError:
            logger.error("CRITICAL: Failed to connect. Is the Docker container running?")
            break
        except Exception as e:
            logger.error(f"Transaction Failed: {e}")

        print("-" * 50)
        # Simulate real-world network delay between transactions (0.5 to 1.5 seconds)
        time.sleep(random.uniform(0.5, 1.5))

    logger.info("Simulation Complete.")

if __name__ == "__main__":
    run_simulation()