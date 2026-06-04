import requests

URL = "http://localhost:8000/predict"
sample_sus_transaction = {
    "TransactionID": 3663549,
    "TransactionDT": 86400,
    "TransactionAmt": 850.50, # High dollar risk
    "ProductCD": "C",         # High-risk product category
    "card1": 4444,            # Using a different card ID for a clean Redis slate
    "card2": 555.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 142.0,
    "card6": "credit",
    "P_emaildomain": "mail.com", # High-risk disposable email
    "R_emaildomain": "mail.com"
}

'''
sample_safe_transaction = {
    "TransactionID": 9876543, 
    "TransactionDT": 86400, 
    "TransactionAmt": 45.00,      # Standard everyday purchase amount
    "ProductCD": "W",             # The most common, lowest-risk product category
    "card1": 8888,                # A fresh card ID to ensure 0 initial velocity in Redis
    "card2": 111.0, 
    "card3": 150.0, 
    "card4": "visa",              # Standard, highly trusted issuer
    "card5": 226.0, 
    "card6": "debit",             # Standard debit transaction
    "P_emaildomain": "gmail.com", # Highly trusted standard email provider
    "R_emaildomain": "gmail.com"
}
'''

def run_simulation(iterations):
    start_dt = 86400
    for i in range(iterations):
        # Create a dynamic transaction that changes over time
        tx = sample_sus_transaction.copy()
        # tx = sample_safe_transaction.copy()
        tx['TransactionDT'] = start_dt + (i * 10) # Advance time by 10 seconds each run
        tx['TransactionID'] = 3663549 + i

        print(f"--- Iteration {i+1}: Sending DT {tx['TransactionDT']} ---")
        response = requests.post(URL, json=tx)

        if response.status_code == 200:
            data = response.json()
            print(f"Prob: {data['fraud_probability']}, Status: {data['status']}")

if __name__ == "__main__":
    run_simulation(100)