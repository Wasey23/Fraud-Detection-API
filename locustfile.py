from locust import HttpUser, task, between

class FraudAPIUser(HttpUser):

    wait_time = between(1, 3)

    payload = {
      "TransactionID": 999999,
      "TransactionDT": 86450,
      "TransactionAmt": 3500.00,
      "ProductCD": "C",
      "card1": 99999,
      "card2": 555.0,
      "card3": 185.0,
      "card4": "discover",
      "card5": 137.0,
      "card6": "credit",
      "P_emaildomain": "mail.com",
      "R_emaildomain": "anonymous.com",
      "C1": 45.0, "C2": 52.0, "C11": 30.0, "C12": 40.0, "C14": 25.0,
      "V70": 5.0, "V91": 2.0, "V147": 8.0, "V156": 4.0, "V172": 1.0, "V258": 15.0, "V294": 7.0,
      "id_17": 225.0,
      "DeviceType": "mobile",
      "DeviceInfo": "iOS Device"
    }

    @task
    def test_xgboost(self):
        self.client.post("/predict/xgboost", json=self.payload)

    @task
    def test_lgbm(self):
        self.client.post("/predict/lgbm", json=self.payload)