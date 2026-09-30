import unittest
from app import app

class AppSmokeTest(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertIn(response.status_code, (200,))

    def test_empty_prediction_rejected(self):
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
