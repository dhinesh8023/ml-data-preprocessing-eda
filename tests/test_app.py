"""
Unit tests for Flask Dashboard application.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app


class TestFlaskDashboard(unittest.TestCase):
    """Test suite for Flask dashboard endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        """Test GET / renders successfully."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Telco Customer Churn", res.data)

    def test_predict_route_high_risk(self):
        """Test POST /predict with high-risk attributes."""
        payload = {
            "tenure": 1,
            "monthly": 95.0,
            "contract": "Month-to-month",
            "internet": "Fiber optic",
            "tech": "No",
            "payment": "Electronic check",
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("probability", data)
        self.assertIn("risk_tier", data)
        self.assertGreater(data["probability"], 0.5)
        self.assertEqual(data["risk_tier"], "High Risk")

    def test_predict_route_low_risk(self):
        """Test POST /predict with low-risk loyal attributes."""
        payload = {
            "tenure": 65,
            "monthly": 25.0,
            "contract": "Two year",
            "internet": "No",
            "tech": "Yes",
            "payment": "Credit card (automatic)",
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertLess(data["probability"], 0.25)
        self.assertEqual(data["risk_tier"], "Low Risk")


if __name__ == "__main__":
    unittest.main()
