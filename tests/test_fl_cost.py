import unittest

import pandas as pd

from src.fl.cost import RoundCost, add_communication_savings, savings_percent


class CommunicationCostTests(unittest.TestCase):
    def test_savings_percent_uses_actual_byte_ratio(self):
        self.assertEqual(savings_percent(25, 100), 75.0)
        self.assertEqual(savings_percent(150, 100), -50.0)

    def test_zero_baseline_returns_zero_savings(self):
        self.assertEqual(savings_percent(0, 0), 0.0)
        self.assertEqual(savings_percent(10, 0), 0.0)

    def test_comparison_savings_match_rounds_and_separate_uplink_total(self):
        results = pd.DataFrame(
            [
                {"method": "fedavg", "round": 1, "upload_bytes": 100, "bytes_sent": 150},
                {"method": "compressed", "round": 1, "upload_bytes": 25, "bytes_sent": 75},
                {"method": "fedavg", "round": 2, "upload_bytes": 0, "bytes_sent": 0},
                {"method": "compressed", "round": 2, "upload_bytes": 10, "bytes_sent": 10},
            ]
        )

        compared = add_communication_savings(results)
        self.assertEqual(
            compared.loc[compared["round"] == 1, "uplink_savings_percent"].tolist(),
            [0.0, 75.0],
        )
        self.assertEqual(
            compared.loc[
                compared["round"] == 1, "total_communication_savings_percent"
            ].tolist(),
            [0.0, 50.0],
        )
        self.assertEqual(
            compared.loc[compared["round"] == 2, "uplink_savings_percent"].tolist(),
            [0.0, 0.0],
        )
        self.assertEqual(
            compared.loc[
                compared["round"] == 2, "total_communication_savings_percent"
            ].tolist(),
            [0.0, 0.0],
        )

    def test_comparison_rejects_missing_matching_baseline_round(self):
        results = pd.DataFrame(
            [
                {"method": "fedavg", "round": 1, "upload_bytes": 1, "bytes_sent": 2},
                {"method": "compressed", "round": 2, "upload_bytes": 1, "bytes_sent": 2},
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing rounds"):
            add_communication_savings(results)

    def test_round_cost_counts_attempted_and_skipped_uploads(self):
        cost = RoundCost()
        cost.record_uplink(b"header", was_skipped=True)
        cost.record_uplink(b"update")
        cost.record_downlink(b"model", recipients=2)

        self.assertEqual(cost.attempted_uploads, 2)
        self.assertEqual(cost.skipped_uploads, 1)
        self.assertEqual(cost.uplink_bytes, 12)
        self.assertEqual(cost.downlink_bytes, 10)
        self.assertEqual(cost.bytes_sent, 22)


if __name__ == "__main__":
    unittest.main()
