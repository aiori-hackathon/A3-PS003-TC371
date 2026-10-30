import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.fl.client import FederatedClient
from src.fl.compression import (
    UPDATE_HEADER,
    _should_gate,
    decode_update,
    encode_model,
    encode_update,
    flatten_parameters,
)
from src.model import get_parameters


class ClientUpdateGateLoggingTests(unittest.TestCase):
    def setUp(self):
        data = {
            "X_train": np.zeros((2, 12, 1), dtype=np.float32),
            "y_train": np.zeros((2, 3), dtype=np.float32),
        }
        with patch("src.fl.client.load_junction", return_value=data):
            self.client = FederatedClient(7)
        self.global_parameters = get_parameters(self.client.model)
        self.global_payload = encode_model(self.global_parameters)

    @staticmethod
    def apply_small_update(model, _inputs, _targets):
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(1e-6)

    def test_update_metrics_callback_records_actual_update_details(self):
        global_vector = flatten_parameters(self.global_parameters)
        global_norm = float(np.linalg.norm(global_vector))
        with patch("src.fl.client.train", side_effect=self.apply_small_update):
            for threshold, expected_skipped in ((0.0005, True), (0.0, False)):
                with self.subTest(threshold=threshold):
                    records = []
                    result = self.client.fit(
                        self.global_payload,
                        method="coflow_fl",
                        gate_threshold=threshold,
                        round_number=12,
                        update_metrics_callback=records.append,
                    )

                    self.assertEqual(len(records), 1)
                    record = records[0]
                    local_parameters = get_parameters(self.client.model)
                    delta = [
                        local - global_value
                        for local, global_value in zip(
                            local_parameters, self.global_parameters
                        )
                    ]
                    update_norm = float(np.linalg.norm(flatten_parameters(delta)))
                    relative_norm = update_norm / global_norm

                    self.assertEqual(record["round"], 12)
                    self.assertEqual(record["client_id"], 7)
                    self.assertEqual(record["method"], "coflow_fl")
                    self.assertAlmostEqual(record["global_model_norm"], global_norm)
                    self.assertAlmostEqual(record["client_update_norm"], update_norm)
                    self.assertAlmostEqual(record["relative_update_norm"], relative_norm)
                    self.assertEqual(record["gate_threshold"], threshold)
                    self.assertEqual(record["skipped"], expected_skipped)
                    self.assertEqual(
                        record["skipped"],
                        _should_gate("coflow_fl", relative_norm, threshold),
                    )
                    _, flags, _, _, _ = UPDATE_HEADER.unpack_from(result.payload)
                    self.assertEqual(result.gated, bool(flags & 1))
                    self.assertEqual(record["skipped"], result.gated)

    def test_update_metrics_callback_requires_round_number(self):
        with patch("src.fl.client.train", side_effect=self.apply_small_update):
            with self.assertRaisesRegex(ValueError, "round_number is required"):
                self.client.fit(
                    self.global_payload,
                    method="coflow_fl",
                    update_metrics_callback=lambda _record: None,
                )

    def test_logged_gate_decision_matches_encoded_update(self):
        updates = (
            (np.array([1e-6, -2e-6], dtype=np.float32), 1.0, 0.0005),
            (np.array([0.5, 0.0], dtype=np.float32), 1.0, 0.5),
        )
        for update, model_norm, threshold in updates:
            with self.subTest(threshold=threshold):
                relative_norm = float(np.linalg.norm(update)) / model_norm
                logged_decision = _should_gate("coflow_fl", relative_norm, threshold)
                payload = encode_update(
                    update,
                    model_norm,
                    method="coflow_fl",
                    gate_threshold=threshold,
                )
                _, encoded_decision = decode_update(payload, update.size)
                self.assertEqual(logged_decision, encoded_decision)

    def test_other_methods_are_not_gated_by_relative_norm(self):
        self.assertFalse(_should_gate("fedavg", 0.0, 0.0005))


if __name__ == "__main__":
    unittest.main()
