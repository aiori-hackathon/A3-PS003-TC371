import unittest

import numpy as np

from src.fl.compression import (
    decode_model,
    decode_update,
    encode_model,
    encode_update,
    flatten_parameters,
    unflatten_parameters,
)


class CompressionProtocolTests(unittest.TestCase):
    def test_model_round_trip(self):
        params = [
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.linspace(1.0, 4.0, 4, dtype=np.float32),
        ]
        payload = encode_model(params)
        decoded = decode_model(payload, [tuple(p.shape) for p in params])
        self.assertEqual(len(decoded), len(params))
        for original, restored in zip(params, decoded):
            np.testing.assert_allclose(original, restored)

    def test_dense_update_round_trip(self):
        update = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        payload = encode_update(update, model_norm=5.0, method="fedavg")
        decoded, gated = decode_update(payload, update.size)
        self.assertFalse(gated)
        np.testing.assert_allclose(decoded, update)

    def test_quantized_update_round_trip_near_exact(self):
        update = np.array([0.3, -0.8, 1.2, -1.7, 0.1], dtype=np.float32)
        payload = encode_update(update, model_norm=2.0, method="quantized_fedavg")
        decoded, gated = decode_update(payload, update.size)
        self.assertFalse(gated)
        self.assertLess(np.max(np.abs(decoded - update)), 0.05)

    def test_sparse_update_round_trip(self):
        update = np.array([0.0, 1.5, 0.0, -2.0, 0.0], dtype=np.float32)
        payload = encode_update(update, model_norm=5.0, method="topk_fedavg", top_k_fraction=0.4)
        decoded, gated = decode_update(payload, update.size)
        self.assertFalse(gated)
        np.testing.assert_allclose(decoded, update, atol=1e-6)

    def test_gating_skips_small_updates(self):
        update = np.array([1e-6, -2e-6, 3e-6], dtype=np.float32)
        payload = encode_update(update, model_norm=1.0, method="gated_fedavg", gate_threshold=1e-5)
        decoded, gated = decode_update(payload, update.size)
        self.assertTrue(gated)
        np.testing.assert_array_equal(decoded, np.zeros_like(update))

    def test_flatten_unflatten_round_trip(self):
        parameters = [
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32),
        ]
        flat = flatten_parameters(parameters)
        self.assertEqual(flat.size, sum(np.prod(item.shape) for item in parameters))
        restored = unflatten_parameters(flat, [tuple(p.shape) for p in parameters])
        for original, rebuilt in zip(parameters, restored):
            np.testing.assert_allclose(original, rebuilt)


if __name__ == "__main__":
    unittest.main()
