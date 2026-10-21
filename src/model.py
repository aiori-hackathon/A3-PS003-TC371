"""Small GRU model and helpers shared by local, centralized, and federated runs."""

import random

import numpy as np
import torch
from torch import nn

SEED = 42


class TrafficGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(1, 32, num_layers=1, batch_first=True)
        self.output = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        return self.output(sequence[:, -1, :])


def get_model() -> TrafficGRU:
    """Return a freshly initialized CPU model."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    return TrafficGRU()


def train(
    model: TrafficGRU,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1,
    lr: float = 0.005,
    batch_size: int = 64,
) -> float:
    """Train with shuffled mini-batches and return the final epoch mean loss."""
    torch.manual_seed(SEED)
    model.train()
    inputs = torch.from_numpy(np.asarray(X, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(y, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    last_loss = 0.0

    for _ in range(epochs):
        order = torch.randperm(len(inputs))
        losses = []
        for start in range(0, len(inputs), batch_size):
            indexes = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(inputs[indexes]), targets[indexes])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        last_loss = float(np.mean(losses)) if losses else 0.0
    return last_loss


def evaluate(
    model: TrafficGRU,
    X: np.ndarray,
    y: np.ndarray,
    mean: float,
    std: float,
) -> dict[str, float]:
    """Evaluate MAE and RMSE after converting values back to original units."""
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()
    predictions = predictions * std + mean
    targets = np.asarray(y, dtype=np.float32) * std + mean
    error = predictions - targets
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }


def get_parameters(model: TrafficGRU) -> list[np.ndarray]:
    """Return model parameters as numpy arrays for Flower."""
    return [parameter.detach().cpu().numpy().copy() for parameter in model.state_dict().values()]


def set_parameters(model: TrafficGRU, params: list[np.ndarray]) -> None:
    """Load a list produced by get_parameters into a model."""
    state = model.state_dict()
    if len(params) != len(state):
        raise ValueError(f"Expected {len(state)} parameter arrays, got {len(params)}.")
    updated = {
        name: torch.from_numpy(np.asarray(value)).to(dtype=state[name].dtype)
        for name, value in zip(state, params)
    }
    model.load_state_dict(updated)


def parameter_bytes(params: list[np.ndarray]) -> int:
    """Return the total serialized array size in bytes."""
    return int(sum(np.asarray(parameter).nbytes for parameter in params))
