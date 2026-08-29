"""Dependency-light reliability metrics for separated arrays."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np

def _inputs(y_true: Sequence[int], probability: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true and probability must be non-empty and have equal length")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must be finite and lie in [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary (0/1)")
    return y, p

def expected_calibration_error(y_true: Sequence[int], probability: Sequence[float], *, bins: int = 10) -> float:
    """Return deterministic equal-width binned ECE."""
    if bins < 1:
        raise ValueError("bins must be positive")
    y, p = _inputs(y_true, probability)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)

def risk_coverage_curve(y_true: Sequence[int], probability: Sequence[float], coverages: Iterable[float] = (0.60, 0.70, 0.80, 0.90)) -> list[dict[str, float | int]]:
    """Return stable selective-risk rows at requested coverage values."""
    y, p = _inputs(y_true, probability)
    requested = [float(c) for c in coverages]
    if not requested or any(c <= 0.0 or c > 1.0 or not np.isfinite(c) for c in requested):
        raise ValueError("coverages must be a non-empty sequence in (0, 1]")
    predicted = (p >= 0.5).astype(int)
    correct = (predicted == y).astype(float)
    order = np.argsort(-np.maximum(p, 1.0 - p), kind="mergesort")
    rows: list[dict[str, float | int]] = []
    for c in requested:
        k = max(1, min(len(y), int(np.ceil(len(y) * c))))
        selected = order[:k]
        rows.append({"requested_coverage": c, "actual_coverage": float(k / len(y)), "risk": float(1.0 - correct[selected].mean()), "n": k})
    return rows

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

__all__ = ["expected_calibration_error", "risk_coverage_curve", "sha256_file"]
