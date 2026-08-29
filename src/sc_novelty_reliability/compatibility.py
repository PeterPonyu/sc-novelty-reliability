"""Metadata-only comparability checks for existing frozen embeddings."""
from __future__ import annotations
from collections import Counter
from typing import Any, Iterable

def _counts(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(str, values)).items())}

def alignment_contract(labels: Iterable[Any], donors: Iterable[Any], primary: dict[str, Any]) -> dict[str, Any]:
    """Return COMPARABLE only when count maps and split names match."""
    y = list(map(str, labels))
    b = list(map(str, donors))
    result: dict[str, Any] = {"status": "NOT_COMPARABLE", "n_cells": len(y), "alignment_basis": "label_and_donor_count_contract", "row_order_verified": False}
    if not y or len(y) != len(b):
        result["reason"] = "label and donor arrays are empty or have different lengths"
        return result
    expected_labels = {str(k): int(v) for k, v in primary.get("type_n_total", {}).items()}
    expected_donors = {str(k): int(v) for k, v in primary.get("gse_donor_n", {}).items()}
    result["label_count_match"] = _counts(y) == dict(sorted(expected_labels.items()))
    result["donor_count_match"] = _counts(b) == dict(sorted(expected_donors.items()))
    donors = set(b)
    result["missing_split_donors"] = [str(d) for d in [primary.get("held_gse_donor"), primary.get("cal_gse_donor"), *primary.get("train_gse_donors", [])] if d is not None and str(d) not in donors]
    if not result["label_count_match"]:
        result["reason"] = "label count contract mismatch"
    elif not result["donor_count_match"]:
        result["reason"] = "donor count contract mismatch"
    elif result["missing_split_donors"]:
        result["reason"] = "registered split donor is absent"
    else:
        result["status"] = "COMPARABLE"
        result["reason"] = "labels, donor counts and registered split names match"
    return result

__all__ = ["alignment_contract"]
