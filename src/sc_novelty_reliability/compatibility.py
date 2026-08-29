"""Metadata-only comparability checks for existing frozen embeddings."""
from __future__ import annotations
import hashlib
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

def sequence_digest(values: Iterable[Any]) -> str:
    """Hash an ordered metadata sequence with unambiguous boundaries."""
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()

def metadata_sequence_contract(labels: Iterable[Any], donors: Iterable[Any], reference_labels: Iterable[Any], reference_donors: Iterable[Any]) -> dict[str, Any]:
    """Verify ordered label/donor metadata without claiming barcode identity."""
    y = list(map(str, labels)); b = list(map(str, donors))
    ry = list(map(str, reference_labels)); rb = list(map(str, reference_donors))
    label_match = y == ry; donor_match = b == rb
    return {
        "status": "PASS" if label_match and donor_match else "NOT_VERIFIED",
        "n_cells": len(y),
        "labels_sequence_match": label_match,
        "donors_sequence_match": donor_match,
        "metadata_sequence_verified": label_match and donor_match,
        "barcode_identity_claim": False,
        "metadata_sequence_sha256": sequence_digest([*y, *b]),
        "reference_metadata_sequence_sha256": sequence_digest([*ry, *rb]),
    }

__all__ = ["alignment_contract", "metadata_sequence_contract", "sequence_digest"]
