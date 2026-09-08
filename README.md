# sc-novelty-reliability

**A reliability contract for single-cell transfer scores.** Version 0.1.2.

A transfer score is only reusable if you know what it measured. This package gives you
one interface that fixes the estimand, the donor-held-out split, the calibration
partition, the abstention rule, and the provenance binding *before* a score is read, and
returns them alongside every value. Point it at a new atlas or a new embedding and you
get back a comparable measurement, its operating range, and an explicit reason when the
input does not qualify.

## What you get

| Capability | Where |
|---|---|
| Binned expected calibration error and deterministic risk--coverage curves, dependency-light | `src/sc_novelty_reliability/metrics.py` |
| Alignment contract for frozen embeddings, returning an explicit comparable / non-comparable state | `src/sc_novelty_reliability/compatibility.py` |
| Ordered label and donor sequence audit, kept separate from any barcode-identity claim | `src/sc_novelty_reliability/compatibility.py` |
| Role-labelled example summaries for three atlas contexts and six frozen configurations | `examples/` |
| Reproducible vector figure builder and a publication layout source | `figures/` |
| Tests, a release manifest, and a continuous-integration check | `tests/`, `PUBLIC_RELEASE_MANIFEST.json` |

## Why it is useful

Report a transfer score at the wrong unit and it stops meaning anything. Pooling cell
types across atlas contexts can return a healthy-looking summary while one context ranks
novel cells below known ones, and a structural covariate can appear predictive across
pooled types while carrying no association inside any single atlas. This package makes
the reporting unit an explicit, checkable part of the output, so those effects are
visible instead of averaged away.

Concretely, it lets you:

- score novel-cell-type detectability under a leakage-controlled, donor-held-out split;
- calibrate on a reserved group only, and read the coverage you actually retained rather
  than the one you requested;
- compare a conventional principal-component readout against a frozen foundation-model
  embedding through the identical split, so the comparison is matched;
- refuse to score an input whose metadata, support, or lineage does not check out, with a
  named reason rather than a silent imputation.

## Quick start

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -e '.[test]'
    python -m pytest -q

Check whether an embedding you already hold is comparable to a registered
measurement, then read calibration and the coverage you actually retained:

```python
from sc_novelty_reliability import (
    alignment_contract, metadata_sequence_contract,
    expected_calibration_error, risk_coverage_curve,
)

reference = {
    "cells_per_type": {"B cell": 2, "T cell": 2},
    "cells_per_donor": {"donor_a": 2, "donor_b": 2},
    "held_out_donor": "donor_b",
    "calibration_donor": "donor_a",
    "training_donors": [],
}
labels = ["B cell", "T cell", "B cell", "T cell"]
donors = ["donor_a", "donor_a", "donor_b", "donor_b"]

gate = alignment_contract(labels, donors, reference)
print(gate["status"], "-", gate["reason"])          # COMPARABLE - ...

# Order is verified without asserting barcode identity.
print(metadata_sequence_contract(labels, donors, labels, donors)["status"])

y_true      = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
probability = [.05, .2, .55, .3, .9, .75, .1, .6, .85, .95]
print(round(expected_calibration_error(y_true, probability), 3))
for row in risk_coverage_curve(y_true, probability, [0.6, 0.8]):
    print(row["requested_coverage"], row["actual_coverage"], row["risk"])
```

A failing gate returns `NOT_COMPARABLE` together with the reason, so an unusable
input is named rather than silently scored.

Rebuild the release figure (requires R):

    Rscript figures/build_figures.R
    Rscript figures/figure_contract.R

The layout wrapper in `figures/tex/` consumes the vector panels and compiles with
PDFLaTeX when a TeX distribution is available.

## Scope of the shipped examples

The examples are compact derived summaries: enough to exercise the public contract and
rebuild the release figure. Raw atlas matrices, donor-identifying metadata, and model
weights stay with their original providers and are not redistributed, so obtain those
independently to reproduce the cached-input estimates. The three primary rows are
atlas-associated measurements with declared roles and are kept separate rather than
pooled; the six frozen-configuration rows demonstrate adapter compatibility on one
registered split and carry ordered-metadata and convergence flags.

## License and citation

Original software is MIT-licensed. Derived example tables and explanatory text are
shared under CC BY 4.0 as described in `LICENSE-CONTENT.md`. See `CITATION.cff` for
machine-readable citation metadata, and cite the archived snapshot at
<https://doi.org/10.5281/zenodo.22224024>.
