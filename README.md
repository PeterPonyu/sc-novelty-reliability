# sc-novelty-reliability

Release-safe reference implementation of a reliability contract for within-atlas,
donor-held-out zero-shot novel-cell-type scoring (v0.1.1).

The package makes the operating conditions travel with a score: a registered estimand,
group split, calibration-only temperature fit, selective risk--coverage points, and a
source/hash binding. It is an application instrument for single-cell analysis. Compact
examples show how a conventional representation and six cached frozen single-cell
foundation-model configurations spanning five families enter the same contract; they
are not a universal model leaderboard.

## Included

- dependency-light metric primitives for binned ECE and deterministic risk--coverage;
- a metadata alignment contract for frozen embeddings with explicit comparable and
  non-comparable states;
- compact, role-labelled atlas and six-configuration compatibility summaries;
- a base-R vector figure builder plus a TXC/TikZ layout source;
- tests, a public manifest, and a GitHub Check.

No raw atlas, donor-identifying metadata, model weight, or private submission material is
distributed. The examples are compact derived summaries intended to exercise the public
contract and reproduce the release figure. Compatibility rows include an ordered metadata
verification flag and a readout-convergence flag; no barcode or private atlas identifier
is distributed.

## Quick start

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -e '.[test]'
    python -m pytest -q
    Rscript figures/build_figures.R
    Rscript figures/figure_contract.R

The TXC wrapper in figures/tex/ consumes the vector panels and can be compiled with
PDFLaTeX when a TeX distribution is available.

## Interpretation

The three compact primary rows are atlas-associated measurements with explicit roles.
They are intentionally kept separate instead of pooled into a tissue or universal
parameter. The six frozen-scFM rows demonstrate adapter compatibility on an existing
split; they do not claim foundation-model superiority or causal biological effects.

## License and citation

Original software is MIT-licensed. Derived example tables and explanatory text are
shared under CC BY 4.0 as described in LICENSE-CONTENT.md. See CITATION.cff for
machine-readable citation metadata.
