# sc-novelty-reliability

Version 0.2.0 implements and evaluates a reporting protocol for novel-cell-type scores. Each result retains its atlas and withheld target, donor roles, calibration, acceptance rule, input checks and provenance. The individual scoring and calibration methods are established; the contribution is their joint evaluation and explicit interpretation.

The median of 37 preserved target-specific AUROCs is 0.740, while the pulmonary, hepatic and adult-cardiac primary medians are 0.777, 0.425 and 0.911. Those values describe fixed donor assignments. The complete assignment records show additional variation; the hepatic median across all scored assignments is 0.573. A pooled log-distance correlation of +0.390 attenuates to -0.039 after atlas centering, with mixed within-atlas signs. This is an aggregation effect, not a demonstrated consistent Simpson reversal.

New pulmonary fits evaluate five matched scores across 120 supported tasks from four donors. At a calibration-selected nominal 80% known-retention threshold, scaled maximum probability retains a median 81.7% of held known cells but accepts 58.4% of targets. Rank-preserving score transformations leave AUROC unchanged while changing fixed-cutoff acceptance and error.

## Reproduce the revision

Download [the complete v0.2.0 numerical archive](https://github.com/PeterPonyu/sc-novelty-reliability/releases/download/v0.2.0/numerical-reproduction-v0.2.0.zip) and extract it into a new directory. It contains all saved prediction arrays, code, source records and a per-file checksum manifest. In that extracted directory run:

```bash
python tools/verify_prediction_results.py --scores-only
python tools/analyse_evidence.py
python tools/rank_shift_stress.py
python tools/make_figures.py
```

The [revision directory](revision/) contains the same analysis code and derived records without the prediction arrays. Its README specifies software and raw-input replay requirements. Local path prefixes are redacted in the public provenance records; original and exported hashes are recorded separately. Every numerical CSV and saved prediction array retains its original bytes.

## Figure readability after v0.2.0

The default branch includes display corrections after the immutable v0.2.0 release: consistent decimals in Figure 4A, colorbar spacing in Figure 7G, legend clearance in Figure 8C, and cell-color-based text contrast in Figure 8G. No plotted scientific arrays, numerical CSVs or saved predictions change. To reproduce the final proof figures from the complete v0.2.0 archive, replace its `tools/make_figures.py` with [the corrected script](revision/tools/make_figures.py), or apply [the exact patch](revision/figure-readability.patch) from the extracted archive root with `patch -p1 < figure-readability.patch`, then run `python tools/make_figures.py`. The supplied manuscript source packages contain the corresponding final artwork. The original tag and release asset remain unchanged.

## Reference interface

The dependency-light interface and compact examples remain available:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

`alignment_contract` checks the declared label/donor counts and split roles; `metadata_sequence_contract` checks ordered labels and donors without asserting barcode identity. `expected_calibration_error` and `risk_coverage_curve` provide the original binary-score diagnostics. Those compact functions do not implement the entire new multiclass experiment; its reproducible scripts are supplied separately in `revision/tools/`.

## Interpretation and scope

Zero-shot is target-relative: the withheld type is absent from fitted preprocessing, known-label readout training and calibration. It does not establish absence from foundation-model pretraining. Fresh raw-input fits use the pulmonary atlas only. Historical hepatic, cardiac, six frozen-representation configurations and three additional cached contexts remain saved-record analyses. The frozen configurations share one pulmonary split. Cached interface execution is not prospective external validation. Repeated tasks and conditional cell/type resampling do not establish donor-population uncertainty or a coverage/error guarantee. Source annotation vocabularies remain separate.

Raw matrices, independent barcodes and model weights are excluded and retain provider terms. Reviewer correspondence and journal submission files are excluded.

## Versioned citation and licensing

Use the [v0.2.0 code and numerical release](https://github.com/PeterPonyu/sc-novelty-reliability/releases/tag/v0.2.0), archived at [Zenodo DOI 10.5281/zenodo.23436000](https://doi.org/10.5281/zenodo.23436000), for the revision analyses. All four deposited file sizes and server MD5 checksums match the verified local files. The numerical ZIP is one unchanged archive, with SHA-256 `c71ff2d5cc93e16560c5a7ec6f3fbcfa4db6326ac12ed25b4f158ced7b0b56b7`. The [historical version DOI](https://doi.org/10.5281/zenodo.22224024) identifies v0.1.1; the [concept DOI](https://doi.org/10.5281/zenodo.22224023) identifies their version lineage. The immutable v0.2.0 tag and release archives retain their original bytes. The separately pinned display patch above reproduces the final figure layout without changing scientific arrays.

Original software is MIT licensed. Original derived documentation and examples are CC BY 4.0; source-provider terms remain applicable. `CITATION.cff` identifies software authorship for the current code snapshot.
