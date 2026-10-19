This Git directory contains the v0.2.0 analysis code and derived records plus the display-only corrections described in the root README. The immutable v0.2.0 archive retains the original figures; figure-readability.patch updates its plotting script and panel geometry to the current proof layout. Saved prediction arrays are supplied in numerical-reproduction-v0.2.0.zip on the v0.2.0 GitHub release. Extract that complete archive to a separate working directory before running saved-prediction checks.

SC NOVELTY RELIABILITY VERSION 0.2.0

This portable research archive supplies code, saved prediction arrays and derived numerical records for A reliability contract for zero-shot novel-cell-type scoring across single-cell atlases.

Recount the saved predictions
  python tools/verify_prediction_results.py --scores-only
Regenerate derived tables and figures
  python tools/analyse_evidence.py
  python tools/rank_shift_stress.py
  python tools/make_figures.py
Run these commands in an extracted working copy. Original incoming hashes are in SHA256SUMS.json.

Fresh raw-input replay requires the exact public pulmonary source object, checked by its recorded SHA-256:
  python tools/reproduce_pulmonary.py --input /absolute/path/to/GSE130148_LungHmDev.h5ad --output /absolute/path/to/new-replay
Add --limit-fit-groups 1 for a bounded replay. The output directory must be new. No dependency installation or raw-data download is performed. The scripts use numpy, scipy, pandas, scikit-learn, anndata, scanpy and matplotlib; vector composition additionally uses existing Inkscape and pikepdf. Software versions are recorded in experiments/SC31/input_inventory.json. Arial was used for the original plots; a different installed font changes appearance and graphic hashes.

Provenance and paths
Public source records replace the original local home-directory prefix with ${SOURCE_ROOT}. This is a provenance-only export; input SHA-256 values, original plan/runner hash strings, cells, settings, effects and prediction arrays are retained. EXPORT_PROVENANCE.json records both original and exported hashes for each redacted file. Those original hash strings do not authenticate the redacted bytes; SHA256SUMS.json authenticates this public export. The replay wrapper explicitly sets a user-provided input location in a copied plan. The original journal-supplied archive preserves the unredacted records separately.

Evidence scope
The fresh experiment uses four pulmonary donors: 72 distinct representation/readout fits and 11 training-label controls, 120 eligible tasks, five matched scores, 3,000 calibration-selected operating points, 18,000 conditional prevalence calculations and 840 controlled score shifts. Tasks overlap donors/cells; they are not independent biological replicates. The plan preceded these fits but followed the original findings. Historical hepatic, cardiac, frozen-representation and cached-context records are saved-result evidence; their inference is not rerun by the saved-score check. No foundation model was trained or fine-tuned. Cell/type resampling is conditional, not donor-population inference. A monotone score stress test and prevalence weighting do not establish prospective biological validation.

Files exclude raw provider matrices, independent cell barcodes, model weights and reviewer correspondence. Provider terms continue to apply. Original software is MIT licensed; original derived documentation/examples are CC BY 4.0.

Version and citation
Code and this archive: https://github.com/PeterPonyu/sc-novelty-reliability/releases/tag/v0.2.0
The historical version DOI https://doi.org/10.5281/zenodo.22224024 identifies v0.1.1, not these files. Its concept DOI is https://doi.org/10.5281/zenodo.22224023. The matching public v0.2.0 archive is published at https://doi.org/10.5281/zenodo.23436000. The immutable deposited archive retains its original display layout; apply figure-readability.patch for the current figures. The patch changes no plotted scientific arrays, numerical CSVs or saved predictions.
