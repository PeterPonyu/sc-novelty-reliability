# Application note: reliability as a first-class output

Single-cell transfer pipelines often expose a score without exposing the conditions that
make the score interpretable. This package treats those conditions as a small contract:

1. declare the estimand and role of the measurement;
2. keep donor/group splits fixed and visible;
3. fit calibration only on the registered calibration partition;
4. report selective risk at explicit coverage points;
5. bind source values and release files to hashes.

The same interface accepts a classical expression representation or an already computed
frozen scFM embedding. Before any readout is fitted, the alignment helper checks complete
label and donor-count maps and the presence of registered split donors. A mismatch is
returned as NOT_COMPARABLE; no performance number is imputed.

This is an application and software contribution: it gives analysts a reproducible way
to decide whether a transfer score is sufficiently specified for a claim and where an
abstention boundary lies. Compact tables are examples of the data shape, not a
replacement for an atlas-specific provenance record.
