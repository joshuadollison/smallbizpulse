# Data

Put large datasets in data/raw locally.  Do not commit large raw data to GitHub.

Suggested pattern:
- data/raw - original source files (local only)
- data/interim - intermediate transforms (local only)
- data/processed - clean, model-ready datasets (local only)
- data/external - external reference data (local only)

If you need to commit data, commit only tiny samples and document provenance.
