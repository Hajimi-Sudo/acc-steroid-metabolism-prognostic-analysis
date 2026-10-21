# ACC Steroid-Metabolism State and Prognostic Analysis

This repository contains the remote-only analysis code and reviewed aggregate
outputs for a public-cohort study of steroid-metabolism expression states and
prognostic model stability in adrenocortical carcinoma.

## Scope

- Discovery cohort: TCGA-ACC.
- Small external OS test: GSE19750.
- Follow-up layer: GSE33371, whose expression samples match GSE10927 biological
  samples and are therefore not counted as an independent expression cohort.
- Independent partial-program OS layer: GSE49278, with 10 of 22 program genes
  represented on GPL16686 and 42 sample-matched positive-time OS records.

The locked four-gene score contains `STAR`, `NR0B1`, `HSD11B2`, and `AKR1C3`.
The GSE49278 analysis is a partial-program analysis and is not complete
four-gene validation. Candidate target results are hypothesis-generating and do
not demonstrate drug sensitivity, treatment efficacy, causality, or clinical
utility.

## Compute boundary

All data downloads, preprocessing, model fitting, endpoint matching, statistical
analysis, and quantitative figure generation were designed to run on a remote
server. Raw matrices, GEO archives, GDC files, and patient-level clinical rows
are not included in this repository.

See [`code/README.md`](code/README.md) for the remote execution boundary and
the reproducible stage order. See [`code/CODE_MANIFEST.md`](code/CODE_MANIFEST.md)
for the script-level manifest. Research declarations are recorded in
[`DECLARATIONS.md`](DECLARATIONS.md).

## Repository contents

- `code/`: acquisition, audit, analysis, and figure-generation scripts.
- `source_data/`: reviewed aggregate tables used by selected figures.
- `figures/`: reviewed publication-oriented figure assets.
- `gene_sets/`: the frozen 22-gene program definition.
- `evidence/`: concise aggregate evidence summaries and artifact provenance.

## Data sources

- TCGA-ACC: <https://portal.gdc.cancer.gov/projects/TCGA-ACC>
- GSE19750: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19750>
- GSE33371: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33371>
- GSE10927: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10927>
- GSE49278: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49278>
- GSE49280: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49280>

## Environment

The tested remote environment used Python 3.11.17. Pinned core dependencies are
listed in [`code/requirements-remote.txt`](code/requirements-remote.txt).

## Reproducibility and use

Do not edit numerical outputs by hand. Re-run the corresponding remote stage,
record the command and environment, and update checksums when aggregate outputs
change. The repository contains research code and aggregate evidence, not a
clinical decision tool.

## License

The code is released under the MIT License. Public dataset terms and access
conditions remain governed by their original providers.
