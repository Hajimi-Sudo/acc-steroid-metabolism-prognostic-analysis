# ACC steroid-metabolism analysis code

This directory contains the analysis and figure-generation scripts for the ACC
steroid-metabolism study. The repository is configured for remote-only
computation.

## Execution boundary

All data downloads, matrix construction, preprocessing, statistical analysis,
model fitting, endpoint matching, and quantitative figure generation must run
under `/root/autodl-tmp/acc-steroid-metabolism` on the assigned server. Do not
run the `remote_*.py` scripts against local data and do not copy raw matrices or
patient-level clinical rows into this repository.

The scripts read the project root from `ACC_PROJECT_ROOT` when supported. The
default remote root is `/root/autodl-tmp/acc-steroid-metabolism`.

## Reproducible run order

Run the stages below on the server after recording the host, Python and package
versions, mounted filesystems, command lines, timestamps, and output checksums.

1. `remote_gate1_inventory.py` audits remote storage and source metadata.
2. `remote_download.py` acquires the frozen public inputs.
3. `remote_extract_inputs.py` creates the frozen TCGA and GEO inputs.
4. `remote_first_wave.py` creates the initial score and cohort audit.
5. `remote_leakage_free_robustness.py` evaluates patient-level internal OOF
   performance and the exploratory steroid state.
6. `remote_state_robustness.py` evaluates patient perturbation, co-assignment,
   and cluster-number evidence.
7. `remote_model_consistency_audit.py` separates fixed-gene OOF performance,
   fold-wise feature-selection OOF performance, and apparent full-data fitting.
8. `remote_gse19750_audit.py`, `remote_gse19750_locked_validation.py`, and
   `remote_gse19750_sensitivity.py` audit and test the small external OS layer.
9. `remote_gse33371_locked_validation.py` audits the GSE10927 follow-up layer
   and prevents duplicate expression samples from being counted independently.
10. `remote_gse49278_supplement_inspect.py`,
    `remote_gse49278_survival_audit.py`, `remote_external_validation.py`, and
    `remote_gse49278_reanalysis.py` audit the GSE49278 endpoint bridge and
    partial-program analysis.
11. `remote_signature_interactions.py`, `remote_biology_associations.py`, and
    `remote_druggability_prioritization.py` run the exploratory association,
    interaction, and candidate-prioritization analyses.
12. `remote_make_final_tables.py`, `remote_export_figure_data.py`, and the
    selected publication renderer create aggregate manuscript outputs.
13. `remote_finalize_evidence.py` records the final evidence manifest and
    checksums.

## Script groups

- `remote_download.py`, `remote_extract_inputs.py`: acquisition and input
  extraction.
- `remote_first_wave.py`, `remote_leakage_free_robustness.py`,
  `remote_state_robustness.py`, `remote_model_consistency_audit.py`: primary
  score, state, and internal validation analyses.
- `remote_gse*.py`, `remote_external_validation.py`: external cohort audits and
  locked-score or partial-program tests.
- `remote_signature_interactions.py`, `remote_biology_associations.py`,
  `remote_druggability_prioritization.py`: exploratory downstream analyses.
- `remote_make_*.py`, `remote_export_figure_data.py`: aggregate tables and
  publication figures. They do not redefine the frozen model.
- `remote_finalize_evidence.py`: evidence-gate manifest generation.
- `render_figure5_interaction_audit.py`: local raster rendering from reviewed
  aggregate files in `paper/figure_data`. It is not an experiment and does not
  access raw data.

The older figure renderers are retained for provenance. The current manuscript
figure choice is recorded in `00_contract/FIGURE_PLAN.md` and the final quality
report in `paper/FINAL_QC_REPORT_v2.md`.

## Scientific boundaries

The locked four-gene score is `STAR`, `NR0B1`, `HSD11B2`, and `AKR1C3`. GSE33371
is a follow-up layer for the same biological expression samples as GSE10927.
GSE49278 supports a native-platform 10 of 22 gene partial-program analysis and
does not support complete locked four-gene validation. Candidate target output
is hypothesis generation only. No script supports a causal, efficacy, or
clinical-deployment claim.

## Environment

The tested remote environment is Python 3.11.17. Package requirements are
listed in `requirements-remote.txt`. Installation and execution belong on the
server environment only.
