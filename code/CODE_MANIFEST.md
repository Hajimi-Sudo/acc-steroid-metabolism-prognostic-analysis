# Code manifest

The scripts below are the reproducible computational components of the study.
They are designed to run on the remote server only and write to
`/root/autodl-tmp/acc-steroid-metabolism`.

## Canonical analysis pipeline

| Stage | Script | Output role |
|---|---|---|
| Source inventory | `remote_gate1_inventory.py` | Source metadata and storage audit |
| Acquisition | `remote_download.py` | Frozen public archives and manifests |
| Input extraction | `remote_extract_inputs.py` | Remote processed cohort inputs |
| Initial score | `remote_first_wave.py` | Program score, survival, and clinical audit |
| Leakage-free robustness | `remote_leakage_free_robustness.py` | Internal OOF and exploratory state results |
| State robustness | `remote_state_robustness.py` | Patient perturbation, co-assignment, and k-selection |
| Model consistency | `remote_model_consistency_audit.py` | Fixed-gene and feature-selection OOF estimates |
| GSE19750 audit | `remote_gse19750_audit.py` | Endpoint and sample audit |
| GSE19750 validation | `remote_gse19750_locked_validation.py` | Frozen-score OS test |
| GSE19750 sensitivity | `remote_gse19750_sensitivity.py` | Descriptive score-construction sensitivity |
| GSE33371 validation | `remote_gse33371_locked_validation.py` | Follow-up-layer and overlap audit |
| GSE49278 supplement audit | `remote_gse49278_supplement_inspect.py` | Supplement discovery and provenance |
| GSE49278 survival audit | `remote_gse49278_survival_audit.py` | Sample-matched endpoint audit |
| GSE49278 mapping | `remote_external_validation.py` | Native-platform gene mapping and partial OS test |
| GSE49278 reanalysis | `remote_gse49278_reanalysis.py` | Partial-program survival summary |
| Biology associations | `remote_biology_associations.py` | Hormone, Weiss, stage, and program contrasts |
| Score interactions | `remote_signature_interactions.py` | Exploratory interaction models |
| Candidate prioritization | `remote_druggability_prioritization.py` | Hypothesis-generating target table |
| Tables | `remote_make_final_tables.py` | Manuscript-facing aggregate tables |
| Figure source export | `remote_export_figure_data.py` | Aggregate source data for figures |
| Evidence freeze | `remote_finalize_evidence.py` | Final artifact manifest and checksums |

## Figure renderers

`remote_make_figures_publication_v2.py` is the current general publication
renderer. `remote_make_figures_reference_style.py` is the current renderer for
the supplied reference layouts. `remote_make_updated_figures_1_5.py` contains
the user-requested Figure 1 and Figure 5 update path. The other
`remote_make_*.py` files are retained as provenance for earlier figure
iterations. `render_figure5_interaction_audit.py` renders only from reviewed
aggregate TSV files in `paper/figure_data` and is not an experiment.

## Outputs safe to publish

Only reviewed aggregate JSON, TSV, figure, manuscript, manifest, and checksum
files belong in a public repository. Exclude all `data/raw`,
`data/processed`, patient-level clinical tables, GEO archives, GDC files,
server logs containing access details, virtual environments, and Python cache
files. The project-level `.gitignore` enforces the main exclusions.

## Provenance rule

Do not edit numerical outputs by hand. If an aggregate result or figure changes,
rerun the corresponding remote stage, update its checksum, and preserve the
remote command and environment record.
