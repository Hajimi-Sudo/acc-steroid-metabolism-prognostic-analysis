#!/usr/bin/env python3
"""Create the remote evidence-freeze manifest for the ACC project."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "evidence_gate"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    "results/robustness/leakage_free_robustness_summary.json",
    "results/model_consistency/model_consistency_summary.json",
    "results/model_consistency/model_consistency_metrics.tsv",
    "results/model_consistency/model_consistency_selection_frequency.tsv",
    "results/external_validation/GSE19750_locked_signature_validation.json",
    "results/external_validation/GSE19750_locked_signature_sensitivity.json",
    "results/external_validation/GSE33371_locked_signature_validation.json",
    "results/external_validation/GSE33371_overlap_audit.json",
    "results/state_robustness/state_robustness_summary.json",
    "results/external_validation/GSE49278_steroid_scores.tsv",
    "results/external_validation/mapping_audit.json",
    "results/signature_interactions/signature_interaction_summary.json",
    "results/druggability/candidate_target_prioritization.tsv",
    "manifests/GSE19750_survival_audit.json",
    "manifests/gse49278_parse_audit.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    env = ROOT / "env" / "acc" / "bin" / "python"
    versions = subprocess.check_output(
        [str(env), "-c", "import numpy,pandas,scipy,sklearn,lifelines; print({k:v.__version__ for k,v in [('numpy',numpy),('pandas',pandas),('scipy',scipy),('sklearn',sklearn),('lifelines',lifelines)]})"],
        text=True,
    ).strip()
    artifacts = []
    for relative in FILES:
        path = ROOT / relative
        artifacts.append({"path": relative, "exists": path.exists(), "sha256": sha256(path) if path.exists() else None, "bytes": path.stat().st_size if path.exists() else None})
    manifest = {
        "project": "acc-steroid-metabolism",
        "timestamp_utc": "2026-10-20",
        "compute_location": "remote_only",
        "hostname": platform.node(),
        "python_packages": versions,
        "decision": "evidence gate passed with caveats; independent validation remains limited",
        "primary_claim": "steroid-metabolism state and model-stability evidence in ACC",
        "external_claim_boundary": "GSE19750 small directional adult OS test; GSE33371 same-biological-sample follow-up layer of GSE10927; GSE49278 independent partial-program OS evidence",
        "blocked_claims": ["definitive external validation", "C1A/C1B interaction", "clinical utility", "drug efficacy", "causal mechanism"],
        "artifacts": artifacts,
    }
    path = OUT / "final_evidence_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
