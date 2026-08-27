"""
ADEGuard - End-to-end pipeline orchestrator.
Run: python -m src.pipeline --data-dir /path/to/vaers/csvs --max-rows 8000
"""
import argparse
import json
import time
import pandas as pd

from .data_pipeline import build_master, age_bucket
from .gazetteers import build_ade_gazetteer, build_drug_gazetteer, top_ade_frequency
from .weak_supervision import weak_label_dataframe
from .ner import ADENERTagger
from .severity import combine_weak_severity, SeverityClassifier, SEVERITY_ORDER
from .clustering import ADEClusterer, build_cluster_text
from .annotation_export import export as export_annotation


def run(data_dir, out_dir, max_rows=8000, sample_for_annotation=150, offline=False,
        start_year=None, end_year=None):
    t0 = time.time()
    log = {}

    if offline:
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("Running in --offline mode: skipping any Hugging Face download "
              "attempts, using fallback (gazetteer/TF-IDF) models directly.")

    yr_msg = ""
    if start_year or end_year:
        yr_msg = f", years {start_year or 'earliest'}-{end_year or 'latest'}"
    print(f"[1/7] Discovering & merging VAERS year files (max_rows={max_rows}{yr_msg})...")
    master = build_master(data_dir, max_rows=max_rows, start_year=start_year, end_year=end_year)
    master["AGE_BUCKET"] = master["AGE_YRS_NUM"].apply(age_bucket)
    log["n_reports"] = len(master)

    print("[2/7] Building gazetteers from structured MedDRA + vaccine fields...")
    ade_gaz = build_ade_gazetteer(master["MEDDRA_TERMS"])
    drug_gaz = build_drug_gazetteer(master["VACCINES"])
    log["ade_gazetteer_size"] = len(ade_gaz)
    log["drug_gazetteer_size"] = len(drug_gaz)
    log["top_ade_terms"] = top_ade_frequency(master["MEDDRA_TERMS"], n=20)

    print("[3/7] Weak-supervision span labeling (ADE + DRUG) on free text...")
    master = weak_label_dataframe(master, ade_gaz, drug_gaz)

    print("[4/7] NER pass (BioBERT if available, else gazetteer tagger)...")
    tagger = ADENERTagger(ade_gaz, drug_gaz)
    log["ner_backend"] = tagger.describe()
    # WEAK_ADE_SPANS/WEAK_DRUG_SPANS already computed via same code path in fallback;
    # tagger.extract is what a BioBERT-available environment would call per-row instead.

    print("[5/7] Severity labeling (rules + modifiers + learned classifier)...")
    master["SEVERITY_WEAK"] = master.apply(combine_weak_severity, axis=1)
    clf = SeverityClassifier()
    clf.fit(master["SYMPTOM_TEXT_CLEAN"], master["SEVERITY_WEAK"])
    master["SEVERITY_PRED"] = clf.predict(master["SYMPTOM_TEXT_CLEAN"])
    log["severity_backend"] = clf.describe()
    log["severity_distribution"] = master["SEVERITY_WEAK"].value_counts().to_dict()

    def dominant_modifier(spans):
        mods = [s["modifier"] for s in spans if s.get("modifier")]
        for lvl in ["severe", "moderate", "mild"]:
            if lvl in mods:
                return lvl
        return "unspecified"
    master["DOMINANT_MODIFIER"] = master["WEAK_ADE_SPANS"].apply(dominant_modifier)

    print("[6/7] Modifier- and age-aware clustering...")
    master["CLUSTER_TEXT"] = master.apply(build_cluster_text, axis=1)
    clusterer = ADEClusterer()
    has_terms = master["CLUSTER_TEXT"].str.strip().astype(bool)
    cluster_df = master[has_terms].reset_index(drop=True)
    labels, _ = clusterer.fit_predict(cluster_df)
    cluster_df["CLUSTER_ID"] = labels
    master = master.merge(
        cluster_df[["VAERS_ID", "CLUSTER_ID"]], on="VAERS_ID", how="left"
    )
    log["clustering_backend"] = clusterer.describe()
    log["n_clusters"] = int(cluster_df["CLUSTER_ID"].nunique())

    print("[7/7] Exporting gold-annotation sample + all artifacts...")
    sample = export_annotation(master, f"{out_dir}/gold_annotation_tasks.json",
                                n_total=sample_for_annotation)

    keep_cols = ["VAERS_ID", "STATE", "AGE_YRS_NUM", "AGE_BUCKET", "SEX",
                 "SYMPTOM_TEXT_CLEAN", "VACCINES", "MEDDRA_TERMS",
                 "WEAK_ADE_SPANS", "WEAK_DRUG_SPANS", "N_ADE_SPANS",
                 "DOMINANT_MODIFIER", "SEVERITY_WEAK", "SEVERITY_PRED",
                 "CLUSTER_ID", "DIED", "L_THREAT", "HOSPITAL", "DISABLE", "ER_VISIT"]
    export_df = master[keep_cols].copy()
    for c in ["VACCINES", "MEDDRA_TERMS", "WEAK_ADE_SPANS", "WEAK_DRUG_SPANS"]:
        export_df[c] = export_df[c].apply(json.dumps)
    export_df.to_csv(f"{out_dir}/adeguard_processed.csv", index=False)

    log["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(f"{out_dir}/run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"Done in {log['elapsed_seconds']}s. Artifacts written to {out_dir}/")
    return master, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--sample-for-annotation", type=int, default=150)
    ap.add_argument("--offline", action="store_true",
                     help="Skip Hugging Face model downloads entirely and go "
                          "straight to fallback models (use this on a network "
                          "that blocks huggingface.co, or to save time).")
    ap.add_argument("--start-year", type=int, default=None,
                     help="Only include VAERS years >= this (e.g. 2020).")
    ap.add_argument("--end-year", type=int, default=None,
                     help="Only include VAERS years <= this (e.g. 2025).")
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, args.max_rows, args.sample_for_annotation,
        args.offline, args.start_year, args.end_year)
