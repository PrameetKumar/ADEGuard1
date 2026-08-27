"""
ADEGuard - Gold Annotation Export
Selects a representative, stratified sample of narratives (across age
groups, vaccine types, and seriousness flags) and exports them in
Label Studio's pre-annotated task format, seeded with the weak ADE/DRUG
spans. A human annotator then only has to correct rather than label from
scratch - this is the standard weak-supervision -> gold bootstrap loop.
"""
import json
import random


LABEL_CONFIG = """
<View>
  <Labels name="label" toName="text">
    <Label value="ADE" background="#ff6666"/>
    <Label value="DRUG" background="#6699ff"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
"""


def stratified_sample(df, n_total=150, seed=42):
    rng = random.Random(seed)
    strata = df.copy()
    strata["_stratum"] = (
        strata["AGE_BUCKET"].astype(str) + "|" +
        strata["DIED"].astype(str) + "|" +
        strata["HOSPITAL"].astype(str)
    )
    groups = strata.groupby("_stratum")
    per_group = max(1, n_total // max(1, groups.ngroups))
    picks = []
    for _, g in groups:
        idx = list(g.index)
        rng.shuffle(idx)
        picks.extend(idx[:per_group])
    if len(picks) > n_total:
        rng.shuffle(picks)
        picks = picks[:n_total]
    return df.loc[picks]


def to_label_studio_tasks(sample_df):
    tasks = []
    for _, row in sample_df.iterrows():
        text = row["SYMPTOM_TEXT_CLEAN"]
        results = []
        for span in row.get("WEAK_ADE_SPANS", []) + row.get("WEAK_DRUG_SPANS", []):
            results.append({
                "from_name": "label", "to_name": "text", "type": "labels",
                "value": {"start": span["start"], "end": span["end"],
                          "text": span["text"], "labels": [span["label"]]},
            })
        tasks.append({
            "data": {"text": text, "vaers_id": row["VAERS_ID"],
                     "age_bucket": row.get("AGE_BUCKET"), "died": bool(row.get("DIED"))},
            "predictions": [{"model_version": "weak-v1", "result": results}],
        })
    return tasks


def export(df, out_path, n_total=150):
    sample = stratified_sample(df, n_total=n_total)
    tasks = to_label_studio_tasks(sample)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2, default=str)
    with open(out_path.replace(".json", "_label_config.xml"), "w") as f:
        f.write(LABEL_CONFIG)
    return sample
