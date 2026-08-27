"""
ADEGuard - Data Pipeline
Loads, merges and cleans VAERS CSV files into a single report-level
dataframe keyed by VAERS_ID. Auto-discovers every year present in
data_dir (e.g. 1990VAERSDATA.csv ... 2025VAERSDATA.csv) rather than
assuming a single year, since VAERS ships one DATA/VAX/SYMPTOMS triplet
per calendar year.
"""
import glob
import os
import re
import pandas as pd
import numpy as np

VAERS_ENCODING = "latin-1"  # VAERS exports are not strictly UTF-8


def _read_csv(path, **kwargs):
    return pd.read_csv(path, dtype=str, encoding=VAERS_ENCODING, on_bad_lines="skip", **kwargs)


def _discover_year_files(data_dir, suffix):
    """Find every <year><suffix>.csv in data_dir, e.g. suffix='VAERSDATA.csv'
    matches 1990VAERSDATA.csv, 2025VAERSDATA.csv, etc. Returns list of
    (year, path) sorted by year."""
    pattern = os.path.join(data_dir, f"*{suffix}")
    found = []
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        m = re.match(r"^(\d{4})", fname)
        year = int(m.group(1)) if m else None
        found.append((year, path))
    found.sort(key=lambda x: (x[0] is None, x[0]))
    return found


def _load_all_years(data_dir, suffix, start_year=None, end_year=None):
    files = _discover_year_files(data_dir, suffix)
    if not files:
        raise FileNotFoundError(
            f"No files matching *{suffix} found in '{data_dir}'. "
            f"Expected e.g. 2024{suffix}, 2025{suffix}..."
        )
    frames = []
    for year, path in files:
        if start_year and year and year < start_year:
            continue
        if end_year and year and year > end_year:
            continue
        print(f"    loading {os.path.basename(path)}...")
        df = _read_csv(path)
        df["_SOURCE_YEAR"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_raw(data_dir, start_year=None, end_year=None):
    data = _load_all_years(data_dir, "VAERSDATA.csv", start_year, end_year)
    vax = _load_all_years(data_dir, "VAERSVAX.csv", start_year, end_year)
    symptoms = _load_all_years(data_dir, "VAERSSYMPTOMS.csv", start_year, end_year)
    return data, vax, symptoms


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collapse_symptoms(symptoms_df):
    """VAERSSYMPTOMS has multiple rows per VAERS_ID (5 terms per row).
    Collapse into one row per ID with a deduped list of MedDRA PT terms."""
    cols = [c for c in symptoms_df.columns if c.startswith("SYMPTOM") and "VERSION" not in c]
    records = {}
    for _, row in symptoms_df.iterrows():
        vid = row["VAERS_ID"]
        terms = records.setdefault(vid, set())
        for c in cols:
            val = row.get(c)
            if isinstance(val, str) and val.strip():
                terms.add(val.strip())
    out = pd.DataFrame(
        {"VAERS_ID": list(records.keys()),
         "MEDDRA_TERMS": [sorted(v) for v in records.values()]}
    )
    return out


def collapse_vax(vax_df):
    """Collapse VAERSVAX to one row per VAERS_ID with list of (VAX_TYPE, VAX_NAME, VAX_MANU)."""
    records = {}
    for _, row in vax_df.iterrows():
        vid = row["VAERS_ID"]
        entry = (row.get("VAX_TYPE") or "", row.get("VAX_NAME") or "", row.get("VAX_MANU") or "")
        records.setdefault(vid, set()).add(entry)
    out = pd.DataFrame(
        {"VAERS_ID": list(records.keys()),
         "VACCINES": [list(v) for v in records.values()]}
    )
    return out


def build_master(data_dir, max_rows=None, start_year=None, end_year=None, random_sample=True, seed=42):
    """max_rows now samples ACROSS the merged multi-year dataset (randomly,
    for a representative cut) rather than just taking the first N rows,
    which would otherwise always be the earliest year(s) only."""
    data, vax, symptoms = load_raw(data_dir, start_year, end_year)

    data["SYMPTOM_TEXT_CLEAN"] = data["SYMPTOM_TEXT"].apply(clean_text)
    data["AGE_YRS_NUM"] = pd.to_numeric(data["AGE_YRS"], errors="coerce")

    vax_c = collapse_vax(vax)
    symp_c = collapse_symptoms(symptoms)

    master = data.merge(vax_c, on="VAERS_ID", how="left")
    master = master.merge(symp_c, on="VAERS_ID", how="left")

    if max_rows and len(master) > max_rows:
        if random_sample:
            master = master.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        else:
            master = master.head(max_rows)

    master["VACCINES"] = master["VACCINES"].apply(lambda x: x if isinstance(x, list) else [])
    master["MEDDRA_TERMS"] = master["MEDDRA_TERMS"].apply(lambda x: x if isinstance(x, list) else [])

    # Weak "ground-truth-ish" seriousness flags straight from structured VAERS fields
    for col in ["DIED", "L_THREAT", "HOSPITAL", "DISABLE", "ER_VISIT", "X_STAY", "BIRTH_DEFECT"]:
        if col not in master.columns:
            master[col] = ""
        master[col] = master[col].fillna("").apply(lambda v: v.strip().upper() == "Y")

    return master


def age_bucket(age):
    if pd.isna(age):
        return "unknown"
    if age < 2:
        return "infant (<2)"
    if age < 12:
        return "child (2-11)"
    if age < 18:
        return "adolescent (12-17)"
    if age < 45:
        return "adult (18-44)"
    if age < 65:
        return "middle-age (45-64)"
    return "older-adult (65+)"
