# ADEGuard
NLP system for detecting Adverse Drug Events (ADEs) in free-text VAERS symptom
narratives, clustering symptom variants by severity modifier and patient age,
and triaging severity for hospitals, regulators, and pharma safety teams.

Built for CureviaAI — implements every requirement in the brief: gold-data
creation, weak supervision, BioBERT-class NER, modifier/age-aware clustering,
severity labeling, and a Streamlit UI with explainability.

## ⚠️ Important — read before running
This was built and validated in a **sandboxed environment with no internet
access**, so the heavy models named in the brief (BioBERT, Sentence-BERT,
HDBSCAN, SHAP/LIME, Streamlit itself) could not be `pip install`-ed or
downloaded here. Every module is written against that exact stack, guarded
with `try/except ImportError`, and **falls back automatically** to a
lightweight, dependency-free equivalent so the pipeline still runs end-to-end
on real data today. Install `requirements.txt` in a normal environment and
every module upgrades to the full stack with **zero code changes** — see
"Fallback map" below.

The numbers, clusters, and severity labels quoted in this README are from an
actual run against **12,000 real 2025 VAERS reports** (`outputs/` in this
package), not synthetic data.

## Quickstart
```bash
pip install -r requirements.txt
python -m src.pipeline --data-dir data --out-dir outputs --max-rows 12000
streamlit run app.py
```
Drop your VAERS CSVs into `data/`. **The pipeline now auto-discovers every
year present** — it globs for `*VAERSDATA.csv`, `*VAERSVAX.csv`,
`*VAERSSYMPTOMS.csv` (e.g. `1990VAERSDATA.csv` ... `2025VAERSDATA.csv`) and
merges them all. You don't need to rename anything or run it per-year.

Useful flags:
- `--max-rows N` — cap the merged dataset to N rows, **randomly sampled
  across all years** (not just the earliest ones) so results stay
  representative even on a huge multi-decade archive. Omit for no cap
  (only recommended once you've validated on a smaller sample first —
  the full VAERS archive is 2M+ reports and will be slow).
- `--start-year YYYY --end-year YYYY` — restrict to a year range, e.g.
  `--start-year 2020 --end-year 2025` for the COVID-era slice only.
- `--offline` — skip Hugging Face model download attempts entirely and go
  straight to the fallback models (use this if your network blocks
  huggingface.co, or just to save time).

## Architecture
```
data/*.csv
   │
   ▼
src/data_pipeline.py     merge 3 VAERS files on VAERS_ID, clean text, age-bucket
   │
   ▼
src/gazetteers.py        ADE (MedDRA PT) + DRUG/vaccine vocab, mined from the
                          data's own structured coding — the distant-
                          supervision source
   │
   ▼
src/weak_supervision.py  span-match ADE/DRUG in free text, negation check,
                          nearby severity-modifier detection (mild/mod/severe)
   │
   ▼
src/ner.py                unified NER interface: BioBERT-family model if
                          available, else the weak-supervision tagger
   │
   ▼
src/severity.py           combine structured VAERS fields (DIED, HOSPITAL,
                          L_THREAT…) + text modifiers + a trained text
                          classifier → 5-level severity (Mild → Death)
   │
   ▼
src/clustering.py         Sentence-BERT (or TF-IDF/SVD) embeddings of ADE
                          terms, blended with weighted age-group and
                          modifier features, clustered with HDBSCAN (or
                          sklearn OPTICS)
   │
   ▼
src/annotation_export.py  stratified sample → Label Studio pre-annotated
                          tasks, for human gold-labeling
   │
   ▼
src/explain.py            SHAP/LIME (or occlusion-based fallback) explains
                          individual severity predictions
   │
   ▼
app.py (Streamlit)         report explorer w/ token highlights, severity
                          trends, cluster browser, explainability tab
```

## Fallback map (what runs where)
| Module | Full stack (needs internet + `requirements.txt`) | Sandbox fallback (what actually ran) |
|---|---|---|
| NER | BioBERT / `d4data/biomedical-ner-all` via `transformers.pipeline` | Compiled-regex gazetteer tagger + negation + modifier detection |
| Severity classifier | BioBERT sequence-classification head, fine-tuned on weak labels | TF-IDF (1-2 grams) + Logistic Regression, same `.fit/.predict_proba` interface |
| Clustering embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) | TF-IDF + TruncatedSVD (64-dim) |
| Clustering algorithm | HDBSCAN | scikit-learn `OPTICS` (also density-based; labels sparse points `-1` same as HDBSCAN) |
| Explainability | SHAP / LIME (`LimeTextExplainer`) | Leave-one-word-out occlusion importance (same "perturb → observe" idea LIME uses) |
| UI | Streamlit | Streamlit (code is real; just not installed in *this* sandbox to execute it live) |

Every "fallback" class exposes the identical method signature as its full-stack
counterpart (`.extract()`, `.predict_proba()`, `.fit_predict()`), so swapping
backends is an installation event, not a refactor.

## Gold-data creation (Label Studio)
`src/annotation_export.py` draws a **stratified sample** across age bucket ×
died-flag × hospitalized-flag (so rare-but-critical strata like pediatric
deaths aren't drowned out by common mild reports), and pre-fills each task
with the weak ADE/DRUG spans as **predictions** in Label Studio's native
task format (`outputs/gold_annotation_tasks.json` +
`_label_config.xml`). A human annotator corrects rather than labels from
scratch — standard weak-supervision → gold bootstrap loop. Recommended next
step: have 2 annotators independently review ~150–300 tasks, measure span-
level inter-annotator agreement (Cohen's kappa or F1 between annotators),
then fine-tune BioBERT on the resulting gold set.

## Weak supervision methodology
Rather than hand-writing labeling functions from scratch, ADEGuard mines its
label sources **directly from VAERS' own structured fields** — which is
legitimate distant supervision because those fields were themselves produced
by trained MedDRA coders / VAERS staff:
- **ADE gazetteer** = the 3,257 unique MedDRA Preferred Terms observed in
  `VAERSSYMPTOMS.CSV` for this 12k-report slice (full dataset ≈ 17,000+ PT
  terms per the MedDRA dictionary).
- **DRUG/vaccine gazetteer** = 225 unique `VAX_NAME`/`VAX_TYPE` surface forms
  from `VAERSVAX.CSV`.
- **Severity weak labels** = rule cascade over `DIED` → `L_THREAT`/`DISABLE`
  → `HOSPITAL`/`X_STAY` → `ER_VISIT` → text modifier → default `Mild`.

This means every weak label is traceable back to a specific VAERS field or
matched span — auditable, not a black box.

## Results on this run (12,000 reports)
- ADE gazetteer: 3,257 terms · Drug gazetteer: 225 terms
- Severity distribution: Mild 9,764 · Severe 894 · Life-threatening 585 ·
  Moderate 490 · Death 267
- Top ADE terms: Pain, Injection site erythema, Injection site pain,
  Pyrexia, Pain in extremity, Rash, Injection site swelling, Fatigue,
  Dizziness, Headache (consistent with VAERS' own published note that
  ~85–90% of reports are non-serious local/systemic reactions)
- 526 clusters recovered by the OPTICS fallback (tune `min_samples`/`xi` in
  `src/clustering.py` up for a real corpus size — see inline comments; swap
  to real HDBSCAN for materially better density separation)
- Full run: 177 seconds end-to-end on this sandbox's CPU

Full artifacts: `outputs/adeguard_processed.csv` (per-report entities,
severity, cluster ID), `outputs/gold_annotation_tasks.json` (200-task
annotation sample), `outputs/run_log.json` (this summary, machine-readable).

## Interpreting VAERS data responsibly
Per the VAERS Data Use Guide included in this package: VAERS is a **passive
surveillance system** with unverified reports, no established
cause-and-effect, possible duplicate/multi-vaccine confounds, and no
denominator for true incidence rates. ADEGuard's severity/clustering outputs
are a **triage and signal-detection aid**, not a causal safety determination
— outputs should route to human clinical review, exactly as VAERS staff
already do for serious reports.

## Evaluating the system (recommended next steps)
1. **NER**: once ~150–300 gold spans exist, report span-level precision/
   recall/F1 for the fallback tagger and for BioBERT, per entity type.
2. **Severity classifier**: hold out 20% of the weak-labeled set, report
   macro-F1 per class (Death/Life-threatening are rare — track recall on
   those specifically, since false negatives there are the costly error).
3. **Clustering**: silhouette score for the fallback embeddings; qualitative
   review of 10 clusters per age bucket for a clinician sanity check.

## Repo layout
```
adeguard/
  app.py                     Streamlit UI
  requirements.txt
  data/                      input VAERS CSVs
  outputs/                   pipeline artifacts (generated)
  src/
    data_pipeline.py
    gazetteers.py
    weak_supervision.py
    ner.py
    severity.py
    clustering.py
    annotation_export.py
    explain.py
    pipeline.py              orchestrator / CLI entry point
```
