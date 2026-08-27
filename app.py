"""
ADEGuard - Streamlit UI
Run with:  streamlit run app.py
Expects outputs/adeguard_processed.csv (produced by `python -m src.pipeline`)
in the same directory, or lets you point at another path in the sidebar.
"""
import json
import ast
import html
import numpy as np
import pandas as pd
import streamlit as st

from src.severity import SeverityClassifier, SEVERITY_ORDER
from src.explain import explain_prediction

st.set_page_config(page_title="ADEGuard", layout="wide", page_icon="\U0001F6E1\ufe0f")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in ["VACCINES", "MEDDRA_TERMS", "WEAK_ADE_SPANS", "WEAK_DRUG_SPANS"]:
        df[c] = df[c].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else [])
    return df


@st.cache_resource
def get_severity_classifier(df):
    clf = SeverityClassifier()
    clf.fit(df["SYMPTOM_TEXT_CLEAN"].fillna(""), df["SEVERITY_WEAK"])
    return clf


def render_highlighted(text, ade_spans, drug_spans):
    """Token/span-level highlight rendering (HTML) for a single narrative."""
    spans = []
    for s in ade_spans:
        spans.append((s["start"], s["end"], "ADE", s.get("modifier")))
    for s in drug_spans:
        spans.append((s["start"], s["end"], "DRUG", None))
    spans.sort(key=lambda x: x[0])

    out = []
    cursor = 0
    color = {"ADE": "#ffd1d1", "DRUG": "#cfe2ff"}
    for s, e, label, modifier in spans:
        if s < cursor:
            continue
        out.append(html.escape(text[cursor:s]))
        tag = label if not modifier else f"{label} \u00b7 {modifier}"
        out.append(
            f'<span style="background-color:{color[label]}; padding:1px 3px; '
            f'border-radius:3px; border:1px solid #999;" title="{tag}">'
            f'{html.escape(text[s:e])}<sub style="font-size:0.6em;">[{tag}]</sub></span>'
        )
        cursor = e
    out.append(html.escape(text[cursor:]))
    return "".join(out)


st.title("\U0001F6E1\ufe0f ADEGuard")
st.caption("NLP-based Adverse Drug Event detection, clustering, and severity triage over VAERS data")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("Processed CSV path", "outputs/adeguard_processed.csv")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"Could not find {data_path}. Run `python -m src.pipeline --data-dir data` first.")
        st.stop()

    st.metric("Reports loaded", len(df))
    age_filter = st.multiselect("Age group", sorted(df["AGE_BUCKET"].dropna().unique()))
    sev_filter = st.multiselect("Severity", [s for s in SEVERITY_ORDER if s in df["SEVERITY_WEAK"].unique()])
    mod_filter = st.multiselect("Modifier", sorted(df["DOMINANT_MODIFIER"].dropna().unique()))

filtered = df.copy()
if age_filter:
    filtered = filtered[filtered["AGE_BUCKET"].isin(age_filter)]
if sev_filter:
    filtered = filtered[filtered["SEVERITY_WEAK"].isin(sev_filter)]
if mod_filter:
    filtered = filtered[filtered["DOMINANT_MODIFIER"].isin(mod_filter)]

tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001F50D Report Explorer", "\U0001F4CA Severity & Trends",
     "\U0001F9E9 Clustering", "\U0001F9E0 Explainability"]
)

# ---------------- Tab 1: Report Explorer ----------------
with tab1:
    st.subheader("Token-level ADE / DRUG highlighting")
    st.write(f"{len(filtered)} reports match current filters.")
    n_show = st.slider("Reports to display", 1, 25, 5)
    for _, row in filtered.head(n_show).iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**VAERS_ID** {row['VAERS_ID']}")
            c2.markdown(f"**Age group** {row['AGE_BUCKET']}")
            c3.markdown(f"**Severity** {row['SEVERITY_WEAK']}")
            c4.markdown(f"**Modifier** {row['DOMINANT_MODIFIER']}")
            html_text = render_highlighted(
                row["SYMPTOM_TEXT_CLEAN"], row["WEAK_ADE_SPANS"], row["WEAK_DRUG_SPANS"]
            )
            st.markdown(html_text, unsafe_allow_html=True)

# ---------------- Tab 2: Severity & Trends ----------------
with tab2:
    st.subheader("Severity distribution")
    sev_counts = filtered["SEVERITY_WEAK"].value_counts().reindex(SEVERITY_ORDER).fillna(0)
    st.bar_chart(sev_counts)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Severity by age group")
        pivot = pd.crosstab(filtered["AGE_BUCKET"], filtered["SEVERITY_WEAK"])
        st.dataframe(pivot)
    with c2:
        st.subheader("Top ADE terms (current filter)")
        term_counts = {}
        for spans in filtered["WEAK_ADE_SPANS"]:
            for s in spans:
                term_counts[s["canonical"]] = term_counts.get(s["canonical"], 0) + 1
        top = pd.Series(term_counts).sort_values(ascending=False).head(15)
        st.bar_chart(top)

# ---------------- Tab 3: Clustering ----------------
with tab3:
    st.subheader("Modifier- & age-aware ADE symptom clusters")
    valid = filtered[filtered["CLUSTER_ID"].notna()]
    st.write(f"{valid['CLUSTER_ID'].nunique()} clusters found "
             f"({(valid['CLUSTER_ID'] == -1).sum()} reports labeled as noise/outliers).")
    summary = (
        valid[valid["CLUSTER_ID"] != -1]
        .groupby("CLUSTER_ID")
        .agg(size=("VAERS_ID", "count"),
             top_age=("AGE_BUCKET", lambda x: x.value_counts().idxmax()),
             top_modifier=("DOMINANT_MODIFIER", lambda x: x.value_counts().idxmax()))
    )
    st.dataframe(summary)

    cluster_pick = st.selectbox("Inspect cluster", sorted(valid[valid["CLUSTER_ID"] != -1]["CLUSTER_ID"].unique()))
    members = valid[valid["CLUSTER_ID"] == cluster_pick]
    term_counts = {}
    for spans in members["WEAK_ADE_SPANS"]:
        for s in spans:
            term_counts[s["canonical"]] = term_counts.get(s["canonical"], 0) + 1
    st.write("Most common ADE terms in this cluster:")
    st.bar_chart(pd.Series(term_counts).sort_values(ascending=False).head(10))

# ---------------- Tab 4: Explainability ----------------
with tab4:
    st.subheader("Why did the severity classifier predict this?")
    clf = get_severity_classifier(df)
    options = filtered["VAERS_ID"].astype(str).tolist()
    pick = st.selectbox("Pick a VAERS_ID", options)
    row = filtered[filtered["VAERS_ID"].astype(str) == pick].iloc[0]
    st.write(f"Predicted: **{row['SEVERITY_PRED']}**  |  Weak label: **{row['SEVERITY_WEAK']}**")
    exp = explain_prediction(clf, row["SYMPTOM_TEXT_CLEAN"])
    st.caption(f"Explainer backend: {exp['backend']}")
    if "word_importances" in exp:
        imp_df = pd.DataFrame(exp["word_importances"], columns=["token", "importance"])
        st.bar_chart(imp_df.set_index("token"))
    elif "explanation" in exp:
        imp_df = pd.DataFrame(exp["explanation"], columns=["token", "importance"])
        st.bar_chart(imp_df.set_index("token"))
