"""
ADEGuard - Gazetteers
Builds ADE (MedDRA Preferred Term) and DRUG/vaccine gazetteers directly
from the VAERS structured files. These are the anchor vocabulary for
weak supervision and for the fallback (non-BioBERT) NER tagger.
"""
import re
from collections import Counter

SEVERITY_MODIFIERS = {
    "severe": ["severe", "critical", "life-threatening", "life threatening", "acute severe",
               "profound", "extreme", "debilitating", "anaphylaxis", "anaphylactic"],
    "moderate": ["moderate", "significant", "persistent", "worsening", "considerable"],
    "mild": ["mild", "slight", "minor", "minimal", "low-grade", "transient", "brief"],
}

NEGATION_CUES = ["no ", "not ", "denies", "denied", "without", "negative for", "ruled out"]


def build_ade_gazetteer(meddra_terms_series):
    """Collect the full vocabulary of MedDRA Preferred Terms seen in the dataset.
    In production this is exactly the MedDRA PT dictionary (17,000+ terms);
    here we derive the observed subset directly from VAERSSYMPTOMS.CSV, which
    is itself MedDRA-coded -> a legitimate, data-grounded gazetteer."""
    terms = set()
    for lst in meddra_terms_series:
        for t in lst:
            terms.add(t)
    # sort longest-first so multi-word terms match before their substrings
    return sorted(terms, key=len, reverse=True)


def build_drug_gazetteer(vaccines_series):
    """Collect vaccine/drug name surface forms (VAX_NAME, VAX_TYPE) from VAERSVAX.CSV."""
    names = set()
    for lst in vaccines_series:
        for vtype, vname, vmanu in lst:
            for cand in (vtype, vname):
                if cand and cand.upper() not in ("UNKNOWN", "UNK", ""):
                    names.add(cand.strip())
                    # also add the bracket-free short form, e.g. "COVID19 (COVID19 (PFIZER-BIONTECH))" -> "COVID19"
                    short = re.split(r"[\(\[]", cand)[0].strip()
                    if short:
                        names.add(short)
    return sorted(names, key=len, reverse=True)


def top_ade_frequency(meddra_terms_series, n=30):
    c = Counter()
    for lst in meddra_terms_series:
        c.update(lst)
    return c.most_common(n)
