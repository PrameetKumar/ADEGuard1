"""
ADEGuard - Weak Supervision
Generates weak (noisy, programmatic) ADE / DRUG span labels over free-text
SYMPTOM_TEXT by matching against the data-grounded gazetteers, with simple
negation handling. This is the labeling-function layer that would normally
feed a Snorkel-style label model; here we use frequency-weighted majority
voting as a lightweight equivalent, then export a stratified sample for
human gold annotation (Label Studio).
"""
import re
from .gazetteers import NEGATION_CUES, SEVERITY_MODIFIERS

_COMPILED_CACHE = {}


def _compiled_pattern(terms):
    """Build ONE alternation regex for the whole gazetteer, longest-term-
    first so the engine prefers the longest match starting at each
    position. Cached per gazetteer identity so it's compiled once, not
    once per document (this is what makes matching 8k+ documents against
    thousands of MedDRA terms tractable without a trie/Aho-Corasick lib)."""
    key = id(terms)
    if key in _COMPILED_CACHE:
        return _COMPILED_CACHE[key]
    usable = sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True)
    alternation = "|".join(re.escape(t.lower()) for t in usable)
    pattern = re.compile(alternation) if alternation else re.compile(r"(?!x)x")
    _COMPILED_CACHE[key] = pattern
    return pattern


def _find_spans(text, terms):
    """Case-insensitive longest-match-first span finder using one compiled
    alternation regex per gazetteer (see _compiled_pattern)."""
    pattern = _compiled_pattern(terms)
    lower = text.lower()
    spans = []
    occupied = [False] * len(text)
    for m in pattern.finditer(lower):
        s, e = m.start(), m.end()
        if any(occupied[s:e]):
            continue
        spans.append((s, e, text[s:e], m.group(0)))
        for i in range(s, e):
            occupied[i] = True
    spans.sort(key=lambda x: x[0])
    return spans


def _is_negated(text, start, window=40):
    prefix = text[max(0, start - window):start].lower()
    return any(cue in prefix for cue in NEGATION_CUES)


def _nearby_modifier(text, start, end, window=25):
    ctx = text[max(0, start - window):min(len(text), end + window)].lower()
    for level, kws in SEVERITY_MODIFIERS.items():
        for kw in kws:
            if kw in ctx:
                return level
    return None


def label_report(text, ade_gazetteer, drug_gazetteer):
    """Return weak ADE and DRUG spans for one SYMPTOM_TEXT string."""
    ade_spans_raw = _find_spans(text, ade_gazetteer)
    drug_spans_raw = _find_spans(text, drug_gazetteer)

    ade_spans = []
    for s, e, surface, term in ade_spans_raw:
        ade_spans.append({
            "start": s, "end": e, "text": surface, "label": "ADE",
            "canonical": term,
            "negated": _is_negated(text, s),
            "modifier": _nearby_modifier(text, s, e),
        })
    drug_spans = []
    for s, e, surface, term in drug_spans_raw:
        drug_spans.append({
            "start": s, "end": e, "text": surface, "label": "DRUG",
            "canonical": term,
        })
    return ade_spans, drug_spans


def weak_label_dataframe(df, ade_gazetteer, drug_gazetteer, text_col="SYMPTOM_TEXT_CLEAN"):
    ade_out, drug_out = [], []
    for text in df[text_col]:
        a, d = label_report(text, ade_gazetteer, drug_gazetteer)
        ade_out.append(a)
        drug_out.append(d)
    df = df.copy()
    df["WEAK_ADE_SPANS"] = ade_out
    df["WEAK_DRUG_SPANS"] = drug_out
    df["N_ADE_SPANS"] = df["WEAK_ADE_SPANS"].apply(len)
    return df
