"""
ADEGuard - Modifier-Aware & Age-Specific Clustering
Primary backend: Sentence-BERT embeddings + HDBSCAN.
Fallback backend (this sandbox): TF-IDF + TruncatedSVD embeddings + sklearn
OPTICS (a density-based clustering algorithm available in scikit-learn,
functionally the closest in-box analogue to HDBSCAN - both find
variable-density clusters and label sparse points as noise, unlike KMeans).
Age group and severity modifier are folded in as weighted categorical
features so that, e.g., "mild fatigue in a child" and "severe fatigue in
an older adult" land in different clusters even when the base symptom text
is similar.
"""
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.cluster import OPTICS

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import hdbscan as _hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def build_cluster_text(row):
    """One representative string per report: its ADE canonical terms."""
    terms = [s["canonical"] for s in row.get("WEAK_ADE_SPANS", [])]
    return " ".join(terms) if terms else row.get("SYMPTOM_TEXT_CLEAN", "")[:300]


class ADEClusterer:
    def __init__(self, n_components=64, age_weight=0.6, modifier_weight=0.8):
        self.backend_embed = "fallback-tfidf-svd"
        self.backend_cluster = "fallback-OPTICS"
        self.n_components = n_components
        self.age_weight = age_weight
        self.modifier_weight = modifier_weight
        self.vectorizer = None
        self.svd = None
        self.ohe = None
        self.sbert_model = None

        if SBERT_AVAILABLE:
            print("    Attempting to load/download Sentence-BERT model "
                  f"'all-MiniLM-L6-v2' (timeout {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s/request)...")
            try:
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend_embed = "sentence-bert:all-MiniLM-L6-v2"
                print(f"    Loaded {self.backend_embed}.")
            except Exception as e:
                print(f"    Could not load Sentence-BERT ({type(e).__name__}: {e}). "
                      f"Falling back to TF-IDF+SVD embeddings.")
                self.sbert_model = None
        if HDBSCAN_AVAILABLE:
            self.backend_cluster = "HDBSCAN"

    def _text_embed(self, texts):
        if self.sbert_model is not None:
            return self.sbert_model.encode(texts, show_progress_bar=False)
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
            X = self.vectorizer.fit_transform(texts)
            n_comp = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
            self.svd = TruncatedSVD(n_components=max(2, n_comp), random_state=42)
            emb = self.svd.fit_transform(X)
        else:
            X = self.vectorizer.transform(texts)
            emb = self.svd.transform(X)
        return normalize(emb)

    def fit_predict(self, df, text_col="CLUSTER_TEXT", age_col="AGE_BUCKET", modifier_col="DOMINANT_MODIFIER"):
        text_emb = self._text_embed(df[text_col].fillna("").tolist())

        cat = df[[age_col, modifier_col]].fillna("unknown")
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        cat_emb = self.ohe.fit_transform(cat).toarray()

        age_cols = sum(1 for c in self.ohe.categories_[0])
        weights = np.ones(cat_emb.shape[1])
        weights[:age_cols] *= self.age_weight
        weights[age_cols:] *= self.modifier_weight
        cat_emb = cat_emb * weights

        full_emb = np.hstack([text_emb, cat_emb])

        if HDBSCAN_AVAILABLE:
            clusterer = _hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3)
            labels = clusterer.fit_predict(full_emb)
        else:
            # xi/min_samples tuned for VAERS-scale corpora (thousands of short
            # narratives). Increase min_samples for larger corpora to keep
            # cluster count interpretable; decrease xi to recover more, smaller
            # clusters if too many points fall into the -1 ("noise") bucket.
            clusterer = OPTICS(min_samples=5, xi=0.015, min_cluster_size=5)
            labels = clusterer.fit_predict(full_emb)

        return labels, full_emb

    def describe(self):
        return {"embedding_backend": self.backend_embed, "cluster_backend": self.backend_cluster}
