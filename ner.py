"""
ADEGuard - NER Module
Primary path: BioBERT / a biomedical NER pipeline (via HuggingFace
transformers) fine-tuned or zero-shot for ADE + DRUG extraction.
Fallback path (used automatically when transformers/torch are not
installed, e.g. this offline sandbox, OR when the model weights can't be
downloaded - e.g. a corporate firewall silently blocks huggingface.co):
the gazetteer + negation + modifier tagger from weak_supervision.py,
which was itself built by distant supervision from VAERS' own MedDRA
coding - i.e. the same labels a human annotator would use to build the
BioBERT training set.
"""
import os
from .weak_supervision import label_report

# Fail fast instead of hanging indefinitely if the network is blocked/slow.
# (huggingface_hub reads this env var to cap connect/read time per request.)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")

try:
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    BIOBERT_AVAILABLE = True
except ImportError:
    BIOBERT_AVAILABLE = False


class ADENERTagger:
    """Unified interface: .extract(text) -> (ade_spans, drug_spans)
    regardless of which backend is active."""

    def __init__(self, ade_gazetteer, drug_gazetteer,
                 model_name="d4data/biomedical-ner-all"):
        self.ade_gazetteer = ade_gazetteer
        self.drug_gazetteer = drug_gazetteer
        self.backend = "fallback-gazetteer"
        self.pipe = None

        if BIOBERT_AVAILABLE:
            print(f"    Attempting to load/download BioBERT model '{model_name}' "
                  f"(timeout {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s/request; "
                  f"first run downloads ~400MB, then it's cached locally)...")
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.pipe = pipeline("ner", model=model, tokenizer=tok,
                                      aggregation_strategy="simple")
                self.backend = f"biobert:{model_name}"
                print(f"    Loaded {self.backend}.")
            except Exception as e:
                # e.g. no internet / blocked proxy / timeout even though
                # transformers is installed - fall back instead of hanging.
                print(f"    Could not load BioBERT ({type(e).__name__}: {e}). "
                      f"Falling back to gazetteer tagger.")
                self.pipe = None
                self.backend = "fallback-gazetteer"

    def extract(self, text):
        if self.pipe is not None:
            ents = self.pipe(text)
            ade_spans, drug_spans = [], []
            for e in ents:
                grp = e.get("entity_group", "").upper()
                item = {"start": e["start"], "end": e["end"], "text": e["word"],
                        "score": float(e["score"])}
                if any(k in grp for k in ("SIGN", "SYMPTOM", "DISEASE", "ADE")):
                    item["label"] = "ADE"
                    ade_spans.append(item)
                elif any(k in grp for k in ("DRUG", "MED", "CHEM")):
                    item["label"] = "DRUG"
                    drug_spans.append(item)
            return ade_spans, drug_spans

        # fallback
        return label_report(text, self.ade_gazetteer, self.drug_gazetteer)

    def describe(self):
        return self.backend
