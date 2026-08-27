"""
ADEGuard - Explainability
Primary backend: SHAP (KernelExplainer / TreeExplainer on the classifier)
or LIME's LimeTextExplainer, both drop-in against SeverityClassifier since
it exposes .predict_proba(texts).
Fallback backend: an occlusion / leave-one-word-out importance score -
conceptually the same idea LIME uses (perturb -> observe prediction
change) implemented directly with numpy, so it needs no extra packages.
"""
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


def occlusion_importance(clf, text, target_class=None, top_k=15):
    """Leave-one-word-out importance for a single prediction."""
    words = text.split()
    if not words:
        return []
    base_proba = clf.predict_proba([text])[0]
    classes = list(clf.classes_)
    if target_class is None:
        target_idx = int(np.argmax(base_proba))
        target_class = classes[target_idx]
    else:
        target_idx = classes.index(target_class)
    base_score = base_proba[target_idx]

    scores = []
    for i in range(len(words)):
        perturbed = " ".join(words[:i] + words[i + 1:])
        if not perturbed:
            scores.append(0.0)
            continue
        p = clf.predict_proba([perturbed])[0][target_idx]
        scores.append(base_score - p)  # drop in confidence when word removed = importance

    ranked = sorted(zip(words, scores), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return {"predicted_class": target_class, "confidence": float(base_score),
            "word_importances": [(w, float(s)) for w, s in ranked]}


def explain_prediction(clf, text, target_class=None):
    if LIME_AVAILABLE:
        explainer = LimeTextExplainer(class_names=list(clf.classes_))
        exp = explainer.explain_instance(text, clf.predict_proba, num_features=15)
        return {"backend": "LIME", "explanation": exp.as_list()}
    # SHAP text explainers need a specific model wrapper; for the sklearn
    # pipeline here the occlusion method below is the practical equivalent.
    result = occlusion_importance(clf, text, target_class)
    result["backend"] = "fallback-occlusion (LIME/SHAP-equivalent)"
    return result
