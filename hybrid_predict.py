import pickle
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from rules import rule_based_score
from scipy.sparse import hstack


# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def hybrid_prediction(job_text):
    cleaned = clean_text(job_text)

    text_vec = vectorizer.transform([cleaned])
    flags = [[0, 0, 0]]

    X_input = hstack([text_vec, flags])

    ml_prob = model.predict_proba(X_input)[0][1]

    row = {
        "has_company_logo": 0,
        "has_questions": 0,
        "combined_text": cleaned
    }

    rule_score = rule_based_score(row)

    # 🔥 Explainability (Reasons)
    reasons = []

    if "fee" in cleaned or "payment" in cleaned:
        reasons.append("Contains payment request")

    if "whatsapp" in cleaned or "telegram" in cleaned:
        reasons.append("Uses informal contact")

    if "guaranteed" in cleaned:
        reasons.append("Unrealistic promises")

    if "work from home" in cleaned:
        reasons.append("Work-from-home keyword")

    # 🔥 Decision
    if ml_prob > 0.7 or rule_score > 50:
        label = "🚨 High Risk Scam"
    elif ml_prob > 0.4 or rule_score > 30:
        label = "⚠️ Suspicious Job"
    else:
        label = "✅ Safe Job"

    return label, ml_prob, reasons