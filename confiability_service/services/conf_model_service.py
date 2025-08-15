import threading
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

from models.schemas import (
    TrainRequest, TrainResponse,
    ClassifyRequest, ContentConfidentiality
)
from repositories.patterns_repo import PatternsRepository
from repositories.config_repo import ConfigRepository

CLASS_LABELS = ["public", "internal", "confidential", "restricted"]

SEED_DEFAULT = 42

def synth_corpus(n_per_class: int, seed: int, vocab_noise: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)

    public_templates = [
        "Press release about product launch and public event schedule.",
        "General guidelines for visitors and FAQ published on website.",
        "Marketing blog post with public information and announcements.",
        "Open dataset description and public API usage examples."
    ]
    internal_templates = [
        "Internal memo about team processes and meeting notes.",
        "Company policy draft for internal distribution only.",
        "Sprint retrospective and internal roadmap items.",
        "Internal knowledge base article and onboarding checklist."
    ]
    confidential_templates = [
        "Contains personal data: email addresses and phone numbers of clients.",
        "Financial report including revenue projections and client contracts.",
        "Customer PII records and account identifiers to be restricted.",
        "HR document with salary bands and employee performance reviews."
    ]
    restricted_templates = [
        "Payment card numbers and authentication secrets for production.",
        "Encryption keys, private certificates, and admin credentials.",
        "M&A documents and undisclosed strategic plans for executives.",
        "Government IDs and bank account IBAN details for high-risk accounts."
    ]

    # add noise tokens to avoid overfitting
    noise_vocab = [f"tok{rng.randint(1000,9999)}" for _ in range(vocab_noise)]

    X, y = [], []
    def expand(templates: List[str], label: str):
        for _ in range(n_per_class):
            base = rng.choice(templates)
            # add small perturbations
            extra = ""
            if label in ("confidential", "restricted") and rng.random() < 0.5:
                extra += " email john.doe@example.com "
            if label == "restricted" and rng.random() < 0.4:
                extra += " credit card 4111 1111 1111 1111 "
            if rng.random() < 0.5:
                extra += " " + " ".join(rng.sample(noise_vocab, k=min(3, len(noise_vocab)))) if noise_vocab else ""
            X.append(base + extra)
            y.append(label)

    expand(public_templates, "public")
    expand(internal_templates, "internal")
    expand(confidential_templates, "confidential")
    expand(restricted_templates, "restricted")
    return X, y

class ConfidentialityModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self.pipeline: Optional[Pipeline] = None
        self.model_version: str = "untrained"
        self.patterns = PatternsRepository()
        self.cfg = ConfigRepository()

    def train(self, req: TrainRequest) -> TrainResponse:
        X, y = synth_corpus(req.n_per_class, req.seed or SEED_DEFAULT, req.vocab_noise)

        # Split
        n = len(X)
        idx = int(0.8 * n)
        X_train, y_train = X[:idx], y[:idx]
        X_test, y_test = X[idx:], y[idx:]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)),
            ("clf", LogisticRegression(max_iter=200, solver="liblinear", multi_class="auto"))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1_macro = float(f1_score(y_test, y_pred, average="macro"))

        with self._lock:
            self.pipeline = pipe
            self.model_version = f"conf-ml-lr-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return TrainResponse(
            model_version=self.model_version,
            metrics={"f1_macro": round(f1_macro, 3)},
            samples=n
        )

    def classify(self, req: ClassifyRequest) -> ContentConfidentiality:
        text = (req.content_pointer.sample_text or "").strip()
        meta = req.content_pointer.metadata or {}

        # Baseline probabilities
        probs: Dict[str, float] = {k: 0.0 for k in CLASS_LABELS}
        label = "internal"
        score = 0.4
        model_version = self.model_version

        with self._lock:
            pipe = self.pipeline

        if pipe:
            try:
                proba_arr = None
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba_arr = pipe.predict_proba([text])[0]
                    for i, cls in enumerate(pipe.named_steps["clf"].classes_):
                        probs[cls] = float(proba_arr[i])
                    label = max(probs, key=probs.get)
                    score = float(max(probs.values()))
                else:
                    # fallback if classifier has no proba
                    label = pipe.predict([text])[0]
                    score = 0.6
                model_version = self.model_version
            except Exception:
                label, score = "internal", 0.4
                model_version = "conf-fallback-ml-error"
        else:
            # Untrained fallback: heuristic on keywords
            kw = text.lower()
            if any(k in kw for k in ["credit card", "iban", "private key", "credential", "ssn"]):
                label, score = "restricted", 0.9
            elif any(k in kw for k in ["salary", "contract", "pii", "personal data", "customer", "financial"]):
                label, score = "confidential", 0.75
            elif any(k in kw for k in ["internal", "memo", "roadmap"]):
                label, score = "internal", 0.55
            else:
                label, score = "public", 0.3
            model_version = "conf-fallback-0.1.0"

        # Pattern scanning (DLP-lite)
        findings = self.patterns.scan(text)
        pattern_tags = self.patterns.tags_from_findings(findings)
        detected = [f["type"] for f in findings]

        # Adjustments based on patterns/metadata
        # Small boosts to score and potential escalation of class
        boost = 0.0
        if any(f["type"] == "credit_card" for f in findings):
            boost += 0.15
            label = escalate(label, "restricted")
        if any(f["type"] in ("email", "national_id") for f in findings):
            boost += 0.08
            label = escalate(label, "confidential")

        # Metadata-informed nudge
        doc_type = (meta.get("doc_type") or "").lower()
        if doc_type in ("hr", "finance", "legal"):
            boost += 0.05
            if doc_type in ("hr", "finance"):
                label = escalate(label, "confidential")

        score = float(np.clip(score + boost, 0.0, 1.0))

        tags = sorted(set(pattern_tags + ([doc_type] if doc_type else [])))
        return ContentConfidentiality(
            classification=label,
            score=round(score, 3),
            tags=tags,
            detected_patterns=detected,
            dlp_findings=findings,
            source_app_context=meta.get("app"),
            user_label=meta.get("user_label"),
            model_version=model_version
        )

def escalate(current: str, target: str) -> str:
    order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    return target if order[target] > order[current] else current