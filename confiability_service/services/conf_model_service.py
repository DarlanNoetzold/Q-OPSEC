from typing import Tuple, Dict, Any, Optional, List
import os
import time
import joblib
import numpy as np
import pandas as pd
import random

from models.schemas import ClassifyRequest, TrainRequest, ContentConfidentiality, TrainResponse
from repositories.config_repo import DATA_DIR, MODELS_DIR, set_best_model, get_best_model_info, read_registry, \
    write_registry, ConfigRepository
from repositories.patterns_repo import PatternsRepository

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

CLASS_LABELS = ["public", "internal", "confidential", "restricted"]


def synth_corpus(n_per_class: int, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)

    templates = {
        "public": [
            "Press release about product launch and public event schedule.",
            "General guidelines for visitors and FAQ published on website.",
            "Marketing blog post with public information and announcements.",
            "Open dataset description and public API usage examples.",
            "Company news and public statements for media distribution.",
            "Public documentation and user manuals available online.",
            "Community forum posts and public support articles.",
            "Open source project documentation and contribution guidelines."
        ],
        "internal": [
            "Internal memo about team processes and meeting notes.",
            "Company policy draft for internal distribution only.",
            "Sprint retrospective and internal roadmap items.",
            "Internal knowledge base article and onboarding checklist.",
            "Team meeting minutes and project status updates.",
            "Internal training materials and process documentation.",
            "Department budget planning and resource allocation.",
            "Internal communication guidelines and best practices."
        ],
        "confidential": [
            "Contains personal data: email addresses and phone numbers of clients.",
            "Financial report including revenue projections and client contracts.",
            "Customer PII records and account identifiers to be restricted.",
            "HR document with salary bands and employee performance reviews.",
            "Client contract negotiations and pricing strategies.",
            "Employee personal information and medical records.",
            "Customer database with contact information and preferences.",
            "Strategic business plans and competitive analysis reports."
        ],
        "restricted": [
            "Payment card numbers and authentication secrets for production.",
            "Encryption keys, private certificates, and admin credentials.",
            "M&A documents and undisclosed strategic plans for executives.",
            "Government IDs and bank account IBAN details for high-risk accounts.",
            "Database passwords and system administrator credentials.",
            "Legal documents under attorney-client privilege.",
            "Trade secrets and proprietary algorithms.",
            "Security incident reports and vulnerability assessments."
        ]
    }

    noise_vocab = [f"token{rng.randint(1000, 9999)}" for _ in range(50)]

    X, y = [], []

    for label, template_list in templates.items():
        for _ in range(n_per_class):
            base = rng.choice(template_list)

            extra = ""
            if label in ("confidential", "restricted") and rng.random() < 0.6:
                if rng.random() < 0.5:
                    extra += f" email {rng.choice(['john.doe', 'jane.smith', 'admin'])}@example.com "
                if label == "restricted" and rng.random() < 0.4:
                    extra += f" credit card {rng.choice(['4111111111111111', '5555555555554444'])} "
                if label == "restricted" and rng.random() < 0.3:
                    extra += f" password {rng.choice(['admin123', 'secret_key', 'P@ssw0rd'])} "

            if rng.random() < 0.3:
                extra += " " + " ".join(rng.sample(noise_vocab, k=min(2, len(noise_vocab))))

            X.append(base + extra)
            y.append(label)

    combined = list(zip(X, y))
    rng.shuffle(combined)
    X, y = zip(*combined)

    return list(X), list(y)


class DatasetManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.train_csv = os.path.join(DATA_DIR, "confidentiality_train.csv")
        self.valid_csv = os.path.join(DATA_DIR, "confidentiality_valid.csv")

    def ensure_or_generate(self, n_per_class: int = 100, seed: int = 42):
        if os.path.exists(self.train_csv) and os.path.exists(self.valid_csv):
            return

        X_train, y_train = synth_corpus(n_per_class, seed)
        train_df = pd.DataFrame({"text": X_train, "label": y_train})

        X_valid, y_valid = synth_corpus(n_per_class // 4, seed + 1000)
        valid_df = pd.DataFrame({"text": X_valid, "label": y_valid})

        train_df.to_csv(self.train_csv, index=False)
        valid_df.to_csv(self.valid_csv, index=False)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.train_csv)
        valid_df = pd.read_csv(self.valid_csv)
        return train_df, valid_df


class NLPModelCandidateFactory:
    @staticmethod
    def candidates(random_state: int = 42) -> List[Tuple[str, Any]]:
        models: List[Tuple[str, Any]] = []

        vectorizers = [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, max_features=5000)),
            ("tfidf_char", TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=3000)),
            ("count", CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, max_features=5000)),
            ("hash", HashingVectorizer(n_features=4096, ngram_range=(1, 2)))
        ]

        classifiers = [
            ("logreg", LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)),
            ("nb_multi", MultinomialNB(alpha=0.1)),
            ("nb_comp", ComplementNB(alpha=0.1)),
            ("svm_linear", LinearSVC(random_state=random_state, C=1.0)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state, max_depth=10))
        ]

        for vec_name, vectorizer in vectorizers:
            for clf_name, classifier in classifiers:
                if vec_name == "hash" and clf_name.startswith("nb"):
                    continue  # NB não funciona bem com features negativas do HashingVectorizer

                model_name = f"{vec_name}_{clf_name}"
                pipeline = Pipeline([
                    ("vectorizer", vectorizer),
                    ("classifier", classifier)
                ])
                models.append((model_name, pipeline))

        if HAS_XGB:
            models.append((
                "tfidf_xgb",
                Pipeline([
                    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                    ("classifier", XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=random_state,
                        eval_metric="mlogloss"
                    ))
                ])
            ))

        if HAS_LGBM:
            models.append((
                "tfidf_lgbm",
                Pipeline([
                    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                    ("classifier", LGBMClassifier(
                        n_estimators=200,
                        num_leaves=31,
                        learning_rate=0.1,
                        random_state=random_state
                    ))
                ])
            ))

        mlp_configs = [
            ("mlp_small", (100, 50)),
            ("mlp_medium", (200, 100, 50)),
            ("mlp_deep", (300, 200, 100, 50)),
            ("mlp_wide", (500, 300)),
            ("mlp_balanced", (150, 150, 75))
        ]

        for mlp_name, layers in mlp_configs:
            models.append((
                f"tfidf_{mlp_name}",
                Pipeline([
                    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
                    ("classifier", MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate="adaptive",
                        max_iter=500,
                        random_state=random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10
                    ))
                ])
            ))

        return models


class NLPModelTrainer:
    def __init__(self, criterion: str = "f1_macro"):
        assert criterion in ("accuracy", "f1_macro"), "criterion must be accuracy or f1_macro"
        self.criterion = criterion

    def train_and_select(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        results = []

        for name, model in NLPModelCandidateFactory.candidates():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average="macro")

                model_path = os.path.join(MODELS_DIR, f"conf_model_{name}_{int(time.time())}.joblib")
                joblib.dump(model, model_path)

                results.append({
                    "name": name,
                    "path": model_path,
                    "metrics": {"accuracy": acc, "f1_macro": f1}
                })
                print(f"✓ {name}: acc={acc:.3f}, f1={f1:.3f}")

            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
                continue

        if not results:
            raise Exception("No models trained successfully")

        key = (lambda r: r["metrics"]["accuracy"]) if self.criterion == "accuracy" else (
            lambda r: r["metrics"]["f1_macro"])
        best = max(results, key=key)

        registry = read_registry()
        all_models = registry.get("conf_models", [])
        all_models.extend(results)
        registry["conf_models"] = sorted(all_models, key=lambda r: r["metrics"]["f1_macro"], reverse=True)

        best["best"] = True
        registry["best_conf_model"] = best
        write_registry(registry)

        return {"best": best, "all": results}


class ConfidentialityModelService:
    def __init__(self):
        self.dataset = DatasetManager()
        self.trainer = NLPModelTrainer(criterion="f1_macro")
        self.patterns = PatternsRepository()
        self.cfg = ConfigRepository()
        self._best_model = None
        self._best_info = None
        self._retrain_lock = False
        self._load_best_from_disk()

    def _load_best_from_disk(self):
        registry = read_registry()
        info = registry.get("best_conf_model")
        if info and os.path.exists(info["path"]):
            try:
                self._best_model = joblib.load(info["path"])
                self._best_info = info
                print(f"Loaded best confidentiality model: {info['name']}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self._best_model = None
                self._best_info = None

    def train(self, req: TrainRequest) -> TrainResponse:
        n_per_class = max(20, min(500, req.n_per_class if hasattr(req, 'n_per_class') else 100))
        seed = req.seed if req.seed is not None else 42

        self.dataset.ensure_or_generate(n_per_class=n_per_class, seed=seed)

        train_df, valid_df = self.dataset.load()
        X_train = train_df["text"].tolist()
        y_train = train_df["label"].tolist()
        X_val = valid_df["text"].tolist()
        y_val = valid_df["label"].tolist()

        result = self.trainer.train_and_select(X_train, y_train, X_val, y_val)
        self._best_info = result["best"]
        self._best_model = joblib.load(self._best_info["path"])

        metrics = self._best_info["metrics"]
        return TrainResponse(
            model_version=f'{self._best_info["name"]}',
            metrics={"accuracy": float(metrics["accuracy"]), "f1_macro": float(metrics["f1_macro"])},
            samples=int(len(train_df) + len(valid_df))
        )

    def scheduled_retrain(self):
        if self._retrain_lock:
            return
        self._retrain_lock = True
        try:
            print("Starting scheduled retrain for confidentiality models...")
            self.dataset.ensure_or_generate()
            train_df, valid_df = self.dataset.load()
            X_train = train_df["text"].tolist()
            y_train = train_df["label"].tolist()
            X_val = valid_df["text"].tolist()
            y_val = valid_df["label"].tolist()

            result = self.trainer.train_and_select(X_train, y_train, X_val, y_val)
            self._best_info = result["best"]
            self._best_model = joblib.load(self._best_info["path"])
            print(f"Scheduled retrain completed. Best model: {self._best_info['name']}")
        except Exception as e:
            print(f"Scheduled retrain failed: {e}")
        finally:
            self._retrain_lock = False

    def classify(self, req: ClassifyRequest) -> Optional[ContentConfidentiality]:
        if self._best_model is None:
            self._load_best_from_disk()
        if self._best_model is None:
            return None

        text = (req.content_pointer.sample_text or "").strip()
        meta = req.content_pointer.metadata or {}

        if not text:
            return self._fallback_classify(req)

        try:
            if hasattr(self._best_model, "predict_proba"):
                proba = self._best_model.predict_proba([text])[0]
                classes = self._best_model.classes_
                probs = {cls: float(prob) for cls, prob in zip(classes, proba)}
                label = max(probs, key=probs.get)
                score = float(max(probs.values()))
            else:
                label = self._best_model.predict([text])[0]
                score = 0.7
                probs = {cls: 0.25 for cls in CLASS_LABELS}
                probs[label] = score

            findings = self.patterns.scan(text)
            pattern_tags = self.patterns.tags_from_findings(findings)
            detected = [f["type"] for f in findings]

            boost = 0.0
            if any(f["type"] == "credit_card" for f in findings):
                boost += 0.15
                label = self._escalate(label, "restricted")
            if any(f["type"] in ("email", "national_id") for f in findings):
                boost += 0.08
                label = self._escalate(label, "confidential")

            doc_type = (meta.get("doc_type") or "").lower()
            if doc_type in ("hr", "finance", "legal"):
                boost += 0.05
                if doc_type in ("hr", "finance"):
                    label = self._escalate(label, "confidential")

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
                model_version=self._best_info["name"] if self._best_info else "unknown"
            )

        except Exception as e:
            print(f"Classification error: {e}")
            return self._fallback_classify(req)

    def _escalate(self, current: str, target: str) -> str:
        order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        return target if order.get(target, 0) > order.get(current, 0) else current

    def _fallback_classify(self, req: ClassifyRequest) -> ContentConfidentiality:
        text = (req.content_pointer.sample_text or "").strip().lower()
        meta = req.content_pointer.metadata or {}

        if any(k in text for k in ["credit card", "iban", "private key", "credential", "ssn", "password"]):
            label, score = "restricted", 0.9
        elif any(k in text for k in ["salary", "contract", "pii", "personal data", "customer", "financial"]):
            label, score = "confidential", 0.75
        elif any(k in text for k in ["internal", "memo", "roadmap", "meeting"]):
            label, score = "internal", 0.55
        else:
            label, score = "public", 0.3

        findings = self.patterns.scan(text)
        pattern_tags = self.patterns.tags_from_findings(findings)
        detected = [f["type"] for f in findings]

        return ContentConfidentiality(
            classification=label,
            score=round(score, 3),
            tags=pattern_tags,
            detected_patterns=detected,
            dlp_findings=findings,
            source_app_context=meta.get("app"),
            user_label=meta.get("user_label"),
            model_version="conf-fallback-heuristic-1.0"
        )

    def cleanup_old_models(self,
                           keep_best_n: int = 10,
                           max_age_days: int = 30,
                           min_accuracy_threshold: float = 0.5,
                           dry_run: bool = True) -> Dict[str, Any]:

        from services.model_cleanup_service import ModelCleanupService

        cleanup_service = ModelCleanupService(
            MODELS_DIR,
            os.path.join(DATA_DIR, "registry.json")
        )

        result = self._cleanup_conf_models_only(
            keep_best_n=keep_best_n,
            max_age_days=max_age_days,
            min_accuracy_threshold=min_accuracy_threshold,
            dry_run=dry_run
        )

        if not dry_run and result["models_removed"] > 0:
            self._load_best_from_disk()

        return result

    def _cleanup_conf_models_only(self, keep_best_n: int, max_age_days: int,
                                  min_accuracy_threshold: float, dry_run: bool) -> Dict[str, Any]:

        cleanup_stats = {
            "models_analyzed": 0,
            "models_removed": 0,
            "models_kept": 0,
            "space_freed_mb": 0,
            "removed_files": [],
            "kept_files": [],
            "orphaned_files_removed": 0,
            "registry_cleaned": False
        }

        try:
            registry = read_registry()
            conf_models = registry.get("conf_models", [])
            cleanup_stats["models_analyzed"] = len(conf_models)

            if not conf_models:
                print("Nenhum modelo de confidencialidade encontrado no registry")
                return cleanup_stats

            valid_models = []
            for model in conf_models:
                if os.path.exists(model["path"]):
                    stat = os.stat(model["path"])
                    model["file_size_mb"] = stat.st_size / (1024 * 1024)
                    model["file_age_days"] = (time.time() - stat.st_mtime) / (24 * 3600)
                    valid_models.append(model)

            models_to_keep = []
            models_to_remove = []

            valid_models.sort(key=lambda x: x["metrics"].get("f1_macro", 0), reverse=True)

            for i, model in enumerate(valid_models):
                keep_reasons = []
                remove_reasons = []

                if i < keep_best_n:
                    keep_reasons.append(f"top_{keep_best_n}_best")

                if model["file_age_days"] > max_age_days:
                    remove_reasons.append(f"older_than_{max_age_days}_days")

                if model["metrics"].get("accuracy", 0) < min_accuracy_threshold:
                    remove_reasons.append(f"accuracy_below_{min_accuracy_threshold}")

                if keep_reasons and not remove_reasons:
                    models_to_keep.append(model)
                elif remove_reasons and not keep_reasons:
                    models_to_remove.append(model)
                elif keep_reasons and remove_reasons:
                    if f"top_{keep_best_n}_best" in keep_reasons:
                        models_to_keep.append(model)
                    else:
                        models_to_remove.append(model)
                else:
                    models_to_keep.append(model)

            for model in models_to_remove:
                file_path = model["path"]
                if os.path.exists(file_path):
                    file_size_mb = model["file_size_mb"]

                    if not dry_run:
                        os.remove(file_path)
                        print(f"Removido: {file_path} ({file_size_mb:.2f}MB)")
                    else:
                        print(f"[DRY RUN] Removeria: {file_path} ({file_size_mb:.2f}MB)")

                    cleanup_stats["models_removed"] += 1
                    cleanup_stats["space_freed_mb"] += file_size_mb
                    cleanup_stats["removed_files"].append(file_path)

            if not dry_run:
                registry["conf_models"] = models_to_keep
                if models_to_keep:
                    registry["best_conf_model"] = models_to_keep[0]
                else:
                    registry["best_conf_model"] = None
                write_registry(registry)
                cleanup_stats["registry_cleaned"] = True

            cleanup_stats["models_kept"] = len(models_to_keep)
            cleanup_stats["kept_files"] = [m["path"] for m in models_to_keep]

            orphaned_count = self._cleanup_orphaned_conf_files(models_to_keep, dry_run)
            cleanup_stats["orphaned_files_removed"] = orphaned_count

            return cleanup_stats

        except Exception as e:
            print(f"Erro durante limpeza de modelos de confidencialidade: {e}")
            cleanup_stats["error"] = str(e)
            return cleanup_stats

    def _cleanup_orphaned_conf_files(self, valid_models: List[Dict], dry_run: bool) -> int:
        valid_paths = {model["path"] for model in valid_models}

        all_conf_files = glob.glob(os.path.join(MODELS_DIR, "conf_model_*.joblib"))

        orphaned_count = 0
        for file_path in all_conf_files:
            if file_path not in valid_paths:
                if not dry_run:
                    os.remove(file_path)
                    print(f"Arquivo órfão de confidencialidade removido: {file_path}")
                else:
                    print(f"[DRY RUN] Removeria arquivo órfão: {file_path}")
                orphaned_count += 1

        return orphaned_count

    def get_cleanup_recommendations(self) -> Dict[str, Any]:
        registry = read_registry()
        models = registry.get("conf_models", [])

        if not models:
            return {"recommendation": "no_conf_models_found"}

        total_models = len(models)
        accuracies = [m["metrics"].get("accuracy", 0) for m in models]
        f1_scores = [m["metrics"].get("f1_macro", 0) for m in models]

        ages_days = []
        total_size_mb = 0

        for model in models:
            if os.path.exists(model["path"]):
                stat = os.stat(model["path"])
                age_days = (time.time() - stat.st_mtime) / (24 * 3600)
                ages_days.append(age_days)
                total_size_mb += stat.st_size / (1024 * 1024)

        recommendations = {
            "total_conf_models": total_models,
            "total_size_mb": round(total_size_mb, 2),
            "accuracy_stats": {
                "min": round(min(accuracies), 3) if accuracies else 0,
                "max": round(max(accuracies), 3) if accuracies else 0,
                "avg": round(sum(accuracies) / len(accuracies), 3) if accuracies else 0
            },
            "f1_macro_stats": {
                "min": round(min(f1_scores), 3) if f1_scores else 0,
                "max": round(max(f1_scores), 3) if f1_scores else 0,
                "avg": round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else 0
            },
            "age_stats_days": {
                "min": round(min(ages_days), 1) if ages_days else 0,
                "max": round(max(ages_days), 1) if ages_days else 0,
                "avg": round(sum(ages_days) / len(ages_days), 1) if ages_days else 0
            }
        }

        if total_models > 15:
            recommendations["suggested_keep_best_n"] = 8
        elif total_models > 8:
            recommendations["suggested_keep_best_n"] = 4
        else:
            recommendations["suggested_keep_best_n"] = max(2, total_models // 2)

        if max(ages_days) if ages_days else 0 > 45:
            recommendations["suggested_max_age_days"] = 30
        else:
            recommendations["suggested_max_age_days"] = 45

        if min(accuracies) if accuracies else 0 < 0.7:
            recommendations["suggested_min_accuracy"] = 0.7
        else:
            recommendations["suggested_min_accuracy"] = 0.6

        return recommendations

    def scheduled_cleanup(self):
        try:
            recommendations = self.get_cleanup_recommendations()

            keep_best = max(4, recommendations.get("suggested_keep_best_n", 4))
            max_age = max(45, recommendations.get("suggested_max_age_days", 45))
            min_acc = min(0.5, recommendations.get("suggested_min_accuracy", 0.6))

            result = self.cleanup_old_models(
                keep_best_n=keep_best,
                max_age_days=max_age,
                min_accuracy_threshold=min_acc,
                dry_run=False
            )

            print(f"Limpeza automática de confidencialidade: {result['models_removed']} removidos, "
                  f"{result['space_freed_mb']:.2f}MB liberados")

            return result

        except Exception as e:
            print(f"Erro na limpeza automática de confidencialidade: {e}")
            return {"error": str(e)}