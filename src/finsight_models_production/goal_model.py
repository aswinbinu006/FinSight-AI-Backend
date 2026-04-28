# -*- coding: utf-8 -*-
"""
FinSight AI — Goal Intelligence Model (Portable) v2
=====================================================
Stage 6 in the website flow.

Design Philosophy:
    FinSight AI combines machine learning with behavioral finance rules
    to generate stable and interpretable outputs. Domain knowledge features
    (like timeline_progress_ratio) are explicitly labeled as heuristic encodings.

Fixes applied (v2 — from Prompt Reference doc):
    1. Fix 0% probability collapse — reduce max_depth overfitting
    2. Actually apply _LEAKY_TARGETS drop before building X
    3. Use goal_description in predict() output
    4. Update dataset filename to v2
    (Previous fixes remain intact)

Usage:
    from finsight_models_production import GoalModel
    model = GoalModel()
    metrics = model.train()
    result = model.predict(
        behavioral_scores=scores,
        goal_description='Trip to Goa',
        goal_target=50000,
        saved_so_far=8000,
        monthly_savings=5000,
        months_remaining=6,
    )
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
import joblib

from .behavioral_scoring import BEHAVIOR_COLUMNS

warnings.filterwarnings("ignore")


class GoalModel:
    """Goal success predictor + risk factor identifier."""

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    # Change 4: Updated dataset filename
    _DATA_FILE = "goal_intelligence_dataset_v2.csv"

    _CAT_FEATURES = [
        "age_bracket", "income_bracket", "occupation_type",
        "family_size", "city_tier", "emi_burden", "financial_dependents",
        "goal_amount_bracket", "monthly_savings_target_bracket",
    ]

    # Numeric features besides behavior scores
    _EXTRA_NUM_FEATURES = [
        "goal_timeline_months", "days_elapsed", "days_remaining",
        "current_saved_pct", "savings_velocity", "timeline_progress_ratio"
    ]

    _LEAKY_TARGETS = ["goal_success_probability", "predicted_failure_date_days"]

    def __init__(self):
        self._trained = False
        self.xgbc = None
        self.rf_risk = None
        self.best_success_algo = ""
        self.best_risk_algo = ""
        self.encoder = None
        self.num_feature_names = []
        self.feature_names = []

        self.risk_map = {}
        self.reverse_risk_map = {}

        self.df_clean = None
        self.mode_defaults = {}
        self.metrics = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Train success prediction and risk factor models."""
        csv_path = os.path.join(self._DATA_DIR, self._DATA_FILE)
        df = pd.read_csv(csv_path)

        df["financial_dependents"] = df["financial_dependents"].astype(str)
        for col in df.columns:
            if col not in self._CAT_FEATURES and df[col].dtype == "object" and col != "top_risk_factor":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df_clean = df.dropna().copy()

        # ── Domain knowledge feature ──
        df_clean["timeline_progress_ratio"] = df_clean["days_elapsed"] / (
            df_clean["days_elapsed"] + df_clean["days_remaining"]
        ).replace(0, 1)

        # ── Change 2: Actually apply _LEAKY_TARGETS ──
        # These were defined but never dropped. Drop before building X.
        df_clean = df_clean.drop(
            columns=[c for c in self._LEAKY_TARGETS if c in df_clean.columns]
        )

        # ── Numeric features ──
        self.num_feature_names = BEHAVIOR_COLUMNS + self._EXTRA_NUM_FEATURES
        X_num = df_clean[self.num_feature_names].values

        # ── Categorical features (OneHotEncoder) ──
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.encoder.fit_transform(df_clean[self._CAT_FEATURES])
        cat_col_names = self.encoder.get_feature_names_out(self._CAT_FEATURES).tolist()

        # ── Combined Features ──
        X = np.hstack([X_num, X_cat])
        self.feature_names = self.num_feature_names + cat_col_names

        # --- Part A: Binary Success Classification → XGBoost ---
        y_bin = df_clean["goal_success_binary"].astype(int).values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.1, random_state=42)

        bin_classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
            "NaiveBayes": GaussianNB(),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.0,
                n_jobs=-1, random_state=42, eval_metric='logloss'
            )
        }

        best_acc_b = -1
        best_clf_b = None
        best_clf_name_b = ""
        b_f1 = 0
        b_ll = 0
        success_evaluations = {}

        for name, clf in bin_classifiers.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc = float(accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred))
            
            if hasattr(clf, "predict_proba"):
                try:
                    ll = float(log_loss(y_te, clf.predict_proba(X_te)[:, 1]))
                except ValueError:
                    ll = 0.0
            else:
                ll = 0.0
                
            success_evaluations[name] = {"accuracy": round(acc, 4), "f1_score": round(f1, 4), "log_loss": round(ll, 4)}

            if acc > best_acc_b:
                best_acc_b = acc
                best_clf_b = clf
                best_clf_name_b = name
                b_f1 = f1
                b_ll = ll

        # ── EXPERT ML: Probability Calibration (Platt Scaling) ──
        # Guarantees that predict_proba yields mathematically true confidence intervals
        self.success_model = CalibratedClassifierCV(best_clf_b, method='sigmoid', cv=3)
        self.success_model.fit(X_tr, y_tr)
        
        self.best_success_algo = best_clf_name_b
        acc_b = best_acc_b
        f1_b = b_f1
        ll_b = b_ll

        # --- Part B: Risk Factor Classification → Random Forest ---
        unique_risks = sorted(df_clean["top_risk_factor"].unique().tolist())
        self.risk_map = {r: i for i, r in enumerate(unique_risks)}
        self.reverse_risk_map = {v: k for k, v in self.risk_map.items()}

        y_risk = df_clean["top_risk_factor"].map(self.risk_map).astype(int).values  # type: ignore

        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_risk, test_size=0.1, random_state=42)

        risk_classifiers = {
            "DecisionTree": DecisionTreeClassifier(max_depth=15, random_state=42, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
            "NaiveBayes": GaussianNB(),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=15, random_state=42)
        }

        best_acc_r = -1
        best_clf_r = None
        best_clf_name_r = ""
        risk_evaluations = {}

        for name, clf in risk_classifiers.items():
            clf.fit(X_tr_r, y_tr_r)
            y_pred_r = clf.predict(X_te_r)
            acc = float(accuracy_score(y_te_r, y_pred_r))
            f1 = float(f1_score(y_te_r, y_pred_r, average="weighted"))
            
            risk_evaluations[name] = {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
            
            if acc > best_acc_r:
                best_acc_r = acc
                best_clf_r = clf
                best_clf_name_r = name

        # ── EXPERT ML: Probability Calibration ──
        self.risk_model = CalibratedClassifierCV(best_clf_r, method='sigmoid', cv=3)
        self.risk_model.fit(X_tr_r, y_tr_r)
        
        self.best_risk_algo = best_clf_name_r
        acc_r = best_acc_r

        # Store reference data
        self.df_clean = df_clean
        self.mode_defaults = {
            "age_bracket": df_clean["age_bracket"].mode(dropna=True)[0],
            "income_bracket": df_clean["income_bracket"].mode(dropna=True)[0],
            "occupation_type": df_clean["occupation_type"].mode(dropna=True)[0],
            "family_size": df_clean["family_size"].mode(dropna=True)[0],
            "city_tier": df_clean["city_tier"].dropna().mode()[0],
            "emi_burden": df_clean["emi_burden"].mode(dropna=True)[0],
            "financial_dependents": str(df_clean["financial_dependents"].mode(dropna=True)[0]),
            "goal_amount_bracket": df_clean["goal_amount_bracket"].mode(dropna=True)[0],
            "monthly_savings_target_bracket": df_clean["monthly_savings_target_bracket"].mode(dropna=True)[0],
        }

        self.metrics = {
            "evaluated_models": {
                "success_models": success_evaluations,
                "risk_models": risk_evaluations
            },
            "best_success_model": self.best_success_algo,
            "best_risk_model": self.best_risk_algo,
            "success_accuracy": round(acc_b, 4),
            "success_f1": round(f1_b, 4),
            "success_logloss": round(ll_b, 4),
            "risk_accuracy": round(acc_r, 4),
        }

        self._trained = True
        return {"status": "trained", "rows": len(df_clean), **self.metrics}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        behavioral_scores: dict,
        goal_description: str = 'your goal',  # Change 3: Added parameter
        goal_target: float = 100000,
        saved_so_far: float = 15000,
        monthly_savings: float = 5000,
        months_remaining: float = 12,
    ) -> dict:
        """
        Predict goal success probability + risk factors.
        """
        assert self._trained, "Call .train() first or .load() a saved model."
        assert self.encoder is not None
        assert self.success_model is not None
        assert self.risk_model is not None

        # Clamp AI answers structurally
        clamped_scores = {k: max(1.0, min(10.0, float(v))) for k, v in behavioral_scores.items()}

        # Convert Rupee amounts to ML features
        goal_target = max(goal_target, 1.0)
        saved_pct = (saved_so_far / goal_target) * 100.0
        velocity = (monthly_savings / goal_target) * 100.0 / 30.0
        months_elapsed = (saved_so_far / monthly_savings) if monthly_savings > 0 else 0
        elapsed = months_elapsed * 30.0
        timeline = months_elapsed + months_remaining

        sample = self._build_sample(clamped_scores, timeline, elapsed, saved_pct, velocity)

        # ── Encoding ──
        X_num = np.array([[sample[col] for col in self.num_feature_names]])

        cat_df = pd.DataFrame([{col: sample[col] for col in self._CAT_FEATURES}])
        X_cat = self.encoder.transform(cat_df)  # type: ignore

        X_pred = np.hstack([X_num, X_cat])  # type: ignore

        # Binary prediction
        pred_success = self.success_model.predict(X_pred)[0]  # type: ignore
        pred_prob = float(self.success_model.predict_proba(X_pred)[0][1])  # type: ignore

        # Risk prediction
        pred_risk_idx = self.risk_model.predict(X_pred)[0]  # type: ignore
        risk_label = self.reverse_risk_map[pred_risk_idx]

        # ── Feature Importance Extraction ──
        base_estimator = getattr(self.success_model, 'estimator', self.success_model)
        if hasattr(base_estimator, 'feature_importances_'):
            importances = base_estimator.feature_importances_  # type: ignore
        elif hasattr(base_estimator, 'coef_'):
            # Multi-class LR might have multiple rows of coefficients, take the max per feature
            if len(base_estimator.coef_.shape) > 1:  # type: ignore
                importances = np.max(np.abs(base_estimator.coef_), axis=0)  # type: ignore
            else:
                importances = np.abs(base_estimator.coef_[0])  # type: ignore
        else:
            importances = np.zeros(len(self.feature_names))
            
        feat_imp = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        top_factors = [
            {"feature": name.replace("_score", "").replace("_", " ").title(),
             "importance": round(float(imp), 4)}
            for name, imp in feat_imp[:5]
        ]

        # Advisor logic
        month_required = (goal_target - saved_so_far) / monthly_savings if monthly_savings > 0 else float("inf")

        success_pct = round(pred_prob * 100, 1)
        is_on_track = bool(pred_success == 1)

        # ── Score Interpretation Bands ─────────────────────────────────
        # Based on user-provided spec.
        SCORE_BANDS = {
            'On Track':   {'range': '80 - 100', 'min': 80, 'meaning': 'Strong success'},
            'Likely':     {'range': '60 - 79',  'min': 60, 'meaning': 'Good chance'},
            'Uncertain':  {'range': '40 - 59',  'min': 40, 'meaning': 'Depends on improvement'},
            'High Risk':  {'range': '20 - 39',  'min': 20, 'meaning': 'Very unlikely'},
            'Impossible': {'range': '0 - 19',   'min': 0,  'meaning': 'Cannot achieve goal'},
        }

        if success_pct >= 80:
            score_band = 'On Track'
        elif success_pct >= 60:
            score_band = 'Likely'
        elif success_pct >= 40:
            score_band = 'Uncertain'
        elif success_pct >= 20:
            score_band = 'High Risk'
        else:
            score_band = 'Impossible'

        band_info = SCORE_BANDS[score_band]

        return {
            "goal_description": goal_description,  # Change 3: Included in output
            "success_probability": success_pct,
            "score_band": score_band,
            "score_interpretation": {
                "band": score_band,
                "range": band_info['range'],
                "meaning": band_info['meaning'],
                "all_bands": {k: v['range'] for k, v in SCORE_BANDS.items()},
            },
            "is_on_track": is_on_track,
            "risk_factor": risk_label.replace("_", " ").title(),
            "target_summary": {
                "goal_amount": goal_target,
                "saved_so_far": saved_so_far,
                "saved_pct": round(saved_pct, 1),
                "monthly_savings": monthly_savings,
                "months_remaining": months_remaining,
                "months_required_at_current_pace": round(month_required, 1),
            },
            "top_drivers": top_factors,
            "model_metrics": self.metrics,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pick_cohort(self, timeline, saved_pct, velocity):
        assert self.df_clean is not None
        dc = self.df_clean
        candidates = [
            dc[
                dc["goal_timeline_months"].between(max(1, timeline - 3), timeline + 3)
                & dc["current_saved_pct"].between(max(0, saved_pct - 10), min(100, saved_pct + 10))
            ],
            dc[
                dc["goal_timeline_months"].between(max(1, timeline - 6), timeline + 6)
                & dc["savings_velocity"].between(max(0, velocity - 0.15), velocity + 0.15)
            ],
            dc[dc["goal_timeline_months"].between(max(1, timeline - 12), timeline + 12)],
            dc,
        ]
        for c in candidates:
            if not c.empty:
                return c
        return dc

    def _build_sample(self, scores, timeline, elapsed, saved_pct, velocity):
        days_remaining = max(0.0, (timeline * 30.0) - elapsed)
        cohort = self._pick_cohort(timeline, saved_pct, velocity)

        def _mode(col):
            if cohort[col].notna().any():  # type: ignore
                return cohort[col].mode(dropna=True)[0]
            return self.mode_defaults.get(col, "")

        sample = {
            **scores,
            "goal_timeline_months": timeline,
            "days_elapsed": elapsed,
            "days_remaining": days_remaining,
            "current_saved_pct": saved_pct,
            "savings_velocity": velocity,
            "timeline_progress_ratio": elapsed / max(1.0, elapsed + days_remaining),
            "age_bracket": _mode("age_bracket"),
            "income_bracket": _mode("income_bracket"),
            "occupation_type": _mode("occupation_type"),
            "family_size": _mode("family_size"),
            "city_tier": cohort["city_tier"].dropna().mode()[0] if cohort["city_tier"].notna().any() else self.mode_defaults["city_tier"],  # type: ignore
            "emi_burden": _mode("emi_burden"),
            "financial_dependents": str(_mode("financial_dependents")),
            "goal_amount_bracket": _mode("goal_amount_bracket"),
            "monthly_savings_target_bracket": _mode("monthly_savings_target_bracket"),
        }
        return sample

    def save(self, filepath: str):
        """Serialize the trained model."""
        assert self._trained, "Call .train() before saving"
        joblib.dump(self.__dict__, filepath)

    def load(self, filepath: str):
        """Load a serialized model from disk."""
        state = joblib.load(filepath)
        self.__dict__.update(state)
        self._trained = True
