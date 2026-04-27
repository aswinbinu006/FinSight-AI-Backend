# -*- coding: utf-8 -*-
"""
FinSight AI — Health Insight Model (Portable) v3
=================================================
Stage 4 in the website flow.

Design Philosophy:
    FinSight AI combines machine learning with behavioral finance rules
    to generate stable and interpretable outputs. Domain knowledge features
    (like emi_stress_index) are explicitly labeled as heuristic encodings,
    not learned representations.

Fixes applied (v3 — from Prompt Reference doc):
    1. Fix score/trend contradiction — reconciliation block
    2. Ensure metrics are always returned from train()
    3. Remove hardcoded description strings — Gemini explainer handles text
    4. Update dataset filename to v3
    (Previous v2 fixes remain intact)

Usage:
    from finsight_models_production import HealthModel
    model = HealthModel()
    metrics = model.train()   # returns RMSE, R², accuracy
    result = model.predict(behavioral_scores, emi_burden, income_bracket)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, mean_absolute_error, f1_score
)
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings

from .behavioral_scoring import BEHAVIOR_COLUMNS

warnings.filterwarnings("ignore")


class HealthModel:
    """Financial Health Score predictor (0-100) + 30-day trend forecast."""

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    # Change 4: Updated dataset filename
    _DATA_FILE = "health_score_dataset_v3.csv"
    _WEIGHTS_FILE = "health_weights.json"

    _CAT_FEATURES = [
        "age_bracket", "income_bracket", "occupation_type",
        "family_size", "city_tier", "emi_burden", "financial_dependents",
    ]

    # Numeric features the model uses (besides the 10 behavioral scores)
    _EXTRA_NUM_FEATURES = ["projected_score_30d", "emi_stress_index"]

    _TREND_MAP = {
        "Storm Warning": 0, "Rainy": 1, "Partly Cloudy": 2,
        "Improving": 3, "Sunny": 4,
    }
    _REVERSE_TREND = {v: k for k, v in _TREND_MAP.items()}

    # Trend order for reconciliation (Change 1)
    _TREND_ORDER = ['Storm Warning', 'Rainy', 'Partly Cloudy', 'Improving', 'Sunny']

    _INCOME_ALIASES = {
        "below_20k": "below_20k", "20k-50k": "20k-50k",
        "50k-100k": "50k-100k", "100k-200k": "100k+",
        "above_200k": "100k+", "100k+": "100k+",
    }
    _EMI_ALIASES = {
        "no_emi": "none", "none": "none",
        "light": "light", "moderate": "light", "heavy": "heavy",
    }

    # ── Trajectory weights: domain-driven (not magic numbers) ────────
    _TRAJECTORY_WEIGHTS = {
        "impulse_control_score": 0.35,
        "future_planning_score": 0.35,
        "emergency_preparedness_score": 0.30,
    }

    def __init__(self):
        self._trained = False
        self.score_model = None      # Regressor for health score
        self.trend_model = None      # Classifier for trend
        self.best_score_algo = ""
        self.best_trend_algo = ""
        self.encoder = None
        self.num_feature_names = []
        self.feature_names = []
        self.weights = {}
        self.metrics = {}
        self.df_clean = None
        self.mode_defaults = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """
        Train both sub-models on the health dataset.

        Returns
        -------
        dict with keys: status, rows, score_rmse, score_r2, trend_accuracy
        """
        csv_path = os.path.join(self._DATA_DIR, self._DATA_FILE)
        df = pd.read_csv(csv_path)

        # Clean
        df["financial_dependents"] = df["financial_dependents"].astype(str)
        for col in df.columns:
            if col not in self._CAT_FEATURES and df[col].dtype == "object" \
               and col not in ["trend_label", "health_band"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df_clean = df.dropna().copy()

        # ── Domain knowledge feature ──
        df_clean["emi_stress_index"] = df_clean["emi_burden"].apply(
            lambda x: 2 if x == "heavy" else (1 if x == "light" else 0)
        )

        # ── Numeric features ──
        self.num_feature_names = BEHAVIOR_COLUMNS + self._EXTRA_NUM_FEATURES
        X_num = df_clean[self.num_feature_names].values

        # ── Categorical features (OneHotEncoder) ──
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.encoder.fit_transform(df_clean[self._CAT_FEATURES])
        cat_col_names = self.encoder.get_feature_names_out(self._CAT_FEATURES).tolist()

        # ── Combined feature matrix ──
        X = np.hstack([X_num, X_cat])
        self.feature_names = self.num_feature_names + cat_col_names

        # ================================================================
        # MODEL A: Health Score Regression → XGBRegressor
        # ================================================================
        y_score = df_clean["health_score"].values

        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
            X, y_score, test_size=0.1, random_state=42,
        )

        regressors = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=800, max_depth=4,
                learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5, gamma=0.2,
                n_jobs=-1, random_state=42,
            )
        }

        best_r2 = -9999
        best_rmse = 9999
        best_reg = None
        best_reg_name = ""
        score_evaluations = {}

        for name, model in regressors.items():
            model.fit(X_tr_s, y_tr_s)
            y_pred = model.predict(X_te_s)
            r2 = float(r2_score(y_te_s, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_te_s, y_pred)))
            mae = float(mean_absolute_error(y_te_s, y_pred))
            
            score_evaluations[name] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}

            if r2 > best_r2:
                best_r2 = r2
                best_rmse = rmse
                best_reg = model
                best_reg_name = name

        self.score_model = best_reg
        self.best_score_algo = best_reg_name
        rmse = best_rmse
        r2 = best_r2

        # ================================================================
        # MODEL B: Trend Classification → XGBClassifier
        # ================================================================
        trend_encoded = df_clean["trend_label"].map(self._TREND_MAP)
        valid_mask = trend_encoded.notna()
        X_trend = X[valid_mask.values]
        y_trend = trend_encoded[valid_mask].astype(int).values

        X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
            X_trend, y_trend, test_size=0.1, random_state=42,
        )

        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
            "NaiveBayes": GaussianNB(),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                         subsample=0.8, n_jobs=-1, random_state=42)
        }

        best_f1 = -1
        best_clf = None
        best_clf_name = ""
        trend_evaluations = {}

        for name, model in classifiers.items():
            model.fit(X_tr_t, y_tr_t)
            y_pred = model.predict(X_te_t)
            acc = float(accuracy_score(y_te_t, y_pred))
            f1_macro = float(f1_score(y_te_t, y_pred, average="macro"))
            f1_weighted = float(f1_score(y_te_t, y_pred, average="weighted"))
            
            trend_evaluations[name] = {
                "accuracy": round(acc, 4), 
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4)
            }

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_clf = model
                best_clf_name = name

        self.trend_model = best_clf
        self.best_trend_algo = best_clf_name
        trend_f1 = best_f1

        # ── Store reference data ──
        self.df_clean = df_clean
        self.mode_defaults = {
            "age_bracket": df_clean["age_bracket"].mode(dropna=True)[0],
            "occupation_type": df_clean["occupation_type"].mode(dropna=True)[0],
            "family_size": df_clean["family_size"].mode(dropna=True)[0],
            "city_tier": df_clean["city_tier"].dropna().mode()[0],
            "financial_dependents": str(df_clean["financial_dependents"].mode(dropna=True)[0]),
        }

        # Load domain weights
        weights_path = os.path.join(self._DATA_DIR, self._WEIGHTS_FILE)
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                self.weights = json.load(f)

        self.metrics = {
            'evaluated_models': {
                'score_models': score_evaluations,
                'trend_models': trend_evaluations,
            },
            'best_score_model': self.best_score_algo,
            'best_trend_model': self.best_trend_algo,
            'score_rmse':      round(rmse, 2),
            'score_r2':        round(r2, 4),
            'trend_f1_macro':  round(trend_f1, 4),
        }

        self._trained = True
        return {'status': 'trained', 'rows': len(df_clean), **self.metrics}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        behavioral_scores: dict,
        emi_burden: str = "light",
        income_bracket: str = "50k-100k",
    ) -> dict:
        """
        Predict health score + trend.

        Parameters
        ----------
        behavioral_scores : dict
            10 BEHAVIOR_COLUMNS -> float values (1.0 - 10.0).
        emi_burden : str
            One of: no_emi, light, moderate, heavy
        income_bracket : str
            One of: below_20k, 20k-50k, 50k-100k, 100k-200k, above_200k

        Returns
        -------
        dict with keys:
            health_score, trend_label, status_label,
            strengths, weaknesses, top_factors, metrics
        """
        assert self._trained, "Call .train() first or .load() a saved model."

        # Clamp AI answers structurally
        clamped_scores = {k: max(1.0, min(10.0, float(v))) for k, v in behavioral_scores.items()}

        income_ds = self._INCOME_ALIASES.get(income_bracket, "50k-100k")
        emi_ds = self._EMI_ALIASES.get(emi_burden, "light")

        # ── Compute projected_score from behavioral trajectory ──
        trajectory_score = sum(
            clamped_scores.get(feat, 5.0) * weight
            for feat, weight in self._TRAJECTORY_WEIGHTS.items()
        ) * 10.0
        trajectory_score = min(100.0, max(0.0, trajectory_score))

        sample = self._build_sample(behavioral_scores, income_ds, emi_ds, trajectory_score)

        # ── Encode with fitted OneHotEncoder ──
        num_vals = [sample.get(col, 5.0) for col in self.num_feature_names]
        X_num = np.array([num_vals])

        cat_df = pd.DataFrame([{col: sample[col] for col in self._CAT_FEATURES}])
        X_cat = self.encoder.transform(cat_df)

        X_pred = np.hstack([X_num, X_cat])

        # ── Score prediction ──────────────────────────────────────────
        pred_score = float(np.clip(self.score_model.predict(X_pred)[0], 0, 100))

        # ── Trend prediction ──────────────────────────────────────────
        pred_trend_idx = int(self.trend_model.predict(X_pred)[0])
        trend_label = self._REVERSE_TREND.get(pred_trend_idx, "Partly Cloudy")

        # ── Status label ──
        if pred_score >= 75:
            status = "Sunny"
        elif pred_score >= 55:
            status = "Improving"
        elif pred_score >= 40:
            status = "Partly Cloudy"
        elif pred_score >= 25:
            status = "Rainy"
        else:
            status = "Storm Warning"

        # ── Change 1: Fix score/trend contradiction ──
        # If status_label and trend_label disagree by more than 1 step,
        # override trend to match score-based status
        score_idx = self._TREND_ORDER.index(status)
        trend_idx = self._TREND_ORDER.index(trend_label)
        if abs(score_idx - trend_idx) > 1:
            trend_label = self._TREND_ORDER[score_idx]

        # Change 3: REMOVED hardcoded descriptions dict entirely.
        # The Gemini explanation layer (explainer.py) handles all
        # user-facing text. No 'trend_description' in return dict.

        # ── Strengths / Weaknesses ──
        score_items = sorted(
            [(k, v) for k, v in behavioral_scores.items() if k in BEHAVIOR_COLUMNS],
            key=lambda x: x[1],
        )
        weaknesses = [
            {"name": k.replace("_score", "").replace("_", " ").title(), "value": round(v, 1)}
            for k, v in score_items[:2]
        ]
        strengths = [
            {"name": k.replace("_score", "").replace("_", " ").title(), "value": round(v, 1)}
            for k, v in score_items[-2:]
        ]

        # ── Feature importance ────────────────────────────────────────
        if hasattr(self.score_model, 'feature_importances_'):
            importances = self.score_model.feature_importances_
        elif hasattr(self.score_model, 'coef_'):
            importances = np.abs(self.score_model.coef_)
        else:
            importances = np.zeros(len(self.feature_names))
            
        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )
        top_factors = [
            {"feature": name.replace("_score", "").replace("_", " ").title(),
             "importance": round(float(imp), 4)}
            for name, imp in feat_imp[:5]
        ]

        # ── Score Interpretation Bands ─────────────────────────────────
        # Based on user-provided spec. Derived strictly from score.
        SCORE_BANDS = {
            'Excellent': {'range': '70 - 100', 'min': 70, 'meaning': 'Highly disciplined'},
            'Good':      {'range': '55 - 69',  'min': 55, 'meaning': 'Stable and improving'},
            'Average':   {'range': '40 - 54',  'min': 40, 'meaning': 'Mixed behavior'},
            'Poor':      {'range': '25 - 39',  'min': 25, 'meaning': 'Weak financial habits'},
            'Critical':  {'range': '0 - 24',   'min': 0,  'meaning': 'Severe financial instability'},
        }

        if pred_score >= 70:
            score_band = 'Excellent'
        elif pred_score >= 55:
            score_band = 'Good'
        elif pred_score >= 40:
            score_band = 'Average'
        elif pred_score >= 25:
            score_band = 'Poor'
        else:
            score_band = 'Critical'

        band_info = SCORE_BANDS[score_band]

        # ── Demographic Context ───────────────────────────────────────
        # Show user where they rank within their income + EMI cohort
        # so a score of 75 in a tough demographic is seen as excellent.
        cohort = self._pick_cohort(income_ds, emi_ds)
        cohort_scores = cohort['health_score']
        percentile_rank = round(float((cohort_scores < pred_score).mean() * 100), 1)
        cohort_avg = round(float(cohort_scores.mean()), 1)
        cohort_max = round(float(cohort_scores.max()), 1)

        return {
            "health_score": round(pred_score, 1),
            "score_band": score_band,
            "score_interpretation": {
                "band": score_band,
                "range": band_info['range'],
                "meaning": band_info['meaning'],
                "all_bands": {k: v['range'] for k, v in SCORE_BANDS.items()},
            },
            "demographic_context": {
                "income_bracket": income_ds,
                "emi_burden": emi_ds,
                "your_percentile": percentile_rank,
                "cohort_average": cohort_avg,
                "cohort_max": cohort_max,
            },
            "status_label": status,
            "trend_label": trend_label,
            # Change 3: 'trend_description' REMOVED
            "trajectory_score": round(trajectory_score, 1),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "top_factors": top_factors,
            "model_metrics": self.metrics,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pick_cohort(self, income_bracket, emi_burden):
        dc = self.df_clean
        candidates = [
            dc[(dc["income_bracket"] == income_bracket) & (dc["emi_burden"] == emi_burden)],
            dc[dc["income_bracket"] == income_bracket],
            dc[dc["emi_burden"] == emi_burden],
            dc,
        ]
        for c in candidates:
            if not c.empty:
                return c
        return dc

    @staticmethod
    def compute_trajectory(scores: dict) -> float:
        """Compute behavioral trajectory index (0-100)."""
        return min(100.0, max(0.0, sum(
            scores.get(f, 5.0) * w
            for f, w in HealthModel._TRAJECTORY_WEIGHTS.items()
        ) * 10.0))

    def _build_sample(self, scores, income, emi, trajectory_score):
        cohort = self._pick_cohort(income, emi)
        filled = {}
        for col in BEHAVIOR_COLUMNS:
            v = scores.get(col)
            filled[col] = float(v) if v is not None else float(cohort[col].median())

        emi_stress = 2 if emi == "heavy" else (1 if emi == "light" else 0)

        sample = {
            **filled,
            "projected_score_30d": trajectory_score,
            "emi_stress_index": emi_stress,
            "age_bracket": cohort["age_bracket"].mode(dropna=True)[0] if cohort["age_bracket"].notna().any() else self.mode_defaults["age_bracket"],
            "income_bracket": income,
            "occupation_type": cohort["occupation_type"].mode(dropna=True)[0] if cohort["occupation_type"].notna().any() else self.mode_defaults["occupation_type"],
            "family_size": cohort["family_size"].mode(dropna=True)[0] if cohort["family_size"].notna().any() else self.mode_defaults["family_size"],
            "city_tier": cohort["city_tier"].dropna().mode()[0] if cohort["city_tier"].notna().any() else self.mode_defaults["city_tier"],
            "emi_burden": emi,
            "financial_dependents": str(cohort["financial_dependents"].mode(dropna=True)[0]) if cohort["financial_dependents"].notna().any() else self.mode_defaults["financial_dependents"],
        }
        return sample
