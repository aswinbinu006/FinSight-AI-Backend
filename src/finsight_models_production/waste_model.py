# -*- coding: utf-8 -*-
"""
FinSight AI — Waste Recovery Model (Portable) v2
==================================================
Stage 5 in the website flow.

Design Philosophy:
    Waste detection uses XGBoost to learn mapping between usage patterns
    and financial efficiency. Domain rules are explicitly encoded as
    numerical scores (0-10) before reaching the ML model.

Fixes applied (v2 — from Prompt Reference doc):
    1. Fix >100 score bug — filter training data to [0, 100]
    2. Remove _get_reasoning() — Gemini explainer handles text
    3. Add answer columns to drop list — raw string trace cols excluded
    4. Update dataset filename to v6
    (Previous fixes remain intact)

Usage:
    from finsight_models_production import WasteModel
    model = WasteModel()
    model.train()
    result = model.predict_subscription({...})
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import json
import joblib

warnings.filterwarnings("ignore")


class WasteModel:
    """Subscription waste-score predictor and multi-sub wealth report."""

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    # Change 4: Updated dataset filename
    _DATA_FILE = "waste_recovery_dataset_v6.csv"

    _CAT_FEATURES = ["subscription_name", "billing_cycle"]
    _NUM_FEATURES = ["amount_paid", "usage_ratio", "awareness_score", "necessity_score", "has_perks"]

    # Change 3: Answer trace columns — NOT model features, drop before training
    _ANSWER_TRACE_COLS = ['usage_frequency_answer', 'awareness_answer', 'necessity_answer']

    def __init__(self):
        self._trained = False
        self.xgb_model = None
        self.best_algo = ""
        self.encoder = None
        self.feature_names = []
        self.metrics = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Train the waste score regression model."""
        csv_path = os.path.join(self._DATA_DIR, self._DATA_FILE)
        df = pd.read_csv(csv_path)

        # Change 1: Fix >100 score bug at training time — kill at source
        df = df[df['waste_score'].between(0, 100)].copy()

        # Change 3: Drop answer trace columns (raw strings, not features)
        df = df.drop(columns=[c for c in self._ANSWER_TRACE_COLS if c in df.columns])

        # ── Categorical encoding ──
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.encoder.fit_transform(df[self._CAT_FEATURES])
        cat_feature_names = self.encoder.get_feature_names_out(self._CAT_FEATURES).tolist()

        # ── Numeric features (NO scaling for XGBoost) ──
        X_num = df[self._NUM_FEATURES].values

        X = np.hstack([X_num, X_cat])
        self.feature_names = self._NUM_FEATURES + cat_feature_names
        y = df["waste_score"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)

        regressors = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=1000, max_depth=6,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3, gamma=0.1,
                n_jobs=-1, random_state=42,
            )
        }

        best_r2 = -9999
        best_reg = None
        best_reg_name = ""
        best_rmse = 9999
        best_mae = 9999
        score_evaluations = {}

        for name, model in regressors.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            r2 = float(r2_score(y_te, preds))
            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            mae = float(mean_absolute_error(y_te, preds))
            
            score_evaluations[name] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}
            
            if r2 > best_r2:
                best_r2 = r2
                best_rmse = rmse
                best_mae = mae
                best_reg = model
                best_reg_name = name

        self.xgb_model = best_reg
        self.best_algo = best_reg_name

        self.metrics = {
            "best_model": self.best_algo,
            "all_models_performance": score_evaluations,
            "mae": round(best_mae, 4),
            "rmse": round(best_rmse, 4),
            "r2": round(best_r2, 4),
        }

        self._trained = True
        return {"status": "trained", "rows": len(df), **self.metrics}

    # ------------------------------------------------------------------
    # Domain Logic Encoders (Explicit Rules)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_usage(raw: str) -> float:
        """Convert human usage description to 0.0–1.0 ratio."""
        raw = str(raw).strip().lower()
        try:
            if "/" in raw:
                parts = raw.split("/")
                nums = "".join(c for c in parts[0] if c.isdigit() or c == ".")
                val = float(nums) if nums else 0.5
                if "week" in raw: return val / 7.0
                if "month" in raw: return val / 30.0
                if "year" in raw: return (val * 30) / 365.0
            if "daily" in raw: return 1.0
            if "rare" in raw: return 0.05
            if "can" in raw and "remember" in raw: return 0.03
            nums = "".join(c for c in raw if c.isdigit() or c == ".")
            val = float(nums) if nums else 0.5
            return val / 30.0 if val > 1.1 else val
        except Exception:
            return 0.5

    @staticmethod
    def _parse_awareness(raw: str) -> int:
        raw = str(raw).strip().lower()
        if "forget" in raw or "can't" in raw: return 2
        if "sometimes" in raw: return 6
        if "remember" in raw or "yes" in raw: return 10
        return 5

    @staticmethod
    def _parse_necessity(raw: str) -> int:
        raw = str(raw).strip().lower()
        if "fun" in raw: return 2
        if "sometimes" in raw: return 6
        if "essential" in raw or "must" in raw or "need" in raw: return 10
        return 5

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_subscription(self, subscription: dict) -> dict:
        """Predict waste score for a single subscription."""
        assert self._trained, "Call .train() first or .load() a saved model."

        name = subscription.get("name", "Other")
        bill = subscription.get("billing_cycle", "monthly").lower()
        price = float(subscription.get("price", 100))
        usage = self._parse_usage(subscription.get("usage_frequency", "sometimes"))
        aware = max(1.0, min(10.0, float(self._parse_awareness(subscription.get("awareness", "sometimes")))))
        neces = max(1.0, min(10.0, float(self._parse_necessity(subscription.get("necessity", "sometimes")))))
        perks = 1 if subscription.get("has_perks", False) else 0

        m_price = price / 12 if "year" in bill else price

        # Preprocessing
        cat_df = pd.DataFrame([{
            "subscription_name": name,
            "billing_cycle": "monthly" if "month" in bill else "yearly"
        }])
        X_cat = self.encoder.transform(cat_df)
        X_num = np.array([[m_price, usage, aware, neces, perks]])

        X_pred = np.hstack([X_num, X_cat])

        # Prediction
        score = float(self.xgb_model.predict(X_pred)[0])
        score = max(0.0, min(100.0, score))

        # ── Score Interpretation Bands ─────────────────────────────────
        # Based on user-provided spec. High score = bad.
        SCORE_BANDS = {
            'Optimized':      {'range': '0 - 20',   'min': 0,  'meaning': 'No waste'},
            'Controlled':     {'range': '21 - 40',  'min': 21, 'meaning': 'Minor inefficiencies'},
            'Moderate Waste': {'range': '41 - 60',  'min': 41, 'meaning': 'Noticeable leakage'},
            'High Waste':     {'range': '61 - 80',  'min': 61, 'meaning': 'Serious inefficiency'},
            'Critical Waste': {'range': '81 - 100', 'min': 81, 'meaning': 'Major financial drain'},
        }

        if score <= 20:   score_band = 'Optimized'
        elif score <= 40: score_band = 'Controlled'
        elif score <= 60: score_band = 'Moderate Waste'
        elif score <= 80: score_band = 'High Waste'
        else:             score_band = 'Critical Waste'

        band_info = SCORE_BANDS[score_band]
        status = score_band # keep status key working for older clients

        # Feature Importance
        if hasattr(self.xgb_model, 'feature_importances_'):
            importances = self.xgb_model.feature_importances_
        elif hasattr(self.xgb_model, 'coef_'):
            importances = np.abs(self.xgb_model.coef_)
        else:
            importances = np.zeros(len(self.feature_names))
            
        feat_imp = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        top_factors = [
            {"feature": name.replace("_", " ").title(), "importance": round(float(imp), 4)}
            for name, imp in feat_imp[:3]
        ]

        # Change 2: 'advisor_reasoning' REMOVED from return dict
        return {
            "name": name,
            "monthly_price": round(m_price, 2),
            "yearly_cost": round(m_price * 12, 2),
            "waste_score": round(score, 1),
            "score_band": score_band,
            "score_interpretation": {
                "band": score_band,
                "range": band_info['range'],
                "meaning": band_info['meaning'],
                "all_bands": {k: v['range'] for k, v in SCORE_BANDS.items()},
            },
            "usage_pct": round(usage * 100, 1),
            "status": status,
            "top_drivers": top_factors,
        }

    def predict_bulk(self, subscriptions: list[dict]) -> dict:
        """Predict for multiple subscriptions + generate a wealth report."""
        results = [self.predict_subscription(s) for s in subscriptions]

        annual_total = sum(r["yearly_cost"] for r in results)
        recoverable = sum(r["yearly_cost"] for r in results if r["waste_score"] > 55)
        avg_waste = sum(r["waste_score"] for r in results) / max(len(results), 1)
        finsight_score = round(100 - avg_waste, 1)

        return {
            "subscriptions": results,
            "annual_total": round(annual_total, 2),
            "recoverable": round(recoverable, 2),
            "finsight_score": finsight_score,
            "model_metrics": self.metrics,
        }

    def save(self, filepath: str):
        """Serialize the trained model."""
        assert self._trained, "Call .train() before saving"
        joblib.dump(self.__dict__, filepath)

    def load(self, filepath: str):
        """Load a serialized model from disk."""
        state = joblib.load(filepath)
        self.__dict__.update(state)
        self._trained = True
