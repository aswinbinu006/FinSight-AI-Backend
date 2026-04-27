# -*- coding: utf-8 -*-
"""
FinSight AI — Behavioral Clustering Model (Portable)
======================================================
Stage 3 (QNA) in the website flow.

Design Philosophy:
    Unsupervised learning (K-Means) to map users into 4 peer-benchmark 
    archetypes. Uses PCA for noise reduction and StandardScaler for 
    distance-dependent logic.

Archetypes:
    0 — Steady Accumulator
    1 — Impulse Architect
    2 — Financial Explorer
    3 — Defensive Guardian

Usage:
    from finsight_models_production import ClusterModel
    model = ClusterModel()
    model.train()
    result = model.predict(behavioral_scores)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

from .behavioral_scoring import BEHAVIOR_COLUMNS

warnings.filterwarnings("ignore")


class ClusterModel:
    """Assigns users to a behavioral archetype cluster using K-Means."""

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    _DATA_FILE = "behavioral_clusters_dataset.csv"
    _MODEL_FILE = "cluster_model.joblib"

    _SCORE_COLS = BEHAVIOR_COLUMNS  # The 10 behavioral dimensions

    def __init__(self):
        self._trained = False
        self.kmeans = None
        self.scaler = None
        self.pca = None
        self.cluster_names = {}
        self.cluster_descs = {}
        self.cluster_profiles = None
        self.metrics = {}

    # ------------------------------------------------------------------
    # Training & Persistence
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Analyze the behavioral landscape and define user clusters."""
        csv_path = os.path.join(self._DATA_DIR, self._DATA_FILE)
        df = pd.read_csv(csv_path)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df_clean = df.dropna(subset=self._SCORE_COLS).copy()
        X = df_clean[self._SCORE_COLS]

        # ── Scaling (Mandatory for distance-based models) ──
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ── Dimensionality Reduction ──
        self.pca = PCA(n_components=3, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # ── K-Means with K=4 (Optimized for Finsight personas) ──
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_pca)

        # ── Performance Metrics ──
        # Silhouette score measures how well-separated our archetypes are
        sil_n = min(10000, len(X_pca))
        sil_idx = np.random.RandomState(42).choice(len(X_pca), size=sil_n, replace=False)
        sil = silhouette_score(X_pca[sil_idx], labels[sil_idx])

        # ── Cluster profiling (Grounding archetypes in data) ──
        df_profiles = df_clean.iloc[:len(labels)].copy()
        df_profiles["cluster"] = labels
        self.cluster_profiles = df_profiles.groupby("cluster")[self._SCORE_COLS].mean()

        for cid in range(4):
            if cid in self.cluster_profiles.index:
                row = self.cluster_profiles.loc[cid]
                name, desc = self._archetype_assignment(cid, row)
                self.cluster_names[cid] = name
                self.cluster_descs[cid] = desc

        self.metrics = {
            "silhouette_score": round(float(sil), 4),
            "variance_explained": round(float(np.sum(self.pca.explained_variance_ratio_)), 4)
        }
        self._trained = True
        return {"status": "trained", "archetypes": 4, **self.metrics}

    def save(self):
        """Persist the trained model state."""
        assert self._trained, "Cannot save an untrained model."
        data = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "cluster_names": self.cluster_names,
            "cluster_descs": self.cluster_descs,
            "cluster_profiles": self.cluster_profiles,
            "metrics": self.metrics
        }
        joblib.dump(data, os.path.join(self._DATA_DIR, self._MODEL_FILE))

    def load(self):
        """Load a previously trained model state."""
        data = joblib.load(os.path.join(self._DATA_DIR, self._MODEL_FILE))
        self.kmeans = data["kmeans"]
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.cluster_names = data["cluster_names"]
        self.cluster_descs = data["cluster_descs"]
        self.cluster_profiles = data["cluster_profiles"]
        self.metrics = data["metrics"]
        self._trained = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, behavioral_scores: dict) -> dict:
        """Assign user to a behavioral cluster."""
        assert self._trained, "Call .train() or .load() first."

        # Clamp AI inputs mathematically before scoring
        clamped_scores = {k: max(1.0, min(10.0, float(behavioral_scores.get(k, 5.0)))) for k in self._SCORE_COLS}
        user_df = pd.DataFrame([clamped_scores])

        scaled = self.scaler.transform(user_df)
        pca_pt = self.pca.transform(scaled)
        cluster_id = int(self.kmeans.predict(pca_pt)[0])

        cluster_avgs = {}
        if cluster_id in self.cluster_profiles.index:
            cluster_avgs = self.cluster_profiles.loc[cluster_id].to_dict()
            cluster_avgs = {k: round(float(v), 2) for k, v in cluster_avgs.items()}

        return {
            "cluster_id": cluster_id,
            "cluster_name": self.cluster_names.get(cluster_id, f"Archetype {cluster_id}"),
            "cluster_description": self.cluster_descs.get(cluster_id, ""),
            "cluster_averages": cluster_avgs,
            "user_scores": {k: round(clamped_scores[k], 1) for k in self._SCORE_COLS},
            "metrics": self.metrics
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _archetype_assignment(cid, profile_row):
        """Map abstract cluster IDs to human-centric archetypes."""
        # Feature-based labeling rather than hardcoded IDs
        planning = profile_row[["goal_history_score", "future_planning_score"]].mean()
        discipline = profile_row[["impulse_control_score", "subscription_awareness_score"]].mean()
        social = profile_row["social_comparison_score"]

        if planning >= 6.5 and discipline >= 6.5:
            return "Steady Accumulator", "Characterized by high discipline and strategic long-term planning."
        if social >= 6.0 and discipline < 5.0:
            return "Impulse Explorer", "Spending reflects social pressures; high potential for wealth leakage."
        if discipline >= 7.0 and planning < 5.0:
            return "Defensive Guardian", "Strong at saving but lacks a clear roadmap for long-term growth."
        return "Developing Strategist", "A balanced profile in transition, building foundational financial habits."
