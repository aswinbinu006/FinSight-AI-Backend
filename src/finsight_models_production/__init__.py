"""
FinSight AI — Portable Model Package (Production)
====================================================
Drop this folder into your Flask/website project.
Import like:
    from finsight_models_production import HealthModel, WasteModel, GoalModel, ClusterModel

Each model exposes:
    .train()            → trains on the bundled CSV dataset
    .predict(inputs)    → returns a structured result dict

v2 Changes (from Prompt Reference doc):
    - HealthModel: score/trend reconciliation, removed hardcoded descriptions
    - WasteModel: >100 score fix, removed _get_reasoning(), answer trace cols
    - GoalModel: reduced overfitting, leaky targets applied, goal_description
    - NEW: explainer.py — Gemini explanation layer + output validators
"""

from finsight_models_production.health_model import HealthModel
from finsight_models_production.waste_model import WasteModel
from finsight_models_production.goal_model import GoalModel
from finsight_models_production.cluster_model import ClusterModel
from finsight_models_production.behavioral_scoring import score_behavioral_answers
from finsight_models_production.explainer import (
    explain_health, explain_waste, explain_goal,
    validate_health_output, validate_waste_output, validate_goal_output,
    check_health_thresholds, check_waste_thresholds, check_goal_thresholds,
    chat_with_copilot
)


__all__ = [
    "HealthModel",
    "WasteModel",
    "GoalModel",
    "ClusterModel",
    "score_behavioral_answers",
    "explain_health",
    "explain_waste",
    "explain_goal",
    "validate_health_output",
    "validate_waste_output",
    "validate_goal_output",
    "check_health_thresholds",
    "check_waste_thresholds",
    "check_goal_thresholds",
    "chat_with_copilot"
]
