# -*- coding: utf-8 -*-
"""
FinSight AI — Shared Behavioral Scoring Engine
================================================
Converts the 10 natural-language QnA answers into a 10-dimensional
behavioral vector [0.0 – 10.0] using Gemini Flash.

This module is the SINGLE source of truth for behavioral scoring.
All 4 models (Health, Waste, Goal, Clustering) use this same function.
"""

import json
import os

BEHAVIOR_COLUMNS = [
    'payday_behavior_score',
    'weekend_spend_score',
    'subscription_awareness_score',
    'impulse_control_score',
    'goal_history_score',
    'stress_response_score',
    'social_comparison_score',
    'emergency_preparedness_score',
    'future_planning_score',
    'learning_orientation_score',
]

BEHAVIOR_QUESTIONS = [
    {
        "id": 1,
        "key": "payday",
        "text": "When your salary arrives, what do you do FIRST?",
        "examples": "save it, pay bills, spend on yourself, send to family",
    },
    {
        "id": 2,
        "key": "weekend",
        "text": "How do you spend on weekends compared to weekdays?",
        "examples": "spend a lot more, roughly same, less than usual",
    },
    {
        "id": 3,
        "key": "subs",
        "text": "Do you track all your apps/subscriptions and what they cost?",
        "examples": "yes I review them, no I forget them, sometimes",
    },
    {
        "id": 4,
        "key": "impulse",
        "text": "How often do you buy things without planning beforehand?",
        "examples": "often impulsively, sometimes, almost never",
    },
    {
        "id": 5,
        "key": "goal",
        "text": "Have you ever set a savings goal and actually completed it?",
        "examples": "yes many times, once, tried but failed, never",
    },
    {
        "id": 6,
        "key": "stress",
        "text": "When you feel stressed about money, what do you do?",
        "examples": "spend to feel better, cut back on expenses, ask for help, ignore it",
    },
    {
        "id": 7,
        "key": "social",
        "text": "Do you spend more to keep up with friends or social media?",
        "examples": "yes I feel pressure, sometimes, no I don't care",
    },
    {
        "id": 8,
        "key": "emergency",
        "text": "Do you have emergency savings to cover 3+ months of expenses?",
        "examples": "yes, partially, no, working on it",
    },
    {
        "id": 9,
        "key": "future",
        "text": "Do you have a budget or financial plan for the next 6 months?",
        "examples": "yes detailed, rough idea, no plan at all",
    },
    {
        "id": 10,
        "key": "learning",
        "text": "Do you read about saving, investing, or financial tips?",
        "examples": "yes regularly, sometimes, no I don't follow these",
    },
]

# Maps the short JSON keys from Gemini to the full column names
_KEY_TO_COLUMN = {
    "payday":    "payday_behavior_score",
    "weekend":   "weekend_spend_score",
    "subs":      "subscription_awareness_score",
    "impulse":   "impulse_control_score",
    "goal":      "goal_history_score",
    "stress":    "stress_response_score",
    "social":    "social_comparison_score",
    "emergency": "emergency_preparedness_score",
    "future":    "future_planning_score",
    "learning":  "learning_orientation_score",
}


def get_default_scores() -> dict:
    """Return a dict of all 10 behavioral columns set to 5.0 (neutral)."""
    return {col: 5.0 for col in BEHAVIOR_COLUMNS}


def score_behavioral_answers(answers: dict, api_key: str | None = None) -> dict:
    """
    Convert 10 natural-language answers into numeric scores.

    Parameters
    ----------
    answers : dict
        Keys are the short question keys: "payday", "weekend", "subs",
        "impulse", "goal", "stress", "social", "emergency", "future", "learning".
        Values are the user's plain-text response strings.

    api_key : str, optional
        Gemini API key.  Falls back to GEMINI_API_KEY env var.

    Returns
    -------
    dict
        Keys are the full BEHAVIOR_COLUMNS names, values are floats 1.0–10.0.
    """
    scores = get_default_scores()

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return scores

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return scores

    prompt = f"""You are a financial behavior analyst. Rate each behavior from 1.0 (very poor) to 10.0 (excellent).

Q1  (payday - save vs spend):              "{answers.get('payday', '')}"
Q2  (weekend spending vs weekdays):         "{answers.get('weekend', '')}"
Q3  (subscription awareness & tracking):   "{answers.get('subs', '')}"
Q4  (impulse purchase control):            "{answers.get('impulse', '')}"
Q5  (goal history - setting & achieving):  "{answers.get('goal', '')}"
Q6  (stress response - healthy vs harmful): "{answers.get('stress', '')}"
Q7  (social comparison spending pressure): "{answers.get('social', '')}"
Q8  (emergency preparedness / savings):    "{answers.get('emergency', '')}"
Q9  (future planning / budgeting):         "{answers.get('future', '')}"
Q10 (learning about finance / investing):  "{answers.get('learning', '')}"

Return ONLY a valid JSON object with exactly these keys (no markdown, no explanation):
{{"payday": 0.0, "weekend": 0.0, "subs": 0.0, "impulse": 0.0, "goal": 0.0, "stress": 0.0, "social": 0.0, "emergency": 0.0, "future": 0.0, "learning": 0.0}}
"""

    try:
        response = model.generate_content(prompt)
        resp_txt = response.text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(resp_txt)
        for short_key, col_name in _KEY_TO_COLUMN.items():
            val = float(parsed.get(short_key, 5.0))
            scores[col_name] = max(1.0, min(10.0, val))
    except Exception:
        pass  # fall back to defaults

    return scores
