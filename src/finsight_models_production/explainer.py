# -*- coding: utf-8 -*-
"""
FinSight AI — Gemini Explanation Layer
=======================================
Sits between the raw ML model output and the frontend.
Each prompt receives the raw ML output dict and returns
a user-facing explanation via Gemini.

Also includes per-prediction output range validators (Section 8.3).

Usage:
    from finsight_models_production.explainer import (
        explain_health, explain_waste, explain_goal,
        validate_health_output, validate_waste_output, validate_goal_output,
    )
"""

import os
import json


# =====================================================================
# Section 8.1 — Metric Thresholds (run after train())
# =====================================================================
HEALTH_THRESHOLDS = {
    'score_rmse':      8.0,   # RMSE above 8 on a 0-100 scale = broken
    'score_r2':        0.85,  # below 0.85 = weak predictive power
    'trend_f1_macro':  0.55,  # macro F1 is harder to fake
}

WASTE_THRESHOLDS = {
    'rmse':  6.0,
    'r2':    0.90,
}

GOAL_THRESHOLDS = {
    'success_accuracy': 0.80,
    'success_f1':       0.75,  # F1 matters — classes can be imbalanced
    'risk_accuracy':    0.75,
}


def check_health_thresholds(metrics: dict) -> dict:
    """Check health model metrics against thresholds. Returns failures."""
    failures = {}
    if metrics.get('score_rmse', 999) > HEALTH_THRESHOLDS['score_rmse']:
        failures['score_rmse'] = f"{metrics['score_rmse']} > {HEALTH_THRESHOLDS['score_rmse']}"
    if metrics.get('score_r2', 0) < HEALTH_THRESHOLDS['score_r2']:
        failures['score_r2'] = f"{metrics['score_r2']} < {HEALTH_THRESHOLDS['score_r2']}"
    if metrics.get('trend_f1_macro', 0) < HEALTH_THRESHOLDS['trend_f1_macro']:
        failures['trend_f1_macro'] = f"{metrics['trend_f1_macro']} < {HEALTH_THRESHOLDS['trend_f1_macro']}"
    return failures


def check_waste_thresholds(metrics: dict) -> dict:
    """Check waste model metrics against thresholds."""
    failures = {}
    if metrics.get('rmse', 999) > WASTE_THRESHOLDS['rmse']:
        failures['rmse'] = f"{metrics['rmse']} > {WASTE_THRESHOLDS['rmse']}"
    if metrics.get('r2', 0) < WASTE_THRESHOLDS['r2']:
        failures['r2'] = f"{metrics['r2']} < {WASTE_THRESHOLDS['r2']}"
    return failures


def check_goal_thresholds(metrics: dict) -> dict:
    """Check goal model metrics against thresholds."""
    failures = {}
    if metrics.get('success_accuracy', 0) < GOAL_THRESHOLDS['success_accuracy']:
        failures['success_accuracy'] = f"{metrics['success_accuracy']} < {GOAL_THRESHOLDS['success_accuracy']}"
    if metrics.get('success_f1', 0) < GOAL_THRESHOLDS['success_f1']:
        failures['success_f1'] = f"{metrics['success_f1']} < {GOAL_THRESHOLDS['success_f1']}"
    if metrics.get('risk_accuracy', 0) < GOAL_THRESHOLDS['risk_accuracy']:
        failures['risk_accuracy'] = f"{metrics['risk_accuracy']} < {GOAL_THRESHOLDS['risk_accuracy']}"
    return failures


# =====================================================================
# Section 8.3 — Per-Prediction Output Range Checks
# =====================================================================
VALID_LABELS = {'Storm Warning', 'Rainy', 'Partly Cloudy', 'Improving', 'Sunny'}


def validate_health_output(result: dict) -> dict:
    """Validate health prediction output before returning to frontend."""
    assert 0 <= result['health_score'] <= 100, \
        f"Health score {result['health_score']} out of range [0, 100]"
    assert result['status_label'] in VALID_LABELS, \
        f"Invalid status_label: {result['status_label']}"
    assert result['trend_label'] in VALID_LABELS, \
        f"Invalid trend_label: {result['trend_label']}"
    # New: validate score band
    valid_bands = {'Excellent', 'Good', 'Average', 'Poor', 'Critical'}
    if 'score_band' in result:
        assert result['score_band'] in valid_bands, \
            f"Invalid score_band: {result['score_band']}"
    if 'demographic_context' in result:
        ctx = result['demographic_context']
        assert 0 <= ctx['your_percentile'] <= 100, \
            f"Percentile {ctx['your_percentile']} out of range"
    return result


def validate_waste_output(result: dict) -> dict:
    """Validate waste prediction output before returning to frontend."""
    assert 0 <= result['waste_score'] <= 100, \
        f"Waste score {result['waste_score']} out of range"
    valid_bands = {'Optimized', 'Controlled', 'Moderate Waste', 'High Waste', 'Critical Waste'}
    if 'score_band' in result:
        assert result['score_band'] in valid_bands, \
            f"Invalid score_band: {result['score_band']}"
    return result


def validate_goal_output(result: dict) -> dict:
    """Validate goal prediction output before returning to frontend."""
    assert 0 <= result['success_probability'] <= 100, \
        f"Success probability {result['success_probability']} out of range"
    assert isinstance(result['is_on_track'], bool), \
        f"is_on_track must be bool, got {type(result['is_on_track'])}"
    valid_bands = {'Impossible', 'High Risk', 'Uncertain', 'Likely', 'On Track'}
    if 'score_band' in result:
        assert result['score_band'] in valid_bands, \
            f"Invalid score_band: {result['score_band']}"
    return result


# =====================================================================
# Section 7 — Gemini Explanation Prompts
# =====================================================================

HEALTH_PROMPT_TEMPLATE = """You are FinSight AI's financial health advisor. A user's 10 behavioral
dimensions were just scored by machine learning. Write a SHORT personal summary.
Think of yourself as a straight-talking financial doctor reading lab results.

USER DATA:
- Health Score: {health_score}/100
- Score Band: {score_band}
  Bands: Excellent (70-100) | Good (55-69) | Average (40-54) | Poor (25-39) | Critical (0-24)
- Status: {status_label}
  Scale: Storm Warning - Rainy - Partly Cloudy - Improving - Sunny
- 30-day Trend: {trend_label}
- Top Strength: {strength_name} (score {strength_value}/10)
- Top Weakness: {weakness_name} (score {weakness_value}/10)
- Income bracket: {income_bracket}
- EMI burden: {emi_burden}
- Demographic percentile: Top {percentile_display}% among similar users
- Biggest ML factor: {top_factor}

WRITE EXACTLY THIS STRUCTURE - no headers, plain paragraphs:

Paragraph 1 (2 sentences): What their score means. Mention their band
({score_band}) and where it sits on the 5-band scale. If they scored
70+, emphasize this is Excellent territory - the top band.

Paragraph 2 (2 sentences): Why they got this score. Name their
specific strength and weakness. Be direct, not generic.

Paragraph 3 (1-2 sentences): One concrete action they can take
THIS WEEK based on their top weakness. Give a specific number or
step if possible. Never say "optimize" or "consider adjusting".

RULES:
- Under 120 words total
- Never start with "Your financial habits are"
- If EMI is heavy, acknowledge the EMI constraint specifically
- If score_band is Critical, be honest but not alarming
- If score_band is Excellent, be encouraging but not sycophantic
- Always mention their percentile rank to give demographic context
"""


WASTE_PROMPT_TEMPLATE = """You are FinSight AI's subscription analyst. A user's subscriptions
were just analyzed. Write a SHORT personal summary.
Be brutally honest but constructive — like a financially savvy friend.

USER DATA:
- Number of subscriptions analyzed: {sub_count}
- Total annual spend: ₹{annual_total}
- Recoverable annual waste: ₹{recoverable}
- FinSight Score: {finsight_score}/100 (higher = less waste)
- Worst subscription: {worst_name}
  waste score {worst_score}/100, used {worst_usage}% of the time
  waste band: {worst_band}
- Best subscription: {best_name} (waste score {best_score}/100)

WRITE EXACTLY THIS STRUCTURE:

Sentence 1: The headline number. How much they're leaking per year.
Be specific — use the ₹ amount directly.

Sentence 2: Name the worst offender and its waste band ({worst_band}). State why it's a problem
(low usage, forgotten charge, or high price for no value).

Sentence 3: The one action to take right now — cancel, downgrade,
or set a monthly review reminder. Be specific, not vague.

RULES:
- Under 90 words total
- Use ₹ symbol for all amounts
- Never say "optimize your subscriptions"
- If finsight_score > 75, acknowledge they're doing well before
  suggesting the one remaining improvement
- If recoverable < ₹500, say they have minimal waste
"""

GOAL_PROMPT_TEMPLATE = """You are FinSight AI's goal coach. A user's savings goal was analyzed.
Be direct and honest. If they're off track, say so clearly.
If they're on track, say that clearly too. No vague encouragement.

USER DATA:
- Goal: {goal_description}
- Target amount: ₹{goal_amount}
- Saved so far: ₹{saved_so_far} ({saved_pct}% complete)
- Monthly savings: ₹{monthly_savings}
- Months remaining: {months_remaining}
- Months required at current pace: {months_required}
- Success probability: {success_probability}%
- Score Band: {score_band}
- Biggest behavioral risk: {risk_factor}
- On track mathematically: {is_on_track}

WRITE EXACTLY THIS STRUCTURE:

Sentence 1: Are they going to make it? State the probability, their band
({score_band}) and what it means in plain words. Reference the actual goal name.

Sentence 2: The math — ahead or behind? By how many months or
how many rupees per month are they short?

Sentence 3: Their biggest behavioral risk by name. Explain why
that specific behavior threatens THIS goal, not goals in general.

Sentence 4: One specific action with a number. Example:
"Move ₹2,000 more per month into a separate savings account".
Never say "stay the course" or "adjust your rate" without a number.

RULES:
- Under 110 words total
- Use ₹ for amounts
- If probability < 30%, be honest — do not soften it
- Always reference the goal by its name from {goal_description}
- If mathematically impossible, say so directly
"""


def _call_gemini(prompt: str, api_key: str | None = None) -> str:
    """Call Gemini and return the text response. Falls back gracefully."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[Gemini API key not set — raw output returned]"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini explanation unavailable: {e}]"


def explain_health(result: dict, api_key: str | None = None) -> str:
    """Generate a Gemini explanation for a health prediction result."""
    result = validate_health_output(result)

    strengths = result.get('strengths', [])
    weaknesses = result.get('weaknesses', [])
    top_factors = result.get('top_factors', [])

    prompt = HEALTH_PROMPT_TEMPLATE.format(
        health_score=result['health_score'],
        score_band=result.get('score_band', 'Fair'),
        status_label=result['status_label'],
        trend_label=result['trend_label'],
        strength_name=strengths[-1]['name'] if strengths else 'N/A',
        strength_value=strengths[-1]['value'] if strengths else 5.0,
        weakness_name=weaknesses[0]['name'] if weaknesses else 'N/A',
        weakness_value=weaknesses[0]['value'] if weaknesses else 5.0,
        income_bracket=result.get('demographic_context', {}).get('income_bracket', result.get('income_bracket', 'unknown')),
        emi_burden=result.get('demographic_context', {}).get('emi_burden', result.get('emi_burden', 'unknown')),
        percentile_display=round(100 - result.get('demographic_context', {}).get('your_percentile', 50), 1),
        top_factor=top_factors[0]['feature'] if top_factors else 'N/A',
    )
    return _call_gemini(prompt, api_key)


def explain_waste(bulk_result: dict, api_key: str | None = None) -> str:
    """Generate a Gemini explanation for a waste bulk prediction result."""
    subs = bulk_result.get('subscriptions', [])
    if not subs:
        return "No subscriptions to analyze."

    # Validate each sub
    for s in subs:
        validate_waste_output(s)

    worst = max(subs, key=lambda s: s['waste_score'])
    best = min(subs, key=lambda s: s['waste_score'])

    prompt = WASTE_PROMPT_TEMPLATE.format(
        sub_count=len(subs),
        annual_total=f"{bulk_result['annual_total']:,.0f}",
        recoverable=f"{bulk_result['recoverable']:,.0f}",
        finsight_score=bulk_result['finsight_score'],
        worst_name=worst['name'],
        worst_score=worst['waste_score'],
        worst_usage=worst['usage_pct'],
        worst_band=worst.get('score_band', 'Moderate Waste'),
        best_name=best['name'],
        best_score=best['waste_score'],
    )
    return _call_gemini(prompt, api_key)


def explain_goal(result: dict, api_key: str | None = None) -> str:
    """Generate a Gemini explanation for a goal prediction result."""
    result = validate_goal_output(result)

    summary = result.get('target_summary', {})
    prompt = GOAL_PROMPT_TEMPLATE.format(
        goal_description=result.get('goal_description', 'your goal'),
        goal_amount=f"{summary.get('goal_amount', 0):,.0f}",
        saved_so_far=f"{summary.get('saved_so_far', 0):,.0f}",
        saved_pct=summary.get('saved_pct', 0),
        monthly_savings=f"{summary.get('monthly_savings', 0):,.0f}",
        months_remaining=summary.get('months_remaining', 0),
        months_required=summary.get('months_required_at_current_pace', 0),
        success_probability=result['success_probability'],
        score_band=result.get('score_band', 'Uncertain'),
        risk_factor=result.get('risk_factor', 'unknown'),
        is_on_track=result['is_on_track'],
    )
    return _call_gemini(prompt, api_key)


COPILOT_SYSTEM_PROMPT = """You are the FinSight AI Co-Pilot, an institutional-grade financial intelligence assistant.
Your goal is to provide deep, contextual insights based on the user's current financial models (Health, Waste, Goal, and Behavioral).

CONSTRAINTS:
- Be concise but extremely insightful.
- Use the provided context (Health Score, Waste Analysis, Goal Status) to back up your claims.
- If a user asks a generic question, relate it back to their specific data.
- Tone: Professional, slightly editorial (like The Economist), but highly accessible.
- Never give generic advice like "save more money" without looking at their specific waste model or goal probability.
- If they are in a "Critical" health band, be direct about the risks.

CONTEXT INJECTION:
{context_summary}

The user's query: {message}
"""


def chat_with_copilot(message: str, context: dict, api_key: str | None = None) -> str:
    """Natural language interface to the FinSight AI suite using Gemini."""
    # Convert context dict into a readable summary for the prompt
    context_lines = []
    if "health" in context:
        h = context["health"]
        context_lines.append(f"- Health: {h.get('score', 'N/A')}/100 ({h.get('band', 'N/A')}), Status: {h.get('status', 'N/A')}")
    if "waste" in context:
        w = context["waste"]
        context_lines.append(f"- Waste: {w.get('total_waste', 'N/A')} recoverable, Subscriptions: {w.get('sub_count', 0)}")
    if "goal" in context:
        g = context["goal"]
        context_lines.append(f"- Goal: {g.get('name', 'N/A')} is {g.get('probability', 'N/A')}% likely ({g.get('band', 'N/A')})")
    
    summary = "\n".join(context_lines) if context_lines else "No specific model data available yet."
    
    prompt = COPILOT_SYSTEM_PROMPT.format(
        context_summary=summary,
        message=message
    )
    
    return _call_gemini(prompt, api_key)
