# -*- coding: utf-8 -*-
"""
FinSight AI — Dataset Generator (All Models)
==============================================
Generates production-scale synthetic datasets for all 4 models.
Run with: python generate_datasets.py --all
Or individually: --health, --waste, --goal, --cluster

Output files (to data/ directory):
    health_score_dataset_v3.csv       150,000 rows
    waste_recovery_dataset_v6.csv      80,000 rows
    goal_intelligence_dataset_v2.csv  150,000 rows
    behavioral_clusters_dataset.csv   120,000 rows (unchanged schema)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# =====================================================================
# HEALTH DATASET v3  (150,000 rows)
# =====================================================================
def generate_health_v3(n_rows=150000):
    print(f"\n{'='*60}")
    print(f"  Generating Health Dataset v3 ({n_rows:,} rows)")
    print(f"{'='*60}")

    age_brackets = ['18-22', '23-27', '28-35', '36-45', '45+']
    income_brackets = ['below_20k', '20k-50k', '50k-100k', '100k+']
    occupation_types = ['student', 'salaried', 'freelancer', 'self_employed']
    family_sizes = ['1', '2-3', '4-5', '6+']
    city_tiers = ['tier_1', 'tier_2', 'tier_3']
    emi_burdens = ['none', 'light', 'heavy']
    financial_deps = ['0', '1', '2', '3+']

    data = []
    for _ in range(n_rows):
        # Correlated behavioral scores
        base_discipline = np.random.beta(2, 2) * 10
        noise = lambda: np.random.normal(0, 1.5)

        payday    = np.clip(base_discipline + noise(), 1, 10)
        weekend   = np.clip(base_discipline + noise(), 1, 10)
        subs      = np.clip(base_discipline + noise(), 1, 10)
        impulse   = np.clip(base_discipline + noise(), 1, 10)
        goal      = np.clip(base_discipline + noise(), 1, 10)
        stress    = np.clip(base_discipline + noise(), 1, 10)
        social    = np.clip(base_discipline + noise(), 1, 10)
        emergency = np.clip(base_discipline + noise(), 1, 10)
        future    = np.clip(base_discipline + noise(), 1, 10)
        learning  = np.clip(base_discipline + noise(), 1, 10)

        age = np.random.choice(age_brackets)
        income = np.random.choice(income_brackets, p=[0.2, 0.35, 0.3, 0.15])
        occupation = np.random.choice(occupation_types)
        family = np.random.choice(family_sizes)
        city = np.random.choice(city_tiers)
        emi = np.random.choice(emi_burdens, p=[0.3, 0.4, 0.3])
        deps = np.random.choice(financial_deps)

        # Formula: 70% behavioral + 30% financial context
        behavior_avg = (payday + weekend + subs + impulse + goal +
                        stress + social + emergency + future + learning) / 10.0
        behavioral_component = (behavior_avg / 10.0) * 70

        income_map = {'below_20k': 2, '20k-50k': 7, '50k-100k': 12, '100k+': 15}
        income_effect = income_map[income]
        emi_map = {'none': 0, 'light': -5, 'heavy': -15}
        emi_effect = emi_map[emi]
        financial_component = max(0, min(30, 15 + income_effect + emi_effect))

        raw_score = behavioral_component + financial_component
        projected_30d = np.clip(raw_score + np.random.normal(0, 8), 0, 100)
        health_score = raw_score * 0.85 + projected_30d * 0.15
        health_score += np.random.normal(0, 2.0)
        health_score = round(np.clip(health_score, 0, 100), 1)

        # Trend labels — 90%+ deterministic for classifier accuracy
        if health_score >= 75:
            health_band = 'Excellent'
            trend_label = np.random.choice(['Sunny', 'Improving'], p=[0.92, 0.08])
        elif health_score >= 55:
            health_band = 'Good'
            trend_label = np.random.choice(['Improving', 'Sunny', 'Partly Cloudy'], p=[0.88, 0.07, 0.05])
        elif health_score >= 40:
            health_band = 'Fair'
            trend_label = np.random.choice(['Partly Cloudy', 'Improving', 'Rainy'], p=[0.88, 0.07, 0.05])
        elif health_score >= 25:
            health_band = 'Poor'
            trend_label = np.random.choice(['Rainy', 'Partly Cloudy', 'Storm Warning'], p=[0.88, 0.07, 0.05])
        else:
            health_band = 'Poor'
            trend_label = np.random.choice(['Storm Warning', 'Rainy'], p=[0.92, 0.08])

        data.append({
            'age_bracket': age, 'income_bracket': income,
            'occupation_type': occupation, 'family_size': family,
            'city_tier': city, 'emi_burden': emi, 'financial_dependents': deps,
            'payday_behavior_score': round(payday, 2),
            'weekend_spend_score': round(weekend, 2),
            'subscription_awareness_score': round(subs, 2),
            'impulse_control_score': round(impulse, 2),
            'goal_history_score': round(goal, 2),
            'stress_response_score': round(stress, 2),
            'social_comparison_score': round(social, 2),
            'emergency_preparedness_score': round(emergency, 2),
            'future_planning_score': round(future, 2),
            'learning_orientation_score': round(learning, 2),
            'health_score': health_score,
            'health_band': health_band,
            'trend_label': trend_label,
            'projected_score_30d': round(projected_30d, 1),
        })

    df = pd.DataFrame(data)
    out = os.path.join(DATA_DIR, 'health_score_dataset_v3.csv')
    df.to_csv(out, index=False)
    print(f"  -> {out}")
    print(f"     Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")
    print(f"     Score range: [{df['health_score'].min():.1f}, {df['health_score'].max():.1f}]")
    print(f"     Trend dist: {dict(df['trend_label'].value_counts())}")
    return df


# =====================================================================
# WASTE DATASET v6  (80,000 rows)
# =====================================================================
def generate_waste_v6(n_rows=80000):
    print(f"\n{'='*60}")
    print(f"  Generating Waste Dataset v6 ({n_rows:,} rows)")
    print(f"{'='*60}")

    names = ['Netflix', 'Spotify', 'Amazon Prime', 'Disney+', 'Gym Membership',
             'Newspaper', 'Cloud Storage', 'Dating App', 'Software License',
             'Gaming Pass', 'Educational Hub', 'Wine Club', 'Zomato Gold',
             'YouTube Premium', 'LinkedIn Premium', 'Adobe CC']

    usage_answers = ['daily', 'few_times_week', 'once_twice_week',
                     'few_times_month', 'rarely', 'cant_remember']
    awareness_answers = ['always_notice', 'sometimes_notice', 'forget']
    necessity_answers = ['essential', 'nice_to_have', 'fun']

    data = []
    for _ in range(n_rows):
        sub_name = np.random.choice(names)
        billing = np.random.choice(['monthly', 'yearly'])
        amount = np.random.uniform(100, 5000)
        usage_ratio = np.random.uniform(0, 1)
        awareness = np.random.randint(1, 11)
        necessity = np.random.randint(1, 11)
        has_perks = np.random.choice([0, 1], p=[0.7, 0.3])

        # v6: answer trace columns (strings for traceability, NOT features)
        usage_freq_answer = np.random.choice(usage_answers)
        awareness_answer = np.random.choice(awareness_answers)
        necessity_answer = np.random.choice(necessity_answers)

        # Waste formula (usage is king)
        base_waste_pct = (1.0 - usage_ratio) * 100
        necessity_mult = 1.0 - ((necessity - 1) / 9.0 * 0.4)
        awareness_penalty = (10 - awareness) * 2.0
        final_score = (base_waste_pct * necessity_mult) + awareness_penalty

        if has_perks == 1:
            final_score *= 0.85

        if usage_ratio > 0.5:
            final_score = min(final_score, 45)
        if usage_ratio < 0.1:
            final_score = max(final_score, 60)

        final_score += np.random.normal(0, 1.5)
        final_score = round(np.clip(final_score, 0, 100), 1)

        data.append({
            'subscription_name': sub_name, 'billing_cycle': billing,
            'amount_paid': round(amount, 2), 'usage_ratio': round(usage_ratio, 3),
            'awareness_score': awareness, 'necessity_score': necessity,
            'has_perks': has_perks, 'waste_score': final_score,
            # v6 trace columns (will be dropped by model before training)
            'usage_frequency_answer': usage_freq_answer,
            'awareness_answer': awareness_answer,
            'necessity_answer': necessity_answer,
        })

    df = pd.DataFrame(data)
    out = os.path.join(DATA_DIR, 'waste_recovery_dataset_v6.csv')
    df.to_csv(out, index=False)
    print(f"  -> {out}")
    print(f"     Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")
    print(f"     Score range: [{df['waste_score'].min():.1f}, {df['waste_score'].max():.1f}]")
    print(f"     Out-of-range (>100): {(df['waste_score'] > 100).sum()}")
    return df


# =====================================================================
# GOAL DATASET v2  (150,000 rows)
# =====================================================================
def generate_goal_v2(n_rows=150000):
    print(f"\n{'='*60}")
    print(f"  Generating Goal Dataset v2 ({n_rows:,} rows)")
    print(f"{'='*60}")

    age_brackets = ['18-22', '23-27', '28-35', '36-45', '45+']
    income_brackets = ['below_20k', '20k-50k', '50k-100k', '100k+']
    occupation_types = ['student', 'salaried', 'freelancer', 'self_employed']
    family_sizes = ['1', '2-3', '4-5', '6+']
    city_tiers = ['tier_1', 'tier_2', 'tier_3']
    emi_burdens = ['none', 'light', 'heavy']
    financial_deps = ['0', '1', '2', '3+']
    goal_amount_brackets = ['under_10k', '10k-50k', '50k-200k', '200k-500k', 'above_500k']
    savings_brackets = ['under_2k', '2k-5k', '5k-15k', '15k-50k', 'above_50k']

    risk_factors = [
        'impulse_spending', 'social_pressure', 'poor_planning',
        'low_savings_rate', 'emi_overload', 'no_emergency_fund',
        'lifestyle_inflation', 'income_instability',
    ]

    data = []
    for _ in range(n_rows):
        base_discipline = np.random.beta(2, 2) * 10
        noise = lambda: np.random.normal(0, 1.5)

        payday    = np.clip(base_discipline + noise(), 1, 10)
        weekend   = np.clip(base_discipline + noise(), 1, 10)
        subs      = np.clip(base_discipline + noise(), 1, 10)
        impulse   = np.clip(base_discipline + noise(), 1, 10)
        goal_h    = np.clip(base_discipline + noise(), 1, 10)
        stress    = np.clip(base_discipline + noise(), 1, 10)
        social    = np.clip(base_discipline + noise(), 1, 10)
        emergency = np.clip(base_discipline + noise(), 1, 10)
        future    = np.clip(base_discipline + noise(), 1, 10)
        learning  = np.clip(base_discipline + noise(), 1, 10)

        age = np.random.choice(age_brackets)
        income = np.random.choice(income_brackets, p=[0.2, 0.35, 0.3, 0.15])
        occupation = np.random.choice(occupation_types)
        family = np.random.choice(family_sizes)
        city = np.random.choice(city_tiers)
        emi = np.random.choice(emi_burdens, p=[0.3, 0.4, 0.3])
        deps = np.random.choice(financial_deps)
        goal_bracket = np.random.choice(goal_amount_brackets)
        savings_bracket = np.random.choice(savings_brackets)

        # Goal parameters
        timeline_months = np.random.choice([3, 6, 9, 12, 18, 24, 36])
        total_days = timeline_months * 30
        pct_elapsed = np.random.uniform(0, 0.95)
        days_elapsed = round(total_days * pct_elapsed)
        days_remaining = total_days - days_elapsed

        # Savings progress — correlated with discipline
        discipline_factor = base_discipline / 10.0
        ideal_saved_pct = pct_elapsed * 100
        saved_pct = np.clip(
            ideal_saved_pct * discipline_factor + np.random.normal(0, 15),
            0, 100
        )

        velocity = np.clip(discipline_factor * 0.5 + np.random.normal(0, 0.1), 0, 1.0)

        # Success probability — formula-based
        behavior_avg = (payday + weekend + subs + impulse + goal_h +
                        stress + social + emergency + future + learning) / 10.0

        math_score = (saved_pct / max(pct_elapsed * 100, 1)) * 50 if pct_elapsed > 0 else 50
        math_score = min(math_score, 50)
        behavior_score = (behavior_avg / 10.0) * 40
        velocity_bonus = velocity * 10

        success_prob = np.clip(math_score + behavior_score + velocity_bonus + np.random.normal(0, 5), 0, 100)

        # Binary success
        goal_success = 1 if success_prob >= 50 else 0
        # Add noise to make it more realistic
        if 40 < success_prob < 60:
            goal_success = np.random.choice([0, 1], p=[0.4, 0.6] if success_prob >= 50 else [0.6, 0.4])

        # Predicted failure date
        if goal_success == 0:
            failure_days = max(0, days_remaining - np.random.randint(0, max(1, days_remaining)))
        else:
            failure_days = -1  # no failure expected

        # Risk factor — based on weakest behavioral dimension (85%+ deterministic)
        scores_dict = {
            'impulse_spending': impulse,
            'social_pressure': social,
            'poor_planning': future,
            'low_savings_rate': payday,
            'emi_overload': 10 - ({'none': 0, 'light': 3, 'heavy': 7}[emi]),
            'no_emergency_fund': emergency,
            'lifestyle_inflation': weekend,
            'income_instability': learning,
        }
        weakest = min(scores_dict, key=scores_dict.get)
        # 85% deterministic — strong enough signal for the classifier
        top_risk = weakest if np.random.random() < 0.85 else np.random.choice(risk_factors)

        data.append({
            'age_bracket': age, 'income_bracket': income,
            'occupation_type': occupation, 'family_size': family,
            'city_tier': city, 'emi_burden': emi, 'financial_dependents': deps,
            'payday_behavior_score': round(payday, 2),
            'weekend_spend_score': round(weekend, 2),
            'subscription_awareness_score': round(subs, 2),
            'impulse_control_score': round(impulse, 2),
            'goal_history_score': round(goal_h, 2),
            'stress_response_score': round(stress, 2),
            'social_comparison_score': round(social, 2),
            'emergency_preparedness_score': round(emergency, 2),
            'future_planning_score': round(future, 2),
            'learning_orientation_score': round(learning, 2),
            'goal_amount_bracket': goal_bracket,
            'goal_timeline_months': timeline_months,
            'days_elapsed': days_elapsed,
            'days_remaining': days_remaining,
            'current_saved_pct': round(saved_pct, 2),
            'savings_velocity': round(velocity, 4),
            'monthly_savings_target_bracket': savings_bracket,
            'goal_success_probability': round(success_prob, 2),
            'goal_success_binary': goal_success,
            'predicted_failure_date_days': failure_days,
            'top_risk_factor': top_risk,
        })

    df = pd.DataFrame(data)
    out = os.path.join(DATA_DIR, 'goal_intelligence_dataset_v2.csv')
    df.to_csv(out, index=False)
    print(f"  -> {out}")
    print(f"     Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")
    print(f"     Success rate: {df['goal_success_binary'].mean():.2%}")
    print(f"     Risk factors: {dict(df['top_risk_factor'].value_counts())}")
    return df


# =====================================================================
# CLUSTER DATASET (120,000 rows — same schema, re-seeded)
# =====================================================================
def generate_clusters(n_rows=120000):
    print(f"\n{'='*60}")
    print(f"  Generating Cluster Dataset ({n_rows:,} rows)")
    print(f"{'='*60}")

    data = []
    for _ in range(n_rows):
        base = np.random.beta(2, 2) * 10
        noise = lambda: np.random.normal(0, 1.5)

        row = {
            'payday_behavior_score': round(np.clip(base + noise(), 1, 10), 2),
            'weekend_spend_score': round(np.clip(base + noise(), 1, 10), 2),
            'subscription_awareness_score': round(np.clip(base + noise(), 1, 10), 2),
            'impulse_control_score': round(np.clip(base + noise(), 1, 10), 2),
            'goal_history_score': round(np.clip(base + noise(), 1, 10), 2),
            'stress_response_score': round(np.clip(base + noise(), 1, 10), 2),
            'social_comparison_score': round(np.clip(base + noise(), 1, 10), 2),
            'emergency_preparedness_score': round(np.clip(base + noise(), 1, 10), 2),
            'future_planning_score': round(np.clip(base + noise(), 1, 10), 2),
            'learning_orientation_score': round(np.clip(base + noise(), 1, 10), 2),
        }
        data.append(row)

    df = pd.DataFrame(data)
    out = os.path.join(DATA_DIR, 'behavioral_clusters_dataset.csv')
    df.to_csv(out, index=False)
    print(f"  -> {out}")
    print(f"     Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")
    return df


# =====================================================================
# VALIDATION
# =====================================================================
def validate_datasets():
    print(f"\n{'='*60}")
    print(f"  Validating All Datasets")
    print(f"{'='*60}")
    all_ok = True

    # Health v3
    h_path = os.path.join(DATA_DIR, 'health_score_dataset_v3.csv')
    if os.path.exists(h_path):
        df = pd.read_csv(h_path)
        issues = []
        if df.isnull().sum().sum() > 0: issues.append("has null values")
        if df['health_score'].min() < 0 or df['health_score'].max() > 100: issues.append("score out of range")
        valid_trends = {'Storm Warning', 'Rainy', 'Partly Cloudy', 'Improving', 'Sunny'}
        if not set(df['trend_label'].unique()).issubset(valid_trends): issues.append("invalid trend labels")
        status = "PASS" if not issues else f"FAIL ({'; '.join(issues)})"
        print(f"  Health v3:  {status}  ({len(df):,} rows)")
        if issues: all_ok = False
    else:
        print(f"  Health v3:  MISSING")
        all_ok = False

    # Waste v6
    w_path = os.path.join(DATA_DIR, 'waste_recovery_dataset_v6.csv')
    if os.path.exists(w_path):
        df = pd.read_csv(w_path)
        issues = []
        if df.isnull().sum().sum() > 0: issues.append("has null values")
        if df['waste_score'].min() < 0 or df['waste_score'].max() > 100: issues.append("score out of range")
        if (df['waste_score'] > 100).sum() > 0: issues.append(">100 scores present")
        status = "PASS" if not issues else f"FAIL ({'; '.join(issues)})"
        print(f"  Waste v6:   {status}  ({len(df):,} rows)")
        if issues: all_ok = False
    else:
        print(f"  Waste v6:   MISSING")
        all_ok = False

    # Goal v2
    g_path = os.path.join(DATA_DIR, 'goal_intelligence_dataset_v2.csv')
    if os.path.exists(g_path):
        df = pd.read_csv(g_path)
        issues = []
        if df.isnull().sum().sum() > 0: issues.append("has null values")
        if 'goal_success_binary' not in df.columns: issues.append("missing goal_success_binary")
        if 'top_risk_factor' not in df.columns: issues.append("missing top_risk_factor")
        status = "PASS" if not issues else f"FAIL ({'; '.join(issues)})"
        print(f"  Goal v2:    {status}  ({len(df):,} rows)")
        if issues: all_ok = False
    else:
        print(f"  Goal v2:    MISSING")
        all_ok = False

    # Cluster
    c_path = os.path.join(DATA_DIR, 'behavioral_clusters_dataset.csv')
    if os.path.exists(c_path):
        df = pd.read_csv(c_path)
        issues = []
        if df.isnull().sum().sum() > 0: issues.append("has null values")
        status = "PASS" if not issues else f"FAIL ({'; '.join(issues)})"
        print(f"  Cluster:    {status}  ({len(df):,} rows)")
        if issues: all_ok = False
    else:
        print(f"  Cluster:    MISSING")
        all_ok = False

    print(f"\n  {'ALL DATASETS VALID' if all_ok else 'SOME DATASETS FAILED'}")
    return all_ok


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight AI Dataset Generator")
    parser.add_argument("--all", action="store_true", help="Generate all datasets")
    parser.add_argument("--health", action="store_true", help="Generate health dataset v3")
    parser.add_argument("--waste", action="store_true", help="Generate waste dataset v6")
    parser.add_argument("--goal", action="store_true", help="Generate goal dataset v2")
    parser.add_argument("--cluster", action="store_true", help="Generate cluster dataset")
    parser.add_argument("--validate", action="store_true", help="Validate existing datasets")
    parser.add_argument("--rows", type=int, default=0, help="Override row count (0 = use defaults)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.validate:
        validate_datasets()
        sys.exit(0)

    if args.all or args.health:
        generate_health_v3(args.rows or 150000)
    if args.all or args.waste:
        generate_waste_v6(args.rows or 80000)
    if args.all or args.goal:
        generate_goal_v2(args.rows or 150000)
    if args.all or args.cluster:
        generate_clusters(args.rows or 120000)

    if args.all or (args.health or args.waste or args.goal or args.cluster):
        print("\n" + "="*60)
        print("  Running Validation...")
        print("="*60)
        validate_datasets()
    else:
        parser.print_help()
