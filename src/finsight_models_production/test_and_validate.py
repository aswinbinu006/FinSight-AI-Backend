# -*- coding: utf-8 -*-
"""
FinSight AI — Complete Model Validation Suite (Production)
===========================================================
Execution Order (from docx § 10):
    Step 4: Retrain all models
    Step 5: Run metric threshold checks (§ 8.1)
    Step 6: Run sanity test cases (§ 8.2)
    Step 7: Verify explainer.py validators (§ 8.3)
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finsight_models_production import (
    HealthModel, WasteModel, GoalModel, ClusterModel,
)
from finsight_models_production.explainer import (
    validate_health_output, validate_waste_output, validate_goal_output,
    check_health_thresholds, check_waste_thresholds, check_goal_thresholds,
    HEALTH_THRESHOLDS, WASTE_THRESHOLDS, GOAL_THRESHOLDS,
)
from finsight_models_production.behavioral_scoring import BEHAVIOR_COLUMNS


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    all_passed = True
    results = {}

    # ==================================================================
    # STEP 4: Retrain all models
    # ==================================================================
    separator("STEP 4: Retraining All Models")

    # 4a: Cluster Model
    print("\n  [1/4] ClusterModel.train()...")
    cluster = ClusterModel()
    try:
        cm = cluster.train()
        print(f"         Status: {cm['status']}")
        print(f"         Archetypes: {cm['archetypes']}")
        print(f"         Silhouette: {cm['silhouette_score']}")
        print(f"         Variance: {cm['variance_explained']}")
        results['cluster'] = cm
    except Exception as e:
        print(f"         FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # 4b: Health Model
    print("\n  [2/4] HealthModel.train()...")
    health = HealthModel()
    try:
        hm = health.train()
        print(f"         Status: {hm['status']}")
        print(f"         Rows: {hm['rows']:,}")
        print(f"         RMSE: {hm['score_rmse']}")
        print(f"         R²: {hm['score_r2']}")
        print(f"         Trend F1 (Macro): {hm['trend_f1_macro']}")
        results['health'] = hm
    except Exception as e:
        print(f"         FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # 4c: Waste Model
    print("\n  [3/4] WasteModel.train()...")
    waste = WasteModel()
    try:
        wm = waste.train()
        print(f"         Status: {wm['status']}")
        print(f"         Rows: {wm['rows']:,}")
        print(f"         RMSE: {wm['rmse']}")
        print(f"         R²: {wm['r2']}")
        print(f"         MAE: {wm['mae']}")
        results['waste'] = wm
    except Exception as e:
        print(f"         FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # 4d: Goal Model
    print("\n  [4/4] GoalModel.train()...")
    goal = GoalModel()
    try:
        gm = goal.train()
        print(f"         Status: {gm['status']}")
        print(f"         Rows: {gm['rows']:,}")
        print(f"         Success Acc: {gm['success_accuracy']}")
        print(f"         Success F1: {gm['success_f1']}")
        print(f"         LogLoss: {gm['success_logloss']}")
        print(f"         Risk Acc: {gm['risk_accuracy']}")
        results['goal'] = gm
    except Exception as e:
        print(f"         FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # ==================================================================
    # STEP 5: Metric Threshold Checks (§ 8.1)
    # ==================================================================
    separator("STEP 5: Metric Threshold Checks (§ 8.1)")

    # Health thresholds
    if 'health' in results:
        hf = check_health_thresholds(results['health'])
        if hf:
            print(f"  Health:  FAIL  {hf}")
            all_passed = False
        else:
            print(f"  Health:  PASS  (RMSE={results['health']['score_rmse']:.2f} <= {HEALTH_THRESHOLDS['score_rmse']}, "
                  f"R²={results['health']['score_r2']:.4f} >= {HEALTH_THRESHOLDS['score_r2']}, "
                  f"TrendF1={results['health']['trend_f1_macro']:.4f} >= {HEALTH_THRESHOLDS['trend_f1_macro']})")

    # Waste thresholds
    if 'waste' in results:
        wf = check_waste_thresholds(results['waste'])
        if wf:
            print(f"  Waste:   FAIL  {wf}")
            all_passed = False
        else:
            print(f"  Waste:   PASS  (RMSE={results['waste']['rmse']:.2f} <= {WASTE_THRESHOLDS['rmse']}, "
                  f"R²={results['waste']['r2']:.4f} >= {WASTE_THRESHOLDS['r2']})")

    # Goal thresholds
    if 'goal' in results:
        gf = check_goal_thresholds(results['goal'])
        if gf:
            print(f"  Goal:    FAIL  {gf}")
            all_passed = False
        else:
            print(f"  Goal:    PASS  (SuccAcc={results['goal']['success_accuracy']:.4f} >= {GOAL_THRESHOLDS['success_accuracy']}, "
                  f"F1={results['goal']['success_f1']:.4f} >= {GOAL_THRESHOLDS['success_f1']}, "
                  f"RiskAcc={results['goal']['risk_accuracy']:.4f} >= {GOAL_THRESHOLDS['risk_accuracy']})")

    # ==================================================================
    # STEP 6: Sanity Test Cases (§ 8.2)
    # ==================================================================
    separator("STEP 6: Sanity Test Cases (§ 8.2)")
    sanity_pass = 0
    sanity_total = 0

    # --- Test 1: Perfect health user must score above 75 ---
    sanity_total += 1
    try:
        perfect = {col: 9.5 for col in BEHAVIOR_COLUMNS}
        r = health.predict(perfect, emi_burden='none', income_bracket='100k+')
        validate_health_output(r)
        assert r['health_score'] > 75, f"Perfect user got {r['health_score']}"
        print(f"  [1] Health — perfect user > 75:          PASS  (got {r['health_score']})")
        sanity_pass += 1
    except AssertionError as e:
        print(f"  [1] Health — perfect user > 75:          FAIL  ({e})")
        all_passed = False
    except Exception as e:
        print(f"  [1] Health — perfect user > 75:          FAIL  ({e})")
        all_passed = False

    # --- Test 2: Bad health user must score below 40 ---
    sanity_total += 1
    try:
        bad = {col: 1.5 for col in BEHAVIOR_COLUMNS}
        r = health.predict(bad, emi_burden='heavy', income_bracket='below_20k')
        validate_health_output(r)
        assert r['health_score'] < 40, f"Bad user got {r['health_score']}"
        print(f"  [2] Health — worst user < 40:            PASS  (got {r['health_score']})")
        sanity_pass += 1
    except AssertionError as e:
        print(f"  [2] Health — worst user < 40:            FAIL  ({e})")
        all_passed = False
    except Exception as e:
        print(f"  [2] Health — worst user < 40:            FAIL  ({e})")
        all_passed = False

    # --- Test 3: Forgotten subscription must score above 70 ---
    sanity_total += 1
    try:
        r = waste.predict_subscription({
            'name': 'TestApp', 'billing_cycle': 'monthly', 'price': 500,
            'usage_frequency': 'cant_remember', 'awareness': 'forget',
            'necessity': 'fun', 'has_perks': False,
        })
        validate_waste_output(r)
        assert r['waste_score'] > 70, f"Forgotten sub got {r['waste_score']}"
        print(f"  [3] Waste — forgotten sub > 70:          PASS  (got {r['waste_score']})")
        sanity_pass += 1
    except AssertionError as e:
        print(f"  [3] Waste — forgotten sub > 70:          FAIL  ({e})")
        all_passed = False
    except Exception as e:
        print(f"  [3] Waste — forgotten sub > 70:          FAIL  ({e})")
        all_passed = False

    # --- Test 4: Mathematically impossible goal must be below 25% ---
    sanity_total += 1
    try:
        r = goal.predict(
            behavioral_scores={col: 5.0 for col in BEHAVIOR_COLUMNS},
            goal_description='Impossible Dream',
            goal_target=1_000_000, saved_so_far=1000,
            monthly_savings=2000, months_remaining=3,
        )
        validate_goal_output(r)
        assert r['success_probability'] < 25, f"Impossible goal got {r['success_probability']}%"
        print(f"  [4] Goal — impossible < 25%:             PASS  (got {r['success_probability']}%)")
        sanity_pass += 1
    except AssertionError as e:
        print(f"  [4] Goal — impossible < 25%:             FAIL  ({e})")
        all_passed = False
    except Exception as e:
        print(f"  [4] Goal — impossible < 25%:             FAIL  ({e})")
        all_passed = False

    # --- Test 5: On-track user must be above 60% ---
    sanity_total += 1
    try:
        r = goal.predict(
            behavioral_scores={col: 8.0 for col in BEHAVIOR_COLUMNS},
            goal_description='Trip to Goa',
            goal_target=50_000, saved_so_far=20_000,
            monthly_savings=10_000, months_remaining=6,
        )
        validate_goal_output(r)
        assert r['success_probability'] > 60, f"On-track user got {r['success_probability']}%"
        assert r['goal_description'] == 'Trip to Goa', f"goal_description wrong: {r['goal_description']}"
        print(f"  [5] Goal — on-track > 60%:               PASS  (got {r['success_probability']}%, desc='{r['goal_description']}')")
        sanity_pass += 1
    except AssertionError as e:
        print(f"  [5] Goal — on-track > 60%:               FAIL  ({e})")
        all_passed = False
    except Exception as e:
        print(f"  [5] Goal — on-track > 60%:               FAIL  ({e})")
        all_passed = False

    # --- Extra Test 6: Cluster model prediction ---
    sanity_total += 1
    try:
        r = cluster.predict({col: 7.0 for col in BEHAVIOR_COLUMNS})
        assert 'cluster_id' in r, "Missing cluster_id"
        assert 'cluster_name' in r, "Missing cluster_name"
        assert 0 <= r['cluster_id'] <= 3, f"Invalid cluster_id: {r['cluster_id']}"
        print(f"  [6] Cluster — valid prediction:          PASS  (cluster='{r['cluster_name']}')")
        sanity_pass += 1
    except Exception as e:
        print(f"  [6] Cluster — valid prediction:          FAIL  ({e})")
        all_passed = False

    # --- Extra Test 7: Health trend/score reconciliation (Change 1) ---
    sanity_total += 1
    try:
        # Deliberately set mediocre scores so status = "Partly Cloudy"
        # After reconciliation, trend must not be > 1 step away
        mid = {col: 5.0 for col in BEHAVIOR_COLUMNS}
        r = health.predict(mid, emi_burden='light', income_bracket='50k-100k')
        validate_health_output(r)
        trend_order = ['Storm Warning', 'Rainy', 'Partly Cloudy', 'Improving', 'Sunny']
        s_idx = trend_order.index(r['status_label'])
        t_idx = trend_order.index(r['trend_label'])
        assert abs(s_idx - t_idx) <= 1, f"status={r['status_label']}, trend={r['trend_label']} disagree by {abs(s_idx - t_idx)} steps"
        print(f"  [7] Health — trend reconciliation:       PASS  (status='{r['status_label']}', trend='{r['trend_label']}')")
        sanity_pass += 1
    except Exception as e:
        print(f"  [7] Health — trend reconciliation:       FAIL  ({e})")
        all_passed = False

    # --- Extra Test 8: Waste model — no advisor_reasoning (Change 2) ---
    sanity_total += 1
    try:
        r = waste.predict_subscription({
            'name': 'Netflix', 'billing_cycle': 'monthly', 'price': 199,
            'usage_frequency': 'daily', 'awareness': 'yes',
            'necessity': 'essential', 'has_perks': True,
        })
        assert 'advisor_reasoning' not in r, f"advisor_reasoning still present in output"
        print(f"  [8] Waste — no advisor_reasoning:        PASS")
        sanity_pass += 1
    except Exception as e:
        print(f"  [8] Waste — no advisor_reasoning:        FAIL  ({e})")
        all_passed = False

    # --- Extra Test 9: Health — no trend_description (Change 3) ---
    sanity_total += 1
    try:
        r = health.predict({col: 7.0 for col in BEHAVIOR_COLUMNS})
        assert 'trend_description' not in r, f"trend_description still present in output"
        print(f"  [9] Health — no trend_description:       PASS")
        sanity_pass += 1
    except Exception as e:
        print(f"  [9] Health — no trend_description:       FAIL  ({e})")
        all_passed = False

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    separator("FINAL RESULTS")
    print(f"  Sanity Tests: {sanity_pass}/{sanity_total} passed")
    print(f"  Overall:      {'ALL PASSED' if all_passed else 'SOME FAILURES'}")

    if all_passed:
        print("\n  ✅ ALL MODELS ARE PRODUCTION-READY")
    else:
        print("\n  ❌ FIX FAILURES BEFORE DEPLOYMENT")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
