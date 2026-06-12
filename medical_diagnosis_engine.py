"""
Medical Diagnosis Reasoning Engine  (No external packages)
===========================================================
Diseases : Flu, COVID, Allergy, Pneumonia, Dengue, Tuberculosis
Symptoms : Fever, Cough, Fatigue, RunnyNose, ShortnessOfBreath,
           ChestPain, NightSweats, Rash, JointPain, WeightLoss

Pure Python – no pandas, no pgmpy, no third-party libraries.
Uses Variable Elimination (exact Bayesian inference) from scratch.
"""

import itertools
from copy import deepcopy

# ──────────────────────────────────────────────
# 1.  Disease priors  P(Disease=yes)
# ──────────────────────────────────────────────

DISEASES = ["Flu", "COVID", "Allergy", "Pneumonia", "Dengue", "Tuberculosis"]

PRIORS = {
    "Flu":          0.10,
    "COVID":        0.05,
    "Allergy":      0.20,
    "Pneumonia":    0.04,
    "Dengue":       0.03,
    "Tuberculosis": 0.02,
}

SYMPTOMS = [
    "Fever", "Cough", "Fatigue", "RunnyNose",
    "ShortnessOfBreath", "ChestPain", "NightSweats",
    "Rash", "JointPain", "WeightLoss",
]

# ──────────────────────────────────────────────
# 2.  Symptom → Parent diseases + strengths
#     strength = P(symptom=yes | only this disease active)
# ──────────────────────────────────────────────

SYMPTOM_PARENTS = {
    "Fever":             [("Flu",0.85), ("COVID",0.80), ("Pneumonia",0.92), ("Dengue",0.95), ("Tuberculosis",0.88)],
    "Cough":             [("Flu",0.80), ("COVID",0.75), ("Allergy",0.65),   ("Pneumonia",0.90), ("Tuberculosis",0.95)],
    "Fatigue":           [("Flu",0.75), ("COVID",0.80), ("Pneumonia",0.85), ("Dengue",0.90),    ("Tuberculosis",0.88)],
    "RunnyNose":         [("Flu",0.80), ("Allergy",0.85)],
    "ShortnessOfBreath": [("COVID",0.60), ("Pneumonia",0.85), ("Allergy",0.50)],
    "ChestPain":         [("Pneumonia",0.70), ("Tuberculosis",0.55)],
    "NightSweats":       [("Tuberculosis",0.75), ("Dengue",0.45)],
    "Rash":              [("Dengue",0.70), ("Allergy",0.40)],
    "JointPain":         [("Dengue",0.80)],
    "WeightLoss":        [("Tuberculosis",0.70)],
}

LEAK = 0.02   # background noise for all symptoms


# ──────────────────────────────────────────────
# 3.  Noisy-OR likelihood  P(symptom | diseases)
# ──────────────────────────────────────────────

def noisy_or_prob(symptom, disease_states):
    """
    Returns P(symptom=yes | disease_states) using Noisy-OR.
    disease_states : dict  {disease: True/False}
    """
    parents = SYMPTOM_PARENTS[symptom]
    p_not_caused = 1 - LEAK
    for disease, strength in parents:
        if disease_states.get(disease, False):
            p_not_caused *= (1 - strength)
    return 1 - p_not_caused


# ──────────────────────────────────────────────
# 4.  Exact Bayesian Inference
#     Enumerate all 2^6 = 64 disease combinations,
#     compute joint probability, then marginalise.
# ──────────────────────────────────────────────

def infer_posteriors(observed):
    """
    observed : dict  {symptom: "yes"/"no"}
    Returns  : dict  {disease: P(disease=yes | observed)}
    """
    # All 64 combinations of 6 diseases
    disease_list = DISEASES
    unnorm_marginals = {d: 0.0 for d in disease_list}
    total_weight = 0.0

    for bits in itertools.product([True, False], repeat=len(disease_list)):
        disease_states = dict(zip(disease_list, bits))

        # P(diseases) = product of independent priors
        p_diseases = 1.0
        for d, active in disease_states.items():
            p = PRIORS[d]
            p_diseases *= (p if active else 1 - p)

        # P(symptoms | diseases) = product over observed symptoms
        p_symptoms = 1.0
        for symptom, value in observed.items():
            p_yes = noisy_or_prob(symptom, disease_states)
            p_symptoms *= (p_yes if value == "yes" else 1 - p_yes)

        weight = p_diseases * p_symptoms
        total_weight += weight

        for d, active in disease_states.items():
            if active:
                unnorm_marginals[d] += weight

    if total_weight == 0:
        return {d: 0.0 for d in disease_list}

    return {d: unnorm_marginals[d] / total_weight for d in disease_list}


# ──────────────────────────────────────────────
# 5.  Display helpers  (no pandas)
# ──────────────────────────────────────────────

def print_table(rows, headers):
    """Print a simple aligned text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep  = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    head = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(head)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        print(line)
    print(sep)


def compute_and_display(observed):
    posteriors = infer_posteriors(observed)
    sorted_diseases = sorted(posteriors, key=lambda d: posteriors[d], reverse=True)
    max_p = max(posteriors.values())

    rows = []
    for d in sorted_diseases:
        prior     = PRIORS[d]
        posterior = posteriors[d]
        delta     = posterior - prior
        star      = "YES" if posterior == max_p else ""
        rows.append([
            d,
            f"{prior:.3f}",
            f"{posterior:.3f}",
            f"{delta:+.3f}",
            star,
        ])

    print_table(rows, ["Disease", "Prior", "Posterior", "Delta", "Most Likely"])


def explain_belief_update(observed):
    print("=" * 65)
    print("  BELIEF UPDATE TRACE")
    print("=" * 65)

    current_evidence = {}
    current_beliefs  = dict(PRIORS)

    print("\n  Prior beliefs (no evidence):")
    for d in DISEASES:
        print(f"    P({d}=yes) = {current_beliefs[d]:.3f}")

    for symptom, value in observed.items():
        current_evidence[symptom] = value
        print(f"\n  Observing: {symptom} = {value}")
        new_beliefs = infer_posteriors(current_evidence)
        for d in DISEASES:
            old_p = current_beliefs[d]
            new_p = new_beliefs[d]
            delta = new_p - old_p
            arrow = "up" if delta > 0.001 else ("dn" if delta < -0.001 else "--")
            print(f"    {d:14s}: {old_p:.3f} -> {new_p:.3f}  [{arrow}] ({delta:+.3f})")
        current_beliefs = new_beliefs

    print("\n" + "=" * 65)
    print("  FINAL DIAGNOSIS LIKELIHOODS")
    print("=" * 65)


def run_scenario(title, symptoms):
    print(f"\n{'='*65}")
    print(f"  SCENARIO : {title}")
    print(f"{'='*65}")
    sym_display = ", ".join(f"{k}={v}" for k, v in symptoms.items() if v == "yes")
    print(f"  Positive symptoms: {sym_display or 'none'}\n")
    explain_belief_update(symptoms)
    compute_and_display(symptoms)
    print()


# ──────────────────────────────────────────────
# 6.  Pre-built scenarios + interactive mode
# ──────────────────────────────────────────────

SCENARIOS = [
    ("Classic Flu", {
        "Fever":"yes","Cough":"yes","Fatigue":"yes","RunnyNose":"yes",
        "ShortnessOfBreath":"no","ChestPain":"no","NightSweats":"no",
        "Rash":"no","JointPain":"no","WeightLoss":"no"}),

    ("Seasonal Allergy", {
        "Fever":"no","Cough":"yes","Fatigue":"no","RunnyNose":"yes",
        "ShortnessOfBreath":"yes","ChestPain":"no","NightSweats":"no",
        "Rash":"yes","JointPain":"no","WeightLoss":"no"}),

    ("COVID Presentation", {
        "Fever":"yes","Cough":"yes","Fatigue":"yes","RunnyNose":"no",
        "ShortnessOfBreath":"yes","ChestPain":"no","NightSweats":"no",
        "Rash":"no","JointPain":"no","WeightLoss":"no"}),

    ("Pneumonia", {
        "Fever":"yes","Cough":"yes","Fatigue":"yes","RunnyNose":"no",
        "ShortnessOfBreath":"yes","ChestPain":"yes","NightSweats":"no",
        "Rash":"no","JointPain":"no","WeightLoss":"no"}),

    ("Dengue Fever", {
        "Fever":"yes","Cough":"no","Fatigue":"yes","RunnyNose":"no",
        "ShortnessOfBreath":"no","ChestPain":"no","NightSweats":"yes",
        "Rash":"yes","JointPain":"yes","WeightLoss":"no"}),

    ("Tuberculosis", {
        "Fever":"yes","Cough":"yes","Fatigue":"yes","RunnyNose":"no",
        "ShortnessOfBreath":"no","ChestPain":"yes","NightSweats":"yes",
        "Rash":"no","JointPain":"no","WeightLoss":"yes"}),
]


if __name__ == "__main__":

    print("\nMEDICAL DIAGNOSIS REASONING ENGINE")
    print("Diseases:", ", ".join(DISEASES))
    print("Symptoms:", ", ".join(SYMPTOMS))

    for title, symptoms in SCENARIOS:
        run_scenario(title, symptoms)

    # ── Interactive Mode ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  INTERACTIVE DIAGNOSIS MODE")
    print("  Answer yes / no for each symptom.")
    print("=" * 65 + "\n")

    user_evidence = {}
    for sym in SYMPTOMS:
        while True:
            ans = input(f"  {sym}? (yes/no): ").strip().lower()
            if ans in ("yes", "no"):
                user_evidence[sym] = ans
                break
            print("  Please type 'yes' or 'no'.")

    run_scenario("Your Symptoms", user_evidence)