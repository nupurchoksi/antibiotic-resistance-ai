import pandas as pd


print("Step 1 — Loading datasets...")
from src.config import PRIMARY_CSV, SECONDARY_XLSX, DATA_PROCESSED

df_primary = pd.read_csv(PRIMARY_CSV)
df_secondary = pd.read_excel(SECONDARY_XLSX)

print("Done loading")


# =========================
# Step 2 — Cleaning
# =========================
print("Step 2 — Cleaning...")
from src.cleaning import clean_primary, clean_secondary

df_primary = clean_primary(df_primary)
df_secondary = clean_secondary(df_secondary)

print("Done cleaning")


# =========================
# Step 3 — Feature Engineering
# =========================
print("Step 3 — Features...")
from src.features import add_mdr_flags, encode_clinical

df_primary = add_mdr_flags(df_primary)
df_primary = encode_clinical(df_primary)

print("Done features")


# =========================
# Step 3.5 — Insights (NEW)
# =========================
print("Step 3.5 — Insights...")
from src.insights import print_insights

print_insights(df_primary)

print("Done insights")


# =========================
# Step 4 — Save Clean Data
# =========================
print("Step 4 — Saving cleaned data...")

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

df_primary.to_csv(DATA_PROCESSED / "primary_cleaned.csv", index=False)
df_secondary.to_csv(DATA_PROCESSED / "secondary_cleaned.csv", index=False)

print("Cleaned data saved to data/processed/")


# =========================
# Step 5 — Visualization
# =========================
print("Step 5 — Visualising...")
from src.visualise import plot_all

plot_all(df_primary, df_secondary)

print("Done visualising")


# =========================
# Step 6 — Drug Comparison
# =========================
print("Step 6 — Drug Comparison...")

from src.drug_comparison import print_comparison

# Example bacteria (change if needed)
print_comparison(df_primary, "Escherichia coli")

print("Done drug comparison")


# =========================
# Step 7 — Model Training
# =========================
print("Step 7 — Training models...")
from src.models import run_all_models

# IMPORTANT: capture returned values
xgb1, X1_te, FEATURES_EXT = run_all_models(df_primary, df_secondary)

print("Done models")


# =========================
# Step 8 — Explainability
# =========================
print("Step 8 — Explainability...")

from src.explainability import explain_prediction, print_explanation

# Take one sample from test set
sample = X1_te[0]

# Generate explanation
contributions = explain_prediction(xgb1, sample, FEATURES_EXT)

# Print explanation
print_explanation(contributions)

print("Done explainability")
# =========================
# Step 9 — Model Demo (Prediction + Recommendations)
# =========================
print("Step 9 — Running Model Demo...")

from backend.prediction import predict_resistance
from backend.recommendation import recommend_antibiotics
from backend.main_demo import compare_drugs   # if exists separately
# =========================
# Prediction
# =========================

# Create dummy input (same logic as your demo)
expected_base_len = xgb1.n_features_in_ - 2
sample_input = [0] * expected_base_len

result = predict_resistance(sample_input)

print("\n🔮 Prediction Result:")
print(f"Prediction : {result['prediction']}")
print(f"Confidence : {result['confidence']*100:.2f}%")

# =========================
# Recommendation
# =========================

bacteria = "Escherichia coli"
print(f"\n🦠 Bacteria: {bacteria}")

recs = recommend_antibiotics(bacteria)

print("\n💊 Recommended Antibiotics:")

for drug, rate in recs:
    if rate < 0.2:
        tag = "🟢 Highly Effective"
    elif rate < 0.4:
        tag = "🟡 Moderate"
    else:
        tag = "🔴 Avoid"

    print(f"{drug} → {rate*100:.1f}% resistance ({tag})")

# =========================
# Best & Worst
# =========================

if recs:
    best = recs[0]
    worst = recs[-1]

    print(f"\n🏆 Best Option: {best[0]} ({best[1]*100:.1f}%)")
    print(f"💀 Worst Option: {worst[0]} ({worst[1]*100:.1f}%)")

# =========================
# Drug Comparison
# =========================

print("\n📊 Drug Comparison:")
compare_drugs(bacteria)

# =========================
# Clinical Insight
# =========================

if result['prediction'] == "Resistant":
    print("\n🚨 High resistance detected — avoid commonly resistant antibiotics.")
elif result['prediction'] == "Susceptible":
    print("\n✅ Most antibiotics are likely effective.")
else:
    print("\n⚠️ Intermediate resistance — use caution in treatment selection.")
# =========================
# FINAL
# =========================
print("\n✅ All done! Check outputs/figures/ and outputs/models/")

