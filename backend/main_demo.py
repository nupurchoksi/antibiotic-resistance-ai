import pandas as pd
from pathlib import Path
from backend.prediction import predict_resistance
from backend.recommendation import recommend_antibiotics
# =========================
# Fix Paths (IMPORTANT)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "primary_cleaned.csv"

print(" Looking for dataset at:", DATA_PATH)

if not DATA_PATH.exists():
    print(" ERROR: Dataset not found!")
    print(" Make sure file exists at: data/processed/primary_cleaned.csv")
    exit()

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")

# =========================
# Antibiotic Columns
# =========================
AB_COLS = [
    "AMC", "CTX/CRO", "FOX", "IPM",
    "AMX/AMP", "CZ", "Furanes",
    "Co-trimoxazole", "colistine"
]

# =========================
# Recommendation System
# =========================
def recommend_antibiotics(bacteria_name, top_n=3):

    df_bac = df[df["Species_clean"] == bacteria_name]

    if df_bac.empty:
        print("⚠️ No data found for this bacteria")
        return []

    resistance = {
        ab: (df_bac[ab] == "R").mean()
        for ab in AB_COLS
    }

    sorted_drugs = sorted(resistance.items(), key=lambda x: x[1])

    return sorted_drugs[:top_n]

# =========================
# Drug Comparison
# =========================
def compare_drugs(bacteria_name):

    df_bac = df[df["Species_clean"] == bacteria_name]

    print("\n Drug Comparison:")
    for ab in AB_COLS:
        rate = (df_bac[ab] == "R").mean()
        print(f"{ab}: {rate*100:.1f}% resistant")

# =========================
# MAIN DEMO
# =========================
if __name__ == "__main__":

    print("\n Running Model Demo...")

    # Dummy input (auto-handled in prediction.py)
    sample = [0] * 14

    # Prediction
    result = predict_resistance(sample)

    print("\n Prediction:")
    print(f"{result['prediction']} ({result['confidence']*100:.2f}%)")

    # Select bacteria
    bacteria = "Escherichia coli"
    print(f"\n Bacteria: {bacteria}")

    # Recommendations
    recs = recommend_antibiotics(bacteria)

    print("\n Recommended Antibiotics:")

    for drug, rate in recs:

        if rate < 0.2:
            tag = "🟢 Highly Effective"
        elif rate < 0.4:
            tag = "🟡 Moderate"
        else:
            tag = "🔴 Avoid"

        print(f"{drug} → {rate*100:.1f}% resistance ({tag})")

    # Best & Worst
    if recs:
        best = recs[0]
        worst = recs[-1]

        print(f"\n Best Option: {best[0]} ({best[1]*100:.1f}%)")
        print(f" Worst Option: {worst[0]} ({worst[1]*100:.1f}%)")

    # Comparison
    compare_drugs(bacteria)

    # Clinical Insight
    if result['prediction'] == "Resistant":
        print("\n High resistance detected — avoid commonly resistant antibiotics.")
    elif result['prediction'] == "Susceptible":
        print("\n Most antibiotics are likely effective.")
    else:
        print("\n Intermediate resistance — use caution in treatment selection.")