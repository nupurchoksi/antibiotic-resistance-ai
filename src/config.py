from pathlib import Path

BASE_DIR       = Path(__file__).parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES_DIR    = BASE_DIR / "outputs" / "figures"
MODELS_DIR     = BASE_DIR / "outputs" / "models"

# Create output directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PRIMARY_CSV    = DATA_RAW / "Bacteria_dataset_Multiresictance.csv"
SECONDARY_XLSX = DATA_RAW / "Dataset.xlsx"

AB_COLS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO",
    "IPM", "GEN", "AN", "Acide nalidixique", "ofx",
    "CIP", "C", "Co-trimoxazole", "Furanes", "colistine"
]

# Keys must match exact column names in df_secondary
CLSI_BREAKPOINTS = {
    "IMIPENEM":      {"R": 19, "S": 23},
    "CEFTAZIDIME":   {"R": 17, "S": 21},
    "GENTAMICIN":    {"R": 12, "S": 15},
    "AUGMENTIN":     {"R": 13, "S": 18},
    "CIPROFLOXACIN": {"R": 20, "S": 26},
}

SPECIES_MAP = {
    "E.COI":                   "Escherichia coli",
    "E.CLI":                   "Escherichia coli",
    "E. COLI":                 "Escherichia coli",
    "ESCHERICHIA COLI":        "Escherichia coli",
    "ENTER.BACTERIA SPP.":     "Enterobacteria spp.",
    "ENTEROBACTERIA SPP.":     "Enterobacteria spp.",
    "ENTEROBACTERIA SPP":      "Enterobacteria spp.",
    "ENTERBACTERIA SPP.":      "Enterobacteria spp.",
    "ENTEROBACTER SPP.":       "Enterobacteria spp.",
    "KLBSIELLA PNEUMONIAE":    "Klebsiella pneumoniae",
    "KLEBSIE.LLA PNEUMONIAE":  "Klebsiella pneumoniae",
    "KLEBSIELLA PNEUMONIAE":   "Klebsiella pneumoniae",
    "PROTUS MIRABILIS":        "Proteus mirabilis",
    "PROEUS MIRABILIS":        "Proteus mirabilis",
    "PROTEUS MIRABILIS":       "Proteus mirabilis",
}