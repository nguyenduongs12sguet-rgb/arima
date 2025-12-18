import pandas as pd
import os

# ====== CONFIG ======
FILE = r"E:\FINAL OF FINAL\data\temperature.csv"
SEP = ';'
GROUP = 'Area'

TARGET_COL = 'Average Mean Surface Air Temperature (Annual Mean )'

OUT_DIR = r"E:\FINAL OF FINAL\output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ALL = os.path.join(OUT_DIR, "temperature_all_data_statistics.csv")
OUT_BY  = os.path.join(OUT_DIR, "temperature_country_statistics.csv")

def log(msg):
    print(f"[INFO] {msg}")

# ====== 1) LOAD ======
log("Reading CSV...")
df = pd.read_csv(FILE, sep=SEP, encoding="latin1")
df.columns = df.columns.str.strip()
log(f"Loaded: {df.shape[0]:,} rows")

# ====== 2) CHECK COLUMN ======
if TARGET_COL not in df.columns or GROUP not in df.columns:
    print("❌ Missing required columns")
    print("Available columns:")
    print(df.columns.tolist())
    raise KeyError("Required column not found")

# ====== 3) CLEAN TARGET COLUMN ======
df[TARGET_COL] = (
    df[TARGET_COL]
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace(',', '.', regex=False)
    .str.strip()
)

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
log(f"Dropped {before - len(df):,} rows with invalid temperature")

# ====== 4) OVERALL STATISTICS ======
log("Computing overall statistics...")
all_stats = (
    df[[TARGET_COL]]
    .agg(['mean', 'median', 'std', 'max', 'min'])
    .round(2)
)

print("\n=== OVERALL DESCRIPTIVE STATISTICS ===")
print(all_stats.to_string())

all_stats.to_csv(OUT_ALL, encoding="utf-8-sig")

# ====== 5) BY COUNTRY STATISTICS ======
log("Computing statistics by country...")
by = (
    df
    .groupby(GROUP)[TARGET_COL]
    .agg(['mean', 'median', 'std', 'max', 'min', 'count'])
    .round(2)
    .reset_index()
)

by.to_csv(OUT_BY, index=False, encoding="utf-8-sig")

# ====== 6) TOP 30 COUNTRIES ======
top30 = by.sort_values('mean', ascending=False).head(30)

print("\n=== TOP 30 COUNTRIES by MEAN TEMPERATURE ===")
print(top30.to_string(index=False))

# ====== DONE ======
log("Saved outputs:")
print(" -", OUT_ALL)
print(" -", OUT_BY)
