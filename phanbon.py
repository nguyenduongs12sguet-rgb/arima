import pandas as pd
import os

# =========================
# 1) LOAD DATA
# =========================
file_name = r"E:\FINAL OF FINAL\data\phanbontieuthu.csv"
delimiter = ';'
df = pd.read_csv(file_name, sep=delimiter)

# =========================
# 2) DEFINE COLUMNS
# =========================
group_col = 'Area'
numerical_col = 'Fertilizer (kg/ha)'

# =========================
# 3) CLEANING: convert to numeric (handle quotes, comma decimals)
# =========================
if df[numerical_col].dtype == 'object':
    df[numerical_col] = (
        df[numerical_col]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip()
    )

df[numerical_col] = pd.to_numeric(df[numerical_col], errors='coerce')

# Keep rows that have numeric value
df = df.dropna(subset=[numerical_col]).copy()

# =========================
# 4) OUTPUT FOLDER (avoid PermissionError)
# =========================
output_dir = r"E:\FINAL CỦA FINAL\output"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 5) OVERALL STATS
# =========================
all_data_stats = df[[numerical_col]].agg(['mean', 'median', 'std', 'max', 'min']).round(2)
overall_out = os.path.join(output_dir, 'phanbontieuthu_all_data_statistics.csv')
all_data_stats.to_csv(overall_out, encoding='utf-8-sig')

# =========================
# 6) BY COUNTRY STATS
# =========================
country_stats = df.groupby(group_col)[[numerical_col]].agg(
    ['mean', 'median', 'std', 'max', 'min', 'count']
).round(2)

# Flatten columns for Excel-friendly CSV
country_stats.columns = [f"{numerical_col}_{stat}" for (col, stat) in country_stats.columns]
country_stats = country_stats.reset_index()

country_out = os.path.join(output_dir, 'phanbontieuthu_country_statistics.csv')
country_stats.to_csv(country_out, index=False, encoding='utf-8-sig')

# =========================
# 7) BONUS: TOP 10 QUỐC GIA THEO MEAN
# =========================
top10 = country_stats.sort_values(f"{numerical_col}_mean", ascending=False).head(10)
top10_out = os.path.join(output_dir, 'phanbontieuthu_top10_by_mean.csv')
top10.to_csv(top10_out, index=False, encoding='utf-8-sig')

# =========================
# 8) PRINT NICE TABLES (terminal-friendly)
# =========================
print("\n--- 1) Thống kê mô tả cho TOÀN BỘ dữ liệu ---")
print(all_data_stats.to_string())

print("\n--- 2) Thống kê mô tả theo TỪNG QUỐC GIA (5 dòng đầu) ---")
print(country_stats.head(5).to_string(index=False))

print("\n✅ Đã xuất 3 file CSV:")
print(f"1) {overall_out}")
print(f"2) {country_out}")
print(f"3) {top10_out}")
