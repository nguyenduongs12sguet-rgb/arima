import pandas as pd
import os

# =========================
# 1. ĐỌC DỮ LIỆU
# =========================
file_name = r"E:\FINAL OF FINAL\data\nangsuat_potatoes_only.csv"
df = pd.read_csv(file_name, sep=';')

numerical_cols = [
    'Area harvested(ha)',
    'Production(tonnes)',
    'Yield (kg/ha)'
]

df[numerical_cols] = df[numerical_cols].apply(
    pd.to_numeric, errors='coerce'
)

# =========================
# 2. TÍNH THỐNG KÊ
# =========================
all_data_stats = df[numerical_cols].agg(
    ['mean', 'median', 'std', 'max', 'min']
).round(2)

country_stats = df.groupby('Area')[numerical_cols].agg(
    ['mean', 'median', 'std', 'max', 'min']
).round(2)

# =========================
# 3. LÀM PHẲNG CỘT (CSV FRIENDLY)
# =========================
country_stats.columns = [
    f"{col[0]}_{col[1]}" for col in country_stats.columns
]

# Reset index để 'Area' thành cột
country_stats.reset_index(inplace=True)

# =========================
# 4. XUẤT CSV
# =========================
output_dir = r"E:\FINAL OF FINAL\output"
os.makedirs(output_dir, exist_ok=True)

all_data_stats.to_csv(
    os.path.join(output_dir, 'all_data_statistics.csv'),
    encoding='utf-8-sig'
)

country_stats.to_csv(
    os.path.join(output_dir, 'country_statistics.csv'),
    index=False,
    encoding='utf-8-sig'
)

print("✅ Đã xuất file CSV thành công!")
