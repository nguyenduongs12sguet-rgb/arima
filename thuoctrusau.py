import pandas as pd
import os

# =========================
# 1) LOAD FILE
# =========================
file_name = r"E:\FINAL OF FINAL\data\thuoctrusau.csv"
delimiter = ';'

df = pd.read_csv(file_name, sep=delimiter)

print("Data head:")
print(df.head())
print("\nData info:")
print(df.info())

# =========================
# 2) COLUMNS SETUP
# =========================
group_col = 'Area'
numerical_cols = [
    'Fungicides and Bactericides',
    'Herbicides',
    'Insecticides',
    'Pesticides (total)'
]

# (An toàn) ép kiểu số phòng khi có dữ liệu bẩn
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# =========================
# 3) OPTIONAL: FILTER BY YEAR (nếu có cột Year)
# =========================
if 'Year' in df.columns:
    # Ví dụ: bạn muốn lọc giai đoạn 2010-2020 thì mở comment 2 dòng dưới
    # df = df[(df['Year'] >= 2010) & (df['Year'] <= 2020)]
    pass

# =========================
# 4) OUTPUT FOLDER (tránh PermissionError)
# =========================
output_dir = r"E:\FINAL CỦA FINAL\output"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 5) GENERAL STATS (TOÀN BỘ)
# =========================
print("\n--- 1. Thống kê mô tả cho TOÀN BỘ dữ liệu ---")
all_data_stats = df[numerical_cols].agg(['mean', 'median', 'std', 'max', 'min']).round(2)
print(all_data_stats)

output_general_file = os.path.join(output_dir, "thuoctrusau_all_data_statistics.csv")
all_data_stats.to_csv(output_general_file, encoding='utf-8-sig')

# =========================
# 6) GROUPED STATS (THEO QUỐC GIA)
# =========================
print(f"\n--- 2. Thống kê mô tả theo TỪNG QUỐC GIA (cột: {group_col}) ---")

if group_col in df.columns:
    country_stats = df.groupby(group_col)[numerical_cols].agg(['mean', 'median', 'std', 'max', 'min']).round(2)

    # Làm phẳng cột để CSV dễ đọc (Excel-friendly)
    country_stats.columns = [f"{c0}_{c1}" for (c0, c1) in country_stats.columns]
    country_stats = country_stats.reset_index()

    print(country_stats.head(10))

    output_grouped_file = os.path.join(output_dir, "thuoctrusau_country_statistics.csv")
    country_stats.to_csv(output_grouped_file, index=False, encoding='utf-8-sig')

    # =========================
    # 7) BONUS: TỔNG HỢP THÊM (sum/count/missing) theo quốc gia
    # =========================
    summary_extra = df.groupby(group_col)[numerical_cols].agg(['sum', 'count']).copy()
    summary_extra.columns = [f"{c0}_{c1}" for (c0, c1) in summary_extra.columns]
    summary_extra = summary_extra.reset_index()

    # Missing values theo quốc gia
    missing = df.groupby(group_col)[numerical_cols].apply(lambda x: x.isna().sum()).reset_index()
    missing.columns = [group_col] + [f"{c}_missing" for c in numerical_cols]

    summary_extra = summary_extra.merge(missing, on=group_col, how='left')

    output_extra_file = os.path.join(output_dir, "thuoctrusau_country_summary_extra.csv")
    summary_extra.to_csv(output_extra_file, index=False, encoding='utf-8-sig')

    print("\n✅ Đã lưu 3 file:")
    print(f"1) {output_general_file}")
    print(f"2) {output_grouped_file}")
    print(f"3) {output_extra_file}")

else:
    print(f"❌ Không tìm thấy cột '{group_col}' để nhóm dữ liệu theo quốc gia.")
