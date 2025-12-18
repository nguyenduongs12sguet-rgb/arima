import pandas as pd

# 1. Đọc dữ liệu
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()

# 2. Tiền xử lý (Sửa lỗi dấu phẩy và ép kiểu số)
cols_to_fix = ['Precipitation (mm)', 'Average Mean Surface Air Temperature (Annual Mean)', 
               'Fertilizer (kg/ha)', 'Pesticides (total)']
for col in cols_to_fix:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.', regex=True).astype(float)

# 3. Chọn các cột sạch (LOẠI BỎ cột AVERAGE PRECIPITATION bị lỗi số lớn)
relevant_cols = ['Yield (kg/ha)', 'Area harvested(ha)', 'Production(tonnes)',
                 'Pesticides (total)', 'Precipitation (mm)', 'Fertilizer (kg/ha)',
                 'Average Mean Surface Air Temperature (Annual Mean)']

# --- PHẦN 1: Thống kê tổng thể ---
all_stats = df[relevant_cols].describe(percentiles=[.25, .5, .75])
all_stats.to_csv('all_data_statistics_final.csv')

# --- PHẦN 2: Thống kê theo Quốc gia ---
country_stats = df.groupby('Area')[relevant_cols].describe(percentiles=[.25, .5, .75])
country_stats.to_csv('country_statistics_final.csv')

print("1. Đã xuất file thống kê tổng thể và quốc gia.")