import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

file_path = r"E:\FINAL OF FINAL\data\temperature.csv"

try:
    df_temp = pd.read_csv(file_path, delimiter = ';')
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1) 

print("Tên các cột trong file Nhiệt độ sau khi đọc:")
print(df_temp.columns.tolist())

# Tên cột gốc nhiệt độ (rất dài) và tên cột mới
ORIGINAL_TEMP_COL = 'Average Mean Surface Air Temperature (Annual Mean )'
NEW_TEMP_COL = 'Annual_Mean_Temperature'

# 1. Logic Đổi tên cột
# Thử đổi tên, nếu thất bại (do có khoảng trắng ở cuối) thì thử tên có khoảng trắng
try:
    df_temp.rename(columns={ORIGINAL_TEMP_COL: NEW_TEMP_COL}, inplace=True)
except KeyError:
    # Nếu KeyError, thử tên cột có khoảng trắng ở cuối
    ORIGINAL_TEMP_COL_WITH_SPACE = 'Average Mean Surface Air Temperature (Annual Mean ) '
    df_temp.rename(columns={ORIGINAL_TEMP_COL_WITH_SPACE: NEW_TEMP_COL}, inplace=True)

# 2. ÉP KIỂU dữ liệu (Sử dụng tên cột mới)
# Ép kiểu từ chuỗi (dùng dấu phẩy) sang số thực
df_temp[NEW_TEMP_COL] = df_temp[NEW_TEMP_COL].astype(str).str.replace(',', '.', regex=True).astype(float)


# 3. THIẾT LẬP CÔNG THỨC ANOVA
# Sử dụng tên cột mới đã được ép kiểu: Annual_Mean_Temperature
formula = f'{NEW_TEMP_COL} ~ C(Area)'

# 4. THỰC HIỆN ANOVA
print("\nBảng kết quả ANOVA (So sánh Nhiệt độ trung bình giữa các Quốc gia):")
lm = ols(formula, data=df_temp).fit()
anova_table = anova_lm(lm)
print(anova_table)