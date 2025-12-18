import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Đường dẫn file (Bạn cần thay đổi nếu cần)
file_path = r"E:\FINAL OF FINAL\data\phanbontieuthu.csv" # Giả sử tên file là new_fertilizer.csv

# 1. Đọc file
try:
    df_fertilizer = pd.read_csv(file_path, delimiter=';')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Kiểm tra TÊN CỘT GỐC
print("Tên các cột trong file Phân bón sau khi đọc:")
print(df_fertilizer.columns.tolist())

# 2. ĐỔI TÊN CỘT GỐC và tạo tên mới
# Cột Phân bón gốc thường là 'Fertilizer (kg/ha)' hoặc 'Fertilizer (kg/ha) '
# Giả sử tên gốc là 'Fertilizer (kg/ha)' (không có khoảng trắng ở cuối)
ORIGINAL_COL_NAME = 'Fertilizer (kg/ha)'
NEW_COL_NAME = 'Fertilizer_kg_ha'

# Thử đổi tên, nếu có lỗi có thể do khoảng trắng
try:
    df_fertilizer.rename(columns={ORIGINAL_COL_NAME: NEW_COL_NAME}, inplace=True)
except KeyError:
    # Nếu tên cột gốc có khoảng trắng ở cuối, thử lại
    ORIGINAL_COL_NAME = 'Fertilizer (kg/ha) '
    df_fertilizer.rename(columns={ORIGINAL_COL_NAME: NEW_COL_NAME}, inplace=True)


# 3. ÉP KIỂU dữ liệu (Cột mới đã được đổi tên)
# Lỗi 'KeyError' ban đầu đã xảy ra tại dòng này:
df_fertilizer[NEW_COL_NAME] = df_fertilizer[NEW_COL_NAME].astype(str).str.replace(',', '.', regex=True).astype(float)


# 4. THIẾT LẬP CÔNG THỨC ANOVA
formula = f'{NEW_COL_NAME} ~ C(Area)'

# 5. THỰC HIỆN ANOVA
print("\nBảng kết quả ANOVA (So sánh Lượng phân bón giữa các Quốc gia):")
lm = ols(formula, data=df_fertilizer).fit()
anova_table = anova_lm(lm)
print(anova_table)