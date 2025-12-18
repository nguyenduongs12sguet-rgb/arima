import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 1. KHẮC PHỤC LỖI ĐƯỜNG DẪN (Sử dụng chuỗi thô)
file_path = r"E:\FINAL OF FINAL\data\newrain.csv"

# 2. KHẮC PHỤC LỖI ĐỌC FILE: Dùng delimiter=';'
try:
    # Đọc file với dấu chấm phẩy
    df_precipitation = pd.read_csv(file_path, delimiter=';')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# 3. CHUẨN BỊ DỮ LIỆU VÀ ĐỔI TÊN CỘT
# Tên cột Lượng mưa gốc có khoảng trắng ở cuối: 'Precipitation (mm) '
# Đổi tên và chuyển đổi dữ liệu sang dạng số (float)
df_precipitation.rename(columns={'Precipitation (mm) ': 'Rainfall_mm'}, inplace=True)

# Chuyển đổi dữ liệu từ chuỗi (có dấu ",") sang số thực (float)
df_precipitation['Rainfall_mm'] = df_precipitation['Rainfall_mm'].astype(str).str.replace(',', '.', regex=True).astype(float)

# 4. THIẾT LẬP CÔNG THỨC ANOVA
# So sánh 'Rainfall_mm' (biến định lượng) giữa các nhóm 'Area' (biến định tính)
# Đã sử dụng tên cột mới không khoảng trắng
formula = 'Rainfall_mm ~ C(Area)'

# 5. THỰC HIỆN ANOVA
print("\nBảng kết quả ANOVA (So sánh Lượng mưa giữa các Quốc gia):")
lm = ols(formula, data=df_precipitation).fit()
anova_table = anova_lm(lm)
print(anova_table)