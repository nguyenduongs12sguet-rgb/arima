import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 1. KHẮC PHỤC LỖI ĐƯỜNG DẪN: SỬ DỤNG CHUỖI THÔ (RAW STRING)
# Bạn cần thay 'path_to_your_file' bằng đường dẫn thực tế của bạn
# Ví dụ: r"E:\FINAL OF FINAL\data\nangsuat_potatoes_only.csv"
file_path = r"E:\FINAL OF FINAL\data\nangsuat_potatoes_only.csv"

# 2. KHẮC PHỤC LỖI ĐỌC FILE: Dùng delimiter=';'
# Tải file năng suất gốc (được biết có cột Area và Yield)
try:
    df_yield = pd.read_csv(file_path, delimiter=';')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    # Kết thúc chương trình nếu không tìm thấy file
    exit()

# 3. ĐỔI TÊN CỘT ĐỂ DỄ DÀNG SỬ DỤNG TRONG CÔNG THỨC
# Đảm bảo tên cột không có khoảng trắng
df_yield.rename(columns={'Yield (kg/ha)': 'Yield_kg_ha'}, inplace=True)

# 4. THIẾT LẬP CÔNG THỨC ANOVA
# So sánh 'Yield_kg_ha' (biến định lượng) giữa các nhóm 'Area' (biến định tính)
# Không cần dùng Q() vì đã đổi tên cột không có khoảng trắng
formula = 'Yield_kg_ha ~ C(Area)'

# 5. THỰC HIỆN ANOVA
print("\nBảng kết quả ANOVA (So sánh Năng suất giữa các Quốc gia):")
lm = ols(formula, data=df_yield).fit()
anova_table = anova_lm(lm)
print(anova_table)

#Giả thuyết (Hypotheses)
#- Giả thuyết không (H0): Không có sự khác biệt về năng suất giữa các khu vực (Area).
#- Giả thuyết thay thế (H1): Có sự khác biệt về năng suất giữa ít nhất hai khu vực (Area).
#Giá trị $P$ là $0.0$, nhỏ hơn rất nhiều so với mức ý nghĩa thông thường $\alpha = 0.05$.
#Do đó, chúng ta bác bỏ giả thuyết không (H0) và kết luận rằng có sự khác biệt có ý nghĩa thống kê về năng suất giữa các khu vực (Area).
#Có sự khác biệt có ý nghĩa thống kê về năng suất khoai tây trung bình giữa các quốc gia Châu Âu trong tập dữ liệu của bạn.
#Nói cách khác, năng suất khoai tây không đồng nhất giữa 30 quốc gia này. Yếu tố Quốc gia/Khu vực có ảnh hưởng đáng kể đến năng suất.