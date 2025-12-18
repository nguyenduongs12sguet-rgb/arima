import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load the data with the semicolon delimiter
df_pesticide = pd.read_csv(r"E:\FINAL OF FINAL\data\thuoctrusau.csv", delimiter=';')

# Thiết lập công thức ANOVA:
# "Pesticides (total)" (Tổng lượng thuốc trừ sâu) là biến định lượng (Y)
# "Area" (Quốc gia) là biến định tính (X)
# Dùng Q("") để xử lý khoảng trắng và ký tự đặc biệt trong tên cột
formula = 'Q("Pesticides (total)") ~ C(Area)'

# Fit the OLS model
lm = ols(formula, data=df_pesticide).fit()

# Tạo bảng ANOVA
anova_table = anova_lm(lm)
print(anova_table)