import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load the data
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')

# Data Cleaning and Preparation 
df.columns = df.columns.str.strip()
df = df.dropna()

# --- CHỌN BIẾN ĐÃ SỬA (KHẮC PHỤC VIF) ---
# Chỉ giữ lại Pesticides (total) và các biến quan trọng khác
selected_cols = [
    'Yield (kg/ha)', 'Year', 'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 'Fertilizer (kg/ha)'
]
df_arimax = df[selected_cols]

# Sắp xếp dữ liệu theo năm (bắt buộc cho mô hình chuỗi thời gian)
df_sorted = df_arimax.sort_values(by=['Year']).reset_index(drop=True)

# Định nghĩa X và y
X_ar = df_sorted.drop('Yield (kg/ha)', axis=1)
y_ar = df_sorted['Yield (kg/ha)']

# Thêm hằng số
X_ar = sm.add_constant(X_ar)

# --- CHẠY MÔ HÌNH ARIMAX (KHẮC PHỤC TỰ TƯƠNG QUAN) ---
# order=(1, 0, 0): Tự hồi quy bậc 1 (AR=1)
# BẢN SỬA LỖI: Thêm trend='n' để ngăn ARIMA tự thêm hằng số lần nữa.
arma_model = sm.tsa.arima.ARIMA(
    y_ar, 
    exog=X_ar, 
    order=(1, 0, 0),
    trend='n' # <--- ĐÃ SỬA LỖI VALUEERROR
).fit()

print("\n--- KẾT QUẢ ARIMAX (KHẮC PHỤC TỰ TƯƠNG QUAN) ---")
print(arma_model.summary().as_text())