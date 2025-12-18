import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 1. Tải và làm sạch dữ liệu
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna()

# 2. Chuẩn bị biến
selected_cols = [
    'Yield (kg/ha)', 'Year', 'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 'Fertilizer (kg/ha)'
]
df_sorted = df[selected_cols].sort_values(by=['Year']).reset_index(drop=True)

# --- KHẮC PHỤC PHƯƠNG SAI THAY ĐỔI & PHÂN PHỐI ---
# Lấy Logarithm của biến mục tiêu
df_sorted['Log_Yield'] = np.log(df_sorted['Yield (kg/ha)'])

y = df_sorted['Log_Yield']
X = df_sorted.drop(['Yield (kg/ha)', 'Log_Yield'], axis=1)
X = sm.add_constant(X)

# --- KHẮC PHỤC TỰ TƯƠNG QUAN (ARIMA p,d,q) ---
# Theo gợi ý, chúng ta thêm thành phần q (MA) để làm mượt phần dư
# Thử nghiệm với order=(5, 1, 1) hoặc dùng auto_arima để tìm con số này
model = ARIMA(
    y, 
    exog=X, 
    order=(5, 1, 1), # Thêm q=1 để xử lý lỗi như bạn đã nêu
    trend='n'
)
results = model.fit()

# 3. In kết quả đánh giá
print("\n--- KẾT QUẢ MÔ HÌNH ARIMAX TỐI ƯU ---")
print(results.summary())

# 4. Kiểm tra lại phần dư (Diagnostics)
results.plot_diagnostics(figsize=(12, 8))
plt.show()

# 5. Hàm dự báo (Lưu ý phải dùng np.exp để đưa về đơn vị gốc)
# forecast_log = results.forecast(steps=1, exog=new_X)
# forecast_original = np.exp(forecast_log)