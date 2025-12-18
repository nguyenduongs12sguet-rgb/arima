import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Tải và làm sạch
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna().sort_values('Year').reset_index(drop=True)

# 2. Chuẩn bị biến (Dùng 100% dữ liệu)
y = np.log(df['Yield (kg/ha)'])
exog_cols = ['Pesticides (total)', 'Average Mean Surface Air Temperature (Annual Mean)', 
             'Precipitation (mm)', 'Fertilizer (kg/ha)', 'Year']
X = sm.add_constant(df[exog_cols])

# 3. Huấn luyện mô hình trên TOÀN BỘ dữ liệu
model = ARIMA(y, exog=X, order=(5, 1, 1))
results = model.fit()

# 4. Tính toán Metrics trên toàn mẫu (In-sample metrics)
y_pred_log = results.fittedvalues # Giá trị mô hình khớp được
y_orig = np.exp(y)
y_pred_orig = np.exp(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_orig, y_pred_orig))
r2 = r2_score(y_orig, y_pred_orig)

print("\n--- KẾT QUẢ MÔ HÌNH TOÀN MẪU (FULL-SAMPLE) ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.2f} kg/ha")
print("\n", results.summary())

# 5. Vẽ biểu đồ Thực tế vs Khớp (Actual vs Fitted)
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], y_orig, 'b-', label='Thực tế', alpha=0.7)
plt.plot(df['Year'], y_pred_orig, 'r--', label='Mô hình khớp (Fitted)')
plt.fill_between(df['Year'], y_orig, y_pred_orig, color='gray', alpha=0.2, label='Sai số')

plt.title('ĐỘ KHỚP CỦA MÔ HÌNH ARIMAX TRÊN TOÀN BỘ DỮ LIỆU', fontsize=14, fontweight='bold')
plt.xlabel('Năm')
plt.ylabel('Năng suất (kg/ha)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()