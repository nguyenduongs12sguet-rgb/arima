import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. TẢI DỮ LIỆU ---
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna().sort_values(by=['Year']).reset_index(drop=True)

# --- 2. CHUẨN BỊ BIẾN ---
# Biến mục tiêu (Log để ổn định)
y = np.log(df['Yield (kg/ha)'])

# Các biến độc lập (Exogenous)
exog_cols = ['Pesticides (total)', 'Average Mean Surface Air Temperature (Annual Mean)', 
             'Precipitation (mm)', 'Fertilizer (kg/ha)']
X = sm.add_constant(df[exog_cols])

# Chia dữ liệu theo thời gian (80% Train, 20% Test)
split_idx = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

# --- 3. CHẠY MÔ HÌNH THỐNG KÊ ARIMAX ---
# Sử dụng order (1,1,1) là cấu trúc chuẩn, vững chắc
model = ARIMA(y_train, exog=X_train, order=(1, 1, 1))
results = model.fit()

# --- 4. TÍNH TOÁN SAI SỐ (METRICS) ---
y_pred_log = results.forecast(steps=len(y_test), exog=X_test)
y_test_orig = np.exp(y_test)      # Trả về đơn vị gốc kg/ha
y_pred_orig = np.exp(y_pred_log)  # Trả về đơn vị gốc kg/ha

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)

# Tạo bảng Metric
metrics_table = pd.DataFrame({'Chỉ số': ['RMSE (Sai số)', 'R-squared (Độ khớp)'],
                              'Giá trị': [f"{rmse:.2f} kg/ha", f"{r2:.4f}"]})

# --- 5. VẼ BIỂU ĐỒ CỘT RMSE ---
plt.figure(figsize=(8, 6))
sns.set_style("white")
bars = plt.bar(['ARIMAX Model'], [rmse], color='#5dade2', width=0.4, edgecolor='black')

# Điểm nhấn: Ghi số RMSE lên đầu cột
plt.text(0, rmse + (rmse*0.05), f'{rmse:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('ĐÁNH GIÁ SAI SỐ MÔ HÌNH (RMSE)', fontsize=15, fontweight='bold')
plt.ylabel('Giá trị RMSE (kg/ha)', fontsize=12)
plt.ylim(0, rmse * 1.3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()

print("\n--- BẢNG THỐNG KÊ KẾT QUẢ ---")
print(metrics_table)