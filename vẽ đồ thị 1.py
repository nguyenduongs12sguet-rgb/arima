import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Đọc dữ liệu
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna()

selected_cols = ['Yield (kg/ha)', 'Year', 'Pesticides (total)', 
                 'Average Mean Surface Air Temperature (Annual Mean)', 
                 'Precipitation (mm)', 'Fertilizer (kg/ha)']
df_sorted = df[selected_cols].sort_values(by=['Year']).reset_index(drop=True)

# 2. Biến đổi Log và Huấn luyện
df_sorted['Log_Yield'] = np.log(df_sorted['Yield (kg/ha)'])
y = df_sorted['Log_Yield']
X = df_sorted.drop(['Yield (kg/ha)', 'Log_Yield'], axis=1)
X = sm.add_constant(X)

# Sử dụng bậc đơn giản hơn để tránh lỗi Singular Matrix (Đã hạ bậc AR xuống 2)
model = ARIMA(y, exog=X, order=(2, 1, 1), trend='n') 
results = model.fit()

# 3. Chuẩn bị dự báo
forecast_steps = 5
last_year = int(df_sorted['Year'].max())
future_years = np.arange(last_year + 1, last_year + 1 + forecast_steps)

future_exog = pd.DataFrame({
    'const': 1.0,
    'Year': future_years,
    'Pesticides (total)': df_sorted['Pesticides (total)'].iloc[-1],
    'Average Mean Surface Air Temperature (Annual Mean)': df_sorted['Average Mean Surface Air Temperature (Annual Mean)'].mean(),
    'Precipitation (mm)': df_sorted['Precipitation (mm)'].mean(),
    'Fertilizer (kg/ha)': df_sorted['Fertilizer (kg/ha)'].iloc[-1]
})

forecast_res = results.get_forecast(steps=forecast_steps, exog=future_exog)
forecast_final = np.exp(forecast_res.predicted_mean)
conf_int = forecast_res.conf_int()
lower_bound = np.exp(conf_int.iloc[:, 0])
upper_bound = np.exp(conf_int.iloc[:, 1])

# --- 4. VẼ ĐỒ THỊ (SỬA LỖI NAMEERROR VÀ TRỤC X) ---
plt.figure(figsize=(12, 6))

# ĐỊNH NGHĨA history_plot Ở ĐÂY
history_plot = df_sorted.tail(15) 

# Vẽ dữ liệu lịch sử
plt.plot(history_plot['Year'], history_plot['Yield (kg/ha)'], 
         color='blue', label='Lịch sử (15 năm cuối)', marker='o', linewidth=2)

# Vẽ dự báo
plt.plot(future_years, forecast_final, 
         color='red', linestyle='--', label='Dự báo 5 năm tới', marker='s')

# Vẽ khoảng tin cậy
plt.fill_between(future_years, lower_bound, upper_bound, color='red', alpha=0.1)

# Cấu hình trục X để không bị dồn cục
all_years = np.concatenate([history_plot['Year'].values, future_years])
plt.xticks(all_years, rotation=45) 

plt.title('Dự báo Sản lượng Nông nghiệp (ARIMAX)', fontsize=14)
plt.xlabel('Năm')
plt.ylabel('Yield (kg/ha)')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Chèn đoạn này vào cuối file "vẽ đồ thị 1.py" của bạn
plt.figure(figsize=(12, 7))
# ... (giữ nguyên các lệnh plt.plot cũ)

# Tối ưu giao diện cho báo cáo
plt.title('Dự báo Sản lượng Nông nghiệp (ARIMAX Model)', fontsize=16, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Xuất ảnh cực nét
plt.savefig('forecast_report.png', dpi=600, bbox_inches='tight')
print("Đã lưu ảnh forecast_report.png chất lượng cao!")