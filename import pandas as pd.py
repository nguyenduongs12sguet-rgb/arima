import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. XỬ LÝ DỮ LIỆU ---
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", delimiter=';')
df.columns = df.columns.str.strip()
df = df.dropna().sort_values('Year')

Y = df['Yield (kg/ha)']
X = df[['Year', 'Precipitation (mm)', 'Average Mean Surface Air Temperature (Annual Mean)', 
        'Fertilizer (kg/ha)', 'Pesticides (total)']]

unique_years = sorted(df['Year'].unique())
split_index = int(len(unique_years) * 0.8)
train_year_cut_off = unique_years[split_index]

X_train, Y_train = X[df['Year'] <= train_year_cut_off], Y[df['Year'] <= train_year_cut_off]
X_test, Y_test = X[df['Year'] > train_year_cut_off], Y[df['Year'] > train_year_cut_off]

# --- 2. MÔ HÌNH (ĐÃ TỐI ƯU CHỐNG OVERFITTING) ---
rf_model = RandomForestRegressor(n_estimators=500, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf_model.fit(X_train, Y_train)

Y_pred_train = rf_model.predict(X_train)
Y_pred_test = rf_model.predict(X_test)
test_r2 = r2_score(Y_test, Y_pred_test)

# --- 3. VẼ BIỂU ĐỒ CÓ ĐIỂM NHẤN ---
sns.set_style("whitegrid") # Tạo nền trắng sạch sẽ
plt.figure(figsize=(14, 7))

# Vẽ đường Thực tế và Dự báo với độ dày khác nhau để tạo điểm nhấn
plt.plot(df['Year'], Y, color="#4f502c", linewidth=2.5, label='Thực tế (Actual)', marker='o', markersize=6, alpha=0.8)
plt.plot(X_train['Year'], Y_pred_train, color="#e65a22", linewidth=2, linestyle='--', label='Dự báo Train')
plt.plot(X_test['Year'], Y_pred_test, color="#e73c97", linewidth=3, label='Dự báo Test (Tương lai)', marker='s')

# Tạo vùng màu nền để phân tách Train và Test
plt.axvspan(unique_years[0], train_year_cut_off, color='gray', alpha=0.1, label='Giai đoạn Huấn luyện')
plt.axvspan(train_year_cut_off, unique_years[-1], color='yellow', alpha=0.1, label='Giai đoạn Dự báo Test')

# Thêm đường kẻ dọc phân cách rõ ràng
plt.axvline(x=train_year_cut_off, color='black', linestyle='-', linewidth=1)
plt.text(train_year_cut_off - 0.5, plt.ylim()[1]*0.95, 'QUÁ KHỨ', horizontalalignment='right', fontweight='bold', color='gray')
plt.text(train_year_cut_off + 0.5, plt.ylim()[1]*0.95, 'DỰ BÁO MỚI', horizontalalignment='left', fontweight='bold', color='red')

# Hiển thị chỉ số R2 ngay trên biểu đồ
plt.annotate(f'Độ chính xác (R²): {test_r2:.2f}', 
             xy=(0.05, 0.9), xycoords='axes fraction',
             fontsize=14, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

# Tinh chỉnh trục và tiêu đề
plt.title('PHÂN TÍCH VÀ DỰ BÁO NĂNG SUẤT CÂY TRỒNG (RANDOM FOREST)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Năm', fontsize=12)
plt.ylabel('Năng suất (kg/ha)', fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.show()