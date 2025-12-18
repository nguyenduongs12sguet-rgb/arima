import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the data, using the correct filename and avoiding absolute path
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", delimiter=';')

# Data Cleaning (as performed in previous successful steps)
df.columns = df.columns.str.strip()
df = df.dropna()

# --- CHỌN BIẾN ---
Y = df['Yield (kg/ha)']

X = df[[
    'Year', 
    'Precipitation (mm)', 
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Fertilizer (kg/ha)',
    'Pesticides (total)'
]]

# --- PHÂN CHIA TRAIN/TEST (80/20 theo Thời gian) ---
# Sử dụng logic chia theo 80% năm duy nhất (đã xác định là 2011 trước đó)
unique_years = sorted(df['Year'].unique())
split_index = int(len(unique_years) * 0.8)
train_year_cut_off = unique_years[split_index]

# Chia dữ liệu
X_train = X[df['Year'] <= train_year_cut_off]
Y_train = Y[df['Year'] <= train_year_cut_off]

X_test = X[df['Year'] > train_year_cut_off]
Y_test = Y[df['Year'] > train_year_cut_off]

# Kiểm tra kích thước
print(f"Train set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# --- HUẤN LUYỆN MÔ HÌNH VÀ DỰ BÁO (Phần còn thiếu và gây lỗi) ---
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, Y_train)

# Dự báo trên cả tập Train và Test
Y_pred_train = rf_model.predict(X_train)
Y_pred_test = rf_model.predict(X_test)

# Gộp kết quả Train và Test
df_train_results = pd.DataFrame({'Year': X_train['Year'], 'Actual_Yield': Y_train, 'Predicted_Yield': Y_pred_train, 'Data_Type': 'Train'})
df_test_results = pd.DataFrame({'Year': X_test['Year'], 'Actual_Yield': Y_test, 'Predicted_Yield': Y_pred_test, 'Data_Type': 'Test'})

df_performance = pd.concat([df_train_results, df_test_results])

# Xuất dữ liệu ra file CSV trong thư mục hiện tại (tránh đường dẫn tuyệt đối)
df_performance.to_csv('rf_performance_data.csv', index=False, sep=';')
print("Đã xuất 'rf_performance_data.csv' (cho biểu đồ Actual vs Predicted) thành công.")