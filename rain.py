import pandas as pd

# Load the data
file_name = "E:\\FINAL OF FINAL\\data\\newrain.csv"
delimiter = ';'
df = pd.read_csv(file_name, delimiter=delimiter)

# Làm sạch tên cột (loại bỏ khoảng trắng thừa)
df.columns = df.columns.str.strip()

# Xác định các cột số liệu đã làm sạch
numerical_cols = ['Precipitation (mm)']#, 'AVERAGE PRECIPITATION']

# Làm sạch dữ liệu: thay thế dấu phẩy bằng dấu chấm và chuyển đổi sang kiểu số
for col in numerical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('"', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Loại bỏ các hàng có giá trị không hợp lệ
df.dropna(subset=numerical_cols, inplace=True)

# 1. Tính thống kê cho toàn bộ bộ dữ liệu
all_data_stats = df[numerical_cols].agg(['mean', 'median', 'std', 'max', 'min'])
all_data_stats.to_csv('newrain_all_data_statistics.csv')

# 2. Tính thống kê theo từng quốc gia
country_stats = df.groupby('Area')[numerical_cols].agg(['mean', 'median', 'std', 'max', 'min'])
country_stats.to_csv('newrain_country_statistics.csv')

# In kết quả
print("\n--- 1. Thống kê mô tả cho TOÀN BỘ dữ liệu ---")
print(all_data_stats)
print("\n--- 2. Thống kê mô tả theo TỪNG QUỐC GIA ---")
print(country_stats)