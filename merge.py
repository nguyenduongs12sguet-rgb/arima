import pandas as pd

# Đọc các file CSV với dấu phân cách ';'
df_pesticides = pd.read_csv(r"E:\FINAL OF FINAL\data\thuoctrusau.csv", sep=';')  # Thuốc trừ sâu
df_rain = pd.read_csv(r"E:\FINAL OF FINAL\data\newrain.csv", sep=';')  # Lượng mưa
df_potatoes = pd.read_csv(r"E:\FINAL OF FINAL\data\nangsuat_potatoes_only.csv", sep=';')  # Năng suất khoai tây
df_temp = pd.read_csv(r"E:\FINAL OF FINAL\data\temperature.csv", sep=';')  # Nhiệt độ
df_fertilizer = pd.read_csv(r"E:\FINAL OF FINAL\data\phanbontieuthu.csv", sep=';')  # Phân bón

# Chuẩn hóa tên cột (xử lý khoảng trắng hoặc cột dư thừa)
df_rain = df_rain.rename(columns={'Precipitation (mm) ': 'Precipitation (mm)'})  # Xóa khoảng trắng
df_temp = df_temp.rename(columns={'Average Mean Surface Air Temperature (Annual Mean )': 'Average Mean Surface Air Temperature (Annual Mean)'})  # Xóa khoảng trắng
df_fertilizer = df_fertilizer.drop(columns=['Indicator Name'], errors='ignore')  # Bỏ cột không cần thiết nếu có

# Merge tất cả DataFrame dựa trên 'Area' và 'Year' (outer join để giữ đầy đủ dữ liệu)
merged_df = df_pesticides.merge(df_rain, on=['Area', 'Year'], how='outer')
merged_df = merged_df.merge(df_potatoes, on=['Area', 'Year'], how='outer')
merged_df = merged_df.merge(df_temp, on=['Area', 'Year'], how='outer')
merged_df = merged_df.merge(df_fertilizer, on=['Area', 'Year'], how='outer')

# Sắp xếp theo 'Area' và 'Year' để dễ đọc
merged_df = merged_df.sort_values(by=['Area', 'Year'])

# Lưu file kết hợp mới (dùng ';' làm dấu phân cách để tương thích)
merged_df.to_csv('merged_data.csv', index=False, sep=';')

# In preview (đầu 10 dòng) và thông tin
print(merged_df.head(10))
print("\nKích thước DataFrame kết hợp:", merged_df.shape)
print("\nCác cột:", list(merged_df.columns))