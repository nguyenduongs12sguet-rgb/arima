import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- 1. ĐỌC DỮ LIỆU (Định nghĩa biến df ở đây) ---
try:
    df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
    df.columns = df.columns.str.strip()
    df = df.dropna()
    print("Cơ sở dữ liệu đã được tải thành công!")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file tại đường dẫn E:\FINAL OF FINAL\data\merged_data.csv")

# --- 2. CHUẨN BỊ BIẾN ---
# Chỉ lấy các yếu tố tác động thực sự
exog_cols = [
    'Pesticides (total)', 
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 
    'Fertilizer (kg/ha)'
]

X = df[exog_cols]
y = df['Yield (kg/ha)']

# --- 3. TÍNH TOÁN FEATURE IMPORTANCE ---
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)

# Tạo bảng dữ liệu quan trọng
importance_df = pd.DataFrame({
    'Yếu tố': ['Thuốc trừ sâu', 'Nhiệt độ TB', 'Lượng mưa', 'Phân bón'],
    'Độ quan trọng': rf.feature_importances_
}).sort_values(by='Độ quan trọng', ascending=True)

# --- 4. VẼ BIỂU ĐỒ "COOK" CỰC CHUẨN ---
plt.figure(figsize=(12, 7))
sns.set_style("white")

# Sử dụng bảng màu chuyên nghiệp
colors = sns.color_palette("viridis", len(importance_df))

# Vẽ thanh ngang với bo góc nhẹ
bars = plt.barh(importance_df['Yếu tố'], importance_df['Độ quan trọng'], 
                color=colors, edgecolor='none', height=0.6)

# Thêm đường kẻ phụ mờ cho chuyên nghiệp
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Ghi giá trị % lên đầu thanh để tạo điểm nhấn
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width*100:.1f}%', 
             va='center', fontsize=13, fontweight='bold', color='#2c3e50')

# Tinh chỉnh tiêu đề và nhãn
plt.title('MỨC ĐỘ QUAN TRỌNG CỦA CÁC YẾU TỐ ĐẾN NĂNG SUẤT', 
          fontsize=18, fontweight='bold', pad=30, loc='center')
plt.xlabel('Tỷ lệ đóng góp (Feature Importance)', fontsize=14, color='gray')
plt.xlim(0, max(importance_df['Độ quan trọng']) + 0.1)

# Xóa khung biểu đồ cho thoáng
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()