import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Nhập dữ liệu từ file CSV
file_path = r"C:\Users\ht\Downloads\sanluong_caytrong.csv"
# Sử dụng sep=None để tự động nhận diện dấu phân cách như đã phân tích trước đó
df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')

# 2. Loại bỏ các cột trống
df = df.dropna(axis=1, how='all')

# 3. Tiền xử lý dữ liệu
# Chuyển đổi tên cột để khớp với logic (loại bỏ khoảng trắng nếu có)
df.columns = [c.strip() for c in df.columns]

# Chuyển đổi SanLuong(Tan) sang số
df['SanLuong(Tan)'] = pd.to_numeric(df['SanLuong(Tan)'], errors='coerce')

# Loại bỏ các hàng có giá trị NaN (bao gồm cả dòng có 'abc' đã chuyển thành NaN)
df = df.dropna(subset=['SanLuong(Tan)', 'Nam', 'LoaiCayTrong'])

print("--- Dữ liệu sau khi tiền xử lý ---")
print(df.head())

# 4. Vẽ biểu đồ riêng biệt cho từng loại cây trồng
unique_crops = df['LoaiCayTrong'].unique()

for crop in unique_crops:
    plt.figure(figsize=(10, 6))
    subset = df[df['LoaiCayTrong'] == crop]
    
    # Vẽ biểu đồ phân tán
    plt.scatter(subset['Nam'], subset['SanLuong(Tan)'], color='blue', alpha=0.6, label='Dữ liệu thực tế')
    
    # Tạo mô hình hồi quy cho từng loại cây trồng (Biến độc lập là Năm)
    X_subset = sm.add_constant(subset[['Nam']])
    model_subset = sm.OLS(subset['SanLuong(Tan)'], X_subset).fit()
    
    # Dự đoán sản lượng
    predicted_yield = model_subset.predict(X_subset)
    
    # Vẽ đường hồi quy
    plt.plot(subset['Nam'], predicted_yield, color='red', linewidth=2, label='Đường hồi quy (Xu hướng)')
    
    plt.title(f'Mối quan hệ giữa Năm và Sản lượng - {crop}')
    plt.xlabel('Năm')
    plt.ylabel('Sản lượng (Tấn)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'hoi_quy_{crop}.png')
    plt.close()

# 5. Phân tích mô tả
print("\nThống kê mô tả sản lượng theo loại cây trồng:")
print(df.groupby('LoaiCayTrong')['SanLuong(Tan)'].describe())

# 6. Vẽ biểu đồ sản lượng theo loại cây trồng (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='LoaiCayTrong', y='SanLuong(Tan)', data=df)
plt.title('Sản lượng cây trồng theo loại cây')
plt.ylabel('Sản lượng (Tấn)')
plt.xlabel('Loại cây trồng')
plt.savefig('boxplot_san_luong.png')
plt.close()

# 7. Phân tích hồi quy tổng thể (Xét ảnh hưởng của Năm tới Sản lượng nói chung)
X = df[['Nam']]
y = df['SanLuong(Tan)']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("\nKết quả hồi quy tổng thể:")
print(model.summary())

# 8. Đánh giá hiệu quả (Sản lượng trung bình)
mean_yield = df.groupby('LoaiCayTrong')['SanLuong(Tan)'].mean()
print("\nSản lượng trung bình theo loại cây trồng:")
print(mean_yield)

# 9. Tìm năm có sản lượng cao nhất (Tương ứng logic tìm liều lượng tối ưu)
optimal_years = {}
for crop in df['LoaiCayTrong'].unique():
    subset = df[df['LoaiCayTrong'] == crop]
    if not subset.empty:
        # Tìm năm có sản lượng thực tế cao nhất trong dữ liệu
        max_idx = subset['SanLuong(Tan)'].idxmax()
        optimal_years[crop] = {
            'Nam': subset.loc[max_idx, 'Nam'],
            'SanLuong': subset.loc[max_idx, 'SanLuong(Tan)']
        }

print("\nNăm đạt sản lượng cao nhất cho từng loại cây trồng:")
for crop, info in optimal_years.items():
    print(f"{crop}: Năm {info['Nam']} đạt {info['SanLuong']:.2f} tấn")