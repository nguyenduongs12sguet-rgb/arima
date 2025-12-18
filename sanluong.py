import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =================================================================
# 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# =================================================================
try:
    # sep=None và engine='python' giúp tự động phát hiện dấu phân cách (phẩy hoặc chấm phẩy)
    # encoding='utf-8-sig' để đọc đúng tiếng Việt có dấu
    df = pd.read_csv(r"C:\\Users\\ht\\Downloads\\sanluong_caytrong.csv", sep=None, engine='python', encoding='utf-8-sig')
    
    # Xóa khoảng trắng thừa trong tên cột nếu có
    df.columns = [c.strip() for c in df.columns]
    
    print("--- Cấu trúc dữ liệu ban đầu ---")
    print(df.head())

    # Chuyển đổi sản lượng về dạng số, các giá trị lỗi như 'abc' sẽ thành NaN
    df['SanLuong(Tan)'] = pd.to_numeric(df['SanLuong(Tan)'], errors='coerce')

    # Xử lý giá trị thiếu (NaN) bằng phương pháp nội suy (Interpolation) 
    # Điều này giúp điền các năm trống dựa trên xu hướng của các năm trước và sau đó
    df['SanLuong(Tan)'] = df.groupby('LoaiCayTrong')['SanLuong(Tan)'].transform(
        lambda x: x.interpolate().fillna(method='bfill').fillna(method='ffill')
    )

    print("\n--- Dữ liệu sau khi làm sạch ---")
    print(df.info())

except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'sanluong_caytrong.csv'. Hãy kiểm tra tên file.")
    exit()
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
    exit()

# =================================================================
# 2. PHÂN TÍCH XU HƯỚNG (ANALYSIS)
# =================================================================
# Tính tỷ lệ tăng trưởng hàng năm (%)
df = df.sort_values(['LoaiCayTrong', 'Nam'])
df['%_Tang_Truong'] = df.groupby('LoaiCayTrong')['SanLuong(Tan)'].pct_change() * 100

print("\n--- Thống kê sản lượng trung bình hàng năm ---")
print(df.groupby('LoaiCayTrong')['SanLuong(Tan)'].mean())

# =================================================================
# 3. DỰ BÁO TƯƠNG LAI (PREDICTION) & VẼ BIỂU ĐỒ
# =================================================================
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['Arial'] # Hỗ trợ hiển thị tiếng Việt (tùy máy)

# Các năm cần dự báo
nam_du_bao = np.array(range(2024, 2029)).reshape(-1, 1)

for loai_cay in df['LoaiCayTrong'].unique():
    subset = df[df['LoaiCayTrong'] == loai_cay]
    
    # Chuẩn bị dữ liệu cho mô hình Hồi quy tuyến tính
    X = subset[['Nam']].values
    y = subset['SanLuong(Tan)'].values
    
    # Tạo và huấn luyện mô hình
    model = LinearRegression() 
    model.fit(X, y)
    
    # Dự báo cho 5 năm tới
    y_pred = model.predict(nam_du_bao)
    
    # Vẽ dữ liệu lịch sử
    plt.plot(subset['Nam'], y, marker='o', label=f'{loai_cay} (Quá khứ)')
    
    # Vẽ dữ liệu dự báo (đường đứt đoạn)
    plt.plot(range(2024, 2029), y_pred, linestyle='--', marker='s', label=f'{loai_cay} (Dự báo 2028)')
    
    # In kết quả dự báo ra màn hình
    print(f"\nDự báo cho {loai_cay} năm 2028: {y_pred[-1]:.2f} Tấn")

# Chú thích thêm về năm hạn hán 2003
han_han_2003 = df[(df['Nam'] == 2003) & (df['LoaiCayTrong'] == 'Lúa')]
if not han_han_2003.empty:
    plt.annotate('Sự kiện Hạn hán', xy=(2003, han_han_2003['SanLuong(Tan)'].iloc[0]), 
                 xytext=(2000, 6000), arrowprops=dict(facecolor='red', shrink=0.05))

plt.title('PHÂN TÍCH XU HƯỚNG VÀ DỰ BÁO SẢN LƯỢNG CÂY TRỒNG', fontsize=14)
plt.xlabel('Năm')
plt.ylabel('Sản lượng (Tấn)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Hiển thị và lưu biểu đồ
plt.show()
plt.savefig('bieu_do_xu_huong.png')

# Xuất dữ liệu đã làm sạch ra file mới
df.to_csv('du_lieu_nong_nghiep_sach.csv', index=False, encoding='utf-8-sig')
print("\nHoàn tất! Biểu đồ đã được lưu và dữ liệu sạch đã được xuất ra file CSV.")