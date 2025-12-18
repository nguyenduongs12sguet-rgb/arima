import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Giả định df_sorted, y, X đã có từ bước trước
# Chúng ta sẽ thử nghiệm các bậc thấp hơn để so sánh BIC
orders_to_compare = [(5, 1, 1), (2, 1, 1), (1, 1, 1), (0, 1, 1)]
bic_results = []

print("--- SO SÁNH CHỈ SỐ BIC ---")
for order in orders_to_compare:
    try:
        tmp_model = ARIMA(y, exog=X, order=order, trend='n').fit()
        bic_results.append({'Order': order, 'BIC': tmp_model.bic, 'AIC': tmp_model.aic})
    except:
        continue

bic_df = pd.DataFrame(bic_results).sort_values(by='BIC')
print(bic_df)

# --- KIỂM TRA OVERFITTING BẰNG TRAIN/TEST SPLIT ---
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

model_train = ARIMA(y_train, exog=X_train, order=(5, 1, 1), trend='n').fit()
predictions_log = model_train.forecast(steps=len(y_test), exog=X_test)

# Tính toán sai số (RMSE) trên tập Test
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(predictions_log)))
print(f"\n--- KIỂM TRA OVERFITTING ---")
print(f"RMSE trên tập Test (đơn vị gốc kg/ha): {rmse:.2f}")