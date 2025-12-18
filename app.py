# app.py (Python 3.8 compatible)
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Dự báo Năng suất 20 năm (ARIMAX)", layout="wide")

# =========================
# 1) CẤU HÌNH CỘT (SỬA NẾU FILE CỦA BẠN KHÁC TÊN)
# =========================
COL_AREA = "Area"
COL_YEAR = "Year"
COL_YIELD = "Yield (kg/ha)"
COL_PEST = "Pesticides (total)"
COL_TEMP = "Average Mean Surface Air Temperature (Annual Mean)"
COL_RAIN = "Precipitation (mm)"
COL_FERT = "Fertilizer (kg/ha)"

REQUIRED_COLS = [COL_AREA, COL_YEAR, COL_YIELD, COL_PEST, COL_TEMP, COL_RAIN, COL_FERT]


# =========================
# 2) HÀM TIỆN ÍCH
# =========================
def clean_numeric_series(s):
    """Chuyển chuỗi có dấu ',' thành float; ép kiểu an toàn."""
    if s is None:
        return s
    if s.dtype == "object":
        s = (
            s.astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    """
    Load CSV từ upload hoặc từ file cục bộ.
    - Ưu tiên ';' (dataset của bạn thường dùng)
    - Fallback sang ',' nếu cần
    """
    if file_bytes is not None:
        # Đọc từ upload
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(
                    pd.io.common.BytesIO(file_bytes),
                    sep=sep,
                    encoding="latin1"
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        raise ValueError("Không đọc được file upload. Hãy kiểm tra định dạng CSV/encoding.")
    else:
        if not file_name:
            raise ValueError("Chưa có file dữ liệu.")
        # Đọc từ file cục bộ
        try:
            return pd.read_csv(file_name, sep=";", encoding="latin1")
        except Exception:
            return pd.read_csv(file_name, encoding="latin1")


def prepare_df(df_raw):
    """Làm sạch dữ liệu và đảm bảo đủ cột cần thiết."""
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Thiếu cột: {}\n"
            "=> Hãy sửa các biến COL_* ở đầu file app.py cho khớp tên cột trong CSV.\n"
            "Danh sách cột hiện có:\n{}".format(missing, df.columns.tolist())
        )

    # Ép kiểu Year
    df[COL_YEAR] = clean_numeric_series(df[COL_YEAR])
    df = df.dropna(subset=[COL_YEAR])
    df[COL_YEAR] = df[COL_YEAR].astype(int)

    # Ép kiểu số cho các cột numeric
    for c in [COL_YIELD, COL_PEST, COL_TEMP, COL_RAIN, COL_FERT]:
        df[c] = clean_numeric_series(df[c])

    # Bỏ dòng thiếu
    df = df.dropna(subset=[COL_AREA, COL_YEAR, COL_YIELD, COL_PEST, COL_TEMP, COL_RAIN, COL_FERT])

    # Loại bỏ Yield <= 0 để tránh lỗi log
    df = df[df[COL_YIELD] > 0]

    # Sắp xếp
    df = df.sort_values([COL_AREA, COL_YEAR]).reset_index(drop=True)
    return df


def build_xy(df_one_area):
    """Tạo y=log(Yield) và X=exog + const."""
    df_one_area = df_one_area.sort_values(COL_YEAR).reset_index(drop=True)

    y = np.log(df_one_area[COL_YIELD].astype(float))
    X = df_one_area[[COL_YEAR, COL_PEST, COL_TEMP, COL_RAIN, COL_FERT]].copy()
    X = sm.add_constant(X, has_constant="add")
    return y, X


@st.cache_resource(show_spinner=False)
def fit_arimax(y, X, order):
    """Fit ARIMAX."""
    model = ARIMA(y, exog=X, order=order, trend="n")
    res = model.fit()
    return res


def make_future_exog(last_row, years, temp_delta_total, rain_pct, fert_pct, pest_pct):
    """
    Tạo exog tương lai theo kịch bản đơn giản:
    - Year tăng dần
    - Temp tăng tuyến tính, tổng tăng = temp_delta_total sau N năm
    - Rain/Fert/Pest giữ nền và điều chỉnh theo %
    """
    last_year = int(last_row[COL_YEAR])
    future_years = np.arange(last_year + 1, last_year + 1 + years)

    base_temp = float(last_row[COL_TEMP])
    base_rain = float(last_row[COL_RAIN])
    base_fert = float(last_row[COL_FERT])
    base_pest = float(last_row[COL_PEST])

    # temp tăng tuyến tính tới tổng delta
    temp_trend = np.linspace(0.0, float(temp_delta_total), years)
    temp_vals = base_temp + temp_trend

    rain_val = base_rain * (1.0 + float(rain_pct) / 100.0)
    fert_val = base_fert * (1.0 + float(fert_pct) / 100.0)
    pest_val = base_pest * (1.0 + float(pest_pct) / 100.0)

    Xf = pd.DataFrame({
        COL_YEAR: future_years,
        COL_PEST: np.full(years, pest_val),
        COL_TEMP: temp_vals,
        COL_RAIN: np.full(years, rain_val),
        COL_FERT: np.full(years, fert_val),
    })
    Xf = sm.add_constant(Xf, has_constant="add")
    return Xf


def forecast_future(results, X_future):
    """Dự báo và trả về đơn vị gốc (kg/ha) + CI 95%."""
    fc = results.get_forecast(steps=len(X_future), exog=X_future)
    pred_log = fc.predicted_mean
    ci = fc.conf_int()

    out = pd.DataFrame({
        "Year": X_future[COL_YEAR].astype(int).values,
        "Forecast_Yield": np.exp(pred_log).values,
        "Lower_95": np.exp(ci.iloc[:, 0]).values,
        "Upper_95": np.exp(ci.iloc[:, 1]).values,
    })
    return out


# =========================
# 3) UI
# =========================
st.title("🌾 Web dự đoán Năng suất 20 năm tương lai (ARIMAX)")

with st.sidebar:
    st.header("📁 Dữ liệu")
    uploaded = st.file_uploader("Upload CSV (khuyến nghị)", type=["csv"])

    candidates = ["merged_data.csv", "du_lieu_nong_nghiep_sach.csv"]
    local_file = None
    if uploaded is None:
        for fn in candidates:
            if os.path.exists(fn):
                local_file = fn
                break
        st.caption("Không upload → dùng file cục bộ: {}".format(local_file) if local_file else
                   "Không tìm thấy file cục bộ, hãy upload CSV.")

    st.divider()
    st.header("🧠 Cấu hình mô hình (p,d,q)")
    p = st.number_input("p", min_value=0, max_value=10, value=5)
    d = st.number_input("d", min_value=0, max_value=2, value=1)
    q = st.number_input("q", min_value=0, max_value=10, value=1)

    st.divider()
    st.header("🕹️ Kịch bản tương lai")
    years_to_forecast = st.slider("Số năm dự báo", 1, 30, 20)

    temp_delta_total = st.slider("Tổng tăng nhiệt độ sau N năm (°C)", -2.0, 5.0, 0.0, 0.1)
    rain_pct = st.slider("Lượng mưa thay đổi (%)", -50, 50, 0)
    fert_pct = st.slider("Phân bón thay đổi (%)", -50, 100, 0)
    pest_pct = st.slider("Thuốc trừ sâu thay đổi (%)", -50, 100, 0)


# =========================
# 4) LOAD + RUN
# =========================
try:
    if uploaded is not None:
        df_raw = load_data(uploaded.getvalue(), uploaded.name)
    else:
        if local_file is None:
            st.error("Bạn chưa upload CSV và cũng không có file cục bộ (merged_data.csv hoặc du_lieu_nong_nghiep_sach.csv).")
            st.stop()
        df_raw = load_data(None, local_file)

    df = prepare_df(df_raw)

    # chọn Area
    areas = sorted(df[COL_AREA].astype(str).unique().tolist())
    area = st.selectbox("Chọn quốc gia (Area) để dự báo", areas)

    df_area = df[df[COL_AREA].astype(str) == str(area)].copy()
    if df_area.shape[0] < 15:
        st.warning("Area '{}' có ít dữ liệu ({} dòng). Dự báo có thể kém ổn định.".format(area, df_area.shape[0]))

    # build X,y
    y, X = build_xy(df_area)

    # fit
    with st.spinner("Đang fit ARIMAX..."):
        order = (int(p), int(d), int(q))
        res = fit_arimax(y, X, order=order)

    # future exog
    last_row = df_area.sort_values(COL_YEAR).iloc[-1]
    X_future = make_future_exog(
        last_row=last_row,
        years=years_to_forecast,
        temp_delta_total=float(temp_delta_total),
        rain_pct=float(rain_pct),
        fert_pct=float(fert_pct),
        pest_pct=float(pest_pct),
    )

    fc_df = forecast_future(res, X_future)

    # =========================
    # 5) VIZ
    # =========================
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📈 Biểu đồ dự báo")
        hist = df_area.sort_values(COL_YEAR)[[COL_YEAR, COL_YIELD]].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist[COL_YEAR],
            y=hist[COL_YIELD],
            mode="lines+markers",
            name="Lịch sử"
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["Year"],
            y=fc_df["Forecast_Yield"],
            mode="lines+markers",
            name="Dự báo"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([fc_df["Year"].values, fc_df["Year"].values[::-1]]),
            y=np.concatenate([fc_df["Upper_95"].values, fc_df["Lower_95"].values[::-1]]),
            fill="toself",
            name="CI 95%",
            hoverinfo="skip",
            opacity=0.2,
            line=dict(width=0),
            showlegend=True
        ))

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Năm",
            yaxis_title="Yield (kg/ha)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🧾 Bảng dự báo")
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

        st.subheader("🔎 Tóm tắt mô hình")
        st.code(res.summary().as_text(), language="text")

    st.caption("Nếu lỗi do tên cột: hãy sửa các biến COL_* ở đầu file app.py cho đúng với CSV của bạn.")

except Exception as e:
    st.error("Lỗi: {}".format(e))
    st.stop()
