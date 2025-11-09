# --- Librerías ---
import streamlit as st
import numpy as np
import pandas as pd
import joblib, re
from pathlib import Path
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- Parámetros fijos ---
LOOK_BACK   = 10
FLETE_USD   = 25.0
SEGURO_PCT  = 0.005
AEC_PCT     = 0.15
ALADI_F     = 0.88

# --- Utilidades de fechas hábiles ---
def _bd_up(d):
    d = pd.to_datetime(d).normalize()
    while d.weekday() > 4:
        d += pd.Timedelta(days=1)
    return d

def _bd_down(d):
    d = pd.to_datetime(d).normalize()
    while d.weekday() > 4:
        d -= pd.Timedelta(days=1)
    return d

def _eom(dt):
    return pd.to_datetime(dt).normalize() + pd.offsets.MonthEnd(0)

def _som_next(dt):
    return pd.to_datetime(dt).normalize() + pd.offsets.MonthBegin(1)

def quincena_siguiente(ultima_fecha):
    ultima_fecha = pd.to_datetime(ultima_fecha).normalize()
    if ultima_fecha.day <= 15:
        ini_raw = pd.Timestamp(ultima_fecha.year, ultima_fecha.month, 16)
        fin_raw = _eom(ultima_fecha)
    else:
        ini_raw = _som_next(ultima_fecha)
        fin_raw = ini_raw + pd.Timedelta(days=14)
    return _bd_up(ini_raw), _bd_down(fin_raw)

# --- Cálculo de arancel ---
def calcular_arancel(CR, piso, techo, flete=FLETE_USD, seguro_pct=SEGURO_PCT, aec=AEC_PCT, aladi=ALADI_F):
    seguro = seguro_pct * (CR + flete)
    CIF = CR + flete + seguro
    AEC = aec * 100.0
    derecho_adic = rebaja = 0.0
    if CIF < piso:
        derecho_adic = ((piso - CIF) * (1 + aec)) / CIF * 100.0
        arancel_final_pct = AEC + derecho_adic
    elif CIF > techo:
        rebaja = ((CIF - techo) * (1 + aec)) / CIF * 100.0
        arancel_final_pct = max(AEC - rebaja, 0.0)
    else:
        arancel_final_pct = AEC

    arancel_pagado_pct = arancel_final_pct * aladi

    return dict(
        CR=CR, CIF=CIF, AEC_pct=AEC,
        Derecho_variable_pct=derecho_adic, Rebaja_pct=rebaja,
        Arancel_DIAN_pct=arancel_final_pct,
        Arancel_ALADI_pct=arancel_pagado_pct
    )

# --- Configuración de la app ---
st.set_page_config(page_title="Pronóstico y Arancel Azúcar B", layout="wide")
st.markdown("<h1 style='text-align: center;'>HERRAMIENTA: CÁLCULO DEL ARANCEL DE IMPORTACIÓN DE AZÚCAR ORIGEN BRASIL</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Cargar modelo y scaler ---
modelo_path = Path("modelo_lstm_azucar_b.keras")
scaler_path = Path("scaler_azucar_b.pkl")
try:
    model = load_model(modelo_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error cargando modelo o scaler: {e}")
    st.stop()

# --- Parser coma decimal ---
def parse_decimal_comma(x) -> float | None:
    if pd.isna(x): return None
    s = str(x).strip().replace('\u00A0', ' ')
    s = re.sub(r'[^0-9,.\-+]', '', s)
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif s.count(",") == 1 and s.count(".") >= 1 and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return None

# --- Lector CSV tolerante ---
def read_csv_tolerant(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=",", quotechar='"', engine="python")
    if df.shape[1] == 1:
        file.seek(0)
        df = pd.read_csv(file, sep=";", quotechar='"', engine="python")
    if df.shape[1] == 1:
        file.seek(0)
        df = pd.read_csv(
            file,
            sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)|;(?=(?:[^"]*"[^"]*")*[^"]*$)',
            engine="python",
            header=0
        )
    return df

# --- Carga y normalización de archivo ---
def _load_and_normalize_csv(file) -> pd.DataFrame:
    raw = read_csv_tolerant(file)
    if raw.shape[1] < 2:
        raise ValueError("El CSV debe tener al menos 2 columnas.")
    df = raw.iloc[:, :2].copy()
    df.columns = ["COL_FECHA_RAW", "COL_PRECIO_RAW"]
    df["FECHA"] = pd.to_datetime(df["COL_FECHA_RAW"].astype(str).str.strip(), format="%d.%m.%Y", errors="coerce")
    df["AZUCAR_B"] = df["COL_PRECIO_RAW"].apply(parse_decimal_comma)
    df = df.loc[df["FECHA"].notna() & df["AZUCAR_B"].notna(), ["FECHA", "AZUCAR_B"]]
    df = df.drop_duplicates(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    if df.empty:
        raise ValueError("Sin datos válidos tras limpiar.")
    return df

# --- Interfaz principal ---
st.markdown("### Carga de archivo y parámetros")
col_input, col_plot = st.columns([1, 2])

with col_input:
    csv_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])
    piso_val = st.number_input("Precio PISO (USD/t)", value=595.0)
    techo_val = st.number_input("Precio TECHO (USD/t)", value=683.0)
    ejecutar = st.button("Ejecutar pronóstico")

with col_plot:
    if csv_file:
        try:
            df = _load_and_normalize_csv(csv_file)
        except Exception as e:
            st.error(f"Error preparando el CSV: {e}")
            st.stop()

        if ejecutar:
            last_date = df["FECHA"].max()
            ini_hab, fin_hab = quincena_siguiente(last_date)
            fechas_q = pd.bdate_range(ini_hab, fin_hab, freq="B")
            vals_scaled = scaler.transform(df[["AZUCAR_B"]]).ravel()
            seq = vals_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
            preds_s = []
            for _ in range(len(fechas_q)):
                yhat = model.predict(seq, verbose=0)[0, 0]
                preds_s.append(yhat)
                seq = np.concatenate([seq[:, 1:, :], np.array([[[yhat]]])], axis=1)
            preds = scaler.inverse_transform(np.array(preds_s).reshape(-1, 1)).ravel()
            forecast_df = pd.DataFrame({
                "FECHA": fechas_q,
                "PRONOSTICO_AZUCAR_B": preds
            })

            st.markdown("<h3 style='text-align: center;'>Comportamiento de la serie</h3>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["FECHA"], y=df["AZUCAR_B"], mode="lines+markers", name="Histórico", marker=dict(color="blue")))
            fig.add_trace(go.Scatter(x=forecast_df["FECHA"], y=forecast_df["PRONOSTICO_AZUCAR_B"], mode="markers", name="Pronóstico", marker=dict(color="red", size=8)))
            fig.update_layout(xaxis_title="Fecha", yaxis_title="USD/t", height=400)
            st.plotly_chart(fig, use_container_width=True)

            prom_q = float(forecast_df["PRONOSTICO_AZUCAR_B"].mean())
            ar = calcular_arancel(prom_q, piso_val, techo_val)

            st.markdown("<h3 style='text-align: center;'>Resultados del pronóstico</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="CR promedio (USD/t)", value=round(prom_q, 4))
                st.dataframe(
                    forecast_df[["FECHA", "PRONOSTICO_AZUCAR_B"]]
                    .rename(columns={"FECHA": "Fecha", "PRONOSTICO_AZUCAR_B": "CR Pronosticado (USD/t)"})
                )
            with col2:
                st.metric(label="Arancel DIAN (%)", value=round(ar["Arancel_DIAN_pct"], 4))
                resumen = pd.DataFrame({
                    "Detalle": [
                        "CR [USD/t]", "CIF [USD/t]", "AEC [%]",
                        "Derecho variable [%]", "Rebaja [%]",
                        "Arancel DIAN [%]", "Arancel efectivamente pagado [%]"
                    ],
                    "Valor": [round(ar[k], 4) for k in [
                        "CR", "CIF", "AEC_pct",
                        "Derecho_variable_pct", "Rebaja_pct",
                        "Arancel_DIAN_pct", "Arancel_ALADI_pct"
                    ]]
                })
                st.dataframe(resumen)

            st.download_button(
                label="Descargar pronóstico en CSV",
                data=forecast_df[["FECHA", "PRONOSTICO_AZUCAR_B"]]
                    .rename(columns={"FECHA": "Fecha", "PRONOSTICO_AZUCAR_B": "CR_Pronosticado_USD_t"})
                    .to_csv(index=False).encode("utf-8"),
                file_name="pronostico_quincena_proxima.csv",
                mime="text/csv"
            )