import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# ---- Parámetros fijos ----
LOOK_BACK   = 10
FLETE_USD   = 25.0
SEGURO_PCT  = 0.005
AEC_PCT     = 0.15
ALADI_F     = 0.88

# ---- Funciones de fechas ----
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

# ---- Cálculo de arancel ----
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
    arancel_final_usd  = (arancel_final_pct/100.0) * CIF
    arancel_pagado_usd = (arancel_pagado_pct/100.0) * CIF
    return dict(
        CR=CR, Flete=flete, Seguro=seguro, CIF=CIF, PP=piso, PT=techo, AEC_pct=AEC,
        Derecho_variable_pct=derecho_adic, Rebaja_pct=rebaja,
        Arancel_DIAN_pct=arancel_final_pct, Arancel_DIAN_usd=arancel_final_usd,
        Arancel_ALADI_pct=arancel_pagado_pct, Arancel_ALADI_usd=arancel_pagado_usd,
        Total_DIAN_usd=CIF + arancel_final_usd,
        Total_ALADI_usd=CIF + arancel_pagado_usd
    )

# ---- Interfaz Streamlit ----
st.set_page_config(page_title="Pronóstico y Arancel Azúcar B", layout="wide")
st.title("Pronóstico de Azúcar B y Cálculo de Arancel")

# Cargar modelo y scaler
modelo_path = Path("modelo_lstm_azucar_b.keras")
scaler_path = Path("scaler_azucar_b.pkl")
model = load_model(modelo_path)
scaler = joblib.load(scaler_path)

# Subir archivo Excel
excel_file = st.file_uploader("Sube tu archivo Excel con columnas FECHA y AZUCAR_B", type=["xlsx"])
if excel_file:
    df = pd.read_excel(excel_file)[["FECHA", "AZUCAR_B"]].dropna()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values("FECHA").reset_index(drop=True)

    # Inputs de franja
    col1, col2 = st.columns(2)
    piso = col1.number_input("Precio PISO (USD/t)", value=400.0)
    techo = col2.number_input("Precio TECHO (USD/t)", value=600.0)

    if techo < piso:
        st.error("El TECHO no puede ser menor que el PISO.")
    else:
        ejecutar = st.button("Ejecutar pronóstico ")
        if ejecutar:
            last_date = df["FECHA"].iloc[-1]
            ini_hab, fin_hab = quincena_siguiente(last_date)
            fechas_q = pd.bdate_range(ini_hab, fin_hab, freq="B")

            if len(fechas_q) == 0:
                st.error("No hay días hábiles en la próxima quincena.")
            else:
                vals_scaled = scaler.transform(df[["AZUCAR_B"]]).ravel()
                if len(vals_scaled) < LOOK_BACK:
                    st.error("Datos insuficientes para la ventana LOOK_BACK.")
                else:
                    seq = vals_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
                    preds_s = []
                    for _ in range(len(fechas_q)):
                        yhat = model.predict(seq, verbose=0)[0,0]
                        preds_s.append(yhat)
                        seq = np.concatenate([seq[:,1:,:], np.array([[[yhat]]])], axis=1)

                    preds = scaler.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()
                    forecast_df = pd.DataFrame({"FECHA": fechas_q, "PRONOSTICO_AZUCAR_B": preds})
                    prom_q = float(forecast_df["PRONOSTICO_AZUCAR_B"].mean())
                    ar = calcular_arancel(prom_q, piso, techo)

                    st.subheader("Próxima quincena hábil")
                    st.write(f"{ini_hab.date()} → {fin_hab.date()}  (n={len(fechas_q)})")

                    st.subheader("Pronóstico diario")
                    st.dataframe(forecast_df)

                    st.subheader("Promedio quincenal pronosticado (CR)")
                    st.metric(label="CR promedio (USD/t)", value=round(prom_q, 4))

                    tabla = pd.DataFrame({
                        "Detalle": [
                            "CR [USD/t]","Flete [USD/t]","Seguro [USD/t]","CIF [USD/t]",
                            "PP [USD/t]","PT [USD/t]","AEC [%]","Derecho variable [%]","Rebaja [%]",
                            "Arancel DIAN [%]","Arancel DIAN [USD/t]","Arancel pagado ALADI 88% [%]","Arancel pagado ALADI 88% [USD/t]",
                            "Total DIAN = CIF + Arancel [USD/t]","Total ALADI = CIF + Arancel pagado [USD/t]"
                        ],
                        "Valor": [round(ar[k], 4) for k in [
                            "CR","Flete","Seguro","CIF","PP","PT","AEC_pct","Derecho_variable_pct","Rebaja_pct",
                            "Arancel_DIAN_pct","Arancel_DIAN_usd","Arancel_ALADI_pct","Arancel_ALADI_usd",
                            "Total_DIAN_usd","Total_ALADI_usd"
                        ]]
                    })
                    st.subheader("Resumen de arancel y costos")
                    st.dataframe(tabla)

                    st.download_button(
                        label="Descargar pronóstico en CSV",
                        data=forecast_df.to_csv(index=False).encode("utf-8"),
                        file_name="pronostico_quincena_proxima.csv",
                        mime="text/csv"
                    )