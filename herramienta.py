
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # opcional: silenciar aviso oneDNN

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

# ---- Utilidades de fechas hábiles ----
def _bd_up(d):
    d = pd.to_datetime(d).normalize()
    while d.weekday() > 4:  # 0=Lun ... 6=Dom
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
    """Con base en la última fecha del dataset, calcula la siguiente quincena hábil."""
    ultima_fecha = pd.to_datetime(ultima_fecha).normalize()
    if ultima_fecha.day <= 15:
        ini_raw = pd.Timestamp(ultima_fecha.year, ultima_fecha.month, 16)
        fin_raw = _eom(ultima_fecha)  # fin de mes
    else:
        ini_raw = _som_next(ultima_fecha)  # 1 del mes siguiente
        fin_raw = ini_raw + pd.Timedelta(days=14)  # 1..15
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

# ---- Configuración de la app ----
st.set_page_config(page_title="Pronóstico y Arancel Azúcar B", layout="wide")
st.title("Pronóstico de Azúcar B y Cálculo de Arancel")

# ---- Cargar modelo y scaler ----
modelo_path = Path("modelo_lstm_azucar_b.keras")
scaler_path = Path("scaler_azucar_b.pkl")
try:
    model = load_model(modelo_path)
except Exception as e:
    st.error(f"No pude cargar el modelo Keras ({modelo_path}). Detalle: {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"No pude cargar el scaler ({scaler_path}). Detalle: {e}")
    st.stop()

# ---- Carga de archivo CSV (solo 2 primeras columnas) ----
csv_file = st.file_uploader(
    "Sube tu archivo CSV (delimitado por comas). Se tomarán SOLO las 2 primeras columnas: Fecha (M/D/Y) y Precio.",
    type=["csv"]
)

def _load_and_normalize_csv(file) -> pd.DataFrame:
    # 1) Leer CSV tal cual
    raw = pd.read_csv(file)

    if raw.shape[1] < 2:
        raise ValueError(f"El CSV debe tener al menos 2 columnas. Columnas detectadas: {list(raw.columns)}")

    # 2) Tomar SOLO las dos primeras columnas
    df = raw.iloc[:, :2].copy()
    df.columns = ["COL_FECHA_RAW", "COL_PRECIO_RAW"]

    # 3) Parsear fecha: origen viene como Mes/Día/Año (month-first)
    #    Guardamos fecha datetime para cálculo y una versión string D/M/Y para visual
    df["FECHA"] = pd.to_datetime(df["COL_FECHA_RAW"], errors="coerce", dayfirst=False, infer_datetime_format=True)
    df["FECHA_DMY"] = df["FECHA"].dt.strftime("%d/%m/%Y")

    # 4) Limpiar precio -> float (quita comas de miles, $, %, guiones largos)
    s = df["COL_PRECIO_RAW"].astype(str)
    s = (s.str.replace(",", "", regex=False)
           .str.replace("%", "", regex=False)
           .str.replace("$", "", regex=False)
           .str.replace("—", "", regex=False)
           .str.strip())
    df["AZUCAR_B"] = pd.to_numeric(s, errors="coerce")

    # 5) Filtrar filas válidas y ordenar ASC por fecha
    df = df.loc[df["FECHA"].notna() & df["AZUCAR_B"].notna(), ["FECHA", "FECHA_DMY", "AZUCAR_B"]]
    df = df.drop_duplicates(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)

    if df.empty:
        raise ValueError("Después de limpiar, no quedaron datos válidos. Revisa el formato de fecha y precio.")
    return df

# ---- Ejecución principal ----
if csv_file:
    try:
        df = _load_and_normalize_csv(csv_file)
    except Exception as e:
        st.error(f"Error preparando el CSV: {e}")
        st.stop()

    # Verificación rápida del rango de fechas
    st.caption(f"Datos desde {df['FECHA'].min().date()} hasta {df['FECHA'].max().date()} (formato mostrado: D/M/Y)")

    # Inputs de franja (piso/techo)
    col1, col2 = st.columns(2)
    piso = col1.number_input("Precio PISO (USD/t)", value=595.0)
    techo = col2.number_input("Precio TECHO (USD/t)", value=683.0)

    if techo < piso:
        st.error("El TECHO no puede ser menor que el PISO.")
        st.stop()

    if st.button("Ejecutar pronóstico"):
        # Usar SIEMPRE la fecha máxima (por si el CSV viene en orden descendente)
        last_date = df["FECHA"].max()

        ini_hab, fin_hab = quincena_siguiente(last_date)
        fechas_q = pd.bdate_range(ini_hab, fin_hab, freq="B")
        if len(fechas_q) == 0:
            st.error("No hay días hábiles en la próxima quincena.")
            st.stop()

        # Escalar y validar ventana
        vals_scaled = scaler.transform(df[["AZUCAR_B"]]).ravel()
        if len(vals_scaled) < LOOK_BACK:
            st.error(f"Datos insuficientes para LOOK_BACK={LOOK_BACK}. Proporciona más histórico.")
            st.stop()

        # Predicción recursiva para cada día hábil de la quincena
        seq = vals_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
        preds_s = []
        for _ in range(len(fechas_q)):
            yhat = model.predict(seq, verbose=0)[0, 0]
            preds_s.append(yhat)
            seq = np.concatenate([seq[:, 1:, :], np.array([[[yhat]]])], axis=1)

        preds = scaler.inverse_transform(np.array(preds_s).reshape(-1, 1)).ravel()
        forecast_df = pd.DataFrame({
            "FECHA": fechas_q,
            "FECHA_DMY": pd.to_datetime(fechas_q).strftime("%d/%m/%Y"),
            "PRONOSTICO_AZUCAR_B": preds
        })
        prom_q = float(forecast_df["PRONOSTICO_AZUCAR_B"].mean())
        ar = calcular_arancel(prom_q, piso, techo)

        st.subheader("📅 Próxima quincena hábil")
        st.write(f"Última fecha en datos: **{last_date.strftime('%d/%m/%Y')}**")
        st.write(f"{ini_hab.strftime('%d/%m/%Y')} → {fin_hab.strftime('%d/%m/%Y')}  (n={len(fechas_q)})")

        st.subheader("📈 Pronóstico diario")
        st.dataframe(forecast_df[["FECHA_DMY", "PRONOSTICO_AZUCAR_B"]].rename(columns={"FECHA_DMY": "FECHA (D/M/Y)"}))

        st.subheader("💰 Promedio quincenal pronosticado (CR)")
        st.metric(label="CR promedio (USD/t)", value=round(prom_q, 4))

        tabla = pd.DataFrame({
            "Detalle": [
                "CR [USD/t]","Flete [USD/t]","Seguro [USD/t]","CIF [USD/t]",
                "PP [USD/t]","PT [USD/t]","AEC [%]","Derecho variable [%]","Rebaja [%]",
                "Arancel DIAN [%]","Arancel DIAN [USD/t]",
                "Arancel pagado ALADI 88% [%]","Arancel pagado ALADI 88% [USD/t]",
                "Total DIAN = CIF + Arancel [USD/t]","Total ALADI = CIF + Arancel pagado [USD/t]"
            ],
            "Valor": [round(ar[k], 4) for k in [
                "CR","Flete","Seguro","CIF","PP","PT","AEC_pct","Derecho_variable_pct","Rebaja_pct",
                "Arancel_DIAN_pct","Arancel_DIAN_usd","Arancel_ALADI_pct","Arancel_ALADI_usd",
                "Total_DIAN_usd","Total_ALADI_usd"
            ]]
        })

        st.subheader("📊 Resumen de arancel y costos")
        st.dataframe(tabla)

        st.download_button(
            label="Descargar pronóstico en CSV",
            data=forecast_df[["FECHA_DMY", "PRONOSTICO_AZUCAR_B"]].rename(columns={"FECHA_DMY": "FECHA"}).to_csv(index=False).encode("utf-8"),
            file_name="pronostico_quincena_proxima.csv",
            mime="text/csv"
        )
