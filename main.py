# ===================== PRONÓSTICO QUINCENAL + ARANCEL (sin calendar) =====================
# pip install tensorflow pandas numpy scikit-learn joblib openpyxl

import numpy as np, pandas as pd, joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# ---- Rutas y parámetros ----
MODELO_PATH = Path("/content/modelo_lstm_azucar_b.keras")
SCALER_PATH = Path("/content/scaler_azucar_b.pkl")
EXCEL_PATH  = Path("/content/Datos ARIMA.xlsx")   # Debe tener FECHA y AZUCAR_B
LOOK_BACK   = 10                                   # Igual al usado al entrenar
FLETE_USD   = 25.0
SEGURO_PCT  = 0.005  # 0.5%
AEC_PCT     = 0.15   # 15%
ALADI_F     = 0.88   # 88%

# ---- Utilidades de fechas (hábiles lun-vie) ----
def _bd_up(d):
    d = pd.to_datetime(d).normalize()
    while d.weekday() > 4:  # 0-4: L-V
        d += pd.Timedelta(days=1)
    return d

def _bd_down(d):
    d = pd.to_datetime(d).normalize()
    while d.weekday() > 4:  # 0-4: L-V
        d -= pd.Timedelta(days=1)
    return d

def _eom(dt):
    """Fin de mes con pandas (equivalente a calendar.monthrange)."""
    dt = pd.to_datetime(dt).normalize()
    return (dt + pd.offsets.MonthEnd(0))

def _som_next(dt):
    """Primer día del mes siguiente con pandas."""
    dt = pd.to_datetime(dt).normalize()
    return (dt + pd.offsets.MonthBegin(1))

def quincena_siguiente(ultima_fecha: pd.Timestamp):
    """
    Si última fecha está en la 1ª quincena, devuelve [16, fin de mes actual].
    Si está en la 2ª quincena, devuelve [1, 15] del mes siguiente.
    Ajusta a días hábiles con _bd_up/_bd_down.
    """
    ultima_fecha = pd.to_datetime(ultima_fecha).normalize()
    if ultima_fecha.day <= 15:
        ini_raw = pd.Timestamp(ultima_fecha.year, ultima_fecha.month, 16)
        fin_raw = _eom(ultima_fecha)
    else:
        ini_raw = _som_next(ultima_fecha)               # 1° del próximo mes
        fin_raw = ini_raw + pd.Timedelta(days=14)       # 15 del próximo mes
    return _bd_up(ini_raw), _bd_down(fin_raw)

# ---- Arancel según tu especificación ----
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

# ---- Main ----
if __name__ == "__main__":
    # 1) Cargar modelo, scaler y datos
    model  = load_model(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_excel(EXCEL_PATH)[["FECHA","AZUCAR_B"]].dropna()
    df["FECHA"] = pd.to_datetime(df["FECHA"]); df = df.sort_values("FECHA").reset_index(drop=True)

    # 2) Fechas hábiles próxima quincena
    last_date = df["FECHA"].iloc[-1]
    ini_hab, fin_hab = quincena_siguiente(last_date)
    fechas_q = pd.bdate_range(ini_hab, fin_hab, freq="B")
    if len(fechas_q) == 0:
        raise RuntimeError("No hay días hábiles en la próxima quincena.")

    # 3) Preparar última ventana y pronosticar n pasos = días hábiles
    vals_scaled = scaler.transform(df[["AZUCAR_B"]]).ravel()
    if len(vals_scaled) < LOOK_BACK:
        raise RuntimeError("Datos insuficientes para la ventana LOOK_BACK.")
    seq = vals_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)

    preds_s = []
    for _ in range(len(fechas_q)):
        yhat = model.predict(seq, verbose=0)[0,0]
        preds_s.append(yhat)
        seq = np.concatenate([seq[:,1:,:], np.array([[[yhat]]])], axis=1)

    preds = scaler.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()
    forecast_df = pd.DataFrame({"FECHA": fechas_q, "PRONOSTICO_AZUCAR_B": preds})
    prom_q = float(forecast_df["PRONOSTICO_AZUCAR_B"].mean())

    # 4) Inputs de franja
    def _inp(prompt, default):
        s = input(f"{prompt} [{default}]: ").strip()
        return float(s) if s else float(default)
    piso  = _inp("Precio PISO (USD/t)", 400.0)
    techo = _inp("Precio TECHO (USD/t)", 600.0)
    if techo < piso:
        raise ValueError("El TECHO no puede ser menor que el PISO.")

    # 5) Arancel
    ar = calcular_arancel(prom_q, piso, techo)

    # 6) Salidas
    print(f"\n📆 Próxima quincena hábil: {ini_hab.date()} → {fin_hab.date()}  (n={len(fechas_q)})")
    print("\n📅 Pronóstico (quincena):")
    print(forecast_df.to_string(index=False))
    print("\n📊 Promedio quincenal pronosticado (CR):", round(prom_q,4), "USD/t")

    tabla = pd.DataFrame({
        "Detalle":[
            "CR [USD/t]","Flete [USD/t]","Seguro [USD/t]","CIF [USD/t]",
            "PP [USD/t]","PT [USD/t]","AEC [%]","Derecho variable [%]","Rebaja [%]",
            "Arancel DIAN [%]","Arancel DIAN [USD/t]","Arancel pagado ALADI 88% [%]","Arancel pagado ALADI 88% [USD/t]",
            "Total DIAN = CIF + Arancel [USD/t]","Total ALADI = CIF + Arancel pagado [USD/t]"
        ],
        "Valor":[
            round(ar["CR"],4), round(ar["Flete"],4), round(ar["Seguro"],4), round(ar["CIF"],4),
            round(ar["PP"],4), round(ar["PT"],4), round(ar["AEC_pct"],4), round(ar["Derecho_variable_pct"],4), round(ar["Rebaja_pct"],4),
            round(ar["Arancel_DIAN_pct"],4), round(ar["Arancel_DIAN_usd"],4), round(ar["Arancel_ALADI_pct"],4), round(ar["Arancel_ALADI_usd"],4),
            round(ar["Total_DIAN_usd"],4), round(ar["Total_ALADI_usd"],4)
        ]
    })
    print("\n===== Resumen arancel y costos (quincena) =====")
    print(tabla.to_string(index=False))

    out_csv = Path("pronostico_quincena_proxima.csv")
    forecast_df.to_csv(out_csv, index=False)
    print(f"\n💾 Pronóstico guardado en: {out_csv.resolve()}")
