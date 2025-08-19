import pandas as pd

def _guess_value_col(df: pd.DataFrame):
    for key in ["monto", "neto", "total", "importe", "facturacion", "ingreso", "venta"]:
        for c in df.columns:
            if key in str(c).lower():
                return c
    return None

def _guess_category_cols(df: pd.DataFrame):
    cats = [c for c in df.columns if df[c].nunique(dropna=False) <= 30]
    priority = ["fecha","mes","tipo","cliente","patente","estado","proceso","vehiculo","unidad"]
    cats_sorted = sorted(cats, key=lambda c: (0 if any(p in c.lower() for p in priority) else 1, c))
    return cats_sorted

def _group_sum_safe(df: pd.DataFrame, cat_col: str, val_col: str) -> pd.Series:
    """
    Agrupa sumando de forma robusta:
    - fuerza a numérico (errores -> NaN)
    - suma por categoría
    - ordena descendente
    """
    vals = pd.to_numeric(df[val_col], errors="coerce")
    g = (
        df.assign(__v=vals)
          .groupby(cat_col, dropna=False)["__v"]
          .sum()
          .sort_values(ascending=False)
    )
    # Asegura tipo float (evita TypeError al dividir)
    return g.astype(float)

def analizar_datos_taller(data: dict) -> dict:
    res = {}
    for hoja, df in data.items():
        if df is None or df.empty:
            continue

        info = {"columnas": list(df.columns), "filas_totales": int(len(df)), "insights": []}
        val = _guess_value_col(df)

        if val:
            cats = _guess_category_cols(df)
            for cat in cats[:3]:
                try:
                    g = _group_sum_safe(df, cat, val)
                    if g.empty:
                        continue
                    total = float(g.sum())
                    top = g.head(3)
                    top_sum = float(top.sum())
                    conc = (top_sum / total) * 100.0 if total > 0 else 0.0

                    info["insights"].append({
                        "hoja": hoja,
                        "categoria": cat,
                        "valor": val,
                        "top3": {str(k): float(v) for k, v in top.to_dict().items()},
                        "concentracion_top3_pct": round(conc, 2)
                    })
                except Exception:
                    # No bloqueamos el análisis por una columna problemática
                    continue

        res[hoja] = info
    return res
