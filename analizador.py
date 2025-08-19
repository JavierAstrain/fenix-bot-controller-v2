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
                g = df.groupby(cat)[val].sum().sort_values(ascending=False)
                total = g.sum()
                top = g.head(3)
                conc = float(top.sum()/total) if total else 0.0
                info["insights"].append({
                    "hoja": hoja,
                    "categoria": cat,
                    "valor": val,
                    "top3": top.to_dict(),
                    "concentracion_top3_pct": round(conc*100, 2)
                })
        res[hoja] = info
    return res
