import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gspread
import json
import re
import unicodedata
from typing import Dict, Any, Optional
from google.oauth2.service_account import Credentials
from openai import OpenAI
from analizador import analizar_datos_taller

st.set_page_config(layout="wide", page_title="Controller Financiero IA")

# ---------------------------
# LOGIN
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.markdown("## 🔐 Iniciar sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        try_user = st.secrets.get("USER", None)
        try_pass = st.secrets.get("PASSWORD", None)
        if try_user is None or try_pass is None:
            st.error("Secrets USER/PASSWORD no configurados. Agrega USER y PASSWORD en secrets.toml / Cloud.")
            return
        if username == try_user and password == try_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas")

if not st.session_state.authenticated:
    login()
    st.stop()

# ---------------------------
# ESTADO PERSISTENTE
# ---------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []
if "data" not in st.session_state:
    st.session_state.data = None
if "sheet_url" not in st.session_state:
    st.session_state.sheet_url = ""
if "__ultima_vista__" not in st.session_state:
    st.session_state["__ultima_vista__"] = None

# ---------------------------
# CARGA DE DATOS (CACHE)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_excel(file):
    return pd.read_excel(file, sheet_name=None)

@st.cache_data(show_spinner=False)
def load_gsheet(json_keyfile: str, sheet_url: str):
    creds_dict = json.loads(json_keyfile)
    scope = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sheet.worksheets()}

# ---------------------------
# OPENAI
# ---------------------------
def ask_gpt(prompt: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    messages = [{"role": "system", "content": "Eres un controller financiero experto de un taller de desabolladura y pintura. Sé conciso y accionable."}]
    for h in st.session_state.historial[-8:]:  # contexto útil y corto
        messages.append({"role": "user", "content": h["pregunta"]})
        messages.append({"role": "assistant", "content": h["respuesta"]})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------------------------
# NORMALIZACIÓN & UTILIDADES
# ---------------------------
def _norm(s: str) -> str:
    s = str(s).replace("\u00A0", " ").strip()   # NBSP -> espacio
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s).lower()
    return s

def find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Match exacto robusto por normalización (espacios invisibles, acentos, mayus/minus)."""
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    return None

def _build_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    schema = {}
    for hoja, df in data.items():
        if df is None or df.empty:
            continue
        cols = []
        samples = {}
        for c in df.columns:
            cols.append(str(c))
            vals = df[c].dropna().astype(str).head(3).tolist()
            if vals:
                samples[str(c)] = vals
        schema[hoja] = {"columns": cols, "samples": samples}
    return schema

def _choose_chart_auto(df: pd.DataFrame, cat_col: str, val_col: str) -> str:
    if pd.api.types.is_datetime64_any_dtype(df[cat_col]) or "fecha" in _norm(cat_col):
        return "linea"
    nunique = df[cat_col].nunique(dropna=False)
    return "torta" if 2 <= nunique <= 6 else "barras"

# ---------------------------
# VISUALIZACIONES (EXCEL-LIKE)
# ---------------------------
def _fmt_miles(x, pos=None):
    try:
        return f"${int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    resumen = df.groupby(col_categoria, dropna=False)[col_valor].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.pie(
        resumen.values,
        labels=[str(x) for x in resumen.index],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig)

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None):
    resumen = df.groupby(col_categoria, dropna=False)[col_valor].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.bar(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_ylabel(col_valor)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_miles))
    ax.tick_params(axis='x', rotation=45, ha='right')
    for b in bars:
        ax.annotate(_fmt_miles(b.get_height()), xy=(b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

def mostrar_tabla(df, col_categoria, col_valor, titulo=None):
    resumen = (
        df.groupby(col_categoria, dropna=False)[col_valor]
          .sum().sort_values(ascending=False).reset_index()
    )
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    try:
        resumen[col_val] = resumen[col_val].astype(float).round(0).astype(int)
    except Exception:
        pass
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)

# ---------------------------
# PLANNER (IA → JSON) Y EJECUTOR
# ---------------------------
def plan_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Eres un controller financiero. Te doy el ESQUEMA de las hojas (columnas y ejemplos) y una PREGUNTA.
Devuélveme SOLO un JSON con la acción óptima:

{{
  "action": "table" | "chart" | "text",
  "sheet": "<nombre_hoja_o_vacia>",
  "category_col": "<col cat o vacio>",
  "value_col": "<col valor o vacio>",
  "date_col": "<col fecha si aplica o vacio>",
  "agg": "sum" | "avg" | "count",
  "chart": "barras" | "torta" | "linea" | "auto",
  "title": "<titulo sugerido>"
}}

Reglas:
- Usa nombres EXACTOS del esquema (case-insensitive).
- Si piden “por …”, usa eso de categoría.
- Si no especifican tipo de gráfico, usa "chart":"auto".
- Si no hay valor claro, elige monto/importe/neto/total.
- Si la pregunta es solo textual, "action":"text".

ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    raw = ask_gpt(prompt).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {}
    try:
        plan = json.loads(m.group(0))
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass
    return {}

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any]) -> bool:
    action = plan.get("action")
    if action not in ("table", "chart", "text"):
        return False

    if action == "text":
        return False

    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    date_col = plan.get("date_col") or ""
    agg = (plan.get("agg") or "sum").lower()
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None

    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty:
            continue

        # Resolver columnas
        cat_real = find_col(df, cat) if cat else None
        val_real = find_col(df, val) if val else None
        date_real = find_col(df, date_col) if date_col else None

        # Parseo fecha si llegó
        if date_real:
            df = df.copy()
            df[date_real] = pd.to_datetime(df[date_real], errors="coerce")

        # Autodetección si faltan
        if not val_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["monto", "importe", "neto", "total", "facturacion", "ingreso"]):
                    val_real = c; break
        if not cat_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["patente","cliente","tipo","estado","mes","proceso","servicio","vehiculo","hoja"]):
                    cat_real = c; break

        if not cat_real and cat in df.columns: cat_real = cat
        if not val_real and val in df.columns: val_real = val

        if not val_real or not cat_real or cat_real not in df.columns or val_real not in df.columns:
            continue

        # Tipo de gráfico si es automático
        if action == "chart" and chart == "auto":
            chart = _choose_chart_auto(df, cat_real, val_real)

        # Ejecutar
        if action == "table":
            mostrar_tabla(df, cat_real, val_real, title)
            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type": "tabla"}
            return True

        if action == "chart":
            if chart == "barras":
                mostrar_grafico_barras(df, cat_real, val_real, title)
            elif chart == "torta":
                mostrar_grafico_torta(df, cat_real, val_real, title)
            elif chart == "line
