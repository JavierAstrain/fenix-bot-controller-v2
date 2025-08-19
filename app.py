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
    for h in st.session_state.historial[-8:]:
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
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    return None

# alias para evitar confusión con los bloques del planner
def _find_col(df, name: str) -> Optional[str]:
    return find_col(df, name)

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
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
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
    resumen = (df.groupby(col_categoria, dropna=False)[col_valor]
                 .sum().sort_values(ascending=False).reset_index())
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    try:
        resumen[col_val] = resumen[col_val].astype(float).round(0).astype(int)
    except Exception:
        pass
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)

# ---------------------------
# PARSER (texto → render)
# ---------------------------
def parse_and_render_instructions(respuesta_texto: str, data_dict: dict):
    """
    Soporta viñetas, code blocks y opcional @HOJA.
    Formatos:
      - grafico_torta[:|@HOJA:]cat|val|titulo
      - grafico_barras[:|@HOJA:]cat|val|titulo
      - tabla[:|@HOJA:]cat|val[|titulo]
    """
    patt = re.compile(r'(grafico_torta|grafico_barras|tabla)(?:@([^\s:]+))?\s*:\s*([^\n\r]+)', re.IGNORECASE)

    def safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
        cat = find_col(df, cat_raw)
        val = find_col(df, val_raw)
        if not cat or not val:
            st.warning(f"❗ No se pudo generar la visualización en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'.")
            return
        try:
            plot_fn(df, cat, val, titulo)
        except Exception as e:
            st.error(f"Error generando visualización en '{hoja}': {e}")

    for m in patt.finditer(respuesta_texto):
        kind = m.group(1).lower()
        hoja_sel = m.group(2)
        body = m.group(3).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]

        if kind in ("grafico_torta", "grafico_barras"):
            if len(parts) != 3:
                st.warning("Instrucción de gráfico inválida.")
                continue
            cat_raw, val_raw, title = parts
            if hoja_sel and hoja_sel in data_dict:
                if find_col(data_dict[hoja_sel], cat_raw) and find_col(data_dict[hoja_sel], val_raw):
                    safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                              hoja_sel, data_dict[hoja_sel], cat_raw, val_raw, title)
                else:
                    st.warning(f"No se encontraron columnas en la hoja '{hoja_sel}' para: {cat_raw} | {val_raw}")
            else:
                dibujado = False
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                                  hoja, df, cat_raw, val_raw, title)
                        dibujado = True
                if not dibujado:
                    st.warning("No se pudo generar el gráfico en ninguna hoja (verifica nombres de columnas).")

        else:  # tabla
            if len(parts) not in (2, 3):
                st.warning("Instrucción de tabla inválida.")
                continue
            cat_raw, val_raw = parts[0], parts[1]
            title = parts[2] if len(parts) == 3 else None

            def draw_table_on(df, hoja):
                cat = find_col(df, cat_raw)
                val = find_col(df, val_raw)
                if cat and val:
                    mostrar_tabla(df, cat, val, titulo=title or f"Tabla: {val} por {cat} ({hoja})")
                    return True
                return False

            if hoja_sel and hoja_sel in data_dict:
                if not draw_table_on(data_dict[hoja_sel], hoja_sel):
                    st.warning(f"No se pudo generar la tabla en '{hoja_sel}'. Revisar columnas: '{cat_raw}' y '{val_raw}'.")
            else:
                ok = False
                for hoja, df in data_dict.items():
                    ok = draw_table_on(df, hoja) or ok
                if not ok:
                    st.warning("No se pudo generar la tabla en ninguna hoja (verifica nombres de columnas).")

# ---------------------------
# PLANNER (IA → JSON)
# ---------------------------
def plan_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Eres un controller financiero. Te doy el ESQUEMA de las hojas (columnas y ejemplos) y una PREGUNTA del usuario.
Devuélveme SOLO un JSON (sin explicaciones) con la mejor acción para responder, con esta forma:

{{
  "action": "table" | "chart" | "text",
  "sheet": "<nombre_hoja_o_vacia_si_no_aplica>",
  "category_col": "<col cat o vacio>",
  "value_col": "<col valor o vacio>",
  "date_col": "<col fecha si aplica o vacio>",
  "agg": "sum" | "avg" | "count",
  "chart": "barras" | "torta" | "linea" | "auto",
  "title": "<titulo sugerido>"
}}

Reglas:
- Usa NOMBRES EXACTOS del esquema (insensible a mayúsculas).
- Si piden “por …”, úsalo como categoría.
- Si no se indica tipo de gráfico, usa "chart":"auto".
- Si dudas del valor, usa monto/importe/neto/total.
- Si es puramente textual, usa "action":"text".

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
        return plan if isinstance(plan, dict) else {}
    except Exception:
        return {}

# ---------------------------
# EJECUTOR DEL PLAN
# ---------------------------
def execute_plan(plan: Dict[str, Any], data: Dict[str, Any]) -> bool:
    action = plan.get("action")
    if action not in ("table", "chart", "text"):
        return False

    if action == "text":
        # Nada que renderizar; la IA responderá en texto
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

        # Resolver columnas robustamente
        cat_real  = _find_col(df, cat)  if cat  else None
        val_real  = _find_col(df, val)  if val  else None
        date_real = _find_col(df, date_col) if date_col else None

        # Intentar parsear fecha si corresponde
        if date_real:
            df = df.copy()
            df[date_real] = pd.to_datetime(df[date_real], errors="coerce")

        # Autodetección si faltan columnas
        if not val_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["monto","importe","neto","total","facturacion","ingreso","venta"]):
                    val_real = c; break
        if not cat_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["fecha","mes","patente","cliente","tipo","estado","proceso","servicio","vehiculo","unidad"]):
                    cat_real = c; break

        # Último intento: usar nombres crudos si existen
        if not cat_real and cat in df.columns: cat_real = cat
        if not val_real and val in df.columns: val_real = val

        # Si no están ambas, probar otra hoja
        if not val_real or not cat_real or cat_real not in df.columns or val_real not in df.columns:
            continue

        # Elegir tipo de gráfico si es automático
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
            elif chart == "linea":
                # Serie temporal → agregamos por mes; si no es fecha, caemos a barras
                df2 = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(df2[cat_real]):
                    try:
                        df2[cat_real] = pd.to_datetime(df2[cat_real], errors="coerce")
                    except Exception:
                        pass
                if pd.api.types.is_datetime64_any_dtype(df2[cat_real]):
                    serie = (
                        df2.set_index(cat_real)
                           .groupby(pd.Grouper(freq="M"))[val_real]
                           .sum()
                           .dropna()
                    )
                    fig, ax = plt.subplots()
                    ax.plot(serie.index, serie.values, marker="o")
                    ax.set_title(title or f"{val_real} por tiempo")
                    ax.set_ylabel(val_real)
                    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_miles))
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                else:
                    mostrar_grafico_barras(df, cat_real, val_real, title)
            else:
                # Fallback seguro
                mostrar_grafico_barras(df, cat_real, val_real, title)

            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type": f"chart:{chart}"}
            return True

    return False

# ---------------------------
# UI
# ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 📁 Subir archivo")
    tipo_fuente = st.radio("Fuente de datos", ["Excel", "Google Sheets"], key="k_fuente")

    if tipo_fuente == "Excel":
        file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"], key="k_excel")
        if file:
            st.session_state.data = load_excel(file)
    else:
        with st.form(key="form_gsheet"):
            url = st.text_input("URL de Google Sheet", value=st.session_state.sheet_url, key="k_url")
            conectar = st.form_submit_button("Conectar")
        if conectar and url:
            try:
                nuevo = load_gsheet(st.secrets["GOOGLE_CREDENTIALS"], url)
                if nuevo and len(nuevo) > 0:
                    st.session_state.sheet_url = url
                    st.session_state.data = nuevo
                    st.success("Google Sheet conectado.")
                else:
                    st.warning("La hoja no tiene datos.")
            except Exception as e:
                st.error(f"Error conectando Google Sheet: {e}")

data = st.session_state.data

with col2:
    if data:
        st.markdown("### 📄 Vista previa")
        for name, df in data.items():
            st.markdown(f"#### 📘 Hoja: {name}")
            st.dataframe(df.head(10))

with col3:
    if data:
        st.markdown("### 🤖 Consulta con IA")
        pregunta = st.text_area("Pregunta")

        if st.button("📊 Análisis General Automático"):
            analisis = analizar_datos_taller(st.session_state.data)
            texto_analisis = json.dumps(analisis, indent=2, ensure_ascii=False)
            prompt = f"""
Eres un controller financiero senior.
Con base en los datos calculados (reales) a continuación, entrega un análisis profesional, directo y accionable.
Si una visualización ayuda a entender mejor, incluye UNA instrucción exacta (sin viñetas, sola en una línea):
- grafico_torta:col_categoria|col_valor|titulo
- grafico_barras:col_categoria|col_valor|titulo
- tabla:col_categoria|col_valor
No inventes datos.

Datos calculados:
{texto_analisis}
"""
            respuesta = ask_gpt(prompt)
            st.markdown(respuesta)
            st.session_state.historial.append({"pregunta": "Análisis general", "respuesta": respuesta})
            parse_and_render_instructions(respuesta, st.session_state.data)

        if st.button("Responder") and pregunta:
            # Atajos de continuidad (según la última vista)
            if any(k in _norm(pregunta) for k in ["ahora","segun lo anterior","según lo anterior","mismo","misma","anterior"]):
                last = st.session_state.get("__ultima_vista__")
                if last:
                    hoja = last["sheet"]; cat = last["cat"]; val = last["val"]
                    if "top" in _norm(pregunta) and any(n in _norm(pregunta) for n in ["5","cinco"]):
                        df = st.session_state.data[hoja]
                        resumen = df.groupby(cat)[val].sum().sort_values(ascending=False).head(5).reset_index()
                        st.dataframe(resumen, use_container_width=True)
                        st.stop()

            schema = _build_schema(st.session_state.data)
            plan = plan_from_llm(pregunta, schema)

            executed = False
            if plan:
                executed = execute_plan(plan, st.session_state.data)

            if not executed:
                analisis = analizar_datos_taller(st.session_state.data)
                texto_analisis = json.dumps(analisis, indent=2, ensure_ascii=False)
                prompt = f"""
Actúa como controller financiero senior. Responde a la pregunta.
Si puedes responder con una tabla o gráfico, incluye UNA instrucción exacta en una sola línea:
- grafico_torta:col_categoria|col_valor|titulo
- grafico_barras:col_categoria|col_valor|titulo
- tabla:col_categoria|col_valor[|titulo]
Si sólo piden un gráfico sin tipo, elige el más adecuado (barras/torta/línea).
No expliques el método si puedes entregar el resultado agregado.

Pregunta: {pregunta}

Esquema de columnas:
{json.dumps(schema, ensure_ascii=False)}
Datos resumen:
{texto_analisis}
"""
                respuesta = ask_gpt(prompt)
                st.markdown(respuesta)
                st.session_state.historial.append({"pregunta": pregunta, "respuesta": respuesta})
                parse_and_render_instructions(respuesta, st.session_state.data)
            else:
                respuesta = ask_gpt(f"En 5-8 frases, interpreta el objeto visual generado para la pregunta: {pregunta}. Sé concreto y accionable.")
                st.markdown(respuesta)
                st.session_state.historial.append({"pregunta": pregunta, "respuesta": respuesta})

    if st.session_state.historial:
        with st.expander("🧠 Historial de la sesión"):
            for i, h in enumerate(st.session_state.historial[-10:], 1):
                st.markdown(f"**Q{i}:** {h['pregunta']}")
                st.markdown(f"**A{i}:** {h['respuesta']}")
