import streamlit as st
import pickle
import Orange
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Clasificador de Vulnerabilidad Urbana - RENAMU 2026",
    layout="wide"
)

st.title("Clasificador de Vulnerabilidad Urbana Distrital (RENAMU 2026)")
st.markdown(
    "Aplicación web basada en **Regresión Logística** entrenada en **Orange Data Mining** "
    "con microdatos oficiales del Registro Nacional de Municipalidades (INEI)."
)

@st.cache_resource
def cargar_modelo():
    with open("Nivel_vulnerabilidad.pkcls", "rb") as archivo:
        return pickle.load(archivo)

try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró el archivo 'Nivel_vulnerabilidad.pkcls' en la carpeta del proyecto.")
    st.stop()

def convertir_fila_a_dominio_orange(fila_dict, modelo_orange):
    """Traduce las variables originales (Tipomuni, P19_1_T, P23_1...P70_1)
    al formato codificado (Continuize) que espera el modelo guardado en Orange."""
    dominio = modelo_orange.domain
    valores_orange = []

    # Normalizar claves de entrada
    datos_norm = {str(k).strip().lower(): v for k, v in fila_dict.items()}

    for attr in dominio.attributes:
        nombre_attr = attr.name.strip()

        if "=" in nombre_attr:
            # Ejemplo: "P70_1=2" o "Tipomuni=2" -> var_base="p70_1", val_esperado="2"
            var_base, val_esperado = nombre_attr.split("=", 1)
            var_base_norm = var_base.strip().lower()
            val_esperado_norm = val_esperado.strip().lower()

            if var_base_norm in datos_norm and pd.notna(datos_norm[var_base_norm]):
                raw_val = datos_norm[var_base_norm]
                s_val = str(int(raw_val)) if isinstance(raw_val, float) and raw_val.is_integer() else str(raw_val).strip()
                s_val_norm = s_val.lower()

                # Si Tipomuni viene como texto ("Provincial" / "Distrital") y el modelo espera "2" / "1"
                if var_base_norm == "tipomuni":
                    if s_val_norm == "distrital":
                        s_val_norm = "1"
                    elif s_val_norm == "provincial":
                        s_val_norm = "2"

                valores_orange.append(1.0 if s_val_norm == val_esperado_norm else 0.0)
            else:
                valores_orange.append(0.0)
        else:
            # Variable numérica continua directa (ej. P19_1_T)
            nombre_norm = nombre_attr.lower()
            if nombre_norm in datos_norm and pd.notna(datos_norm[nombre_norm]):
                valores_orange.append(float(datos_norm[nombre_norm]))
            else:
                valores_orange.append(0.0)

    for _ in dominio.class_vars:
        valores_orange.append(float("nan"))

    for meta in dominio.metas:
        m_norm = meta.name.strip().lower()
        if m_norm in datos_norm and pd.notna(datos_norm[m_norm]):
            valores_orange.append(str(datos_norm[m_norm]))
        else:
            valores_orange.append("")

    return valores_orange

def predecir_dataframe(df_datos, modelo_orange):
    dominio = modelo_orange.domain
    filas_orange = [convertir_fila_a_dominio_orange(fila.to_dict(), modelo_orange) for _, fila in df_datos.iterrows()]

    tabla = Orange.data.Table.from_list(dominio, filas_orange)
    predicciones_idx = modelo_orange(tabla)
    probabilidades = modelo_orange(tabla, modelo_orange.Probs)

    clases = dominio.class_var.values
    nombres_predichos = [dominio.class_var.str_val(idx) for idx in predicciones_idx]
    return nombres_predichos, probabilidades, clases

tab1, tab2 = st.tabs(["Ingreso Manual de Distrito", "Carga Masiva desde Excel (.xlsx)"])

# ==========================================
# PESTAÑA 1: INGRESO MANUAL
# ==========================================
with tab1:
    st.subheader("Ingrese los indicadores operativos de la municipalidad")

    col1, col2 = st.columns(2)
    with col1:
        tipomuni = st.selectbox("Tipo de Municipalidad (Tipomuni)", ["Distrital", "Provincial"])
        personal = st.number_input("Total de personal municipal (P19_1_T)", min_value=0, max_value=15000, value=60, step=1)
        p23 = st.selectbox("Servicio de Serenazgo (P23_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")
        p29 = st.selectbox("Recojo de residuos sólidos (P29_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")

    with col2:
        p30 = st.selectbox("Limpieza pública (P30_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")
        p34 = st.selectbox("Parques y jardines (P34_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")
        p48 = st.selectbox("Abastecimiento de agua potable (P48_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")
        p70 = st.selectbox("Alcantarillado y saneamiento (P70_1)", ["1", "2"], format_func=lambda x: "1 - Sí presta el servicio" if x == "1" else "2 - No presta el servicio")

    if personal == 0:
        st.warning("Advertencia: El personal municipal registrado es 0. Verifique que el dato sea correcto.")
    elif personal > 5000:
        st.warning("Advertencia: El personal municipal supera los 5,000 trabajadores (valor atípico en distritos estándar).")

    if st.button("Evaluar Vulnerabilidad Urbana", type="primary"):
        df_manual = pd.DataFrame([{
            "Tipomuni": tipomuni,
            "P19_1_T": personal,
            "P23_1": p23,
            "P29_1": p29,
            "P30_1": p30,
            "P34_1": p34,
            "P48_1": p48,
            "P70_1": p70
        }])

        st.write("**Resumen de datos ingresados:**")
        st.dataframe(df_manual, use_container_width=True)

        preds, probs, clases = predecir_dataframe(df_manual, modelo)
        resultado = preds[0]
        prob_fila = probs[0]

        st.divider()
        texto_resultado = f"Nivel de Vulnerabilidad Predicho: **{resultado.upper()}**"
        if resultado == "Alto":
            st.error(texto_resultado)
        elif resultado == "Medio":
            st.warning(texto_resultado)
        else:
            st.success(texto_resultado)

        st.write("**Probabilidades estimadas por nivel de vulnerabilidad:**")
        df_probs = pd.DataFrame({"Nivel": clases, "Probabilidad": prob_fila}).set_index("Nivel")
        st.bar_chart(df_probs)

# ==========================================
# PESTAÑA 2: CARGA MASIVA DESDE EXCEL
# ==========================================
with tab2:
    st.subheader("Cargar archivo Excel con registros municipales")
    archivo_subido = st.file_uploader("Seleccione un archivo .xlsx", type=["xlsx"])

    columnas_requeridas = ["Tipomuni", "P19_1_T", "P23_1", "P29_1", "P30_1", "P34_1", "P48_1", "P70_1"]
    columnas_meta = ["Ubigeo", "Departamento", "Provincia", "Distrito"]

    if archivo_subido is not None:
        df_excel = pd.read_excel(archivo_subido)
        st.write("**Vista previa de los datos cargados:**")
        st.dataframe(df_excel.head(10), use_container_width=True)

        cols_excel_lower = [c.strip().lower() for c in df_excel.columns]
        faltantes = [col for col in columnas_requeridas if col.lower() not in cols_excel_lower]
        if faltantes:
            st.error(f"El archivo no contiene las siguientes columnas obligatorias: {', '.join(faltantes)}")
        else:
            if st.button("Ejecutar Predicción Masiva"):
                preds, probs, clases = predecir_dataframe(df_excel, modelo)

                df_resultados = df_excel.copy()
                df_resultados["Vulnerabilidad_Predicha"] = preds
                for i, clase in enumerate(clases):
                    df_resultados[f"Prob_{clase}"] = np.round(probs[:, i], 4)

                st.success(f"Clasificación completada para {len(df_resultados)} registros.")
                st.dataframe(df_resultados, use_container_width=True)

                cols_exportar = [c for c in df_resultados.columns if c not in columnas_meta]
                csv_bytes = df_resultados[cols_exportar].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Descargar Resultados en CSV",
                    data=csv_bytes,
                    file_name="predicciones_vulnerabilidad_renamu.csv",
                    mime="text/csv"
                )

st.divider()
st.caption(
    "Aviso: Esta herramienta tiene fines exclusivamente académicos (Curso de Inteligencia Artificial Aplicada - "
    "Universidad de Lima) y constituye un apoyo analítico preliminar para la gestión pública."
)