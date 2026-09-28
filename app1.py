import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulación Cerebral con TVB",
    page_icon="🧠",
    layout="wide"
)

# Título principal
st.title("🧠 Simulación Cerebral con The Virtual Brain")

st.write(
    "Interfaz para la visualización de resultados de una "
    "simulación de actividad cerebral en reposo."
)

st.divider()
# CONTROLES DE ESTIMULACIÓN DBS

st.subheader("⚡ Controles de estimulación DBS")

frecuencia_dbs = st.slider(
    "Frecuencia de estimulación (Hz)",
    min_value=0,
    max_value=200,
    value=130,
    step=1
)

amplitud_dbs = st.slider(
    "Amplitud de estimulación (u.a.)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.1
)

st.write("Frecuencia seleccionada:", frecuencia_dbs, "Hz")
st.write("Amplitud seleccionada:", amplitud_dbs, "u.a.")

st.divider()

# RESECCIÓN VIRTUAL

st.subheader("🧠 Resección virtual")

# Nombres de las 76 regiones de la conectividad de TVB
regiones = [f"Región {i}" for i in range(1, 77)]

nodos_desconectados = st.multiselect(
    "Selecciona los nodos que deseas desconectar:",
    options=regiones
)

if nodos_desconectados:
    st.warning(
        "Nodos seleccionados para resección virtual: "
        + ", ".join(nodos_desconectados)
    )
else:
    st.info("No hay nodos desconectados.")

# Crear dos columnas
col1, col2 = st.columns(2)

# Columna izquierda
with col1:
    st.subheader("Actividad cerebral")
    st.info("Espacio reservado para el gráfico de actividad cerebral.")

# Columna derecha
with col2:
    st.subheader("Conectividad cerebral")
    st.info("Espacio reservado para el gráfico de conectividad cerebral.")

st.divider()

st.caption("Proyecto desarrollado con The Virtual Brain y Streamlit.")