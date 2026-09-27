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