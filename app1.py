import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tvb.simulator.lab import connectivity

# Configuración de la página

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
# --------------------------------------------------
# VISOR 3D DEL CONECTOMA
# --------------------------------------------------
st.success("PRUEBA: el bloque 3D se está ejecutando")
st.subheader("🌐 Conectoma cerebral 3D")

st.write(
    "Visualización interactiva de las regiones cerebrales y "
    "sus principales conexiones estructurales."
)

# Cargar conectividad de TVB
con = connectivity.Connectivity.from_file()
con.configure()

coordenadas = con.centres
pesos = con.weights

# Coordenadas de los nodos
x = coordenadas[:, 0]
y = coordenadas[:, 1]
z = coordenadas[:, 2]

# Crear nodos
nodos_3d = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="markers",
    marker=dict(
        size=6,
        opacity=0.9
    ),
    text=con.region_labels,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "X: %{x:.2f}<br>"
        "Y: %{y:.2f}<br>"
        "Z: %{z:.2f}"
        "<extra></extra>"
    ),
    name="Regiones cerebrales"
)

# Seleccionar las conexiones más fuertes
pesos_positivos = pesos[pesos > 0]
umbral = np.percentile(pesos_positivos, 90)

lineas_x = []
lineas_y = []
lineas_z = []

for i in range(con.number_of_regions):
    for j in range(i + 1, con.number_of_regions):

        if pesos[i, j] >= umbral:
            lineas_x.extend([
                coordenadas[i, 0],
                coordenadas[j, 0],
                None
            ])

            lineas_y.extend([
                coordenadas[i, 1],
                coordenadas[j, 1],
                None
            ])

            lineas_z.extend([
                coordenadas[i, 2],
                coordenadas[j, 2],
                None
            ])

# Crear conexiones
conexiones_3d = go.Scatter3d(
    x=lineas_x,
    y=lineas_y,
    z=lineas_z,
    mode="lines",
    line=dict(width=1),
    opacity=0.35,
    hoverinfo="skip",
    name="Conexiones"
)

# Crear figura
fig_3d = go.Figure(
    data=[conexiones_3d, nodos_3d]
)

fig_3d.update_layout(
    height=700,
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=20
    )
)

# Mostrar en Streamlit
st.plotly_chart(
    fig_3d,
    use_container_width=True
)

st.subheader("🌐 Conectoma cerebral 3D")
st.write(
    "Visualización interactiva de las regiones cerebrales y "
    "sus principales conexiones estructurales."
)
st.caption("Proyecto desarrollado con The Virtual Brain y Streamlit.")
# Cargar conectividad cerebral de TVB
con = connectivity.Connectivity.from_file()
con.configure()

coordenadas = con.centres
pesos = con.weights
# Coordenadas X, Y, Z
x = coordenadas[:, 0]
y = coordenadas[:, 1]
z = coordenadas[:, 2]

# Crear los nodos cerebrales
nodos_3d = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="markers",
    marker=dict(
        size=6,
        opacity=0.9
    ),
    text=con.region_labels,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "X: %{x:.2f}<br>"
        "Y: %{y:.2f}<br>"
        "Z: %{z:.2f}"
        "<extra></extra>"
    ),
    name="Regiones cerebrales"
)# Seleccionar las conexiones más fuertes
pesos_positivos = pesos[pesos > 0]
umbral = np.percentile(pesos_positivos, 90)

lineas_x = []
lineas_y = []
lineas_z = []
for i in range(con.number_of_regions):
    for j in range(i + 1, con.number_of_regions):
        if pesos[i, j] >= umbral:
            lineas_x.extend([
                coordenadas[i, 0],
                coordenadas[j, 0],
                None
            ])

            lineas_y.extend([
                coordenadas[i, 1],
                coordenadas[j, 1],
                None
            ])

            lineas_z.extend([
                coordenadas[i, 2],
                coordenadas[j, 2],
                None
            ])
            # Crear las conexiones 3D
conexiones_3d = go.Scatter3d(
    x=lineas_x,
    y=lineas_y,
    z=lineas_z,
    mode="lines",
    line=dict(width=1),
    opacity=0.35,
    hoverinfo="skip",
    name="Conexiones"
)# Crear la figura del conectoma
fig_3d = go.Figure(
    data=[conexiones_3d, nodos_3d]
)

fig_3d.update_layout(
    height=700,
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=20
    )
)
# Mostrar el conectoma 3D en Streamlit
st.plotly_chart(
    fig_3d,
    use_container_width=True
)