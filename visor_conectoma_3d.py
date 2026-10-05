import numpy as np
import plotly.graph_objects as go

from tvb.simulator.lab import connectivity


# --------------------------------------------------
# 1. CARGAR CONECTIVIDAD DE TVB
# --------------------------------------------------

con = connectivity.Connectivity.from_file()
con.configure()

print("Conectividad cargada correctamente")
print("Número de regiones:", con.number_of_regions)


# --------------------------------------------------
# 2. OBTENER COORDENADAS DE LOS NODOS
# --------------------------------------------------

coordenadas = con.centres

print("Forma de las coordenadas:", coordenadas.shape)

print("\nPrimeras 5 regiones:")

for i in range(5):
    print(
        i,
        con.region_labels[i],
        "->",
        coordenadas[i]
    )
    # --------------------------------------------------
# 3. CREAR NODOS DEL CONECTOMA 3D
# --------------------------------------------------

x = coordenadas[:, 0]
y = coordenadas[:, 1]
z = coordenadas[:, 2]

nodos = go.Scatter3d(
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
    )
)
# --------------------------------------------------
# 4. CREAR CONEXIONES DEL CONECTOMA
# --------------------------------------------------

pesos = con.weights

# Mostrar solo las conexiones más fuertes para evitar
# saturar visualmente el conectoma
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

conexiones = go.Scatter3d(
    x=lineas_x,
    y=lineas_y,
    z=lineas_z,
    mode="lines",
    line=dict(
        width=1
    ),
    opacity=0.35,
    hoverinfo="skip"
)
fig = go.Figure(data=[conexiones, nodos])
fig.update_layout(
    title="Conectoma cerebral 3D - The Virtual Brain",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()