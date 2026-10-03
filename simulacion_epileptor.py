import numpy as np
import matplotlib.pyplot as plt

from tvb.simulator.lab import (
    connectivity,
    coupling,
    integrators,
    monitors,
    simulator
)

from tvb.simulator.models.epileptor import Epileptor


# --------------------------------------------------
# 1. CARGAR CONECTIVIDAD CEREBRAL
# --------------------------------------------------

con = connectivity.Connectivity.from_file()

con.configure()

print("Conectividad cargada correctamente")
print("Número de regiones:", con.number_of_regions)
print("\nRegiones disponibles:")

for i, region in enumerate(con.region_labels):
    print(i, "-", region)
    # --------------------------------------------------
# 2. CONFIGURAR MODELO EPILEPTOR
# --------------------------------------------------

numero_regiones = con.number_of_regions

# Excitabilidad basal de las regiones
x0 = np.full(numero_regiones, -2.4)

# Nodo seleccionado como foco epileptógeno
nodo_foco = 12
x0[nodo_foco] = -1.6

modelo = Epileptor(
    x0=x0,
    Iext=np.array([3.1]),
    Iext2=np.array([0.45])
)

print("\nModelo Epileptor configurado correctamente")
print("Nodo epileptógeno:", nodo_foco, "-", con.region_labels[nodo_foco])
print("x0 del foco:", x0[nodo_foco])
print("x0 del resto de la red:", -2.4)
# --------------------------------------------------
# 3. CONFIGURAR SIMULACIÓN
# --------------------------------------------------

# Acoplamiento entre las regiones cerebrales
acoplamiento = coupling.Difference(a=np.array([0.1]))

# Integrador numérico
integrador = integrators.HeunDeterministic(dt=0.1)

# Monitor para guardar la actividad temporal
monitor = monitors.TemporalAverage(period=1.0)

sim = simulator.Simulator(
    model=modelo,
    connectivity=con,
    coupling=acoplamiento,
    integrator=integrador,
    monitors=(monitor,)
)

sim.configure()

print("\nSimulador configurado correctamente")

# 4. EJECUTAR SIMULACIÓN EPILEPTOR

print("\nEjecutando simulación Epileptor...")

resultados = sim.run(simulation_length=5000.0)

tiempo, datos = resultados[0]

print("Simulación terminada correctamente")
print("Forma de los datos:", datos.shape)

# Extraer la primera variable de estado de Epileptor
actividad = datos[:, 0, :, 0]

print("Nodo foco analizado:", con.region_labels[nodo_foco])

# Guardar resultados
np.save("resultados/tiempo_epileptor.npy", tiempo)
np.save("resultados/actividad_epileptor.npy", actividad)

print("Resultados guardados correctamente")

# 5. GRAFICAR ACTIVIDAD EPILEPTIFORME

plt.figure(figsize=(11, 6))

# Nodo foco
plt.plot(
    tiempo,
    actividad[:, nodo_foco],
    label=f"Foco: {con.region_labels[nodo_foco]}",
    linewidth=2
)

# Algunos nodos de comparación
nodos_comparacion = [13, 25, 50]

for nodo in nodos_comparacion:
    plt.plot(
        tiempo,
        actividad[:, nodo],
        label=con.region_labels[nodo],
        alpha=0.7
    )

plt.title("Simulación de actividad epileptiforme con Epileptor")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Actividad simulada (x1)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resultados/propagacion_epileptor.png",
    dpi=300
)

plt.show()

print("Gráfica guardada en resultados/propagacion_epileptor.png")