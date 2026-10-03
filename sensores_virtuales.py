import numpy as np
import matplotlib.pyplot as plt


# 1. CARGAR RESULTADOS DE EPILEPTOR

tiempo = np.load("resultados/tiempo_epileptor.npy")
actividad = np.load("resultados/actividad_epileptor.npy")

print("Resultados de Epileptor cargados correctamente")
print("Forma de la actividad:", actividad.shape)
print("Número de muestras:", len(tiempo))


# 2. DEFINIR REGIONES DE INTERÉS

# Corteza motora primaria derecha
rM1 = 12

# Corteza motora primaria izquierda
lM1 = 50

print("\nSensores virtuales configurados:")
print("Sensor 1 -> rM1, nodo", rM1)
print("Sensor 2 -> lM1, nodo", lM1)
# --------------------------------------------------
# 3. EXTRAER SEÑALES DE LOS SENSORES VIRTUALES
# --------------------------------------------------

senal_rM1 = actividad[:, rM1]
senal_lM1 = actividad[:, lM1]

print("\nSeñales extraídas correctamente")
print("Muestras sensor rM1:", len(senal_rM1))
print("Muestras sensor lM1:", len(senal_lM1))

# --------------------------------------------------
# 4. GRAFICAR MONITOREO VIRTUAL
# --------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    tiempo,
    senal_rM1,
    label="Sensor virtual rM1",
    linewidth=2
)

plt.plot(
    tiempo,
    senal_lM1,
    label="Sensor virtual lM1",
    alpha=0.8
)

plt.title("Monitoreo virtual de la corteza motora primaria")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Actividad simulada (x1)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resultados/sensores_virtuales_M1.png",
    dpi=300
)

plt.show()

print("\nGráfica guardada en resultados/sensores_virtuales_M1.png")
