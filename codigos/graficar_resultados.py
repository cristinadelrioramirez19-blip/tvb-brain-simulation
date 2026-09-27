import numpy as np
import matplotlib.pyplot as plt


# 1. CARGAR LOS RESULTADOS

tiempo = np.load("tiempo_reposo.npy")
datos = np.load("datos_reposo.npy")

print("Archivos cargados correctamente")
print("Forma del tiempo:", tiempo.shape)
print("Forma de los datos:", datos.shape)


# 2. EXTRAER ALGUNAS REGIONES

# Los datos tienen forma:
# (tiempo, variable, region, modo)

region_1 = datos[:, 0, 0, 0]
region_2 = datos[:, 0, 1, 0]
region_3 = datos[:, 0, 2, 0]


# 3. CREAR LA GRAFICA


plt.figure(figsize=(12, 6))

plt.plot(tiempo, region_1, label="Region 1")
plt.plot(tiempo, region_2, label="Region 2")
plt.plot(tiempo, region_3, label="Region 3")

plt.title("Actividad cerebral simulada en reposo")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Actividad neuronal simulada")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# 4. GUARDAR COMO PNG

plt.savefig(
    "actividad_cerebral_reposo.png",
    dpi=300,
    bbox_inches="tight"
)

print("Grafica guardada correctamente:")
print("actividad_cerebral_reposo.png")


# 5. MOSTRAR LA GRAFICA

plt.show()