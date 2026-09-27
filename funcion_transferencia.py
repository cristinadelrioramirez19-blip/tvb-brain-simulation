import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# DATOS FICTICIOS
# Potencia espectral de la corteza motora
# --------------------------------------------------

potencia_espectral = np.array([
    1.0, 1.5, 2.0, 2.5, 3.0,
    3.5, 4.0, 4.5, 5.0, 5.5
])

# --------------------------------------------------
# FUNCIÓN DE TRANSFERENCIA
#
# f_temblor = a * P_motora + b
#
# P_motora: potencia espectral de la corteza motora
# f_temblor: frecuencia estimada del temblor (Hz)
# a: sensibilidad del modelo
# b: frecuencia basal
# --------------------------------------------------

a = 0.8
b = 2.0

frecuencia_temblor = a * potencia_espectral + b

# Mostrar resultados en la terminal
print("Potencia espectral:", potencia_espectral)
print("Frecuencia de temblor estimada:", frecuencia_temblor)

# --------------------------------------------------
# GRÁFICA
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    potencia_espectral,
    frecuencia_temblor,
    marker="o"
)

plt.title("Potencia espectral vs frecuencia de temblor")
plt.xlabel("Potencia espectral de la corteza motora (u.a.)")
plt.ylabel("Frecuencia de temblor estimada (Hz)")
plt.grid(True)

plt.tight_layout()

# Guardar la gráfica
plt.savefig("resultados/funcion_transferencia_temblor.png", dpi=300)

plt.show()

print("Gráfica guardada correctamente.")