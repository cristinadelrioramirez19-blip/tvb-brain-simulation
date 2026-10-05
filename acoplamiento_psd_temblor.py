import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 
# 1. CARGAR DATOS GENERADOS POR TVB
# 

datos = np.load("Reporte/datos_reposo.npy")
tiempo = np.load("Reporte/tiempo_reposo.npy")

print("Datos cargados correctamente")
print("Forma de los datos:", datos.shape)

# Seleccionar la señal utilizada para el análisis
senal_motora = datos[:, 0, 0, 0]

# Calcular frecuencia de muestreo
dt_ms = np.mean(np.diff(tiempo))
fs = 1000.0 / dt_ms

print("Frecuencia de muestreo:", fs, "Hz")
# 2. CALCULAR DENSIDAD ESPECTRAL DE POTENCIA (PSD)

frecuencias, psd = welch(
    senal_motora,
    fs=fs,
    nperseg=min(256, len(senal_motora))
)

print("PSD calculada correctamente")
print("Número de frecuencias analizadas:", len(frecuencias))
# 3. CALCULAR POTENCIA EN BANDAS BETA Y GAMMA

beta_min = 13
beta_max = 30

gamma_min = 30
gamma_max = 80

indices_beta = (frecuencias >= beta_min) & (frecuencias < beta_max)
indices_gamma = (frecuencias >= gamma_min) & (frecuencias <= gamma_max)

potencia_beta = np.trapezoid(
    psd[indices_beta],
    frecuencias[indices_beta]
)

potencia_gamma = np.trapezoid(
    psd[indices_gamma],
    frecuencias[indices_gamma]
)

# Potencia espectral utilizada como entrada
# de la función de transferencia
potencia_motora = potencia_beta + potencia_gamma

print("\nPotencia Beta:", potencia_beta)
print("Potencia Gamma:", potencia_gamma)
print("Potencia espectral motora total:", potencia_motora)

# 4. FUNCIÓN DE TRANSFERENCIA BIOMECÁNICA


# f_temblor = a * P_motora + b

a = 0.8
b = 2.0

frecuencia_temblor = a * potencia_motora + b

print("\n--- RESULTADO DEL ACOPLAMIENTO ---")
print("Potencia motora de entrada:", potencia_motora)
print("Frecuencia de temblor estimada:", frecuencia_temblor, "Hz")

# 5. CÁLCULO DINÁMICO DEL INDICADOR DE TEMBLOR


# Dividir la señal en ventanas temporales
tamano_ventana = 250
paso_ventana = 125

tiempos_dinamicos = []
frecuencias_temblor_dinamicas = []

print("\nCalculando indicador dinámico...")
for inicio in range(
    0,
    len(senal_motora) - tamano_ventana + 1,
    paso_ventana
):
    fin = inicio + tamano_ventana

    # Extraer ventana de la señal
    ventana = senal_motora[inicio:fin]

    # Calcular PSD de la ventana
    frecuencias_ventana, psd_ventana = welch(
        ventana,
        fs=fs,
        nperseg=min(256, len(ventana))
    )

    # Seleccionar bandas Beta y Gamma
    beta = (
        (frecuencias_ventana >= beta_min)
        & (frecuencias_ventana < beta_max)
    )

    gamma = (
        (frecuencias_ventana >= gamma_min)
        & (frecuencias_ventana <= gamma_max)
    )

    # Calcular potencia espectral
    potencia_beta_ventana = np.trapezoid(
        psd_ventana[beta],
        frecuencias_ventana[beta]
    )

    potencia_gamma_ventana = np.trapezoid(
        psd_ventana[gamma],
        frecuencias_ventana[gamma]
    )

    potencia_motora_ventana = (
        potencia_beta_ventana + potencia_gamma_ventana
    )

    # Aplicar función de transferencia
    frecuencia_temblor_ventana = (
        a * potencia_motora_ventana + b
    )

    # Tiempo correspondiente al centro de la ventana
    tiempo_centro = tiempo[inicio + tamano_ventana // 2]

    tiempos_dinamicos.append(tiempo_centro)
    frecuencias_temblor_dinamicas.append(
        frecuencia_temblor_ventana
    )
    # Convertir resultados a arreglos de NumPy
tiempos_dinamicos = np.array(tiempos_dinamicos)
frecuencias_temblor_dinamicas = np.array(
    frecuencias_temblor_dinamicas
)

print("\n--- INDICADOR DINÁMICO DE TEMBLOR ---")

for t, f in zip(
    tiempos_dinamicos,
    frecuencias_temblor_dinamicas
):
    print(
        f"Tiempo: {t:.1f} ms | "
        f"Frecuencia estimada: {f:.6f} Hz"
    )
    
# 6. GRÁFICA DEL INDICADOR DINÁMICO


plt.figure(figsize=(9, 5))

plt.plot(
    tiempos_dinamicos,
    frecuencias_temblor_dinamicas,
    marker="o"
)

plt.title(
    "Indicador dinámico de impacto en el temblor"
)

plt.xlabel("Tiempo (ms)")
plt.ylabel("Frecuencia estimada de temblor (Hz)")
plt.ticklabel_format(
    axis="y",
    style="plain",
    useOffset=False
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "resultados/impacto_temblor_dinamico.png",
    dpi=300
)

plt.show()

print(
    "\nGráfica guardada en "
    "resultados/impacto_temblor_dinamico.png"
)