import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --------------------------------------------------
# 1. CARGAR DATOS GENERADOS POR TVB
# --------------------------------------------------

datos = np.load("Reporte/datos_reposo.npy")
tiempo = np.load("Reporte/tiempo_reposo.npy")

print("Datos cargados correctamente")
print("Forma de los datos:", datos.shape)

# Seleccionar una región cerebral como señal de ejemplo
eeg = datos[:, 0, 0, 0]

# --------------------------------------------------
# 2. FRECUENCIA DE MUESTREO
# --------------------------------------------------

# El tiempo de TVB está expresado en milisegundos
dt_ms = np.mean(np.diff(tiempo))
fs = 1000.0 / dt_ms

print("Frecuencia de muestreo:", fs, "Hz")

# --------------------------------------------------
# 3. CALCULAR PSD MEDIANTE EL MÉTODO DE WELCH
# --------------------------------------------------

frecuencias, psd = welch(
    eeg,
    fs=fs,
    nperseg=min(256, len(eeg))
)

# --------------------------------------------------
# 4. DEFINIR BANDAS DE FRECUENCIA
# --------------------------------------------------

# Beta: 13 - 30 Hz
beta_min = 13
beta_max = 30

# Gamma: 30 - 80 Hz
gamma_min = 30
gamma_max = 80

# Seleccionar frecuencias pertenecientes a cada banda
indices_beta = (frecuencias >= beta_min) & (frecuencias < beta_max)
indices_gamma = (frecuencias >= gamma_min) & (frecuencias <= gamma_max)

# --------------------------------------------------
# 5. CALCULAR POTENCIA EN CADA BANDA
# --------------------------------------------------

potencia_beta = np.trapezoid(
    psd[indices_beta],
    frecuencias[indices_beta]
)

potencia_gamma = np.trapezoid(
    psd[indices_gamma],
    frecuencias[indices_gamma]
)

print("Potencia Beta:", potencia_beta)
print("Potencia Gamma:", potencia_gamma)

# --------------------------------------------------
# 6. GENERAR GRÁFICA DE LA PSD
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(frecuencias, psd, label="PSD")

plt.axvspan(
    beta_min,
    beta_max,
    alpha=0.3,
    label="Beta (13-30 Hz)"
)

plt.axvspan(
    gamma_min,
    gamma_max,
    alpha=0.3,
    label="Gamma (30-80 Hz)"
)

plt.xlim(0, 100)

plt.title("Densidad espectral de potencia del EEG simulado")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (potencia/Hz)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resultados/psd_beta_gamma.png",
    dpi=300
)

plt.show()

print("Gráfica guardada en resultados/psd_beta_gamma.png")