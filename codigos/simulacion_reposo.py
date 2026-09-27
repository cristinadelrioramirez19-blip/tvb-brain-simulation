from tvb.simulator.lab import *
import numpy as np

# ---------------------------------
# 1. CARGAR CONECTIVIDAD CEREBRAL
# ---------------------------------

conn = connectivity.Connectivity.from_file()
conn.configure()

print("Conectividad cargada:")
print("Numero de regiones:", len(conn.region_labels))


# ---------------------------------
# 2. MODELO DE ACTIVIDAD NEURONAL
# ---------------------------------

modelo = models.Generic2dOscillator()


# ---------------------------------
# 3. ACOPLAMIENTO ENTRE REGIONES
# ---------------------------------

acoplamiento = coupling.Linear(a=np.array([0.015]))


# ---------------------------------
# 4. INTEGRADOR
# ---------------------------------

integrador = integrators.HeunDeterministic(
    dt=0.1
)


# ---------------------------------
# 5. MONITOR
# ---------------------------------

monitor = monitors.TemporalAverage(
    period=1.0
)


# ---------------------------------
# 6. CREAR EL SIMULADOR
# ---------------------------------

sim = simulator.Simulator(
    model=modelo,
    connectivity=conn,
    coupling=acoplamiento,
    integrator=integrador,
    monitors=(monitor,)
)

sim.configure()

print("Simulador configurado correctamente")


# ---------------------------------
# 7. EJECUTAR SIMULACION
# ---------------------------------

print("Iniciando simulacion...")

resultados = sim.run(
    simulation_length=1000.0
)

tiempo, datos = resultados[0]


# ---------------------------------
# 8. GUARDAR RESULTADOS
# ---------------------------------

np.save("tiempo_reposo.npy", tiempo)
np.save("datos_reposo.npy", datos)

print("Simulacion terminada correctamente")
print("Forma de los datos:", datos.shape)
print("Resultados guardados:")
print("- tiempo_reposo.npy")
print("- datos_reposo.npy")