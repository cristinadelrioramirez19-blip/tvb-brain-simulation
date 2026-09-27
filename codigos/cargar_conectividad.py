from tvb.simulator.lab import connectivity

# Cargar la conectividad cerebral estándar incluida en TVB
conn = connectivity.Connectivity.from_file()

# Preparar la conectividad
conn.configure()

# Mostrar información
print("Conectividad cargada correctamente")
print("Numero de regiones:", len(conn.region_labels))
print("Primeras 10 regiones:")
print(conn.region_labels[:10])

print("Tamaño de la matriz de conectividad:")
print(conn.weights.shape)
