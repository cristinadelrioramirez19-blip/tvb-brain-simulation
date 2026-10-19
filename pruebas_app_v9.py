"""
Pruebas cuantitativas de funciones reales extraídas de app_v9.py.
No inicia Streamlit ni modifica app_v9.py.
Ejecutar desde la carpeta que contiene ambos archivos:
    python pruebas_app_v9.py

IMPORTANTE:
- El script extrae las funciones generar_atlas_conectomico_76 y
  simular_crisis_multifase directamente desde app_v9.py.
- Las señales y la conectividad del código actual son sintéticas.
- Las pruebas no constituyen validación clínica.
"""
import ast
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

APP_FILE = Path(__file__).with_name("app_v9.py")
CSV_FILE = Path(__file__).with_name("resultados_app_v9.csv")

def cargar_funciones_reales(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta.name}. Coloca pruebas_app_v9.py junto a app_v9.py."
        )
    source = ruta.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(ruta))
    wanted = {"generar_atlas_conectomico_76", "simular_crisis_multifase"}
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            node.decorator_list = []  # Quita @st.cache_data solo en la copia de prueba
            found[node.name] = node
    missing = wanted - set(found)
    if missing:
        raise RuntimeError("No se encontraron estas funciones en app_v9.py: " + ", ".join(sorted(missing)))
    mini_tree = ast.Module(body=[found[name] for name in (
        "generar_atlas_conectomico_76", "simular_crisis_multifase"
    )], type_ignores=[])
    ast.fix_missing_locations(mini_tree)
    namespace = {"np": np}
    exec(compile(mini_tree, str(ruta), "exec"), namespace)
    return namespace

def main():
    funcs = cargar_funciones_reales(APP_FILE)
    atlas_fn = funcs["generar_atlas_conectomico_76"]
    sim_fn = funcs["simular_crisis_multifase"]
    results = []

    def check(test_id, name, passed, detail):
        results.append({
            "ID": test_id,
            "Prueba": name,
            "Resultado": "APROBADA" if passed else "FALLIDA",
            "Detalle": detail
        })
        print(f"[{'OK' if passed else 'FALLO'}] {test_id} {name}: {detail}")

    print("\nPRUEBAS DE FUNCIONES EXTRAÍDAS DE app_v9.py\n")
    regions, names, coords, conn = atlas_fn()
    n = len(names)
    check("V1", "Atlas con 76 regiones", n == 76, f"regiones={n}")
    check("V2", "Coordenadas finitas y forma correcta",
          coords.shape == (76, 3) and bool(np.isfinite(coords).all()),
          f"forma={coords.shape}; valores finitos={bool(np.isfinite(coords).all())}")
    check("V3", "Matriz de conectividad válida",
          conn.shape == (76, 76) and bool(np.isfinite(conn).all())
          and bool(np.allclose(conn, conn.T))
          and bool(np.allclose(np.diag(conn), 0)),
          f"forma={conn.shape}; simétrica={bool(np.allclose(conn, conn.T))}; diagonal cero={bool(np.allclose(np.diag(conn), 0))}")

    ez = names[0]
    pz = names[8]
    duration = 10
    fs = 250
    interventions = [
        "Ninguna (Línea Base / Crisis Natural)",
        "Resección Virtual del Foco (EZ)",
        "Neuromodulación DBS Talamocortical (130 Hz)",
        "Aislamiento Conectómico de Red (Ablación de Vías)"
    ]
    metrics = {}
    for k, intervention in enumerate(interventions):
        # La semilla hace repetible esta ejecución de prueba.
        np.random.seed(2026)
        t, eeg, f1, f2, f3 = sim_fn(duration, ez, pz, intervention, names, conn)
        valid = eeg.shape == (duration * fs, 76) and bool(np.isfinite(eeg).all())
        check(f"S{k+1}", f"Salida válida: {intervention}", valid,
              f"forma={eeg.shape}; finitos={bool(np.isfinite(eeg).all())}")
        ez_signal = eeg[:, names.index(ez)]
        pz_signal = eeg[:, names.index(pz)]
        # RMS en la ventana de inicio ictal, evitando la primera fase.
        start, end = int(f1 * fs), max(int(f2 * fs), int(f1 * fs) + 1)
        ez_rms = float(np.sqrt(np.mean(ez_signal[start:end] ** 2)))
        pz_rms = float(np.sqrt(np.mean(pz_signal[start:end] ** 2)))
        freqs, psd = welch(ez_signal, fs=fs, nperseg=min(256, len(ez_signal)))
        mask_hfo = (freqs >= 30) & (freqs <= 80)
        hfo_power = float(np.trapezoid(psd[mask_hfo], freqs[mask_hfo])) if np.count_nonzero(mask_hfo) > 1 else 0.0
        metrics[intervention] = {
            "EZ_RMS_inicio": ez_rms,
            "PZ_RMS_inicio": pz_rms,
            "Potencia_HFO_30_80": hfo_power,
            "Muestras": int(eeg.shape[0]),
            "Regiones": int(eeg.shape[1])
        }
        results[-1]["Detalle"] += (
            f"; RMS_EZ_inicio={ez_rms:.6f}; RMS_PZ_inicio={pz_rms:.6f}; "
            f"potencia_HFO_30_80={hfo_power:.8g}"
        )

    baseline = interventions[0]
    resection = interventions[1]
    base_ez = metrics[baseline]["EZ_RMS_inicio"]
    resect_ez = metrics[resection]["EZ_RMS_inicio"]
    reduction_pct = 100 * (base_ez - resect_ez) / base_ez if base_ez > 0 else float("nan")
    check("M1", "Cambio de RMS EZ: línea base vs resección",
          bool(np.isfinite(reduction_pct)),
          f"RMS base={base_ez:.6f}; RMS resección={resect_ez:.6f}; cambio={reduction_pct:.2f}% "
          "(comparación descriptiva, no prueba de eficacia clínica)")

    # Verificación informativa: fs=250 Hz implica Nyquist de 125 Hz.
    nyquist = fs / 2
    check("L1", "Detección del límite de muestreo para DBS 130 Hz", 130 >= nyquist,
          f"fs={fs} Hz; Nyquist={nyquist} Hz; correctamente detectado: 130 Hz NO es representable sin aliasing")

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Prueba", "Resultado", "Detalle"])
        writer.writeheader()
        writer.writerows(results)

    passed = sum(r["Resultado"] == "APROBADA" for r in results)
    failed = len(results) - passed
    print("\nMÉTRICAS POR INTERVENCIÓN")
    for intervention, m in metrics.items():
        print(f"- {intervention}: RMS EZ={m['EZ_RMS_inicio']:.6f}; "
              f"RMS PZ={m['PZ_RMS_inicio']:.6f}; HFO={m['Potencia_HFO_30_80']:.8g}")
    print(f"\nRESUMEN: {passed}/{len(results)} condiciones aprobadas; {failed} fallidas.")
    print(f"CSV guardado en: {CSV_FILE}")
    print("NOTA: son verificaciones computacionales sobre señales sintéticas; no validan clínicamente el modelo.")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Verifica que app_v9.py esté en la misma carpeta y que el archivo no haya sido renombrado.")
        sys.exit(2)
