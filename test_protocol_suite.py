"""
Protocolo inicial de pruebas in silico para el proyecto modular.
Usa datos sintéticos; no constituye validación clínica.
Ejecutar: python test_protocol_suite.py
Dependencias: numpy, scipy, pandas
"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

REGIONS = [f"{h}_{i}" for h in ("L", "R") for i in range(34)]

def connectivity():
    rng = np.random.default_rng(42)
    a = rng.lognormal(-1.5, 0.8, (68, 68))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    a /= a.max()
    a[a < .05] = 0
    return a

def coordinates():
    rng = np.random.default_rng(42)
    p = np.zeros((68, 3))
    p[:34, 0] = rng.uniform(-65, -15, 34)
    p[:34, 1] = rng.uniform(-90, 50, 34)
    p[:34, 2] = rng.uniform(-40, 65, 34)
    p[34:, 0] = rng.uniform(15, 65, 34)
    p[34:, 1] = rng.uniform(-90, 50, 34)
    p[34:, 2] = rng.uniform(-40, 65, 34)
    return p

def resect(sc, indices):
    out = sc.copy()
    for i in indices:
        out[i, :] = 0
        out[:, i] = 0
    return out

def simulate(sc, indices, resected=False, seconds=10, fs=250):
    n, steps = 68, int(seconds * fs)
    dt = 1 / fs
    x0, z = np.full(n, -2.4), np.full(n, 3.2)
    for i in indices:
        x0[i], z[i] = (-3.5, 4.0) if resected else (-1.35, 1.5)
    x, y = np.full(n, -1.2), np.full(n, -9.0)
    eeg = np.zeros((n, steps))
    rng = np.random.default_rng(123)
    for t in range(steps):
        for _ in range(4):
            coupling = .03 * (sc @ x)
            f = np.where(x < 0, x**3 - 3*x**2, (x - .6*(z-4)**2)*x)
            dx = y - f - z + 3.1 + coupling
            dy = 1 - 5*x**2 - y
            dz = .0003 * (4*(x-x0)-z)
            dts = dt / 4
            x += dts*dx + np.sqrt(dts)*.02*rng.standard_normal(n)
            y += dts*dy
            z += dts*dz
        eeg[:, t] = -x + .1*rng.standard_normal(n)
    return np.arange(steps)*dt, eeg

def leadfield(pos, contacts):
    return np.array([[1/(np.linalg.norm(pos-c, axis=1)**2 + 1e-3)
                      for c in contacts]])[0]

def i_rom(distance, direct_ratio, w1=.6, w2=.4, lam=15, conn=.8):
    risk = w1*np.exp(-distance/lam) + w2*direct_ratio*conn
    score = float(np.clip(100*(1-risk), 0, 100))
    if distance < 6 and direct_ratio > .8:
        cat = "Alerta crítica"
    elif score < 85:
        cat = "Riesgo moderado"
    else:
        cat = "Margen seguro"
    return score, cat

def main():
    results = []
    def check(module, ident, name, passed, detail):
        results.append({"Módulo":module, "ID":ident, "Prueba":name,
                        "Resultado":"APROBADA" if passed else "FALLIDA",
                        "Detalle":detail})
        print(f"[{'OK' if passed else 'FALLO'}] {ident} {name}: {detail}")

    sc = connectivity()
    # Selección de cuatro nodos sintéticos representativos
    idx = [4, 14, 31, 13]
    post = resect(sc, idx)
    check("A","A1","Nodos resecados aislados",
          bool(np.all(post[idx,:] == 0) and np.all(post[:,idx] == 0)),
          "filas y columnas seleccionadas deben ser cero")
    check("A","A2","Simetría y diagonal nula",
          bool(np.allclose(post, post.T) and np.all(np.diag(post)==0)),
          f"simetría={np.allclose(post,post.T)}")
    check("A","A3","Pérdida de conectividad no negativa",
          bool(np.all(sc-post >= -1e-12)), f"pérdida máxima={np.max(sc-post):.4f}")

    t, eeg = simulate(sc, idx)
    check("B","B1","Estabilidad numérica", bool(np.isfinite(eeg).all()),
          f"dimensiones EEG={eeg.shape}; todos los valores finitos")
    amp = float(np.ptp(eeg[idx[0]]))
    check("B","B2","Amplitud del foco", amp > 2,
          f"amplitud pico a pico={amp:.3f}; umbral programado=2.0")
    _, eeg_post = simulate(post, idx, resected=True)
    amp_post = float(np.ptp(eeg_post[:, int(len(t)*.2):]))
    check("B","B3","Amplitud después de resección", amp_post < 1.2,
          f"amplitud pico a pico={amp_post:.3f}; umbral=1.2")

    pos = coordinates()
    contacts = np.array([[-35.,-15.,-20.],[-30.,-15.,-18.],[-25.,-15.,-16.]])
    lf = leadfield(pos, contacts)
    d = np.linalg.norm(pos-contacts[0], axis=1)
    ratio = float(lf[0, np.argmin(d)] / lf[0, np.argmax(d)])
    check("C","C1","Atenuación espacial sEEG", ratio > 10,
          f"relación cerca/lejos={ratio:.2f}x; umbral=10x")
    hull = ConvexHull(pos)
    check("C","C2","Envolvente 3D", hull.volume > 10000,
          f"volumen geométrico={hull.volume:.1f} unidades cúbicas")

    v,c = i_rom(4,1)
    check("D","D1","Escenario crítico", c=="Alerta crítica" and v<30,
          f"I_ROM={v:.1f}%; categoría={c}")
    v,c = i_rom(12,0)
    check("D","D2","Escenario moderado", 30<=v<=80 and c=="Riesgo moderado",
          f"I_ROM={v:.1f}%; categoría={c}")
    v,c = i_rom(40,0)
    check("D","D3","Escenario distante", c=="Margen seguro" and v>=85,
          f"I_ROM={v:.1f}%; categoría={c}")

    df = pd.DataFrame(results)
    df.to_csv("resultados_pruebas.csv", index=False, encoding="utf-8-sig")
    passed = int((df["Resultado"]=="APROBADA").sum())
    total = len(df)
    print(f"\nRESUMEN: {passed}/{total} aprobadas ({100*passed/total:.1f}%).")
    print("Se creó resultados_pruebas.csv en la misma carpeta.")
    print("Aviso: son pruebas con datos sintéticos, no validación clínica.")
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
