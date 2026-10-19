"""
===============================================================================
 Virtual Neurosurgery & Epileptogenic Zone Resection in Connectomes
 Using The Virtual Brain (TVB) & Streamlit Integration
 Version 2.0 (Compatible con Streamlit y ejecución directa en terminal)
===============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Detectar de forma precisa si el script se está ejecutando mediante 'streamlit run'
IS_STREAMLIT = False
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        IS_STREAMLIT = True
except Exception:
    pass

# Si NO estamos en Streamlit, configurar backend no interactivo seguro
if not IS_STREAMLIT:
    matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# 1. Parcelación Cortical (Desikan-Killiany - 68 regiones)
# -----------------------------------------------------------------------------
DESIKAN_REGIONS = [
    'L_bankstemporal', 'L_caudalanteriorcingulate', 'L_caudalmiddlefrontal',
    'L_cuneus', 'L_entorhinal', 'L_fusiform', 'L_inferiorparietal',
    'L_inferiortemporal', 'L_isthmuscingulate', 'L_lateraloccipital',
    'L_lateralorbitofrontal', 'L_lingual', 'L_medialorbitofrontal',
    'L_middletemporal', 'L_parahippocampal', 'L_paracentral',
    'L_parsopercularis', 'L_parsorbitalis', 'L_parstriangularis',
    'L_pericalcarine', 'L_postcentral', 'L_posteriorcingulate',
    'L_precentral', 'L_precuneus', 'L_rostralanteriorcingulate',
    'L_rostralmiddlefrontal', 'L_superiorfrontal', 'L_superiorparietal',
    'L_superiortemporal', 'L_supramarginal', 'L_frontalpole',
    'L_temporalpole', 'L_transversetemporal', 'L_insula',
    'R_bankstemporal', 'R_caudalanteriorcingulate', 'R_caudalmiddlefrontal',
    'R_cuneus', 'R_entorhinal', 'R_fusiform', 'R_inferiorparietal',
    'R_inferiortemporal', 'R_isthmuscingulate', 'R_lateraloccipital',
    'R_lateralorbitofrontal', 'R_lingual', 'R_medialorbitofrontal',
    'R_middletemporal', 'R_parahippocampal', 'R_paracentral',
    'R_parsopercularis', 'R_parsorbitalis', 'R_parstriangularis',
    'R_pericalcarine', 'R_postcentral', 'R_posteriorcingulate',
    'R_precentral', 'R_precuneus', 'R_rostralanteriorcingulate',
    'R_rostralmiddlefrontal', 'R_superiorfrontal', 'R_superiorparietal',
    'R_superiortemporal', 'R_supramarginal', 'R_frontalpole',
    'R_temporalpole', 'R_transversetemporal', 'R_insula'
]

def load_or_generate_sc():
    """Genera una matriz de conectividad estructural (SC) sintética y simétrica de 68x68."""
    np.random.seed(42)
    n = len(DESIKAN_REGIONS)
    sc = np.random.lognormal(mean=-1.5, sigma=0.8, size=(n, n))
    sc = (sc + sc.T) / 2.0
    np.fill_diagonal(sc, 0)
    sc = sc / np.max(sc)
    sc[sc < 0.05] = 0.0
    return sc

# -----------------------------------------------------------------------------
# 2. Resección Virtual (Cirugía Virtual)
# -----------------------------------------------------------------------------
def perform_virtual_resection(sc_matrix, ez_region_names):
    """
    Pone a cero las conexiones asociadas a los nodos de la Zona Epileptógena (EZ).
    """
    sc_resected = sc_matrix.copy()
    ez_indices = [DESIKAN_REGIONS.index(r) for r in ez_region_names if r in DESIKAN_REGIONS]
    
    for idx in ez_indices:
        sc_resected[idx, :] = 0.0
        sc_resected[:, idx] = 0.0
        
    return sc_resected, ez_indices

# -----------------------------------------------------------------------------
# 3. Integración con TVB Simulator
# -----------------------------------------------------------------------------
def run_tvb_simulation(sc_weights, use_epileptor=True, sim_length=2000):
    """
    Ejecuta el simulador TVB si tvb-library está instalada, o retorna una simulación estimada.
    """
    try:
        from tvb.simulator.lab import connectivity, models, coupling, integrators, monitors, simulator
        
        conn = connectivity.Connectivity(
            weights=sc_weights,
            region_labels=np.array(DESIKAN_REGIONS)
        )
        conn.configure()
        
        model = models.Epileptor() if use_epileptor else models.ReducedWongWang()
        
        sim = simulator.Simulator(
            connectivity=conn,
            model=model,
            coupling=coupling.Difference(a=np.array([1.0])),
            integrator=integrators.HeunStochastic(dt=0.05),
            monitors=[monitors.TemporalAverage(period=1.0)]
        )
        sim.configure()
        (time_series, data), = sim.run(simulation_length=sim_length)
        return time_series, data
        
    except ImportError:
        # Fallback estimativo de FC cuando tvb.simulator no está instalado localmente
        n = len(DESIKAN_REGIONS)
        time_steps = int(sim_length)
        raw_signals = np.random.randn(time_steps, n) @ sc_weights
        fc = np.corrcoef(raw_signals.T)
        return np.arange(time_steps), fc

# -----------------------------------------------------------------------------
# 4. Generación de Gráficos de Conectomas
# -----------------------------------------------------------------------------
def create_connectome_figure(sc_pre, sc_post, ez_indices):
    """Crea y retorna un objeto Figure de Matplotlib."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    
    # 1. SC Pre-Op
    im0 = axes[0].imshow(sc_pre, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('SC Pre-Operatoria', fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. SC Post-Resección
    im1 = axes[1].imshow(sc_post, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('SC Post-Resección Virtual', fontsize=11, fontweight='bold')
    for idx in ez_indices:
        axes[1].axhline(idx, color='red', linestyle='--', alpha=0.5, linewidth=0.7)
        axes[1].axvline(idx, color='red', linestyle='--', alpha=0.5, linewidth=0.7)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Diferencia
    diff = sc_pre - sc_post
    im2 = axes[2].imshow(diff, cmap='Reds', aspect='auto', vmin=0, vmax=max(np.max(diff), 0.01))
    axes[2].set_title('Tractos/Conexiones Resecadas', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 5. Punto de Entrada Principal
# -----------------------------------------------------------------------------
def main():
    sc_pre = load_or_generate_sc()
    default_ez = ['L_entorhinal', 'L_parahippocampal', 'L_temporalpole', 'L_middletemporal']

    if IS_STREAMLIT:
        # Modo Streamlit Web App
        st.title("🧠 Cirugía Virtual y Resección en Epilepsia con TVB")
        st.markdown("Modelo de conectoma pre y post resección virtual basado en la plataforma *The Virtual Brain*.")

        st.sidebar.header("Parámetros de Resección")
        ez_target = st.sidebar.multiselect(
            "Selecciona la Zona Epileptógena (EZ):",
            options=DESIKAN_REGIONS,
            default=default_ez
        )

        sc_post, ez_idx = perform_virtual_resection(sc_pre, ez_target)

        st.write(f"**Regiones resecadas en la Zona Epileptógena:** {ez_target}")
        st.write(f"**Índices de nodos resecados:** {ez_idx}")

        # Renderizar en la interfaz web usando st.pyplot()
        fig = create_connectome_figure(sc_pre, sc_post, ez_idx)
        st.pyplot(fig)
        plt.close(fig)

    else:
        # Modo Terminal / Script de Python
        sc_post, ez_idx = perform_virtual_resection(sc_pre, default_ez)
        print(f"Resección aplicada a la Zona Epileptógena: {default_ez}")
        print(f"Nodos afectados: {ez_idx}")
        
        fig = create_connectome_figure(sc_pre, sc_post, ez_idx)
        output_filename = "resultado_reseccion_v2.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Gráfica generada exitosamente: {output_filename}")

if __name__ == '__main__':
    main()
