import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Detect Streamlit environment
IS_STREAMLIT = False
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        IS_STREAMLIT = True
except Exception:
    pass

# =============================================================================
# 1. PARCELACIÓN DESIKAN-KILLIANY Y COORDENADAS MNI 3D
# =============================================================================
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

# Definición de Áreas Elocuentes / Críticas (Lenguaje, Motor, Sensorial, Visual)
ELOQUENT_REGIONS = [
    'L_precentral', 'R_precentral',          # Corteza Motora Primaria
    'L_postcentral', 'R_postcentral',        # Corteza Somatosensorial
    'L_parsopercularis', 'L_parstriangularis', # Área de Broca (Lenguaje)
    'L_superiortemporal', 'R_superiortemporal',# Área de Wernicke / Auditiva
    'L_pericalcarine', 'R_pericalcarine'     # Corteza Visual Primaria
]

def get_mni_coordinates():
    """Genera coordenadas 3D en espacio MNI (mm) para las 68 regiones de Desikan-Killiany."""
    np.random.seed(101)
    base_coords = {
        'bankstemporal': (-52, -35, -2),
        'caudalanteriorcingulate': (-6, 22, 24),
        'caudalmiddlefrontal': (-34, 12, 44),
        'cuneus': (-8, -80, 27),
        'entorhinal': (-25, -10, -32),
        'fusiform': (-36, -40, -20),
        'inferiorparietal': (-42, -66, 38),
        'inferiortemporal': (-52, -30, -22),
        'isthmuscingulate': (-9, -48, 16),
        'lateraloccipital': (-36, -84, 10),
        'lateralorbitofrontal': (-32, 30, -14),
        'lingual': (-14, -68, -4),
        'medialorbitofrontal': (-6, 38, -14),
        'middletemporal': (-54, -32, -10),
        'parahippocampal': (-22, -26, -18),
        'paracentral': (-8, -30, 60),
        'parsopercularis': (-46, 14, 18),
        'parsorbitalis': (-44, 30, -6),
        'parstriangularis': (-46, 26, 12),
        'pericalcarine': (-12, -78, 8),
        'postcentral': (-42, -24, 52),
        'posteriorcingulate': (-8, -38, 28),
        'precentral': (-38, -10, 52),
        'precuneus': (-10, -60, 42),
        'rostralanteriorcingulate': (-6, 38, 14),
        'rostralmiddlefrontal': (-30, 48, 18),
        'superiorfrontal': (-14, 30, 44),
        'superiorparietal': (-22, -60, 56),
        'superiortemporal': (-54, -20, 6),
        'supramarginal': (-52, -38, 32),
        'frontalpole': (-18, 62, -10),
        'temporalpole': (-38, 14, -34),
        'transversetemporal': (-42, -20, 10),
        'insula': (-35, 6, 2)
    }
    
    pos_matrix = np.zeros((len(DESIKAN_REGIONS), 3))
    for idx, reg in enumerate(DESIKAN_REGIONS):
        hemi = reg[0]
        name = reg[2:]
        x, y, z = base_coords.get(name, (-30, 0, 0))
        if hemi == 'R':
            x = -x
        pos_matrix[idx] = [x, y, z]
        
    return pos_matrix

# =============================================================================
# 2. CONECTOMA & SENSORES VIRTUALES (sEEG)
# =============================================================================
def generate_synthetic_sc(pos_matrix):
    """Genera matriz SC simétrica decayendo con la distancia Euclidiana 3D."""
    n = len(pos_matrix)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(pos_matrix[i] - pos_matrix[j])
            
    sc = np.exp(-dists / 35.0)
    np.fill_diagonal(sc, 0)
    sc = sc / np.max(sc)
    sc[sc < 0.08] = 0.0
    return sc, dists

def create_virtual_seeg_sensors(ez_pos):
    """Crea un arreglo de electrodos profundos virtuales (sEEG) dirigidos a la Zona Epileptógena."""
    target = np.mean(ez_pos, axis=0)
    entry = target + np.array([-40, 0, 15]) # Punto de entrada craneal
    num_contacts = 8
    
    contacts = []
    for t in np.linspace(0, 1, num_contacts):
        contacts.append(entry * (1 - t) + target * t)
        
    return np.array(contacts)

def compute_leadfield(sensors_pos, regions_pos):
    """Calcula la matriz Leadfield / Ganancia (Inverso del cuadrado de la distancia)."""
    num_sensors = len(sensors_pos)
    num_regions = len(regions_pos)
    gain_matrix = np.zeros((num_sensors, num_regions))
    
    for s in range(num_sensors):
        for r in range(num_regions):
            d = np.linalg.norm(sensors_pos[s] - regions_pos[r])
            gain_matrix[s, r] = 1.0 / (d**2 + 1e-3)
            
    gain_matrix /= np.max(gain_matrix, axis=1, keepdims=True)
    return gain_matrix

# =============================================================================
# 3. VISUALIZACIÓN INTERACTIVA 3D CON PLOTLY
# =============================================================================
def build_plotly_3d_connectome(pos, sc_pre, sc_post, ez_indices, seeg_contacts):
    """Construye figura Plotly 3D interactiva para la web (Streamlit)."""
    fig = go.Figure()
    
    # 1. Glass Brain (Convex Hull 3D)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pos)
    fig.add_trace(go.Mesh3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
        color='lightgray', opacity=0.08, hoverinfo='none', name='Glass Brain (Corteza)'
    ))
    
    # 2. Conectividad Conservada
    edge_x, edge_y, edge_z = [], [], []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if sc_post[i, j] > 0.15:
                edge_x.extend([pos[i, 0], pos[j, 0], None])
                edge_y.extend([pos[i, 1], pos[j, 1], None])
                edge_z.extend([pos[i, 2], pos[j, 2], None])
                
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(0, 180, 216, 0.35)', width=2),
        name='Conectividad Conservada', hoverinfo='none'
    ))
    
    # 3. Tractos Resecados
    resected_x, resected_y, resected_z = [], [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            if sc_pre[i, j] > 0.1 and sc_post[i, j] == 0:
                resected_x.extend([pos[i, 0], pos[j, 0], None])
                resected_y.extend([pos[i, 1], pos[j, 1], None])
                resected_z.extend([pos[i, 2], pos[j, 2], None])
                
    fig.add_trace(go.Scatter3d(
        x=resected_x, y=resected_y, z=resected_z,
        mode='lines',
        line=dict(color='rgba(230, 57, 70, 0.65)', width=3, dash='dash'),
        name='Tractos Resecados (EZ)', hoverinfo='none'
    ))
    
    # 4. Nodos Corticales Clasificados
    node_colors, node_sizes, categories = [], [], []
    for idx, reg in enumerate(DESIKAN_REGIONS):
        if idx in ez_indices:
            node_colors.append('#E63946') # Rojo - EZ
            node_sizes.append(10)
            categories.append('Zona Epileptógena (EZ)')
        elif reg in ELOQUENT_REGIONS:
            node_colors.append('#FFB703') # Amarillo - Área Elocuente
            node_sizes.append(9)
            categories.append('Área Elocuente (Crítica)')
        else:
            node_colors.append('#2A9D8F') # Verde/Azul - Corteza
            node_sizes.append(6)
            categories.append('Corteza Estándar')
            
    hover_text = [
        f"<b>Región:</b> {reg}<br><b>Categoría:</b> {cat}<br><b>MNI:</b> ({pos[i,0]:.1f}, {pos[i,1]:.1f}, {pos[i,2]:.1f})"
        for i, (reg, cat) in enumerate(zip(DESIKAN_REGIONS, categories))
    ]
    
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, opacity=0.9, line=dict(color='white', width=1)),
        text=hover_text, hoverinfo='text', name='Nodos Corticales (Atlas 3D)'
    ))
    
    # 5. Electrodo sEEG Virtual
    fig.add_trace(go.Scatter3d(
        x=seeg_contacts[:, 0], y=seeg_contacts[:, 1], z=seeg_contacts[:, 2],
        mode='markers+lines',
        marker=dict(size=7, color='#7209B7', symbol='diamond'),
        line=dict(color='#7209B7', width=5),
        name='Electrodo sEEG Virtual',
        text=[f"Contacto sEEG #{i+1}" for i in range(len(seeg_contacts))],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Estudio de Conectividad Neuronal 3D & Resección Virtual (TVB)</b>", x=0.5, font=dict(size=16, color='white')),
        paper_bgcolor='#0F172A', plot_bgcolor='#0F172A',
        scene=dict(
            xaxis=dict(title='X (Izq - Der)', backgroundcolor='#0F172A', gridcolor='#334155'),
            yaxis=dict(title='Y (Post - Ant)', backgroundcolor='#0F172A', gridcolor='#334155'),
            zaxis=dict(title='Z (Inf - Sup)', backgroundcolor='#0F172A', gridcolor='#334155'),
            aspectmode='data'
        ),
        legend=dict(font=dict(color='white'), bgcolor='rgba(15, 23, 42, 0.8)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# =============================================================================
# 4. RENDERIZADO MATPLOTLIB 3D PARA EXPORTACIÓN ESTÁTICA
# =============================================================================
def plot_matplotlib_3d_connectome(pos, sc_pre, sc_post, ez_indices, seeg_contacts, filename="virtual_epilepsy_resection_3d.png"):
    """Genera un renderizado 3D estático de alta calidad en Matplotlib para guardar en disco."""
    fig = plt.figure(figsize=(12, 9), facecolor='#0F172A', dpi=300)
    ax = fig.add_subplot(111, projection='3d', facecolor='#0F172A')
    
    # Renderizar Conexiones Conservadas
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if sc_post[i, j] > 0.2:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], [pos[i,2], pos[j,2]],
                        color='#00B4D8', alpha=0.15, linewidth=0.8)
                
    # Renderizar Tractos Resecados (EZ)
    for i in range(len(pos)):
        for j in range(len(pos)):
            if sc_pre[i, j] > 0.1 and sc_post[i, j] == 0:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], [pos[i,2], pos[j,2]],
                        color='#E63946', alpha=0.5, linestyle='--', linewidth=1.2)
                
    # Renderizar Nodos Corticales
    ez_pos = pos[ez_indices]
    eloquent_indices = [idx for idx, reg in enumerate(DESIKAN_REGIONS) if reg in ELOQUENT_REGIONS and idx not in ez_indices]
    standard_indices = [idx for idx in range(len(DESIKAN_REGIONS)) if idx not in ez_indices and idx not in eloquent_indices]
    
    ax.scatter(pos[standard_indices, 0], pos[standard_indices, 1], pos[standard_indices, 2],
               color='#2A9D8F', s=35, label='Corteza Estándar', alpha=0.8, edgecolors='none')
    ax.scatter(pos[eloquent_indices, 0], pos[eloquent_indices, 1], pos[eloquent_indices, 2],
               color='#FFB703', s=65, label='Área Elocuente (Crítica)', alpha=0.9, edgecolors='white', linewidths=0.5)
    ax.scatter(pos[ez_indices, 0], pos[ez_indices, 1], pos[ez_indices, 2],
               color='#E63946', s=90, label='Zona Epileptógena (EZ Resecada)', alpha=1.0, edgecolors='white', linewidths=1.0)
    
    # Renderizar Electrodo sEEG Virtual
    ax.plot(seeg_contacts[:, 0], seeg_contacts[:, 1], seeg_contacts[:, 2],
            color='#7209B7', linewidth=4, label='Electrodo sEEG Virtual')
    ax.scatter(seeg_contacts[:, 0], seeg_contacts[:, 1], seeg_contacts[:, 2],
               color='#D8B4FE', s=50, marker='D')
    
    # Formato de la Escena 3D
    ax.set_title("Mapeo Cortical 3D, Sensores Virtuales y Resección en Epilepsia", color='white', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('X (MNI mm)', color='white')
    ax.set_ylabel('Y (MNI mm)', color='white')
    ax.set_zlabel('Z (MNI mm)', color='white')
    ax.tick_params(colors='white')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
    
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='white', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close(fig)

# =============================================================================
# 5. EJECUCIÓN PRINCIPAL Y STREAMLIT
# =============================================================================
def main():
    pos = get_mni_coordinates()
    sc_pre, dists = generate_synthetic_sc(pos)
    
    default_ez = ['L_entorhinal', 'L_parahippocampal', 'L_temporalpole', 'L_middletemporal']
    ez_indices = [DESIKAN_REGIONS.index(r) for r in default_ez]
    
    sc_post = sc_pre.copy()
    for idx in ez_indices:
        sc_post[idx, :] = 0.0
        sc_post[:, idx] = 0.0
        
    seeg_contacts = create_virtual_seeg_sensors(pos[ez_indices])
    leadfield = compute_leadfield(seeg_contacts, pos)
    
    if IS_STREAMLIT:
        st.title("🧠 Mapeo Quirúrgico 3D & Sensores Virtuales (TVB)")
        st.markdown("""
        **Evaluación Clínica y Planeación Quirúrgica:**
        * **Rojo**: Zona Epileptógena (EZ) resecada virtualmente.
        * **Amarillo**: Áreas Elocuentes Críticas (Preservadas: Motor, Lenguaje, Somatosensorial).
        * **Morado**: Electrodos Profundos sEEG Virtuales.
        * **Líneas Rojas Discontinuas**: Vías y tractos desconectados.
        """)
        
        ez_selected = st.sidebar.multiselect("Seleccionar Zona Epileptógena (EZ):", DESIKAN_REGIONS, default=default_ez)
        ez_selected_idx = [DESIKAN_REGIONS.index(r) for r in ez_selected]
        
        sc_post_user = sc_pre.copy()
        for idx in ez_selected_idx:
            sc_post_user[idx, :] = 0.0
            sc_post_user[:, idx] = 0.0
            
        fig_3d = build_plotly_3d_connectome(pos, sc_pre, sc_post_user, ez_selected_idx, seeg_contacts)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.subheader("📡 Ganancia/Sensibilidad de Contactos sEEG Virtuales")
        st.dataframe({
            f"Contacto sEEG #{i+1}": leadfield[i, :] for i in range(len(seeg_contacts))
        })
    else:
        out_png = "virtual_epilepsy_resection_3d-v3.png"
        plot_matplotlib_3d_connectome(pos, sc_pre, sc_post, ez_indices, seeg_contacts, filename=out_png)

if __name__ == '__main__':
    main()
