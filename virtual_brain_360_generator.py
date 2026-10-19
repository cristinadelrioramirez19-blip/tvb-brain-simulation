import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# PALETA DE COLORES VIBRANTES (Basada en la imagen de referencia)
# =============================================================================
COLOR_DEEP_INDIGO = '#382A85'  # Indigo Profundo
COLOR_PERIWINKLE  = '#6272F0'  # Azul Periwinkle / Red Normal
COLOR_GOLD        = '#FFC107'  # Amarillo Oro / Áreas Elocuentes
COLOR_ORANGE      = '#FF8C00'  # Naranja Cálido / Electrodo
COLOR_VIVID_RED   = '#FF3D00'  # Rojo Neón / Foco Epiléptico
COLOR_BG_DARK     = '#0B0F19'  # Fondo Oscuro Quirúrgico

# =============================================================================
# COORDENADAS Y MATRIZ DE CONECTIVIDAD (ATLAS DESIKAN-KILLIANY)
# =============================================================================
DESIKAN_REGIONS = [
    'L_bankstemporal', 'L_caudalanteriorcingulate', 'L_caudalmiddlefrontal', 'L_cuneus', 'L_entorhinal',
    'L_fusiform', 'L_inferiorparietal', 'L_inferiortemporal', 'L_isthmuscingulate', 'L_lateraloccipital',
    'L_lateralorbitofrontal', 'L_lingual', 'L_medialorbitofrontal', 'L_middletemporal', 'L_parahippocampal',
    'L_paracentral', 'L_parsopercularis', 'L_parsorbitalis', 'L_parstriangularis', 'L_pericalcarine',
    'L_postcentral', 'L_posteriorcingulate', 'L_precentral', 'L_precuneus', 'L_rostralanteriorcingulate',
    'L_rostralmiddlefrontal', 'L_superiorfrontal', 'L_superiorparietal', 'L_superiortemporal', 'L_supramarginal',
    'L_frontalpole', 'L_temporalpole', 'L_transversetemporal', 'L_insula',
    'R_bankstemporal', 'R_caudalanteriorcingulate', 'R_caudalmiddlefrontal', 'R_cuneus', 'R_entorhinal',
    'R_fusiform', 'R_inferiorparietal', 'R_inferiortemporal', 'R_isthmuscingulate', 'R_lateraloccipital',
    'R_lateralorbitofrontal', 'R_lingual', 'R_medialorbitofrontal', 'R_middletemporal', 'R_parahippocampal',
    'R_paracentral', 'R_parsopercularis', 'R_parsorbitalis', 'R_parstriangularis', 'R_pericalcarine',
    'R_postcentral', 'R_posteriorcingulate', 'R_precentral', 'R_precuneus', 'R_rostralanteriorcingulate',
    'R_rostralmiddlefrontal', 'R_superiorfrontal', 'R_superiorparietal', 'R_superiortemporal', 'R_supramarginal',
    'R_frontalpole', 'R_temporalpole', 'R_transversetemporal', 'R_insula'
]

ELOQUENT_REGIONS = [
    'L_precentral', 'R_precentral', 'L_postcentral', 'R_postcentral',
    'L_parsopercularis', 'L_parstriangularis', 'L_superiortemporal', 'R_superiortemporal'
]

def get_mni_coordinates():
    np.random.seed(101)
    base_coords = {
        'bankstemporal': (-52, -35, -2), 'caudalanteriorcingulate': (-6, 22, 24),
        'caudalmiddlefrontal': (-34, 12, 44), 'cuneus': (-8, -80, 27),
        'entorhinal': (-25, -10, -32), 'fusiform': (-36, -40, -20),
        'inferiorparietal': (-42, -66, 38), 'inferiortemporal': (-52, -30, -22),
        'isthmuscingulate': (-9, -48, 16), 'lateraloccipital': (-36, -84, 10),
        'lateralorbitofrontal': (-32, 30, -14), 'lingual': (-14, -68, -4),
        'medialorbitofrontal': (-6, 38, -14), 'middletemporal': (-54, -32, -10),
        'parahippocampal': (-22, -26, -18), 'paracentral': (-8, -30, 60),
        'parsopercularis': (-46, 14, 18), 'parsorbitalis': (-44, 30, -6),
        'parstriangularis': (-46, 26, 12), 'pericalcarine': (-12, -78, 8),
        'postcentral': (-42, -24, 52), 'posteriorcingulate': (-8, -38, 28),
        'precentral': (-38, -10, 52), 'precuneus': (-10, -60, 42),
        'rostralanteriorcingulate': (-6, 38, 14), 'rostralmiddlefrontal': (-30, 48, 18),
        'superiorfrontal': (-14, 30, 44), 'superiorparietal': (-22, -60, 56),
        'superiortemporal': (-54, -20, 6), 'supramarginal': (-52, -38, 32),
        'frontalpole': (-18, 62, -10), 'temporalpole': (-38, 14, -34),
        'transversetemporal': (-42, -20, 10), 'insula': (-35, 6, 2)
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

def generate_synthetic_sc(pos_matrix):
    n = len(pos_matrix)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(pos_matrix[i] - pos_matrix[j])
    sc = np.exp(-dists / 35.0)
    np.fill_diagonal(sc, 0)
    sc = sc / np.max(sc)
    sc[sc < 0.08] = 0.0
    return sc

def create_virtual_seeg_sensors(ez_pos):
    target = np.mean(ez_pos, axis=0)
    entry = target + np.array([-40, 5, 20])
    num_contacts = 8
    contacts = []
    for t in np.linspace(0, 1, num_contacts):
        contacts.append(entry * (1 - t) + target * t)
    return np.array(contacts)

# =============================================================================
# GENERADOR DE ANIMACIÓN 360° GIF / MP4
# =============================================================================
def generate_360_pitch_gif(pos, sc, ez_indices, seeg_contacts, output_filename="virtual_brain_360.gif"):
    fig = plt.figure(figsize=(9, 8), facecolor=COLOR_BG_DARK, dpi=120)
    ax = fig.add_subplot(111, projection='3d', facecolor=COLOR_BG_DARK)

    num_nodes = len(pos)
    
    # 1. Tractos de Fibra del Conectoma
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sc[i, j] > 0.17:
                is_ez = (i in ez_indices) or (j in ez_indices)
                color = COLOR_VIVID_RED if is_ez else COLOR_PERIWINKLE
                alpha = 0.55 if is_ez else 0.15
                lw = 1.5 if is_ez else 0.7
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], [pos[i,2], pos[j,2]],
                        color=color, alpha=alpha, linewidth=lw)

    # 2. Electrodo sEEG Profundo con Destellos
    ax.plot(seeg_contacts[:, 0], seeg_contacts[:, 1], seeg_contacts[:, 2],
            color=COLOR_ORANGE, linewidth=4.5, label='Electrodo sEEG Profundo')
    ax.scatter(seeg_contacts[:, 0], seeg_contacts[:, 1], seeg_contacts[:, 2],
               color=COLOR_GOLD, s=60, marker='D', edgecolors=COLOR_ORANGE, depthshade=False)

    # 3. Clasificación de Nodos y Efecto Resplandor en Zona Epileptógena (EZ)
    eloquent_idx = [idx for idx, reg in enumerate(DESIKAN_REGIONS) if reg in ELOQUENT_REGIONS and idx not in ez_indices]
    standard_idx = [idx for idx in range(num_nodes) if idx not in ez_indices and idx not in eloquent_idx]

    # Corteza Estándar
    ax.scatter(pos[standard_idx, 0], pos[standard_idx, 1], pos[standard_idx, 2],
               color=COLOR_PERIWINKLE, s=35, alpha=0.85, label='Corteza Estándar', depthshade=True)
    
    # Áreas Elocuentes Críticas
    ax.scatter(pos[eloquent_idx, 0], pos[eloquent_idx, 1], pos[eloquent_idx, 2],
               color=COLOR_GOLD, s=65, alpha=0.95, label='Áreas Elocuentes Críticas', depthshade=False)

    # Zona Epileptógena (EZ - Foco) con Halo Neón de doble capa
    ax.scatter(pos[ez_indices, 0], pos[ez_indices, 1], pos[ez_indices, 2],
               color=COLOR_VIVID_RED, s=200, alpha=0.22, depthshade=False)  # Resplandor exterior
    ax.scatter(pos[ez_indices, 0], pos[ez_indices, 1], pos[ez_indices, 2],
               color=COLOR_VIVID_RED, s=80, alpha=1.0, label='Zona Epileptógena (EZ)', depthshade=False)

    # Formato Visual
    ax.set_title("GEMELO VIRTUAL DE CEREBRO — PITCH 3D", color='white', fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')
    ax.legend(facecolor=COLOR_BG_DARK, edgecolor='#334155', labelcolor='white', loc='upper left', fontsize=9)

    def update_rotation(angle):
        ax.view_init(elev=18, azim=angle)
        return ax,

    frames = np.linspace(0, 360, 36, endpoint=False)
    anim = FuncAnimation(fig, update_rotation, frames=frames, interval=80)
    anim.save(output_filename, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"GIF de rotación 360° generado con éxito: {output_filename}")

if __name__ == '__main__':
    pos = get_mni_coordinates()
    sc = generate_synthetic_sc(pos)
    ez_indices = [DESIKAN_REGIONS.index(r) for r in ['L_entorhinal', 'L_parahippocampal', 'L_temporalpole']]
    seeg_contacts = create_virtual_seeg_sensors(pos[ez_indices])
    generate_360_pitch_gif(pos, sc, ez_indices, seeg_contacts)
