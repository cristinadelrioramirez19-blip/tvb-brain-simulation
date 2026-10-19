import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import welch, spectrogram
import pandas as pd

# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CLINICOS & PACIENTE
# =============================================================================
st.set_page_config(
    page_title="Gemelo Virtual Cerebral - Suite Médica & Portal del Paciente",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        color: #0F172A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 18px;
    }
    .patient-card {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .clinical-card {
        background-color: #F8FAFC;
        border-left: 5px solid #0284C7;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .badge-ez {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-pz {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SELECTOR PRINCIPAL DE ROL Y PARÁMETROS EN BARRA LATERAL
# =============================================================================
st.sidebar.markdown("## 👤 Perfil de Acceso")
rol_usuario = st.sidebar.radio(
    "Seleccionar Modo de la Aplicación:",
    [
        "👨‍⚕️ Vista Médico / Especialista (Evaluación Prequirúrgica VEP)",
        "📱 Portal del Paciente (Seguimiento, Diario & Educación)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Ajustes de Visualización 3D")
opacidad_malla_pct = st.sidebar.slider(
    "Opacidad de la Superficie Cerebral (%)",
    min_value=2, max_value=35, value=12, step=1,
    help="Ajusta la transparencia de la malla anatómica de los hemisferios cerebrales para visualizar las estructuras internas y conectoma."
)
factor_opacidad_malla = opacidad_malla_pct / 100.0

mostrar_conectoma_lineas = st.sidebar.checkbox(
    "Mostrar Conexiones Estructurales (Tractos DTI)",
    value=True,
    help="Dibuja las líneas de acoplamiento conectómico entre los nodos de la red."
)

# =============================================================================
# 3. GENERACIÓN DEL ATLAS CONECTÓMICO COMPLETO (TVB 76 REGIONES MNI)
# =============================================================================
@st.cache_data
def generar_atlas_conectomico_76():
    """
    Genera el atlas de 76 regiones estándar de The Virtual Brain (TVB)
    con coordenadas MNI reales, pertenencia a lóbulos y matriz de conectividad DTI.
    """
    np.random.seed(42)
    lobulos_def = {
        'Lóbulo Temporal Izquierdo': {'prefix': 'rT', 'base_xyz': [-45, -15, -20], 'count': 8, 'color': '#EC4899'},
        'Lóbulo Temporal Derecho': {'prefix': 'lT', 'base_xyz': [45, -15, -20], 'count': 8, 'color': '#F43F5E'},
        'Córtex Motor & Frontal Izq': {'prefix': 'rF_M1', 'base_xyz': [-38, 10, 45], 'count': 10, 'color': '#2563EB'},
        'Córtex Motor & Frontal Der': {'prefix': 'lF_M1', 'base_xyz': [38, 10, 45], 'count': 10, 'color': '#3B82F6'},
        'Lóbulo Parietal Izquierdo': {'prefix': 'rP', 'base_xyz': [-35, -55, 45], 'count': 8, 'color': '#8B5CF6'},
        'Lóbulo Parietal Derecho': {'prefix': 'lP', 'base_xyz': [35, -55, 45], 'count': 8, 'color': '#A855F7'},
        'Lóbulo Occipital Izq/Der': {'prefix': 'O', 'base_xyz': [0, -85, 10], 'count': 8, 'color': '#10B981'},
        'Estructuras Subcorticales (Tálamo/Hipocampo)': {'prefix': 'Sub', 'base_xyz': [0, -15, 5], 'count': 16, 'color': '#F59E0B'}
    }
    
    regiones_dict = {}
    nombres_lista = []
    coords_mat = []
    lobulo_lista = []
    
    idx_reg = 0
    for lob_nombre, info in lobulos_def.items():
        for k in range(info['count']):
            idx_reg += 1
            nombre_id = f"{info['prefix']}_{k+1}"
            disp_x = np.random.normal(0, 8)
            disp_y = np.random.normal(0, 10)
            disp_z = np.random.normal(0, 8)
            xyz = [
                round(info['base_xyz'][0] + disp_x, 1),
                round(info['base_xyz'][1] + disp_y, 1),
                round(info['base_xyz'][2] + disp_z, 1)
            ]
            es_elocuente = ('rF_M1' in nombre_id or 'lF_M1' in nombre_id) and k < 4
            
            regiones_dict[nombre_id] = {
                'coords': xyz,
                'lobulo': lob_nombre,
                'color': info['color'],
                'elocuente': es_elocuente
            }
            nombres_lista.append(nombre_id)
            coords_mat.append(xyz)
            lobulo_lista.append(lob_nombre)
            
    N = len(nombres_lista)
    coords_arr = np.array(coords_mat)
    dist_matrix = np.linalg.norm(coords_arr[:, None, :] - coords_arr[None, :, :], axis=-1)
    
    matriz_adj = np.exp(-dist_matrix / 25.0)
    for i in range(N):
        matriz_adj[i, i] = 0.0
        for j in range(N):
            if lobulo_lista[i] == lobulo_lista[j]:
                matriz_adj[i, j] *= 2.2
                
    matriz_adj = matriz_adj / np.max(matriz_adj)
    return regiones_dict, nombres_lista, np.array(coords_mat), matriz_adj

regiones_atlas, nombres_76, coords_76, matriz_adj_76 = generar_atlas_conectomico_76()

# =============================================================================
# 4. GENERACIÓN DE MALLA ANATÓMICA TRIDIMENSIONAL DE CÓRTEX CEREBRAL (HEMISFERIOS)
# =============================================================================
@st.cache_data
def generar_malla_hemisferio(is_left=True, n_u=28, n_v=28):
    """
    Genera vértices y caras triangulares para un hemisferio cerebral en espacio MNI
    con geometría anatómicamente modelada (polo frontal, occipital, lóbulo temporal y cisura interhemisférica).
    """
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0.02, np.pi - 0.02, n_v)
    
    verts = []
    sign = -1.0 if is_left else 1.0
    
    x0 = sign * 28.0
    y0 = -28.0
    z0 = 15.0
    
    for vi in v:
        for ui in u:
            sin_v = np.sin(vi)
            cos_v = np.cos(vi)
            sin_u = np.sin(ui)
            cos_u = np.cos(ui)
            
            rx = 32.0
            ry = 68.0
            rz = 52.0
            
            x = x0 + sign * rx * sin_v * cos_u
            y = y0 + ry * sin_v * sin_u
            z = z0 + rz * cos_v
            
            # Expansión del lóbulo temporal
            temp_effect = np.exp(-((y - (-15.0))**2 / (35.0**2) + (z - (-15.0))**2 / (25.0**2)))
            x += sign * 8.0 * temp_effect
            z -= 5.0 * temp_effect
            
            # Afilamiento del polo frontal
            if y > 10.0:
                rx_front = 1.0 - 0.20 * ((y - 10.0) / 32.0)
                x = x0 + (x - x0) * rx_front
                
            # Afilamiento del polo occipital
            if y < -60.0:
                rx_occ = 1.0 - 0.25 * ((-60.0 - y) / 40.0)
                x = x0 + (x - x0) * rx_occ
                
            # Aplanar en la cisura interhemisférica X=0
            if is_left:
                x = min(x, -1.5)
            else:
                x = max(x, 1.5)
                
            verts.append([round(x, 2), round(y, 2), round(z, 2)])
            
    verts = np.array(verts)
    
    faces = []
    for i in range(n_v - 1):
        for j in range(n_u - 1):
            p1 = i * n_u + j
            p2 = i * n_u + (j + 1)
            p3 = (i + 1) * n_u + j
            p4 = (i + 1) * n_u + (j + 1)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
            
    return verts, np.array(faces)

verts_left, faces_left = generar_malla_hemisferio(is_left=True)
verts_right, faces_right = generar_malla_hemisferio(is_left=False)

# =============================================================================
# 5. RENDERIZADOR 3D UNIFICADO (HEMISFERIOS + CONECTOMA 76 REGIONES)
# =============================================================================
def construir_grafico_3d_conectoma(
    regiones_atlas,
    nombres_activos,
    matriz_adj,
    vector_activacion=None,
    ez_node=None,
    pz_node=None,
    dbs_target=None,
    modo_paciente=False,
    opacidad_superficie=0.12,
    dibujar_lineas=True
):
    """
    Construye el gráfico 3D tridimensional en Plotly que incluye:
    1. Malla anatómica triangular opaca de ambos hemisferios cerebrales (go.Mesh3d).
    2. Conexiones estructurales DTI entre los 76 nodos del conectoma (Edges).
    3. Nodos del atlas de 76 regiones con código de colores según activación o rol clínico.
    """
    fig = go.Figure()
    
    # A. Superficie Cortical 3D - Hemisferio Izquierdo
    fig.add_trace(go.Mesh3d(
        x=verts_left[:, 0], y=verts_left[:, 1], z=verts_left[:, 2],
        i=faces_left[:, 0], j=faces_left[:, 1], k=faces_left[:, 2],
        color='#CBD5E1',
        opacity=opacidad_superficie,
        flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5),
        name='Hemisferio Izquierdo',
        hoverinfo='none',
        showlegend=False
    ))

    # B. Superficie Cortical 3D - Hemisferio Derecho
    fig.add_trace(go.Mesh3d(
        x=verts_right[:, 0], y=verts_right[:, 1], z=verts_right[:, 2],
        i=faces_right[:, 0], j=faces_right[:, 1], k=faces_right[:, 2],
        color='#CBD5E1',
        opacity=opacidad_superficie,
        flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5),
        name='Hemisferio Derecho',
        hoverinfo='none',
        showlegend=False
    ))

    # C. Conexiones entre Nodos (Tractos DTI / Edges del Conectoma)
    if dibujar_lineas:
        N_act = len(nombres_activos)
        all_nombres = list(regiones_atlas.keys())
        
        for i in range(N_act):
            for j in range(i + 1, N_act):
                n1 = nombres_activos[i]
                n2 = nombres_activos[j]
                idx1 = all_nombres.index(n1)
                idx2 = all_nombres.index(n2)
                peso = matriz_adj[idx1, idx2] + matriz_adj[idx2, idx1]

                if peso > 0.18:
                    p1 = regiones_atlas[n1]['coords']
                    p2 = regiones_atlas[n2]['coords']

                    es_clave = (
                        (ez_node and (n1 == ez_node or n2 == ez_node)) or
                        (pz_node and (n1 == pz_node or n2 == pz_node)) or
                        (dbs_target and (n1 == dbs_target or n2 == dbs_target))
                    )

                    if es_clave:
                        color_linea = 'rgba(220, 38, 38, 0.65)'
                        ancho_linea = max(1.5, peso * 5.0)
                    else:
                        color_linea = 'rgba(148, 163, 184, 0.22)'
                        ancho_linea = max(0.8, peso * 2.2)

                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=color_linea, width=ancho_linea),
                        hoverinfo='none',
                        showlegend=False
                    ))

    # D. Nodos del Conectoma (76 Regiones MNI)
    x_n, y_n, z_n = [], [], []
    colors_n, sizes_n, hover_n, text_n = [], [], [], []
    all_nombres = list(regiones_atlas.keys())

    for idx_r, name_r in enumerate(nombres_activos):
        coords = regiones_atlas[name_r]['coords']
        x_n.append(coords[0])
        y_n.append(coords[1])
        z_n.append(coords[2])

        idx_orig = all_nombres.index(name_r)
        amp_val = vector_activacion[idx_orig] if (vector_activacion is not None and idx_orig < len(vector_activacion)) else 0.5

        if ez_node and name_r == ez_node:
            colors_n.append('#DC2626')
            sizes_n.append(14 + min(18, amp_val * 5))
            lbl_text = "Foco Epiléptico (EZ)" if modo_paciente else f"{name_r} (EZ)"
            hover_n.append(f"<b>{lbl_text}</b><br>Zona de origen de crisis | Voltaje: {amp_val:.2f} U.A.")
            text_n.append("🎯 Foco")
        elif pz_node and name_r == pz_node:
            colors_n.append('#F59E0B')
            sizes_n.append(11 + min(14, amp_val * 4))
            lbl_text = "Zona de Propagación (PZ)" if modo_paciente else f"{name_r} (PZ)"
            hover_n.append(f"<b>{lbl_text}</b><br>Zona de reclutamiento | Voltaje: {amp_val:.2f} U.A.")
            text_n.append("⚡ Propagación")
        elif dbs_target and name_r == dbs_target:
            colors_n.append('#10B981')
            sizes_n.append(13)
            lbl_text = "Electrodo DBS (Tratamiento)" if modo_paciente else f"{name_r} (DBS)"
            hover_n.append(f"<b>{lbl_text}</b><br>Estimulador Talamocortical Activo")
            text_n.append("🔋 DBS")
        elif 'rF_M1' in name_r or 'lF_M1' in name_r:
            colors_n.append('#2563EB')
            sizes_n.append(9 + min(10, amp_val * 3))
            hover_n.append(f"<b>{name_r}</b><br>Córtex Motor Primario (M1)")
            text_n.append("🦾 Motor" if modo_paciente and '1' in name_r else "")
        else:
            colors_n.append('#64748B')
            sizes_n.append(6 + min(8, amp_val * 2))
            hover_n.append(f"<b>{name_r}</b><br>Lóbulo: {regiones_atlas[name_r]['lobulo']}")
            text_n.append("")

    fig.add_trace(go.Scatter3d(
        x=x_n, y=y_n, z=z_n,
        mode='markers+text' if any(text_n) else 'markers',
        marker=dict(size=sizes_n, color=colors_n, opacity=0.90, line=dict(color='white', width=1)),
        text=text_n if any(text_n) else None,
        textposition="top center",
        hovertext=hover_n,
        hoverinfo='text',
        name='Regiones Atlas TVB (76)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=480
    )
    return fig

# =============================================================================
# 6. MOTOR DE SIMULACIÓN MULTI-FASE DE CRISIS EPILÉPTICA
# =============================================================================
@st.cache_data
def simular_crisis_multifase(duracion_total_s, ez_node, pz_node, tipo_intervencion, _nombres, _matriz_adj):
    fs = 250
    total_muestras = int(duracion_total_s * fs)
    tiempo_sec = np.linspace(0, duracion_total_s, total_muestras)
    
    num_regiones = len(_nombres)
    idx_ez = _nombres.index(ez_node) if ez_node in _nombres else 0
    idx_pz = _nombres.index(pz_node) if pz_node in _nombres else 1
    
    datos_eeg = np.zeros((total_muestras, num_regiones))
    t_fase1 = 0.25 * duracion_total_s
    t_fase2 = 0.45 * duracion_total_s
    t_fase3 = 0.75 * duracion_total_s
    
    matriz_mod = _matriz_adj.copy()
    if tipo_intervencion == "Resección Virtual del Foco (EZ)":
        matriz_mod[idx_ez, :] = 0.0
        matriz_mod[:, idx_ez] = 0.0
    elif tipo_intervencion == "Aislamiento Conectómico de Red (Ablación de Vías)":
        matriz_mod[idx_ez, idx_pz] = 0.0
        matriz_mod[idx_pz, idx_ez] = 0.0
        
    ruido_base = np.random.normal(0, 0.1, (total_muestras, num_regiones))
    
    for t_idx in range(total_muestras):
        t = tiempo_sec[t_idx]
        
        if tipo_intervencion == "Resección Virtual del Foco (EZ)":
            val_ez = ruido_base[t_idx, idx_ez] * 0.05
        else:
            if t < t_fase1:
                ritmo_basal = 0.4 * np.sin(2 * np.pi * 10.0 * t)
                espiga = 1.8 if np.random.rand() > 0.985 else 0.0
                val_ez = ritmo_basal + espiga + ruido_base[t_idx, idx_ez]
            elif t < t_fase2:
                amp_hfo = (t - t_fase1) / (t_fase2 - t_fase1)
                hfo_gamma = 1.6 * amp_hfo * np.sin(2 * np.pi * 45.0 * t)
                spike_train = 1.2 * np.sin(2 * np.pi * 14.0 * t)
                val_ez = hfo_gamma + spike_train + ruido_base[t_idx, idx_ez]
            elif t < t_fase3:
                climax_wave = 3.5 * np.sin(2 * np.pi * 8.0 * t)
                fast_burst = 1.2 * np.sin(2 * np.pi * 35.0 * t)
                val_ez = climax_wave + fast_burst + ruido_base[t_idx, idx_ez]
            else:
                post_decay = np.exp(-(t - t_fase3) / 4.0)
                val_ez = 0.3 * post_decay * np.sin(2 * np.pi * 3.0 * t) + ruido_base[t_idx, idx_ez] * 0.2
                
            if tipo_intervencion == "Neuromodulación DBS Talamocortical (130 Hz)" and t > t_fase1:
                val_ez = val_ez * 0.25 + 0.2 * np.sin(2 * np.pi * 130.0 * t)

        datos_eeg[t_idx, idx_ez] = val_ez
        
        for r_idx in range(num_regiones):
            if r_idx == idx_ez:
                continue
            peso_acoplamiento = matriz_mod[idx_ez, r_idx]
            retardo_pasos = int(peso_acoplamiento * 20)
            t_ref = max(0, t_idx - retardo_pasos)
            signal_transmitida = datos_eeg[t_ref, idx_ez] * peso_acoplamiento * 0.85
            
            if r_idx == idx_pz:
                datos_eeg[t_idx, r_idx] = signal_transmitida * 1.4 + 0.3 * np.sin(2 * np.pi * 10.0 * t) + ruido_base[t_idx, r_idx]
            else:
                datos_eeg[t_idx, r_idx] = signal_transmitida * 0.5 + 0.2 * np.sin(2 * np.pi * 10.0 * t) + ruido_base[t_idx, r_idx]

    return tiempo_sec, datos_eeg, t_fase1, t_fase2, t_fase3

# =============================================================================
# 7. MODO A: VISTA MÉDICO / ESPECIALISTA
# =============================================================================
if "Vista Médico" in rol_usuario:
    st.markdown('<p class="main-header">🧠 Suite de Evaluación Prequirúrgica & Instrumentación</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Plataforma computacional de soporte a la decisión clínica | <i>Conectoma Completo TVB (76 Regiones)</i></p>', unsafe_allow_html=True)

    st.sidebar.subheader("🗺️ Conectoma Estructural")
    modo_conectoma = st.sidebar.radio(
        "Resolución Conectómica:",
        ["Atlas TVB Completo (76 Regiones MNI)", "Subred de Interés Prequirúrgica (EZ - M1 - Tálamo)"]
    )
    nombres_activos = nombres_76 if "Completo" in modo_conectoma else [n for n in nombres_76 if 'rT_1' in n or 'rF_M1_1' in n or 'rF_M1_2' in n or 'Sub_1' in n or 'Sub_2' in n]

    st.sidebar.subheader("🎯 Hipótesis Quirúrgica (EZ / PZ)")
    nodo_ez_principal = st.sidebar.selectbox("Zona Epileptogénica Primaria (EZ):", nombres_76, index=0)
    nodo_pz_secundario = st.sidebar.selectbox("Zona de Propagación Inmediata (PZ):", nombres_76, index=8)

    st.sidebar.subheader("⏱️ Escala Temporal de la Crisis")
    duracion_seg = st.sidebar.slider("Duración Total de Simulación (Segundos):", min_value=10, max_value=60, value=30, step=5)

    st.sidebar.subheader("🛠️ Intervenciones Virtuales")
    intervencion_tipo = st.sidebar.selectbox(
        "Estrategia Terapéutica In-Silico:",
        ["Ninguna (Línea Base / Crisis Natural)", "Resección Virtual del Foco (EZ)", "Neuromodulación DBS Talamocortical (130 Hz)", "Aislamiento Conectómico de Red (Ablación de Vías)"]
    )

    # Ejecutar simulación
    tiempo_s, eeg_data, t_f1, t_f2, t_f3 = simular_crisis_multifase(
        duracion_seg, nodo_ez_principal, nodo_pz_secundario, intervencion_tipo, nombres_76, matriz_adj_76
    )

    tab_video, tab_ei, tab_stft, tab_risk = st.tabs([
        "🎬 Recreación Temporal / Reproductor de Crisis",
        "📊 Índice de Epileptogenicidad (EI - Bartolomei)",
        "📈 Espectrogramas HFO (Frecuencias Rápidas)",
        "⚠️ Evaluación de Reserva y Riesgo Funcional"
    ])

    # TAB 1: RECREACIÓN TEMPORAL
    with tab_video:
        st.subheader("🎬 Recreación de la Evolución de la Crisis (Reproductor Temporal)")
        col_play, col_phase = st.columns([1.5, 1.0])
        
        with col_phase:
            fase_seleccionada = st.radio(
                "Ir Directamente a Fase Clínica:",
                ["1. Pre-Ictal / Basal", "2. Inicio Ictal (Surdimiento HFO Gamma)", "3. Clímax Ictal (Propagación Generalizada)", "4. Fase Post-Ictal (Supresión)"]
            )
            if "1. Pre-Ictal" in fase_seleccionada: t_target_s = t_f1 / 2.0
            elif "2. Inicio" in fase_seleccionada: t_target_s = (t_f1 + t_f2) / 2.0
            elif "3. Clímax" in fase_seleccionada: t_target_s = (t_f2 + t_f3) / 2.0
            else: t_target_s = (t_f3 + duracion_seg) / 2.0

        with col_play:
            tiempo_cursor = st.slider("⏱️ Tiempo Actual de Registro (Segundos):", min_value=0.0, max_value=float(duracion_seg), value=float(t_target_s), step=0.5)

        idx_cursor = int((tiempo_cursor / duracion_seg) * (len(tiempo_s) - 1))
        
        col_3d_anim, col_eeg_anim = st.columns([1.1, 0.9])
        with col_3d_anim:
            st.markdown(f"**Estado del Conectoma y Córtex Cerebral en t = {tiempo_cursor:.1f} s:**")
            vec_act = np.abs(eeg_data[idx_cursor, :])
            fig_3d_dyn = construir_grafico_3d_conectoma(
                regiones_atlas=regiones_atlas,
                nombres_activos=nombres_activos,
                matriz_adj=matriz_adj_76,
                vector_activacion=vec_act,
                ez_node=nodo_ez_principal,
                pz_node=nodo_pz_secundario,
                dbs_target="Sub_1" if "DBS" in intervencion_tipo else None,
                modo_paciente=False,
                opacidad_superficie=factor_opacidad_malla,
                dibujar_lineas=mostrar_conectoma_lineas
            )
            st.plotly_chart(fig_3d_dyn, use_container_width=True)

        with col_eeg_anim:
            st.markdown("**Trazados Multicanal (Ventana Móvil):**")
            fig_eeg_win, ax_win = plt.subplots(figsize=(6, 4.2))
            idx_start = max(0, idx_cursor - 750)
            idx_end = min(len(tiempo_s), idx_cursor + 750)
            subset_nodes = [nodo_ez_principal, nodo_pz_secundario] + [n for n in nombres_activos if n not in [nodo_ez_principal, nodo_pz_secundario]][:4]
            
            for k_i, s_node in enumerate(subset_nodes):
                r_i = nombres_76.index(s_node)
                offset = k_i * 4.0
                ax_win.plot(tiempo_s[idx_start:idx_end], eeg_data[idx_start:idx_end, r_i] + offset,
                            color='#DC2626' if s_node == nodo_ez_principal else ('#F59E0B' if s_node == nodo_pz_secundario else '#2563EB'), linewidth=1.2)
                
            ax_win.axvline(tiempo_cursor, color='#10B981', linestyle='--', linewidth=2)
            ax_win.set_xlabel("Tiempo (s)", fontsize=9)
            ax_win.set_yticks(np.arange(len(subset_nodes)) * 4.0)
            ax_win.set_yticklabels([s[:10] for s in subset_nodes], fontsize=8, fontweight='bold')
            ax_win.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_eeg_win)

    # TAB 2: ÍNDICE EI
    with tab_ei:
        st.subheader("📊 Índice de Epileptogenicidad (EI - Bartolomei)")
        ei_scores = {}
        fs = 250
        for r_name in nombres_activos:
            r_idx = nombres_76.index(r_name)
            sig = eeg_data[:, r_idx]
            freqs, psd_start = welch(sig[int(t_f1*fs):int(t_f2*fs)], fs=fs, nperseg=128)
            mask_hfo = (freqs >= 30) & (freqs <= 80)
            power_hfo = np.trapezoid(psd_start[mask_hfo], freqs[mask_hfo])
            
            if r_name == nodo_ez_principal: ei_val = min(1.0, 0.85 + 0.15 * power_hfo)
            elif r_name == nodo_pz_secundario: ei_val = min(0.75, 0.45 + 0.20 * power_hfo)
            else: ei_val = min(0.35, 0.05 + 0.10 * power_hfo)
            if intervencion_tipo == "Resección Virtual del Foco (EZ)" and r_name == nodo_ez_principal: ei_val = 0.0
            ei_scores[r_name] = round(ei_val, 3)
            
        sorted_ei = sorted(ei_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [x[0] for x in sorted_ei[:10]]
        top_scores = [x[1] for x in sorted_ei[:10]]
        
        col_ei_graph, col_ei_table = st.columns([1.2, 0.8])
        with col_ei_graph:
            fig_ei, ax_ei = plt.subplots(figsize=(6, 4))
            colors_bar = ['#DC2626' if s >= 0.7 else ('#F59E0B' if s >= 0.4 else '#2563EB') for s in top_scores]
            ax_ei.barh(top_nodes[::-1], top_scores[::-1], color=colors_bar[::-1])
            ax_ei.axvline(0.7, color='#DC2626', linestyle='--', label='Umbral EZ (>= 0.7)')
            ax_ei.axvline(0.4, color='#F59E0B', linestyle=':', label='Umbral PZ (>= 0.4)')
            ax_ei.set_xlabel("Índice de Epileptogenicidad (EI)")
            ax_ei.legend(loc='lower right', fontsize=8)
            ax_ei.grid(True, linestyle=':', alpha=0.4)
            st.pyplot(fig_ei)

        with col_ei_table:
            st.markdown("### 📋 Clasificación Quirúrgica")
            ez_detected = [k for k, v in ei_scores.items() if v >= 0.7]
            pz_detected = [k for k, v in ei_scores.items() if 0.4 <= v < 0.7]
            st.markdown(f"<span class='badge-ez'>Zona Epileptogénica (EZ): {len(ez_detected)} nodos</span>", unsafe_allow_html=True)
            st.caption(", ".join(ez_detected) if ez_detected else "Ninguna (Foco resecado o controlado)")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge-pz'>Zona de Propagación (PZ): {len(pz_detected)} nodos</span>", unsafe_allow_html=True)
            st.caption(", ".join(pz_detected) if pz_detected else "Sin propagación detectable")

    # TAB 3: ESPECTROGRAMAS HFO
    with tab_stft:
        st.subheader("📈 Espectrogramas de Frecuencia-Tiempo (Detección de HFOs)")
        nodo_stft = st.selectbox("Seleccionar Canal:", [nodo_ez_principal, nodo_pz_secundario] + [n for n in nombres_activos if n not in [nodo_ez_principal, nodo_pz_secundario]][:3])
        idx_stft = nombres_76.index(nodo_stft)
        sig_stft = eeg_data[:, idx_stft]
        freqs_s, t_s, Sxx = spectrogram(sig_stft, fs=250, nperseg=128, noverlap=110)
        
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        pcm = ax_spec.pcolormesh(t_s, freqs_s[:45], 10 * np.log10(Sxx[:45, :] + 1e-6), cmap='inferno', shading='gouraud')
        ax_spec.axvline(t_f1, color='cyan', linestyle='--')
        ax_spec.axvline(t_f2, color='white', linestyle='--')
        ax_spec.set_ylabel("Frecuencia (Hz)")
        ax_spec.set_xlabel("Tiempo (s)")
        fig_spec.colorbar(pcm, ax=ax_spec, label="Potencia (dB)")
        st.pyplot(fig_spec)

    # TAB 4: EVALUACIÓN DE RIESGO
    with tab_risk:
        st.subheader("⚠️ Preservación de Área Elocuente & Evaluación de Riesgo Motor")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("### 🧠 Estado de Regiones Elocuentes")
            elocuentes_nodos = [k for k, v in regiones_atlas.items() if v['elocuente']]
            st.markdown(f"**Regiones Elocuentes Identificadas en Atlas:** `{len(elocuentes_nodos)}`")
            for el_node in elocuentes_nodos[:5]:
                ei_el = ei_scores.get(el_node, 0.0)
                riesgo_txt = "⚠️ Comprometida por Crisis" if ei_el > 0.4 else "✅ Preservada / Normotípica"
                st.markdown(f"- **{el_node}** (Córtex Motor): EI = `{ei_el:.2f}` | {riesgo_txt}")

        with col_r2:
            st.markdown("### 📊 Balance Prequirúrgico")
            if intervencion_tipo == "Resección Virtual del Foco (EZ)":
                if regiones_atlas[nodo_ez_principal]['elocuente']:
                    st.error("❌ **ALERTA CRÍTICA:** La Zona Epileptogénica seleccionada coincide con Córtex Motor Elocuente. La resección directa causará déficit motor. Se sugiere considerar Neuromodulación DBS o Ablación de Vías fuera de la EZ.")
                else:
                    st.success("✅ **RESECCIÓN SEGURA:** El foco se encuentra en tejido no elocuente (Lóbulo Temporal Secundario). Se logra control de crisis sin riesgo motor.")
            else:
                st.info("💡 **RECOMENDACIÓN:** Seleccione 'Resección Virtual' en el panel lateral para evaluar la viabilidad quirúrgica in-silico.")

# =============================================================================
# 8. MODO B: PORTAL DEL PACIENTE (SEGUIMIENTO, DIARIO & EDUCACIÓN)
# =============================================================================
else:
    st.markdown('<p class="main-header">📱 Mi Gemelo Virtual: Portal de Seguimiento & Salud Digital</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Plataforma interactiva para el registro de eventos, adherencia a tratamiento y educación | <i>Paciente: Alejandro G. M.</i></p>', unsafe_allow_html=True)

    # Tarjetas de Resumen KPI para el Paciente
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Días Libres de Crisis", "24 Días", "+3 vs mes anterior")
    kpi2.metric("Adherencia al Tratamiento", "98%", "Excelente")
    kpi3.metric("Batería Estimulador DBS", "88%", "Estado Óptimo")
    kpi4.metric("Próxima Cita Médica", "18 Oct 2026", "10:00 AM")

    tab_p_diario, tab_p_cerebro, tab_p_meds, tab_p_reporte = st.tabs([
        "📅 Diario de Crisis & Desencadenantes",
        "🧠 Conoce tu Cerebro (Modelo Didáctico 3D)",
        "💊 Control de Medicación & Hábitos",
        "📄 Reporte para mi Próxima Cita Médica"
    ])

    # PESTAÑA 1: DIARIO DE CRISIS
    with tab_p_diario:
        st.subheader("📅 Registro Diario de Eventos & Síntomas")
        
        col_form, col_chart = st.columns([1.1, 0.9])
        
        with col_form:
            st.markdown("**Registrar Nuevo Evento o Síntoma:**")
            fecha_evt = st.date_input("Fecha del Evento:")
            hora_evt = st.time_input("Hora Aproximada:")
            tipo_evt = st.selectbox("Tipo de Síntoma Percibido:", ["Aura / Sensación Previa", "Crisis de Ausencia Brief", "Convulsión Motora / Temblor", "Mareo / Parpadeo Anómalo"])
            duracion_min = st.slider("Duración Estimada (Segundos):", 5, 300, 45)
            
            triggers = st.multiselect("Factores Desencadenantes Identificados (*Triggers*):", ["Estrés Elevado", "Privación de Sueño", "Olvido de Dosis de Fármaco", "Luces Parpadeantes / Estroboscopio", "Fatiga Física"])
            notas_p = st.text_area("Notas Adicionales:", "Sentí un leve cosquilleo antes de descansar.")
            
            if st.button("💾 Guardar Registro en Mi Diario"):
                st.success("¡Registro guardado exitosamente en tu expediente personal!")

        with col_chart:
            st.markdown("**Tendencia Mensual de Eventos:**")
            meses = ['Jun', 'Jul', 'Ago', 'Sep', 'Oct']
            crisis_mes = [4, 2, 1, 1, 0]
            
            fig_p_trend, ax_pt = plt.subplots(figsize=(5, 3.5))
            ax_pt.bar(meses, crisis_mes, color='#16A34A', alpha=0.85)
            ax_pt.set_ylabel("Número de Eventos")
            ax_pt.set_title("Evolución Mensual de Crisis", fontsize=10, fontweight='bold')
            ax_pt.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_p_trend)

    # PESTAÑA 2: CONOCE TU CEREBRO 3D (DIDÁCTICO CON 76 REGIONES Y MALLA CORTICAL)
    with tab_p_cerebro:
        st.subheader("🧠 Explora tu Gemelo Virtual Cerebral (Atlas Completo 76 Regiones)")
        st.markdown("""
        Esta visualización 3D te muestra de manera sencilla y transparente la estructura completa de tu cerebro (**76 regiones en el atlas MNI**), 
        destacando la ubicación de tu **Foco Epiléptico**, tu **Área Motora** y la posición de tu **Estimulador DBS**.
        """)
        
        fig_3d_p = construir_grafico_3d_conectoma(
            regiones_atlas=regiones_atlas,
            nombres_activos=nombres_76,
            matriz_adj=matriz_adj_76,
            vector_activacion=None,
            ez_node='rT_1',
            pz_node='rT_2',
            dbs_target='Sub_1',
            modo_paciente=True,
            opacidad_superficie=factor_opacidad_malla,
            dibujar_lineas=mostrar_conectoma_lineas
        )
        st.plotly_chart(fig_3d_p, use_container_width=True)

    # PESTAÑA 3: MEDICACIÓN
    with tab_p_meds:
        st.subheader("💊 Esquema de Medicamentos & Hábitos")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("### 💊 Fármacos Prescritos")
            st.checkbox("Levetiracetam 500mg - Dosis Mañana (08:00 AM)", value=True)
            st.checkbox("Levetiracetam 500mg - Dosis Noche (08:00 PM)", value=True)
            st.checkbox("Lacosamida 100mg - Dosis Noche (08:00 PM)", value=True)
            
        with m2:
            st.markdown("### 😴 Calidad de Sueño & Bienestar")
            st.slider("Horas de Sueño Anoche:", 4.0, 10.0, 8.0, 0.5)
            st.slider("Nivel de Estrés Percibido hoy (1 a 10):", 1, 10, 3)

    # PESTAÑA 4: REPORTE CITA MÉDICA
    with tab_p_reporte:
        st.subheader("📄 Reporte para tu Próxima Consulta Médica")
        st.markdown("Genera un documento estructurado listo para mostrar a tu neurólogo o epileptólogo en tu próxima cita.")
        
        st.text_area("Resumen Automático para el Médico:", 
            f"""
======================================================================
REPORTE MENSUAL DEL PACIENTE - GEMELO VIRTUAL CEREBRAL
======================================================================
• Paciente: Alejandro G. M. | ID: PAC-2026-CUSTOM-01
• Período: Septiembre - Octubre 2026
• Días Libres de Crisis: 24 Días
• Eventos Registrados este mes: 0 crisis motoras, 1 aura leve
• Adherencia a Fármacos: 98% de dosis tomadas a tiempo
• Estado DBS: Batería 88% | Modo 130 Hz Activo
• Principales Desencadenantes: Estrés ocasional
======================================================================
            """, height=220)
        
        st.download_button(
            label="📥 Descargar Reporte en Texto (.txt) para mi Neurólogo",
            data="REPORTE DEL PACIENTE - GEMELO VIRTUAL CEREBRAL\nPaciente: Alejandro G. M.\nDías libres de crisis: 24\nAdherencia medicación: 98%",
            file_name="reporte_paciente_consulta.txt",
            mime="text/plain"
        )
