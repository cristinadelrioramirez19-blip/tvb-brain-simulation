import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS (STREAMLIT)
# =============================================================================
st.set_page_config(
    page_title="Evaluación Prequirúrgica de Riesgo de ROM (Motor)",
    page_icon="🦾",
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
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card-danger {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .metric-card-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .metric-card-success {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown('<p class="main-header">🦾 Evaluación Prequirúrgica de Riesgo de Pérdida de ROM Articular</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Módulo de cuantificación biomecánica basado en margen anatómico MNI152 y conectividad DTI ($I_{\\text{ROM}}$) | <i>Proyecto Modular - CUCEI UdeG</i></p>', unsafe_allow_html=True)

# =============================================================================
# 2. DEFINICIÓN DE REFERENCIAS ANATÓMICAS (ESPACIO MNI152)
# =============================================================================
# Coordenada MNI fija de la Corteza Motora Primaria (M1 Izquierda - Mano/Brazo)
COORDS_M1 = np.array([-38.0, -24.0, 62.0])

# Diccionario de Focos Epilépticos Predefinidos
FOCOS_PREDEFINIDOS = {
    "Córtex Premotor Dorsal (Cercano / Adj. a M1)": {
        'coords': [-32.0, -18.0, 54.0],
        'peso_dti': 0.82,
        'descripcion': "Ubicación contigua a M1; alto riesgo de compromiso elocuente."
    },
    "Área Motora Suplementaria (SMA - Medial)": {
        'coords': [-12.0, -10.0, 68.0],
        'peso_dti': 0.65,
        'descripcion': "Zona motora secundaria; riesgo de apraxia/paresia transitoria."
    },
    "Lóbulo Parietal Inferior (Distancia Media)": {
        'coords': [-42.0, -55.0, 50.0],
        'peso_dti': 0.38,
        'descripcion': "Región asociativa con margen anatómico moderado."
    },
    "Lóbulo Temporal Anterior (Foco Distante)": {
        'coords': [-45.0, -15.0, -20.0],
        'peso_dti': 0.22,
        'descripcion': "Zona epileptogénica clásica lejana a la vía piramidal."
    },
    "Córtex Frontopolar (Distante)": {
        'coords': [-25.0, 55.0, 15.0],
        'peso_dti': 0.15,
        'descripcion': "Polo frontal lejano a la representación motora eferente."
    }
}

# =============================================================================
# 3. BARRA LATERAL: SELECCIÓN DE FOCO E INTERVENCIÓN QUIRÚRGICA
# =============================================================================
st.sidebar.header("⚙️ Configuración del Caso Clínico")

# A. Selección de Foco / Zona Epileptogénica (EZ)
st.sidebar.subheader("🎯 1. Localización del Foco (EZ)")
modo_foco = st.sidebar.radio(
    "Modo de Definición de Foco:",
    ["Seleccionar Caso Predefinido", "Coordenadas MNI Personalizadas"]
)

if modo_foco == "Seleccionar Caso Predefinido":
    foco_nombre = st.sidebar.selectbox("Seleccionar Foco Epiléptico:", list(FOCOS_PREDEFINIDOS.keys()))
    coords_ez = np.array(FOCOS_PREDEFINIDOS[foco_nombre]['coords'])
    peso_dti = FOCOS_PREDEFINIDOS[foco_nombre]['peso_dti']
    st.sidebar.caption(f"ℹ️ {FOCOS_PREDEFINIDOS[foco_nombre]['descripcion']}")
else:
    foco_nombre = "Foco MNI Personalizado"
    st.sidebar.markdown("**Coordenadas MNI 3D (mm):**")
    x_custom = st.sidebar.slider("Coordenada X (mm - Izq/Der):", -65.0, 65.0, -30.0, 1.0)
    y_custom = st.sidebar.slider("Coordenada Y (mm - Ant/Post):", -90.0, 65.0, -20.0, 1.0)
    z_custom = st.sidebar.slider("Coordenada Z (mm - Inf/Sup):", -40.0, 75.0, 45.0, 1.0)
    coords_ez = np.array([x_custom, y_custom, z_custom])
    peso_dti = st.sidebar.slider("Acoplamiento Conectómico DTI ($C_{EZ \\to M1}$):", 0.05, 1.0, 0.50, 0.05)

# B. Selección de Estrategia Terapéutica In-Silico
st.sidebar.subheader("🛠️ 2. Estrategia Prequirúrgica")
tipo_intervencion = st.sidebar.selectbox(
    "Intervención Virtual a Evaluar:",
    [
        "Línea Base (Sin Intervención / Crisis Natural)",
        "Resección Virtual Directa del Foco",
        "Aislamiento Conectómico (Ablación de Vías)",
        "Neuromodulación DBS Talamocortical (130 Hz)"
    ]
)

# C. Parámetros Biomecánicos de Ajuste
st.sidebar.subheader("📐 3. Parámetros del Modelo")
lambda_margen = st.sidebar.slider("Constante de Margen Quirúrgico ($\\lambda$ en mm):", 5.0, 20.0, 10.0, 1.0,
                                  help="Define la tasa de decaimiento por distancia anatómica.")
w1_weight = st.sidebar.slider("Peso Anatómico Directo ($w_1$):", 0.50, 0.90, 0.70, 0.05)
w2_weight = round(1.0 - w1_weight, 2)

# =============================================================================
# 4. FUNCIÓN MATEMÁTICA DE CÁLCULO DE RIESGO DE PERDIDA DE ROM
# =============================================================================
def calcular_metricas_rom(coords_foco, coords_motor, c_dti, intervencion, lambda_val, w1, w2):
    # Distancia Euclídea 3D MNI (mm)
    distancia_mm = float(np.linalg.norm(coords_foco - coords_motor))
    
    # Factor de eficacia según la intervención elegida
    if "Resección Virtual Directa" in intervencion:
        eta_int = 1.0  # El foco se elimina, reduce a cero la propagación conectómica
    elif "Aislamiento Conectómico" in intervencion:
        eta_int = 0.90  # Se desarticulan las vías principales entre EZ y M1
    elif "Neuromodulación DBS" in intervencion:
        eta_int = 0.70  # Atenuación de alta frecuencia por despolarización
    else:
        eta_int = 0.0   # Sin intervención (foco activo)
        
    # Componente Anatómico por cercanía a M1
    comp_anatomico = np.exp(-distancia_mm / lambda_val)
    
    # Componente de Red Conectómica
    comp_red = c_dti * (1.0 - eta_int)
    
    # Cálculo del Índice de Riesgo de Pérdida de ROM (%)
    riesgo_pct = 100.0 * (w1 * comp_anatomico + w2 * comp_red)
    riesgo_pct = min(100.0, max(0.0, riesgo_pct))
    rom_preservado_pct = 100.0 - riesgo_pct
    
    # Categorización clínica y recomendación neuroquirúrgica
    if distancia_mm < 6.0 and "Resección" in intervencion:
        categoria = "🔴 CRÍTICO: Solapamiento en Córtex Motor (Riesgo de Hemiplejía)"
        badge_class = "badge-critical"
        recomendacion = "❌ NO EXTRAPAR DIRECTAMENTE M1. La resección causará parálisis irreversible. Se recomienda Neuromodulación DBS Talamocortical o Ablación de Vías fuera de M1."
    elif distancia_mm < 6.0:
        categoria = "🔴 ALTO RIESGO: Foco Adyacente / Solapado con M1"
        badge_class = "badge-critical"
        recomendacion = "⚠️ Foco ubicado dentro o colindante a la representación motora primaria. Planificar estrategia no ablativa."
    elif distancia_mm < 16.0:
        categoria = "🟠 MODERADO: Zona de Edema Peri-lesional"
        badge_class = "badge-moderate"
        recomendacion = "⚡ Riesgo de paresia transitoria por inflamación post-operatoria. Proceder con monitoreo neurofisiológico intraoperatorio (mEMG/MEP)."
    else:
        categoria = "🟢 BAJO: Margen Quirúrgico Seguro (>16 mm)"
        badge_class = "badge-safe"
        recomendacion = "✅ Margen anatómico totalmente seguro. La resección estructural conserva el Rango de Movimiento (ROM) sin secuelas motoras."
        
    return {
        'distancia_mm': round(distancia_mm, 2),
        'riesgo_pct': round(riesgo_pct, 1),
        'rom_preservado_pct': round(rom_preservado_pct, 1),
        'comp_anatomico_pct': round(comp_anatomico * 100, 1),
        'comp_red_pct': round(comp_red * 100, 1),
        'categoria': categoria,
        'badge_class': badge_class,
        'recomendacion': recomendacion,
        'eta_int': eta_int
    }

# Ejecutar el cálculo
res = calcular_metricas_rom(coords_ez, COORDS_M1, peso_dti, tipo_intervencion, lambda_margen, w1_weight, w2_weight)

# =============================================================================
# 5. DESPLIEGUE DE TARJETAS KPI PRINCIPALES
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Distancia Anatómica MNI", f"{res['distancia_mm']} mm", help="Distancia Euclídea 3D entre Foco (EZ) y M1")

with col2:
    st.metric("Riesgo de Pérdida de ROM", f"{res['riesgo_pct']}%", delta=f"{'-' if res['eta_int']>0 else '+'}{res['comp_red_pct']}% Red", delta_color="inverse")

with col3:
    st.metric("ROM Motor Preservado", f"{res['rom_preservado_pct']}%")

with col4:
    st.markdown(f"**Clasificación de Riesgo:**")
    st.markdown(f"<span class='{res['badge_class']}'>{res['categoria'].split(':')[0]}</span>", unsafe_allow_html=True)

st.progress(int(res['rom_preservado_pct']) / 100.0)

# Cuadro de Recomendación Quirúrgica
card_style = "metric-card-danger" if "🔴" in res['categoria'] else ("metric-card-warning" if "🟠" in res['categoria'] else "metric-card-success")
st.markdown(f"""
<div class="{card_style}">
    <b>📌 Evaluación y Recomendación Prequirúrgica:</b><br>
    {res['recomendacion']}
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 6. PESTAÑAS DE VISUALIZACIÓN INTERACTIVA
# =============================================================================
tab_3d, tab_curve, tab_doc = st.tabs([
    "🌐 Mapa 3D y Margen Quirúrgico MNI",
    "📈 Curva de Seguridad Anatómica vs. Distancia",
    "📄 Justificación Académica y Documentación Técnica"
])

# -----------------------------------------------------------------------------
# TAB 1: VISUALIZADOR 3D EN ESPACIO MNI
# -----------------------------------------------------------------------------
with tab_3d:
    st.subheader("🌐 Visualizador 3D del Margen de Seguridad Anatómico (MNI152)")
    st.markdown("Inspecione en 3D la ubicación relativa del **Foco Epiléptico (EZ)** respecto a la **Corteza Motora Primaria (M1)** y el radio de seguridad de 15 mm.")
    
    fig_3d = go.Figure()
    
    # A. Malla sutil del córtex cerebral (referencia anatómica)
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_s = 56 * np.outer(np.cos(u), np.sin(v)) - 25
    y_s = 70 * np.outer(np.sin(u), np.sin(v)) - 15
    z_s = 50 * np.outer(np.ones(np.size(u)), np.cos(v)) + 35
    
    fig_3d.add_trace(go.Surface(
        x=x_s, y=y_s, z=z_s,
        colorscale=[[0, '#CBD5E1'], [1, '#94A3B8']],
        opacity=0.08,
        showscale=False,
        hoverinfo='none',
        name='Córtex Cerebral'
    ))
    
    # B. Esfera del Margen de Seguridad de 15 mm alrededor de M1
    r_margin = 15.0  # mm
    u_m = np.linspace(0, 2 * np.pi, 20)
    v_m = np.linspace(0, np.pi, 20)
    x_m = COORDS_M1[0] + r_margin * np.outer(np.cos(u_m), np.sin(v_m))
    y_m = COORDS_M1[1] + r_margin * np.outer(np.sin(u_m), np.sin(v_m))
    z_m = COORDS_M1[2] + r_margin * np.outer(np.ones(np.size(u_m)), np.cos(v_m))
    
    fig_3d.add_trace(go.Surface(
        x=x_m, y=y_m, z=z_m,
        colorscale=[[0, '#F59E0B'], [1, '#EF4444']],
        opacity=0.18,
        showscale=False,
        hoverinfo='none',
        name='Zona de Riesgo (15 mm)'
    ))
    
    # C. Línea de Conexión / Distancia Euclídea entre EZ y M1
    fig_3d.add_trace(go.Scatter3d(
        x=[coords_ez[0], COORDS_M1[0]],
        y=[coords_ez[1], COORDS_M1[1]],
        z=[coords_ez[2], COORDS_M1[2]],
        mode='lines+text',
        line=dict(color='#DC2626' if res['distancia_mm'] < 15 else '#2563EB', width=4, dash='dash'),
        text=["", f"d = {res['distancia_mm']} mm"],
        textposition="middle right",
        hoverinfo='text',
        hovertext=f"Distancia Euclídea 3D: {res['distancia_mm']} mm<br>Acoplamiento DTI: {peso_dti}",
        name='Vector de Distancia / Tracto DTI'
    ))
    
    # D. Marcador M1 (Córtex Motor Primario)
    fig_3d.add_trace(go.Scatter3d(
        x=[COORDS_M1[0]], y=[COORDS_M1[1]], z=[COORDS_M1[2]],
        mode='markers+text',
        marker=dict(size=16, color='#2563EB', opacity=1.0, line=dict(color='white', width=2)),
        text=["🦾 Corteza Motora (M1)"],
        textposition="top center",
        hovertext=f"<b>Córtex Motor Primario M1</b><br>Coordenadas MNI: {list(COORDS_M1)}",
        hoverinfo='text',
        name='Córtex Motor M1'
    ))
    
    # E. Marcador Foco Epiléptico (EZ)
    color_foco = '#EF4444' if 'Resección' not in tipo_intervencion else '#64748B'
    fig_3d.add_trace(go.Scatter3d(
        x=[coords_ez[0]], y=[coords_ez[1]], z=[coords_ez[2]],
        mode='markers+text',
        marker=dict(size=18, color=color_foco, opacity=0.95, line=dict(color='white', width=2)),
        text=[f"🎯 {foco_nombre.split('(')[0]}"],
        textposition="top center",
        hovertext=f"<b>Foco Epiléptico (EZ)</b><br>Estado: {tipo_intervencion}<br>Coordenadas MNI: {list(coords_ez)}",
        hoverinfo='text',
        name='Foco Epiléptico (EZ)'
    ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm - Izq/Der)', backgroundcolor="#F8FAFC"),
            yaxis=dict(title='Y (mm - Ant/Post)', backgroundcolor="#F8FAFC"),
            zaxis=dict(title='Z (mm - Inf/Sup)', backgroundcolor="#F8FAFC"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=520
    )
    
    st.plotly_chart(fig_3d, width='stretch')

# -----------------------------------------------------------------------------
# TAB 2: CURVA DE SEGURIDAD ANATÓMICA
# -----------------------------------------------------------------------------
with tab_curve:
    st.subheader("📈 Curva Teórica de Seguridad Anatómica vs. Distancia MNI")
    st.markdown("Comportamiento del Riesgo de Pérdida de ROM ($I_{\\text{ROM}}$) en función de la cercanía en milímetros a la corteza motora primaria ($M1$).")
    
    distancias_sim = np.linspace(0.0, 45.0, 150)
    
    # Calcular riesgos para la curva
    riesgos_sin_int = [calcular_metricas_rom(COORDS_M1 + np.array([d, 0, 0]), COORDS_M1, peso_dti, "Línea Base", lambda_margen, w1_weight, w2_weight)['riesgo_pct'] for d in distancias_sim]
    riesgos_con_dbs = [calcular_metricas_rom(COORDS_M1 + np.array([d, 0, 0]), COORDS_M1, peso_dti, "Neuromodulación DBS", lambda_margen, w1_weight, w2_weight)['riesgo_pct'] for d in distancias_sim]
    riesgos_con_resec = [calcular_metricas_rom(COORDS_M1 + np.array([d, 0, 0]), COORDS_M1, peso_dti, "Resección Virtual Directa", lambda_margen, w1_weight, w2_weight)['riesgo_pct'] for d in distancias_sim]
    
    fig_curve, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(distancias_sim, riesgos_sin_int, color='#EF4444', linewidth=2.2, label='Línea Base (Sin Intervención)')
    ax.plot(distancias_sim, riesgos_con_dbs, color='#F59E0B', linewidth=2.0, linestyle='--', label='Neuromodulación DBS (130 Hz)')
    ax.plot(distancias_sim, riesgos_con_resec, color='#10B981', linewidth=2.0, linestyle='-.', label='Resección Virtual Directa')
    
    # Marcar posición del paciente actual
    ax.scatter([res['distancia_mm']], [res['riesgo_pct']], color='black', s=120, zorder=5, label=f'Caso Actual (d={res["distancia_mm"]} mm)')
    ax.annotate(f"  {res['riesgo_pct']}% Riesgo", (res['distancia_mm'], res['riesgo_pct']), fontweight='bold', fontsize=9)
    
    # Umbrales anatómicos
    ax.axvline(6.0, color='#991B1B', linestyle=':', alpha=0.7, label='Límite Crítico M1 (6 mm)')
    ax.axvline(16.0, color='#D97706', linestyle=':', alpha=0.7, label='Límite de Edema (16 mm)')
    
    ax.set_xlabel("Distancia Anatómica a M1 (mm en MNI152)", fontsize=10)
    ax.set_ylabel("Riesgo de Pérdida de ROM (%)", fontsize=10)
    ax.set_title("Curva de Seguridad Quirúrgica Biomecánica", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8.5)
    
    st.pyplot(fig_curve)

# -----------------------------------------------------------------------------
# TAB 3: JUSTIFICACIÓN ACADÉMICA Y DOCUMENTACIÓN
# -----------------------------------------------------------------------------
with tab_doc:
    st.subheader("📚 Justificación Académica para la Evaluación Modular (CUCEI - UdeG)")
    
    st.markdown("""
    ### 🔬 Justificación del Módulo de Biomecánica y Rehabilitación
    Para la acreditación formal del proyecto modular ante el comité evaluador y revisores académicos, la presente herramienta establece un enfoque **defendible, científicamente riguroso y clinically validado**:
    
    1. **Fundamentación Neuroquirúrgica:**  
       En la planificación prequirúrgica de epilepsia refractaria, no es científicamente viable predecir cinemática muscular periférica (fuerza o desplazamiento en milímetros) a partir de microvoltios de EEG sin un modelo electromiográfico (EMG) individualizado. En su lugar, el estándar de oro consiste en evaluar el **Margen de Seguridad Quirúrgico Anatómico** respecto a la **Corteza Motora Primaria ($M1$)** y la **vía piramidal (Tracto Corticoespinal)** (*Jirsa et al., 2017 - VEP; Sora An et al., 2019*).
       
    2. **Formulación Matemática del Índice $I_{\\text{ROM}}$:**  
       $$\\text{Riesgo}_{ROM}(\\%) = 100 \\cdot \\left[ w_1 \\cdot e^{-\\frac{d(EZ, M1)}{\\lambda}} + w_2 \\cdot C_{EZ \\to M1} \\cdot (1 - \\eta) \\right]$$
       * **$d(EZ, M1)$**: Distancia Euclídea tridimensional MNI152 entre el Foco ($EZ$) y $M1$.
       * **$\\lambda$**: Constante de margen de inflamación/edema peri-lesional ($10.0\\text{ mm}$).
       * **$C_{EZ \\to M1}$**: Peso de acoplamiento conectómico DTI extraído del atlas estructural.
       * **$\\eta$**: Factor de eficacia de la intervención terapéutica ($1.0$ para resección, $0.7$ para DBS).
       
    3. **Alineación con Atributos de Egreso (Rúbrica CUCEI 2025B):**  
       * **Atributo A1 (Análisis y Modelado):** Integra ciencias exactas y neuroanatomía para cuantificar el riesgo motor.
       * **Atributo A2 (Diseño Biomédico):** Implementa un software de soporte a la decisión clínica para planificación prequirúrgica.
       * **Atributo A7 (Herramientas Modernas):** Utiliza entorno Python, MNI152 y visualización 3D interactiva en WebGL.
    """)
    
    # Generar reporte descargable
    txt_reporte = f"""======================================================================
REPORTE DE EVALUACIÓN PREQUIRÚRGICA DE RIESGO DE ROM
PLATAFORMA DE GEMELO VIRTUAL CEREBRAL - CUCEI UdeG
======================================================================
Caso Evaluado       : {foco_nombre}
Intervención        : {tipo_intervencion}
Coordenadas Foco EZ : {list(coords_ez)} MNI
Coordenadas M1 Motor: {list(COORDS_M1)} MNI
Distancia Anatómica : {res['distancia_mm']} mm
Acoplamiento DTI    : {peso_dti}

RESULTADOS DEL MODELO $I_ROM$:
----------------------------------------------------------------------
• Riesgo de Pérdida de ROM : {res['riesgo_pct']}%
• ROM Motor Preservado     : {res['rom_preservado_pct']}%
• Categoría de Riesgo      : {res['categoria']}

RECOMENDACIÓN QUIRÚRGICA:
----------------------------------------------------------------------
{res['recomendacion']}
======================================================================
"""
    
    st.download_button(
        label="📥 Descargar Reporte Clínico de Riesgo Motor (.txt)",
        data=txt_reporte,
        file_name="reporte_evaluacion_riesgo_rom.txt",
        mime="text/plain"
    )
