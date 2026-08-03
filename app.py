import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import time
import os
import streamlit.components.v1 as components
from mode_engine import run_single_point, run_1d_sweep, run_2d_universal_sweep

# ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(
    page_title="Universal Waveguide Confinement & Mode Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GOOGLE ANALYTICS INTEGRATION ---
def inject_google_analytics(measurement_id):
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{measurement_id}', {{
          'page_path': window.parent.location.pathname,
          'cookie_flags': 'SameSite=None;Secure'
      }});
    </script>
    """
    components.html(ga_code, height=0, width=0)

inject_google_analytics("G-7776KX662W")

st.title("⚡ Universal Waveguide Core & Air Confinement Solver")
st.markdown("### Integrated Optics Multi-Mode Solver with PDF & CSV Export Tools")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🧪 Material & Polarization")
core_material = st.sidebar.selectbox(
    "Core Material",
    options=["Si3N4 (Stoichiometric)", "SiN (Low Stress)", "Al2O3 (Alumina)", "Si (Silicon)"],
    index=0
)

pol_choice = st.sidebar.selectbox(
    "Polarization Selection",
    options=["Both (TE & TM)", "TE", "TM"],
    index=0
)

analysis_type = st.sidebar.radio(
    "Analysis Type",
    options=["Single Point Analysis", "1D Parametric Sweep", "2D Universal Parametric Sweep (v49)"],
    index=0
)

res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

# --- HELPER FUNCTION: CONVERT MATPLOTLIB FIG TO BYTES ---
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# --- HELPER FUNCTION: GENERATE PDF REPORT ---
def create_pdf_report(title, summary_dict, figs_dict):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#0F172A"), spaceAfter=12)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, leading=13)
    
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("<b>Integrated Optics 2D SVFD Mode Solver Summary Report</b>", body_style))
    story.append(Spacer(1, 10))
    
    # Table of Parameters / Summary
    story.append(Paragraph("1. Simulation Parameters & Results Summary", h2_style))
    table_data = [["Parameter", "Value"]]
    for k, v in summary_dict.items():
        table_data.append([str(k), str(v)])
        
    t = Table(table_data, colWidths=[240, 280])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0,0), (1,0), colors.HexColor("#0F172A")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Equations Summary
    story.append(Paragraph("2. Governing Physical Equations", h2_style))
    eq_text = """
    <b>Effective Index:</b> n_clad &lt; n_eff &lt; n_core <br/>
    <b>Confinement Factor:</b> &Gamma;_Core = &iint;_Core |E|&sup2; dx dy / &iint;_Total |E|&sup2; dx dy &times; 100% <br/>
    <b>Effective Mode Area:</b> A_eff = (&iint; |E|&sup2; dx dy)&sup2; / &iint; |E|&sup4; dx dy <br/>
    <b>Mode Field Diameter:</b> MFD = 2 &times; &radic;(A_eff / &pi;)
    """
    story.append(Paragraph(eq_text, body_style))
    story.append(Spacer(1, 12))
    
    # Figures
    story.append(Paragraph("3. Calculated Modal Distributions & Curves", h2_style))
    for fig_name, fig_obj in figs_dict.items():
        if fig_obj is not None:
            story.append(Paragraph(f"<b>{fig_name}</b>", body_style))
            img_buf = fig_to_bytes(fig_obj)
            story.append(RLImage(img_buf, width=480, height=270))
            story.append(Spacer(1, 10))
            
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- HELPER FUNCTION FOR 3 SAMPLE FIELDS DISPLAY ---
def render_sample_3_fields(sample_points_dict):
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    tags = ["Min", "Mid", "Max"]
    
    for idx, tag in enumerate(tags):
        if tag in sample_points_dict:
            s_data = sample_points_dict[tag]
            with cols[idx]:
                st.markdown(f"#### 📍 {tag} Sample Point")
                if 'val' in s_data: 
                    st.caption(f"Parameter Value: **{s_data['val']:.3f}**")
                elif 'p1' in s_data: 
                    st.caption(f"X: **{s_data['p1']:.3f}**, Y: **{s_data['p2']:.3f}**")
                
                sp_r = s_data['res']
                
                if len(sp_r['te_modes']) > 0:
                    m0 = sp_r['te_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    ax.plot([sp_r['x_min'], sp_r['x_max'], sp_r['x_max'], sp_r['x_min'], sp_r['x_min']], [sp_r['y_min'], sp_r['y_min'], sp_r['y_max'], sp_r['y_max'], sp_r['y_min']], 'w--', lw=1.2)
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TE0 Field (n_eff={m0['neff']:.4f})")
                    st.pyplot(fig)
                
                if len(sp_r['tm_modes']) > 0:
                    m0 = sp_r['tm_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    ax.plot([sp_r['x_min'], sp_r['x_max'], sp_r['x_max'], sp_r['x_min'], sp_r['x_min']], [sp_r['y_min'], sp_r['y_min'], sp_r['y_max'], sp_r['y_max'], sp_r['y_min']], 'w--', lw=1.2)
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TM0 Field (n_eff={m0['neff']:.4f})")
                    st.pyplot(fig)

# ==============================================================================
# --- MODE 1: SINGLE POINT ANALYSIS ---
# ==============================================================================
if analysis_type == "Single Point Analysis":
    st.sidebar.header("🛠️ Geometry & Wavelength")
    w_core = st.sidebar.number_input("Waveguide Width [μm]", value=1.5, step=0.1)
    h_core = st.sidebar.number_input("Waveguide Height [μm]", value=0.4, step=0.05)
    lam_um = st.sidebar.number_input("Wavelength [μm]", value=1.55, step=0.01)
    top_ox = st.sidebar.number_input("Oxide Top Thickness [μm]", value=0.1, step=0.05)
    bottom_ox = st.sidebar.number_input("Oxide Bottom Thickness (BOX) [μm]", value=4.0, step=0.5)

    st.sidebar.header("🔍 Advanced Search")
    search_higher_modes = st.sidebar.checkbox("Search for Higher-Order Modes", value=False)

    run_sp_btn = st.sidebar.button("🚀 Calculate Single Point", type="primary", use_container_width=True)

    if run_sp_btn or 'sp_results' in st.session_state:
        if run_sp_btn:
            with st.spinner("Solving optical wave equations..."):
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice, search_higher_modes)
                st.session_state['sp_results'] = res

        r = st.session_state['sp_results']

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bound TE Modes Found", f"{len(r['te_modes'])}")
        m2.metric("Bound TM Modes Found", f"{len(r['tm_modes'])}")
        if len(r['te_modes']) > 0: m3.metric("TE0 n_eff", f"{r['te_modes'][0]['neff']:.4f}")
        if len(r['tm_modes']) > 0: m4.metric("TM0 n_eff", f"{r['tm_modes'][0]['neff']:.4f}")

        st.markdown("---")
        
        tab_list = []
        if len(r['te_modes']) > 0:
            for m in r['te_modes']: tab_list.append(f"TE{m['mode_num']} Mode")
        if len(r['tm_modes']) > 0:
            for m in r['tm_modes']: tab_list.append(f"TM{m['mode_num']} Mode")
            
        tab_list.append("📊 1D Profiles (X & Y Cutlines)")
        tab_list.append("📖 Equations & Definitions")
        
        tabs = st.tabs(tab_list)
        tab_idx = 0
        figs_for_pdf = {}
        
        for m in r['te_modes']:
            with tabs[tab_idx]:
                st.subheader(f"Quasi-TE Mode TE_{m['mode_num']} (n_eff = {m['neff']:.5f})")
                col_info, col_img = st.columns([1, 2])
                with col_info:
                    st.markdown(f"""
                    * **Effective Index ($n_{{\\text{{eff}}}}$):** `{m['neff']:.5f}`
                    * **Core Confinement ($\Gamma_{{\\text{{Core}}}}$):** `{m['gamma_core']:.2f}%`
                    * **Air Confinement ($\Gamma_{{\\text{{Air}}}}$):** `{m['gamma_air']:.2f}%`
                    * **Effective Area ($A_{{\\text{{eff}}}}$):** `{m['a_eff']:.3f} μm²`
                    * **Mode Field Diameter (MFD):** `{m['mfd']:.3f} μm`
                    * **Horizontal FWHM (X):** `{m['fwhm_x']:.3f} μm`
                    * **Vertical FWHM (Y):** `{m['fwhm_y']:.3f} μm`
                    """)
                with col_img:
                    fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
                    im_m = ax_m.imshow(m['field'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                    ax_m.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                    fig_m.colorbar(im_m, ax=ax_m, label='Field (Ex)')
                    st.pyplot(fig_m)
                    figs_for_pdf[f"TE{m['mode_num']} 2D Field Profile"] = fig_m
            tab_idx += 1
            
        for m in r['tm_modes']:
            with tabs[tab_idx]:
                st.subheader(f"Quasi-TM Mode TM_{m['mode_num']} (n_eff = {m['neff']:.5f})")
                col_info, col_img = st.columns([1, 2])
                with col_info:
                    st.markdown(f"""
                    * **Effective Index ($n_{{\\text{{eff}}}}$):** `{m['neff']:.5f}`
                    * **Core Confinement ($\Gamma_{{\\text{{Core}}}}$):** `{m['gamma_core']:.2f}%`
                    * **Air Confinement ($\Gamma_{{\\text{{Air}}}}$):** `{m['gamma_air']:.2f}%`
                    * **Effective Area ($A_{{\\text{{eff}}}}$):** `{m['a_eff']:.3f} μm²`
                    * **Mode Field Diameter (MFD):** `{m['mfd']:.3f} μm`
                    * **Horizontal FWHM (X):** `{m['fwhm_x']:.3f} μm`
                    * **Vertical FWHM (Y):** `{m['fwhm_y']:.3f} μm`
                    """)
                with col_img:
                    fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
                    im_m = ax_m.imshow(m['field'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                    ax_m.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                    fig_m.colorbar(im_m, ax=ax_m, label='Field (Ey)')
                    st.pyplot(fig_m)
                    figs_for_pdf[f"TM{m['mode_num']} 2D Field Profile"] = fig_m
            tab_idx += 1

        with tabs[tab_idx]:
            st.subheader("📊 1D Transverse Field Profiles (Cutlines along X & Y)")
            col_cx, col_cy = st.columns(2)
            
            with col_cx:
                fig_cx, ax_cx = plt.subplots(figsize=(6, 4))
                if len(r['te_modes']) > 0: ax_cx.plot(r['xc'], r['te_modes'][0]['cut_x'], 'b-', lw=2, label='TE0 (Horizontal)')
                if len(r['tm_modes']) > 0: ax_cx.plot(r['xc'], r['tm_modes'][0]['cut_x'], 'r--', lw=2, label='TM0 (Horizontal)')
                ax_cx.grid(True); ax_cx.legend()
                ax_cx.set_xlabel('Horizontal Position X [μm]'); ax_cx.set_ylabel('Normalized Field Intensity')
                ax_cx.set_title("Horizontal Cutline (y = y_center)")
                st.pyplot(fig_cx)
                figs_for_pdf["1D Horizontal Cutline"] = fig_cx

            with col_cy:
                fig_cy, ax_cy = plt.subplots(figsize=(6, 4))
                if len(r['te_modes']) > 0: ax_cy.plot(r['yc'], r['te_modes'][0]['cut_y'], 'b-', lw=2, label='TE0 (Vertical)')
                if len(r['tm_modes']) > 0: ax_cy.plot(r['yc'], r['tm_modes'][0]['cut_y'], 'r--', lw=2, label='TM0 (Vertical)')
                ax_cy.grid(True); ax_cy.legend()
                ax_cy.set_xlabel('Vertical Position Y [μm]'); ax_cy.set_ylabel('Normalized Field Intensity')
                ax_cy.set_title("Vertical Cutline (x = 0)")
                st.pyplot(fig_cy)
                figs_for_pdf["1D Vertical Cutline"] = fig_cy
        tab_idx += 1

        with tabs[tab_idx]:
            st.markdown("""
            ### 📖 Mathematical Equations & Physical Definitions
            * **Effective Index:** $n_{\\text{clad}} < n_{\\text{eff}} < n_{\\text{core}}$
            * **Confinement Factor:** $\\Gamma_{\\text{Core}} = \\frac{\\iint_{\\text{Core}} \vert{}E\vert{}^2 dx dy}{\\iint \vert{}E\vert{}^2 dx dy} \\times 100\\%$
            * **Effective Area:** $A_{\\text{eff}} = \\frac{\\left( \\iint \vert{}E\vert{}^2 dx dy \\right)^2}{\\iint \vert{}E\vert{}^4 dx dy}$
            """)

        # EXPORT SECTION
        st.markdown("---")
        st.subheader("📥 Export
