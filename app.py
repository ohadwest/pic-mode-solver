import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import time
import os
import base64
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
st.markdown("### Integrated Optics Multi-Mode Solver with Trapezoidal Sidewalls & PDF Export")

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
    return buf.getvalue()

# --- HELPER FUNCTION: RENDER FIG WITH INDIVIDUAL DOWNLOAD BUTTON ---
def display_fig_with_download(fig, filename, key_unique):
    st.pyplot(fig)
    img_bytes = fig_to_bytes(fig)
    st.download_button(
        label="💾 Download Image (PNG)",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        key=key_unique
    )

# --- HELPER FUNCTION: DRAW TRAPEZOID OUTLINE ON MATPLOTLIB AXIS ---
def draw_core_outline(ax, sp_r):
    w_top = sp_r.get('w_top', 1.5)
    w_bot = sp_r.get('w_bottom', w_top)
    y_min, y_max = sp_r['y_min'], sp_r['y_max']
    
    # Polygon corners: Bottom-Left, Bottom-Right, Top-Right, Top-Left, Bottom-Left
    x_coords = [-w_bot / 2.0, w_bot / 2.0, w_top / 2.0, -w_top / 2.0, -w_bot / 2.0]
    y_coords = [y_min, y_min, y_max, y_max, y_min]
    
    ax.plot(x_coords, y_coords, 'w--', lw=1.5, label='Core Boundary')

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
    
    story.append(Paragraph("2. Governing Physical Equations", h2_style))
    eq_text = """
    <b>Effective Index:</b> n_clad &lt; n_eff &lt; n_core <br/>
    <b>Confinement Factor:</b> &Gamma;_Core = &iint;_Core |E|&sup2; dx dy / &iint;_Total |E|&sup2; dx dy &times; 100% <br/>
    <b>Effective Mode Area:</b> A_eff = (&iint; |E|&sup2; dx dy)&sup2; / &iint; |E|&sup4; dx dy <br/>
    <b>Mode Field Diameter:</b> MFD = 2 &times; &radic;(A_eff / &pi;)
    """
    story.append(Paragraph(eq_text, body_style))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("3. Calculated Modal Distributions & Curves", h2_style))
    for fig_name, fig_obj in figs_dict.items():
        if fig_obj is not None:
            story.append(Paragraph(f"<b>{fig_name}</b>", body_style))
            img_buf = io.BytesIO(fig_to_bytes(fig_obj))
            story.append(RLImage(img_buf, width=480, height=270))
            story.append(Spacer(1, 10))
            
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- HELPER FUNCTION FOR INDEX PROFILE + 3 SAMPLE FIELDS DISPLAY ---
def render_index_and_3_sample_fields(sample_points_dict, prefix_key=""):
    tags = ["Min", "Mid", "Max"]
    
    if "Mid" in sample_points_dict:
        mid_sp = sample_points_dict["Mid"]["res"]
        st.markdown("#### 📐 Cross-Sectional Refractive Index Distribution $n(x,y)$")
        fig_n, ax_n = plt.subplots(figsize=(7, 3.8))
        n_mesh = np.sqrt(mid_sp['eps_mesh'])
        im_n = ax_n.imshow(n_mesh.T, origin='lower', extent=[mid_sp['xc'][0], mid_sp['xc'][-1], mid_sp['yc'][0], mid_sp['yc'][-1]], cmap='viridis', aspect='auto')
        draw_core_outline(ax_n, mid_sp)
        fig_n.colorbar(im_n, ax=ax_n, label='Refractive Index (n)')
        ax_n.set_xlabel('Horizontal Position X [μm]')
        ax_n.set_ylabel('Vertical Position Y [μm]')
        display_fig_with_download(fig_n, "refractive_index_profile.png", f"{prefix_key}_n_prof")
        st.markdown("---")

    st.markdown("#### 🖼️ Representative Mode Profiles Across Scan Range (Min / Mid / Max)")
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    
    for idx, tag in enumerate(tags):
        if tag in sample_points_dict:
            s_data = sample_points_dict[tag]
            with cols[idx]:
                st.markdown(f"##### 📍 {tag} Sample Point")
                if 'val' in s_data: 
                    st.caption(f"Parameter Value: **{s_data['val']:.3f}**")
                elif 'p1' in s_data: 
                    st.caption(f"X: **{s_data['p1']:.3f}**, Y: **{s_data['p2']:.3f}**")
                
                sp_r = s_data['res']
                
                if len(sp_r['te_modes']) > 0:
                    m0 = sp_r['te_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    draw_core_outline(ax, sp_r)
                    ax.axhline(sp_r['interface_y'], color='r', linestyle='--', lw=1.2, label='Air Boundary')
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TE0 Field (n_eff={m0['neff']:.4f})")
                    display_fig_with_download(fig, f"TE0_field_{tag}.png", f"{prefix_key}_te0_{tag}")
                
                if len(sp_r['tm_modes']) > 0:
                    m0 = sp_r['tm_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    draw_core_outline(ax, sp_r)
                    ax.axhline(sp_r['interface_y'], color='r', linestyle='--', lw=1.2, label='Air Boundary')
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TM0 Field (n_eff={m0['neff']:.4f})")
                    display_fig_with_download(fig, f"TM0_field_{tag}.png", f"{prefix_key}_tm0_{tag}")

# ==============================================================================
# --- MODE 1: SINGLE POINT ANALYSIS ---
# ==============================================================================
if analysis_type == "Single Point Analysis":
    st.sidebar.header("🛠️ Geometry & Wavelength")
    w_core = st.sidebar.number_input("Waveguide Top Width [μm]", value=1.5, step=0.1)
    h_core = st.sidebar.number_input("Waveguide Height [μm]", value=0.4, step=0.05)
    sidewall_angle = st.sidebar.slider("Sidewall Angle [deg]", min_value=30.0, max_value=90.0, value=90.0, step=1.0, help="90° = Rectangular. 45-80° = Trapezoidal sidewall etch.")
    lam_um = st.sidebar.number_input("Wavelength [μm]", value=1.55, step=0.01)
    top_ox = st.sidebar.number_input("Oxide Top Thickness [μm]", value=0.1, step=0.05)
    bottom_ox = st.sidebar.number_input("Oxide Bottom Thickness (BOX) [μm]", value=4.0, step=0.5)

    st.sidebar.header("🔍 Advanced Search")
    search_higher_modes = st.sidebar.checkbox("Search for Higher-Order Modes", value=False)

    run_sp_btn = st.sidebar.button("🚀 Calculate Single Point", type="primary", use_container_width=True)

    if run_sp_btn or 'sp_results' in st.session_state:
        if run_sp_btn:
            with st.spinner("Solving optical wave equations..."):
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice, search_higher_modes, sidewall_angle)
                st.session_state['sp_results'] = res

        r = st.session_state['sp_results']

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bound TE Modes Found", f"{len(r['te_modes'])}")
        m2.metric("Bound TM Modes Found", f"{len(r['tm_modes'])}")
        if len(r['te_modes']) > 0: m3.metric("TE0 n_eff", f"{r['te_modes'][0]['neff']:.4f}")
        if len(r['tm_modes']) > 0: m4.metric("TM0 n_eff", f"{r['tm_modes'][0]['neff']:.4f}")

        if r.get('sidewall_angle_deg', 90.0) < 89.9:
            st.info(f"📐 **Trapezoidal Profile Active:** Top Width = `{r['w_top']:.3f} μm`, Bottom Width = `{r['w_bottom']:.3f} μm` (Angle = `{r['sidewall_angle_deg']:.1f}°`)")

        st.markdown("---")
        
        tab_list = ["📐 Refractive Index Profile n(x,y)"]
        if len(r['te_modes']) > 0:
            for m in r['te_modes']: tab_list.append(f"TE{m['mode_num']} Mode")
        if len(r['tm_modes']) > 0:
            for m in r['tm_modes']: tab_list.append(f"TM{m['mode_num']} Mode")
            
        tab_list.append("📊 1D Profiles (X & Y Cutlines)")
        tab_list.append("📖 Equations & Definitions")
        
        tabs = st.tabs(tab_list)
        tab_idx = 0
        figs_for_pdf = {}

        # TAB 0: Refractive Index Profile
        with tabs[tab_idx]:
            st.subheader(f"Cross-Sectional Refractive Index Distribution n(x,y) - {core_material}")
            fig_idx_prof, ax_idx_prof = plt.subplots(figsize=(7, 4))
            n_mesh = np.sqrt(r['eps_mesh'])
            im_n = ax_idx_prof.imshow(n_mesh.T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='viridis', aspect='auto')
            draw_core_outline(ax_idx_prof, r)
            fig_idx_prof.colorbar(im_n, ax=ax_idx_prof, label='Refractive Index (n)')
            ax_idx_prof.set_xlabel('Horizontal Position X [μm]')
            ax_idx_prof.set_ylabel('Vertical Position Y [μm]')
            display_fig_with_download(fig_idx_prof, f"refractive_index_profile_{core_material}.png", "sp_n_prof")
            figs_for_pdf["Refractive Index Distribution n(x,y)"] = fig_idx_prof
        tab_idx += 1
        
        # TE Modes
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
                    draw_core_outline(ax_m, r)
                    ax_m.axhline(r['interface_y'], color='r', linestyle='--', lw=1.5, label='Air Boundary')
                    fig_m.colorbar(im_m, ax=ax_m, label='Field (Ex)')
                    display_fig_with_download(fig_m, f"TE{m['mode_num']}_field_profile.png", f"sp_te_{m['mode_num']}")
                    figs_for_pdf[f"TE{m['mode_num']} 2D Field Profile"] = fig_m
            tab_idx += 1
            
        # TM Modes
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
                    draw_core_outline(ax_m, r)
                    ax_m.axhline(r['interface_y'], color='r', linestyle='--', lw=1.5, label='Air Boundary')
                    fig_m.colorbar(im_m, ax=ax_m, label='Field (Ey)')
                    display_fig_with_download(fig_m, f"TM{m['mode_num']}_field_profile.png", f"sp_tm_{m['mode_num']}")
                    figs_for_pdf[f"TM{m['mode_num']} 2D Field Profile"] = fig_m
            tab_idx += 1

        # 1D Cutlines
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
                display_fig_with_download(fig_cx, "horizontal_cutline_1d.png", "sp_cut_x")
                figs_for_pdf["1D Horizontal Cutline"] = fig_cx

            with col_cy:
                fig_cy, ax_cy = plt.subplots(figsize=(6, 4))
                if len(r['te_modes']) > 0: ax_cy.plot(r['yc'], r['te_modes'][0]['cut_y'], 'b-', lw=2, label='TE0 (Vertical)')
                if len(r['tm_modes']) > 0: ax_cy.plot(r['yc'], r['tm_modes'][0]['cut_y'], 'r--', lw=2, label='TM0 (Vertical)')
                ax_cy.grid(True); ax_cy.legend()
                ax_cy.set_xlabel('Vertical Position Y [μm]'); ax_cy.set_ylabel('Normalized Field Intensity')
                ax_cy.set_title("Vertical Cutline (x = 0)")
                display_fig_with_download(fig_cy, "vertical_cutline_1d.png", "sp_cut_y")
                figs_for_pdf["1D Vertical Cutline"] = fig_cy
        tab_idx += 1

        # Equations & Definitions
        with tabs[tab_idx]:
            st.markdown("""
            ### 📖 Mathematical Equations & Physical Definitions
            * **Effective Index:** $n_{\\text{clad}} < n_{\\text{eff}} < n_{\\text{core}}$
            * **Confinement Factor:** $\\Gamma_{\\text{Core}} = \\frac{\\iint_{\\text{Core}} |E|^2 dx dy}{\\iint |E|^2 dx dy} \\times 100\\%$
            * **Effective Area:** $A_{\\text{eff}} = \\frac{\\left( \\iint |E|^2 dx dy \\right)^2}{\\iint |E|^4 dx dy}$
            """)

        # EXPORT SECTION
        st.markdown("---")
        st.subheader("📥 Export Complete Reports")
        c_exp1, c_exp2 = st.columns(2)
        
        summary_info = {
            "Core Material": core_material,
            "Wavelength [μm]": lam_um,
            "Top Width [μm]": w_core,
            "Bottom Width [μm]": f"{r['w_bottom']:.3f}",
            "Sidewall Angle": f"{sidewall_angle}°",
            "Core Height [μm]": h_core,
            "Top Oxide [μm]": top_ox,
            "BOX Thickness [μm]": bottom_ox,
            "Bound TE Modes": len(r['te_modes']),
            "Bound TM Modes": len(r['tm_modes'])
        }
        if len(r['te_modes']) > 0:
            summary_info["TE0 Effective Index"] = f"{r['te_modes'][0]['neff']:.5f}"
            summary_info["TE0 Core Confinement"] = f"{r['te_modes'][0]['gamma_core']:.2f}%"
        if len(r['tm_modes']) > 0:
            summary_info["TM0 Effective Index"] = f"{r['tm_modes'][0]['neff']:.5f}"
            summary_info["TM0 Core Confinement"] = f"{r['tm_modes'][0]['gamma_core']:.2f}%"

        with c_exp1:
            pdf_bytes = create_pdf_report("Single Point Mode Analysis Report", summary_info, figs_for_pdf)
            st.download_button(
                label="📄 Download Summary PDF Report",
                data=pdf_bytes,
                file_name=f"single_point_mode_report_{core_material}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with c_exp2:
            csv_df = pd.DataFrame([summary_info])
            st.download_button(
                label="📊 Download Results CSV",
                data=csv_df.to_csv(index=False).encode('utf-8'),
                file_name=f"single_point_results_{core_material}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==============================================================================
# --- MODE 2: FAST 1D PARAMETRIC SWEEP ---
# ==============================================================================
elif analysis_type == "1D Parametric Sweep":
    st.sidebar.header("🎯 1D Scan Parameter Controls")
    param_options = ["Wavelength", "Waveguide Width", "Waveguide Height", "Sidewall Angle", "Oxide Top Thickness"]
    axis_1d = st.sidebar.selectbox("Scanned Parameter", options=param_options, index=0)

    def_min = 1.50 if axis_1d == "Wavelength" else (60.0 if axis_1d == "Sidewall Angle" else (0.6 if axis_1d == "Waveguide Width" else 0.2))
    def_max = 1.60 if axis_1d == "Wavelength" else (90.0 if axis_1d == "Sidewall Angle" else (1.8 if axis_1d == "Waveguide Width" else 0.6))
    def_pts = 11

    c_min, c_max, c_pts = st.sidebar.columns(3)
    val_min = c_min.number_input("Min", value=def_min, step=0.05 if axis_1d!="Sidewall Angle" else 5.0)
    val_max = c_max.number_input("Max", value=def_max, step=0.05 if axis_1d!="Sidewall Angle" else 5.0)
    num_pts = c_pts.number_input("Points", value=def_pts, min_value=3, max_value=51, step=2)

    rem_params_1d = [p for p in param_options if p != axis_1d]
    fixed_dict_1d = {}
    st.sidebar.markdown("**Fixed Parameters:**")
    for p in rem_params_1d:
        def_val = 1.0 if p=="Waveguide Width" else (0.4 if p=="Waveguide Height" else (90.0 if p=="Sidewall Angle" else (1.55 if p=="Wavelength" else 0.1)))
        fixed_dict_1d[p] = st.sidebar.number_input(f"{p} (Fixed)", value=def_val)
    fixed_dict_1d["Oxide Bottom Thickness"] = st.sidebar.number_input("BOX Thickness [μm]", value=4.0)

    run_1d_btn = st.sidebar.button("🚀 Run 1D Sweep", type="primary", use_container_width=True)

    if run_1d_btn or 'sweep_1d_results' in st.session_state:
        if run_1d_btn:
            vec_1d = np.linspace(val_min, val_max, int(num_pts))
            prog_bar_1d = st.progress(0)
            status_txt_1d = st.empty()

            def update_prog_1d(curr, tot):
                pct = int((curr / tot) * 100)
                prog_bar_1d.progress(pct)
                status_txt_1d.markdown(f"⏳ **Running 1D Fundamental Mode Sweep {curr}/{tot} ({pct}%)...**")

            res_1d = run_1d_sweep(axis_1d, vec_1d, fixed_dict_1d, res_mode, core_material, pol_choice, progress_callback=update_prog_1d)
            status_txt_1d.success("✅ 1D Sweep completed!")
            time.sleep(0.5); status_txt_1d.empty(); prog_bar_1d.empty()
            st.session_state['sweep_1d_results'] = res_1d

        s1 = st.session_state['sweep_1d_results']
        figs_for_pdf_1d = {}

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📐 Geometry & Sample Profiles",
            "📈 Effective Index (n_eff)",
            "🔍 Material vs. Modal Index",
            "🎯 Confinement Factor (Γ)",
            "⚡ Group Index (n_g)",
            "🌊 Dispersion (D)",
            "📐 Effective Area (A_eff)"
        ])

        with tab1:
            render_index_and_3_sample_fields(s1['sample_points'], prefix_key="1d_tab1")

        with tab2:
            fig_n, ax_n = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]: ax_n.plot(s1['param_vec'], s1['neff_te'], 'bo-', lw=2, label='Quasi-TE0')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]: ax_n.plot(s1['param_vec'], s1['neff_tm'], 'rs-', lw=2, label='Quasi-TM0')
            ax_n.grid(True); ax_n.legend(); ax_n.set_xlabel(s1['param_name']); ax_n.set_ylabel('Effective Index (n_eff)')
            display_fig_with_download(fig_n, f"1D_neff_vs_{axis_1d}.png", "1d_neff_fig")
            figs_for_pdf_1d["Effective Index Dispersion"] = fig_n

        with tab3:
            fig_mat, ax_mat = plt.subplots(figsize=(7, 4))
            ax_mat.plot(s1['param_vec'], s1['n_core_vec'], 'k--', lw=2, label=f'n_core ({core_material})')
            ax_mat.plot(s1['param_vec'], s1['n_clad_vec'], 'g--', lw=2, label='n_clad (SiO2)')
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]: ax_mat.plot(s1['param_vec'], s1['neff_te'], 'bo-', lw=2, label='n_eff (TE0)')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]: ax_mat.plot(s1['param_vec'], s1['neff_tm'], 'rs-', lw=2, label='n_eff (TM0)')
            ax_mat.grid(True); ax_mat.legend(); ax_mat.set_xlabel(s1['param_name']); ax_mat.set_ylabel('Refractive Index (n)')
            ax_mat.set_title(f"Modal Dispersion n_eff vs. Material Indices ({core_material} & SiO2)")
            display_fig_with_download(fig_mat, f"1D_material_vs_neff_{axis_1d}.png", "1d_mat_fig")
            figs_for_pdf_1d["Material vs Modal Index Comparison"] = fig_mat

        with tab4:
            fig_c, ax_c = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                ax_c.plot(s1['param_vec'], s1['gamma_core_te'], 'b-o', lw=2, label='Γ_Core (TE0)')
                ax_c.plot(s1['param_vec'], s1['gamma_air_te'], 'b--^', lw=1.5, label='Γ_Air (TE0)')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                ax_c.plot(s1['param_vec'], s1['gamma_core_tm'], 'r-s', lw=2, label='Γ_Core (TM0)')
                ax_c.plot(s1['param_vec'], s1['gamma_air_tm'], 'r--v', lw=1.5, label='Γ_Air (TM0)')
            ax_c.grid(True); ax_c.legend(); ax_c.set_xlabel(s1['param_name']); ax_c.set_ylabel('Confinement Factor [%]')
            display_fig_with_download(fig_c, f"1D_confinement_vs_{axis_1d}.png", "1d_conf_fig")
            figs_for_pdf_1d["Confinement Factors"] = fig_c

        with tab5:
            if 'ng_te' in s1 or 'ng_tm' in s1:
                fig_ng, ax_ng = plt.subplots(figsize=(7, 4))
                if 'ng_te' in s1: ax_ng.plot(s1['param_vec'], s1['ng_te'], 'bo-', lw=2, label='n_g (TE0)')
                if 'ng_tm' in s1: ax_ng.plot(s1['param_vec'], s1['ng_tm'], 'rs-', lw=2, label='n_g (TM0)')
                ax_ng.grid(True); ax_ng.legend(); ax_ng.set_xlabel(s1['param_name']); ax_ng.set_ylabel('Group Index (n_g)')
                display_fig_with_download(fig_ng, f"1D_ng_vs_{axis_1d}.png", "1d_ng_fig")
                figs_for_pdf_1d["Group Index (ng)"] = fig_ng
            else: st.info("Group Index n_g is calculated when scanning Wavelength.")

        with tab6:
            if 'D_te' in s1 or 'D_tm' in s1:
                fig_d, ax_d = plt.subplots(figsize=(7, 4))
                if 'D_te' in s1: ax_d.plot(s1['param_vec'], s1['D_te'], 'bo-', lw=2, label='Dispersion D (TE0)')
                if 'D_tm' in s1: ax_d.plot(s1['param_vec'], s1['D_tm'], 'rs-', lw=2, label='Dispersion D (TM0)')
                ax_d.grid(True); ax_d.legend(); ax_d.set_xlabel(s1['param_name']); ax_d.set_ylabel('D [ps/(nm·km)]')
                display_fig_with_download(fig_d, f"1D_dispersion_D_vs_{axis_1d}.png", "1d_disp_fig")
                figs_for_pdf_1d["Chromatic Dispersion (D)"] = fig_d
            else: st.info("Dispersion D is calculated when scanning Wavelength.")

        with tab7:
            fig_a, ax_a = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]: ax_a.plot(s1['param_vec'], s1['a_eff_te'], 'bo-', lw=2, label='A_eff (TE0)')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]: ax_a.plot(s1['param_vec'], s1['a_eff_tm'], 'rs-', lw=2, label='A_eff (TM0)')
            ax_a.grid(True); ax_a.legend(); ax_a.set_xlabel(s1['param_name']); ax_a.set_ylabel('Effective Area A_eff [μm²]')
            display_fig_with_download(fig_a, f"1D_aeff_vs_{axis_1d}.png", "1d_aeff_fig")
            figs_for_pdf_1d["Effective Mode Area"] = fig_a

        # EXPORT SECTION 1D
        st.markdown("---")
        st.subheader("📥 Export Complete Reports")
        c1d_exp1, c1d_exp2 = st.columns(2)
        
        sum_1d = {
            "Core Material": core_material,
            "Scanned Parameter": axis_1d,
            "Range": f"{s1['param_vec'][0]:.3f} to {s1['param_vec'][-1]:.3f}",
            "Points": len(s1['param_vec'])
        }

        with c1d_exp1:
            pdf_bytes_1d = create_pdf_report(f"1D Parametric Sweep ({axis_1d}) Report", sum_1d, figs_for_pdf_1d)
            st.download_button(
                label="📄 Download 1D Sweep PDF Report",
                data=pdf_bytes_1d,
                file_name=f"1D_sweep_report_{axis_1d}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with c1d_exp2:
            data_dict_1d = {axis_1d: s1['param_vec'], 'n_core': s1['n_core_vec'], 'n_clad': s1['n_clad_vec']}
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                data_dict_1d['neff_TE0'] = s1['neff_te']
                data_dict_1d['Gamma_Core_TE0'] = s1['gamma_core_te']
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                data_dict_1d['neff_TM0'] = s1['neff_tm']
                data_dict_1d['Gamma_Core_TM0'] = s1['gamma_core_tm']
                
            csv_df_1d = pd.DataFrame(data_dict_1d)
            st.download_button(
                label="📊 Download 1D Sweep CSV Data",
                data=csv_df_1d.to_csv(index=False).encode('utf-8'),
                file_name=f"1D_sweep_data_{axis_1d}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==============================================================================
# --- MODE 3: FAST 2D UNIVERSAL PARAMETRIC SWEEP ---
# ==============================================================================
else:
    st.sidebar.header("🎯 2D Scan Axes Controls")
    param_options = ["Waveguide Width", "Waveguide Height", "Wavelength", "Sidewall Angle", "Oxide Top Thickness"]
    axis_x = st.sidebar.selectbox("First Scan Axis (X)", options=param_options, index=2)
    axis_y = st.sidebar.selectbox("Second Scan Axis (Y)", options=[p for p in param_options if p != axis_x], index=0)

    st.sidebar.markdown(f"**Axis X ({axis_x}) Range:**")
    cx_min, cx_max, cx_pts = st.sidebar.columns(3)
    vx_min = cx_min.number_input("Min X", value=1.50 if axis_x=="Wavelength" else (60.0 if axis_x=="Sidewall Angle" else 0.6), key="vx_min")
    vx_max = cx_max.number_input("Max X", value=1.60 if axis_x=="Wavelength" else (90.0 if axis_x=="Sidewall Angle" else 1.8), key="vx_max")
    nx_pts = cx_pts.number_input("Pts X", value=7, min_value=3, max_value=21, key="nx_pts")

    st.sidebar.markdown(f"**Axis Y ({axis_y}) Range:**")
    cy_min, cy_max, cy_pts = st.sidebar.columns(3)
    vy_min = cy_min.number_input("Min Y", value=0.2 if axis_y=="Waveguide Height" else (60.0 if axis_y=="Sidewall Angle" else 0.6), key="vy_min")
    vy_max = cy_max.number_input("Max Y", value=0.6 if axis_y=="Waveguide Height" else (90.0 if axis_y=="Sidewall Angle" else 1.8), key="vy_max")
    ny_pts = cy_pts.number_input("Pts Y", value=5, min_value=3, max_value=21, key="ny_pts")

    remaining_params = [p for p in param_options if p not in [axis_x, axis_y]]
    fixed_dict = {}
    st.sidebar.markdown("**Fixed Parameters:**")
    for p in remaining_params:
        def_val = 1.0 if p=="Waveguide Width" else (0.4 if p=="Waveguide Height" else (90.0 if p=="Sidewall Angle" else (1.55 if p=="Wavelength" else 0.1)))
        fixed_dict[p] = st.sidebar.number_input(f"{p} (Fixed)", value=def_val)
    fixed_dict["Oxide Bottom Thickness"] = st.sidebar.number_input("BOX Thickness [μm]", value=4.0)

    run_2d_btn = st.sidebar.button("🚀 Run 2D Confinement Sweep", type="primary", use_container_width=True)

    if run_2d_btn or 'sweep_2d_results' in st.session_state:
        if run_2d_btn:
            vec_x = np.linspace(vx_min, vx_max, int(nx_pts))
            vec_y = np.linspace(vy_min, vy_max, int(ny_pts))
            
            prog_bar = st.progress(0)
            status_txt = st.empty()

            def update_prog(curr, tot):
                pct = int((curr / tot) * 100)
                prog_bar.progress(pct)
                status_txt.markdown(f"⏳ **Running 2D Fundamental Mode Sweep {curr}/{tot} ({pct}%)...**")

            s_res = run_2d_universal_sweep(axis_x, vec_x, axis_y, vec_y, fixed_dict, res_mode, core_material, pol_choice, progress_callback=update_prog)
            status_txt.success("✅ 2D Sweep completed!")
            time.sleep(0.5); status_txt.empty(); prog_bar.empty()
            st.session_state['sweep_2d_results'] = s_res

        sr = st.session_state['sweep_2d_results']
        figs_for_pdf_2d = {}

        tab2d_1, tab2d_2, tab2d_3 = st.tabs([
            "📐 Geometry & Sample Profiles",
            "🗺️ Core Confinement Maps (Γ_Core)",
            "🗺️ Air Cladding Confinement Maps (Γ_Air)"
        ])

        with tab2d_1:
            render_index_and_3_sample_fields(sr['sample_points'], prefix_key="2d_tab1")

        with tab2d_2:
            col_m1, col_m2 = st.columns(2)
            if sr['pol_choice'] in ["TE", "Both (TE & TM)"]:
                fig_c1, ax_c1 = plt.subplots(figsize=(6, 4))
                cp1 = ax_c1.contourf(sr['vec1'], sr['vec2'], sr['gamma_core_te'], levels=10, cmap='jet')
                fig_c1.colorbar(cp1, ax=ax_c1, label='Core Confinement TE [%]')
                ax_c1.set_xlabel(sr['param1_name']); ax_c1.set_ylabel(sr['param2_name']); ax_c1.set_title("TE0 Core Confinement Γ_Core (%)")
                display_fig_with_download(fig_c1, f"2D_TE0_Core_Confinement_{axis_x}_vs_{axis_y}.png", "2d_conf_te_fig")
                figs_for_pdf_2d["TE0 Core Confinement Contour"] = fig_c1

            if sr['pol_choice'] in ["TM", "Both (TE & TM)"]:
                fig_c2, ax_c2 = plt.subplots(figsize=(6, 4))
                cp2 = ax_c2.contourf(sr['vec1'], sr['vec2'], sr['gamma_core_tm'], levels=10, cmap='jet')
                fig_c2.colorbar(cp2, ax=ax_c2, label='Core Confinement TM [%]')
                ax_c2.set_xlabel(sr['param1_name']); ax_c2.set_ylabel(sr['param2_name']); ax_c2.set_title("TM0 Core Confinement Γ_Core (%)")
                display_fig_with_download(fig_c2, f"2D_TM0_Core_Confinement_{axis_x}_vs_{axis_y}.png", "2d_conf_tm_fig")
                figs_for_pdf_2d["TM0 Core Confinement Contour"] = fig_c2

        with tab2d_3:
            col_a1, col_a2 = st.columns(2)
            if sr['pol_choice'] in ["TE", "Both (TE & TM)"]:
                fig_a1, ax_a1 = plt.subplots(figsize=(6, 4))
                cp_a1 = ax_a1.contourf(sr['vec1'], sr['vec2'], sr['gamma_air_te'], levels=10, cmap='jet')
                fig_a1.colorbar(cp_a1, ax=ax_a1, label='Air Confinement TE [%]')
                ax_a1.set_xlabel(sr['param1_name']); ax_a1.set_ylabel(sr['param2_name']); ax_a1.set_title("TE0 Air Confinement Γ_Air (%)")
                display_fig_with_download(fig_a1, f"2D_TE0_Air_Confinement_{axis_x}_vs_{axis_y}.png", "2d_air_te_fig")
                figs_for_pdf_2d["TE0 Air Confinement Contour"] = fig_a1

            if sr['pol_choice'] in ["TM", "Both (TE & TM)"]:
                fig_a2, ax_a2 = plt.subplots(figsize=(6, 4))
                cp_a2 = ax_a2.contourf(sr['vec1'], sr['vec2'], sr['gamma_air_tm'], levels=10, cmap='jet')
                fig_a2.colorbar(cp_a2, ax=ax_a2, label='Air Confinement TM [%]')
                ax_a2.set_xlabel(sr['param1_name']); ax_a2.set_ylabel(sr['param2_name']); ax_a2.set_title("TM0 Air Confinement Γ_Air (%)")
                display_fig_with_download(fig_a2, f"2D_TM0_Air_Confinement_{axis_x}_vs_{axis_y}.png", "2d_air_tm_fig")
                figs_for_pdf_2d["TM0 Air Confinement Contour"] = fig_a2

        # EXPORT SECTION 2D
        st.markdown("---")
        st.subheader("📥 Export Complete Reports")
        c2d_exp1, c2d_exp2 = st.columns(2)
        
        sum_2d = {
            "Core Material": core_material,
            "Axis X Parameter": axis_x,
            "Axis Y Parameter": axis_y,
            "Total Grid Points": f"{len(sr['vec1'])} x {len(sr['vec2'])}"
        }

        with c2d_exp1:
            pdf_bytes_2d = create_pdf_report(f"2D Confinement Sweep ({axis_x} vs {axis_y}) Report", sum_2d, figs_for_pdf_2d)
            st.download_button(
                label="📄 Download 2D Sweep PDF Report",
                data=pdf_bytes_2d,
                file_name=f"2D_sweep_report_{axis_x}_vs_{axis_y}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with c2d_exp2:
            st.download_button(
                label="📊 Download TE0 Core Confinement Matrix CSV",
                data=pd.DataFrame(sr['gamma_core_te'], index=np.round(sr['vec2'], 3), columns=np.round(sr['vec1'], 3)).to_csv().encode('utf-8'),
                file_name=f"2D_TE0_Core_Confinement_{axis_x}_vs_{axis_y}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==============================================================================
# --- PREVIEW / CAROUSEL DISPLAY (INITIAL LOAD) ---
# ==============================================================================
if 'sp_results' not in st.session_state and 'sweep_1d_results' not in st.session_state and 'sweep_2d_results' not in st.session_state:
    st.info("👈 Select core material and physical geometry in the sidebar, then click **Calculate** 🚀")
    
    st.markdown("### 🔬 Reference Modal Profiles & Numerical Benchmarks 🎨")
    st.markdown("Below are standard reference solutions calculated for a single channel optical waveguide:")

    preview_items = [
        {"file": "index_profile.png", "title": "1. Waveguide Refractive Index Distribution n(x,y) 📐"},
        {"file": "even_mode.png", "title": "2. Fundamental Quasi-TE Mode Field Profile ⚡"},
        {"file": "1d_profiles.png", "title": "3. 1D Transverse Field Profile at Core Center 📊"},
        {"file": "dispersion.png", "title": "4. Waveguide Dispersion Characteristics n_eff(λ) 📈"}
    ]
    
    valid_items = [item for item in preview_items if os.path.exists(item["file"])]
    if valid_items:
        encoded_slides = []
        for idx, item in enumerate(valid_items):
            with open(item["file"], "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
            encoded_slides.append(f"""
                <div class="mySlides fade" style="display: {'block' if idx==0 else 'none'}; text-align: center;">
                    <div style="font-weight: 600; font-size: 15px; margin-bottom: 10px; color: #0F172A; font-family: sans-serif;">
                        {item['title']}
                    </div>
                    <img src="data:image/png;base64,{b64}" style="max-width: 80%; height: auto; border-radius: 8px; border: 1px solid #CBD5E1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                </div>
            """)

        carousel_html = f"""
        <div id="slideshow-container" style="max-width: 750px; position: relative; margin: 10px auto; padding: 18px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0;">
            {''.join(encoded_slides)}
        </div>
        <script>
            let slideIndex = 0;
            showSlides();
            function showSlides() {{
                let i;
                let slides = document.getElementsByClassName("mySlides");
                for (i = 0; i < slides.length; i++) {{ slides[i].style.display = "none"; }}
                slideIndex++;
                if (slideIndex > slides.length) {{slideIndex = 1}}    
                if (slides[slideIndex-1]) {{ slides[slideIndex-1].style.display = "block"; }}
                setTimeout(showSlides, 3000);
            }}
        </script>
        """
        st.components.v1.html(carousel_html, height=480)
