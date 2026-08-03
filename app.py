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
st.markdown("### 2D SVFD Mode Solver with TE & TM Polarization Support")

# --- SIDEBAR: MATERIAL & MODE SELECTOR ---
st.sidebar.header("🧪 Material & Mode Selection")
core_material = st.sidebar.selectbox(
    "Core Material",
    options=["Si3N4 (Stoichiometric)", "SiN (Low Stress)", "Al2O3 (Alumina)", "Si (Silicon)"],
    index=0
)

analysis_type = st.sidebar.radio(
    "Analysis Mode",
    options=["Single Point Analysis", "1D Parametric Sweep", "2D Universal Parametric Sweep (v49)"],
    index=0
)

st.sidebar.header("🔬 Numerical Mesh")
res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

# --- MODE 1: SINGLE POINT ANALYSIS ---
if analysis_type == "Single Point Analysis":
    st.sidebar.header("🛠️ Geometry & Wavelength")
    w_core = st.sidebar.number_input("Waveguide Width [μm]", value=1.0, step=0.1)
    h_core = st.sidebar.number_input("Waveguide Height [μm]", value=0.4, step=0.05)
    lam_um = st.sidebar.number_input("Wavelength [μm]", value=1.55, step=0.01)
    top_ox = st.sidebar.number_input("Oxide Top Thickness [μm]", value=0.1, step=0.05)
    bottom_ox = st.sidebar.number_input("Oxide Bottom Thickness (BOX) [μm]", value=4.0, step=0.5)

    run_sp_btn = st.sidebar.button("🚀 Calculate Single Point", type="primary", use_container_width=True)

    if run_sp_btn or 'sp_results' in st.session_state:
        if run_sp_btn:
            with st.spinner("Calculating TE and TM fundamental modes..."):
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material)
                st.session_state['sp_results'] = res

        r = st.session_state['sp_results']

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TE Effective Index (n_eff)", f"{r['neff_te']:.4f}")
        m2.metric("TM Effective Index (n_eff)", f"{r['neff_tm']:.4f}")
        m3.metric("TE Core Confinement (Γ_Core)", f"{r['gamma_core_te']:.2f} %")
        m4.metric("TM Air Confinement (Γ_Air)", f"{r['gamma_air_tm']:.2f} %")

        st.markdown("---")

        fig_te, ax_te = plt.subplots(figsize=(6, 4))
        im_te = ax_te.imshow(r['phi_te'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
        ax_te.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
        ax_te.axhline(r['interface_y'], color='r', linestyle=':', lw=1.5)
        fig_te.colorbar(im_te, ax=ax_te, label='TE Field Intensity (Ex)')
        ax_te.set_title(f"Quasi-TE Mode Profile (Ex) - n_eff={r['neff_te']:.4f}")

        fig_tm, ax_tm = plt.subplots(figsize=(6, 4))
        im_tm = ax_tm.imshow(r['phi_tm'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
        ax_tm.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
        ax_tm.axhline(r['interface_y'], color='r', linestyle=':', lw=1.5)
        fig_tm.colorbar(im_tm, ax=ax_tm, label='TM Field Intensity (Ey)')
        ax_tm.set_title(f"Quasi-TM Mode Profile (Ey) - n_eff={r['neff_tm']:.4f}")

        col_p1, col_p2 = st.columns(2)
        with col_p1: st.pyplot(fig_te)
        with col_p2: st.pyplot(fig_tm)

# --- MODE 2: 1D PARAMETRIC SWEEP ---
elif analysis_type == "1D Parametric Sweep":
    st.sidebar.header("🎯 1D Scan Parameter Selection")
    param_options = ["Wavelength", "Waveguide Width", "Waveguide Height", "Oxide Top Thickness"]
    axis_1d = st.sidebar.selectbox("Scanned Parameter", options=param_options, index=0)

    st.sidebar.header("📊 Scan Vector & Fixed Values")
    def_vec_1d = "1.50, 1.52, 1.54, 1.56, 1.58, 1.60" if axis_1d == "Wavelength" else "0.6, 0.8, 1.0, 1.2, 1.4, 1.6"
    str_vec_1d = st.sidebar.text_input(f"{axis_1d} Vector", value=def_vec_1d)

    rem_params_1d = [p for p in param_options if p != axis_1d]
    fixed_dict_1d = {}
    for p in rem_params_1d:
        def_val = 1.0 if p=="Waveguide Width" else (0.4 if p=="Waveguide Height" else (1.55 if p=="Wavelength" else 0.1))
        fixed_dict_1d[p] = st.sidebar.number_input(f"{p} (Fixed)", value=def_val)
    fixed_dict_1d["Oxide Bottom Thickness"] = st.sidebar.number_input("BOX Thickness [μm]", value=4.0)

    run_1d_btn = st.sidebar.button("🚀 Run 1D Sweep", type="primary", use_container_width=True)

    if run_1d_btn or 'sweep_1d_results' in st.session_state:
        if run_1d_btn:
            vec_1d = np.array([float(v.strip()) for v in str_vec_1d.split(",") if v.strip()])
            prog_bar_1d = st.progress(0)
            status_txt_1d = st.empty()

            def update_prog_1d(curr, tot):
                pct = int((curr / tot) * 100)
                prog_bar_1d.progress(pct)
                status_txt_1d.markdown(f"⏳ **Running 1D Sweep Point {curr}/{tot} ({pct}%)...**")

            res_1d = run_1d_sweep(axis_1d, vec_1d, fixed_dict_1d, res_mode, core_material, progress_callback=update_prog_1d)
            status_txt_1d.success("✅ 1D Sweep completed!")
            time.sleep(0.5)
            status_txt_1d.empty()
            prog_bar_1d.empty()
            st.session_state['sweep_1d_results'] = res_1d

        s1 = st.session_state['sweep_1d_results']

        fig_neff_1d, ax_neff_1d = plt.subplots(figsize=(7, 4))
        ax_neff_1d.plot(s1['param_vec'], s1['neff_te_vec'], 'bo-', lw=2, label='Quasi-TE (n_eff)')
        ax_neff_1d.plot(s1['param_vec'], s1['neff_tm_vec'], 'rs-', lw=2, label='Quasi-TM (n_eff)')
        ax_neff_1d.grid(True)
        ax_neff_1d.set_xlabel(s1['param_name'])
        ax_neff_1d.set_ylabel('Effective Index (n_eff)')
        ax_neff_1d.set_title(f"Effective Index Dispersion vs. {s1['param_name']}")
        ax_neff_1d.legend()

        fig_conf_1d, ax_conf_1d = plt.subplots(figsize=(7, 4))
        ax_conf_1d.plot(s1['param_vec'], s1['gamma_core_te_vec'], 'b-o', lw=2, label='Γ_Core (TE)')
        ax_conf_1d.plot(s1['param_vec'], s1['gamma_core_tm_vec'], 'r-s', lw=2, label='Γ_Core (TM)')
        ax_conf_1d.plot(s1['param_vec'], s1['gamma_air_te_vec'], 'b--^', lw=1.5, label='Γ_Air (TE)')
        ax_conf_1d.plot(s1['param_vec'], s1['gamma_air_tm_vec'], 'r--v', lw=1.5, label='Γ_Air (TM)')
        ax_conf_1d.grid(True)
        ax_conf_1d.set_xlabel(s1['param_name'])
        ax_conf_1d.set_ylabel('Confinement Factor [%]')
        ax_conf_1d.set_title(f"Core & Air Confinement vs. {s1['param_name']}")
        ax_conf_1d.legend()

        col_1d_a, col_1d_b = st.columns(2)
        with col_1d_a: st.pyplot(fig_neff_1d)
        with col_1d_b: st.pyplot(fig_conf_1d)

# --- MODE 3: 2D UNIVERSAL PARAMETRIC SWEEP ---
else:
    st.sidebar.header("🎯 2D Scan Axes Selection")
    param_options = ["Waveguide Width", "Waveguide Height", "Wavelength", "Oxide Top Thickness"]
    axis_x = st.sidebar.selectbox("First Scan Axis (X)", options=param_options, index=2)
    axis_y = st.sidebar.selectbox("Second Scan Axis (Y)", options=[p for p in param_options if p != axis_x], index=0)

    st.sidebar.header("📊 Scan Vectors & Fixed Values")
    def get_default_str(p):
        if p == "Waveguide Width": return "0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8"
        elif p == "Waveguide Height": return "0.2, 0.3, 0.4, 0.5, 0.6"
        elif p == "Wavelength": return "1.50, 1.52, 1.54, 1.56, 1.58, 1.60"
        else: return "0.0, 0.1, 0.2, 0.3, 0.4, 0.5"

    str_vec_x = st.sidebar.text_input(f"{axis_x} Vector (X)", value=get_default_str(axis_x))
    str_vec_y = st.sidebar.text_input(f"{axis_y} Vector (Y)", value=get_default_str(axis_y))

    remaining_params = [p for p in param_options if p not in [axis_x, axis_y]]
    fixed_dict = {}
    st.sidebar.markdown("**Fixed Parameters:**")
    for p in remaining_params:
        def_val = 1.0 if p=="Waveguide Width" else (0.4 if p=="Waveguide Height" else (1.55 if p=="Wavelength" else 0.1))
        fixed_dict[p] = st.sidebar.number_input(f"{p} (Fixed)", value=def_val)
    fixed_dict["Oxide Bottom Thickness"] = st.sidebar.number_input("BOX Thickness [μm]", value=4.0)

    run_2d_btn = st.sidebar.button("🚀 Run 2D Confinement Sweep", type="primary", use_container_width=True)

    if run_2d_btn or 'sweep_2d_results' in st.session_state:
        if run_2d_btn:
            vec_x = np.array([float(v.strip()) for v in str_vec_x.split(",") if v.strip()])
            vec_y = np.array([float(v.strip()) for v in str_vec_y.split(",") if v.strip()])
            
            prog_bar = st.progress(0)
            status_txt = st.empty()

            def update_prog(curr, tot):
                pct = int((curr / tot) * 100)
                prog_bar.progress(pct)
                status_txt.markdown(f"⏳ **Running 2D Sweep Point {curr}/{tot} ({pct}%)...**")

            s_res = run_2d_universal_sweep(axis_x, vec_x, axis_y, vec_y, fixed_dict, res_mode, core_material, progress_callback=update_prog)
            status_txt.success("✅ 2D Sweep completed!")
            time.sleep(0.5)
            status_txt.empty()
            prog_bar.empty()
            st.session_state['sweep_2d_results'] = s_res

        sr = st.session_state['sweep_2d_results']

        st.subheader("🗺️ 2D Confinement Contour Maps (%)")
        
        fig_c1, ax_c1 = plt.subplots(figsize=(6, 4))
        cp1 = ax_c1.contourf(sr['vec1'], sr['vec2'], sr['gamma_core_te_mat'], levels=10, cmap='jet')
        fig_c1.colorbar(cp1, ax=ax_c1, label='Core Confinement TE [%]')
        ax_c1.set_xlabel(sr['param1_name'])
        ax_c1.set_ylabel(sr['param2_name'])
        ax_c1.set_title("TE Core Confinement Γ_Core (%)")

        fig_c2, ax_c2 = plt.subplots(figsize=(6, 4))
        cp2 = ax_c2.contourf(sr['vec1'], sr['vec2'], sr['gamma_air_tm_mat'], levels=10, cmap='jet')
        fig_c2.colorbar(cp2, ax=ax_c2, label='Air Confinement TM [%]')
        ax_c2.set_xlabel(sr['param1_name'])
        ax_c2.set_ylabel(sr['param2_name'])
        ax_c2.set_title("TM Air Confinement Γ_Air (%)")

        col_m1, col_m2 = st.columns(2)
        with col_m1: st.pyplot(fig_c1)
        with col_m2: st.pyplot(fig_c2)

if 'sp_results' not in st.session_state and 'sweep_1d_results' not in st.session_state and 'sweep_2d_results' not in st.session_state:
    st.info("👈 Choose core material, set parameters, and click **Calculate**!")
    st.markdown("### 🌟 Reference Modal Profiles & Numerical Benchmarks")

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
                    <div style="font-weight: 600; font-size: 15px; margin-bottom: 10px; color: #0F172A;">
                        {item['title']}
                    </div>
                    <img src="data:image/png;base64,{b64}" style="max-width: 80%; height: auto; border-radius: 8px; border: 1px solid #CBD5E1;">
                </div>
            """)

        carousel_html = f"""
        <div id="slideshow-container" style="max-width: 750px; margin: 10px auto; padding: 18px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0;">
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
