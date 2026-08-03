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
st.markdown("### Integrated Optics 2D Mode Solver with Dynamic Polarization Sweeps")

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
            with st.spinner("Calculating fundamental modes..."):
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice)
                st.session_state['sp_results'] = res

        r = st.session_state['sp_results']

        m1, m2, m3, m4 = st.columns(4)
        if 'neff_te' in r: m1.metric("TE n_eff", f"{r['neff_te']:.4f}")
        if 'neff_tm' in r: m2.metric("TM n_eff", f"{r['neff_tm']:.4f}")
        if 'gamma_core_te' in r: m3.metric("TE Γ_Core", f"{r['gamma_core_te']:.2f} %")
        if 'gamma_air_tm' in r: m4.metric("TM Γ_Air", f"{r['gamma_air_tm']:.2f} %")

        st.markdown("---")
        t_sp1, t_sp2 = st.tabs(["🖼️ Mode Profiles (2D)", "📊 Field Cutlines (1D)"])

        with t_sp1:
            col_p1, col_p2 = st.columns(2)
            if 'phi_te' in r:
                fig_te, ax_te = plt.subplots(figsize=(6, 4))
                im_te = ax_te.imshow(r['phi_te'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                ax_te.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                fig_te.colorbar(im_te, ax=ax_te, label='TE Field (Ex)')
                ax_te.set_title(f"Quasi-TE Mode Profile (Ex) - n_eff={r['neff_te']:.4f}")
                col_p1.pyplot(fig_te)

            if 'phi_tm' in r:
                fig_tm, ax_tm = plt.subplots(figsize=(6, 4))
                im_tm = ax_tm.imshow(r['phi_tm'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                ax_tm.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                fig_tm.colorbar(im_tm, ax=ax_tm, label='TM Field (Ey)')
                ax_tm.set_title(f"Quasi-TM Mode Profile (Ey) - n_eff={r['neff_tm']:.4f}")
                col_p2.pyplot(fig_tm)

        with t_sp2:
            fig_1d, ax_1d = plt.subplots(figsize=(7, 4))
            mid_y_idx = len(r['yc']) // 2
            if 'phi_te' in r: ax_1d.plot(r['xc'], r['phi_te'][:, mid_y_idx], 'b-', lw=2, label='TE Cutline')
            if 'phi_tm' in r: ax_1d.plot(r['xc'], r['phi_tm'][:, mid_y_idx], 'r--', lw=2, label='TM Cutline')
            ax_1d.grid(True); ax_1d.legend(); ax_1d.set_title("Transverse Field Profile at Core Center")
            st.pyplot(fig_1d)

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

            res_1d = run_1d_sweep(axis_1d, vec_1d, fixed_dict_1d, res_mode, core_material, pol_choice, progress_callback=update_prog_1d)
            status_txt_1d.success("✅ 1D Sweep completed!")
            time.sleep(0.5); status_txt_1d.empty(); prog_bar_1d.empty()
            st.session_state['sweep_1d_results'] = res_1d

        s1 = st.session_state['sweep_1d_results']

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Effective Index (n_eff)",
            "🎯 Confinement Factor (Γ)",
            "⚡ Group Index (n_g)",
            "🌊 Dispersion (D)",
            "📐 Effective Area (A_eff)"
        ])

        with tab1:
            fig_n, ax_n = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]: ax_n.plot(s1['param_vec'], s1['neff_te'], 'bo-', lw=2, label='Quasi-TE')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]: ax_n.plot(s1['param_vec'], s1['neff_tm'], 'rs-', lw=2, label='Quasi-TM')
            ax_n.grid(True); ax_n.legend(); ax_n.set_xlabel(s1['param_name']); ax_n.set_ylabel('Effective Index (n_eff)')
            st.pyplot(fig_n)

        with tab2:
            fig_c, ax_c = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                ax_c.plot(s1['param_vec'], s1['gamma_core_te'], 'b-o', lw=2, label='Γ_Core (TE)')
                ax_c.plot(s1['param_vec'], s1['gamma_air_te'], 'b--^', lw=1.5, label='Γ_Air (TE)')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                ax_c.plot(s1['param_vec'], s1['gamma_core_tm'], 'r-s', lw=2, label='Γ_Core (TM)')
                ax_c.plot(s1['param_vec'], s1['gamma_air_tm'], 'r--v', lw=1.5, label='Γ_Air (TM)')
            ax_c.grid(True); ax_c.legend(); ax_c.set_xlabel(s1['param_name']); ax_c.set_ylabel('Confinement Factor [%]')
            st.pyplot(fig_c)

        with tab3:
            if 'ng_te' in s1 or 'ng_tm' in s1:
                fig_ng, ax_ng = plt.subplots(figsize=(7, 4))
                if 'ng_te' in s1: ax_ng.plot(s1['param_vec'], s1['ng_te'], 'bo-', lw=2, label='n_g (TE)')
                if 'ng_tm' in s1: ax_ng.plot(s1['param_vec'], s1['ng_tm'], 'rs-', lw=2, label='n_g (TM)')
                ax_ng.grid(True); ax_ng.legend(); ax_ng.set_xlabel(s1['param_name']); ax_ng.set_ylabel('Group Index (n_g)')
                st.pyplot(fig_ng)
            else: st.info("Group Index n_g is calculated when scanning Wavelength.")

        with tab4:
            if 'D_te' in s1 or 'D_tm' in s1:
                fig_d, ax_d = plt.subplots(figsize=(7, 4))
                if 'D_te' in s1: ax_d.plot(s1['param_vec'], s1['D_te'], 'bo-', lw=2, label='Dispersion D (TE)')
                if 'D_tm' in s1: ax_d.plot(s1['param_vec'], s1['D_tm'], 'rs-', lw=2, label='Dispersion D (TM)')
                ax_d.grid(True); ax_d.legend(); ax_d.set_xlabel(s1['param_name']); ax_d.set_ylabel('D [ps/(nm·km)]')
                st.pyplot(fig_d)
            else: st.info("Dispersion D is calculated when scanning Wavelength.")

        with tab5:
            fig_a, ax_a = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]: ax_a.plot(s1['param_vec'], s1['a_eff_te'], 'bo-', lw=2, label='A_eff TE')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]: ax_a.plot(s1['param_vec'], s1['a_eff_tm'], 'rs-', lw=2, label='A_eff TM')
            ax_a.grid(True); ax_a.legend(); ax_a.set_xlabel(s1['param_name']); ax_a.set_ylabel('Effective Area A_eff [μm²]')
            st.pyplot(fig_a)

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

            s_res = run_2d_universal_sweep(axis_x, vec_x, axis_y, vec_y, fixed_dict, res_mode, core_material, pol_choice, progress_callback=update_prog)
            status_txt.success("✅ 2D Sweep completed!")
            time.sleep(0.5); status_txt.empty(); prog_bar.empty()
            st.session_state['sweep_2d_results'] = s_res

        sr = st.session_state['sweep_2d_results']

        tab2d_1, tab2d_2 = st.tabs(["🗺️ Core Confinement Maps (Γ_Core)", "🗺️ Air Cladding Confinement Maps (Γ_Air)"])

        with tab2d_1:
            col_m1, col_m2 = st.columns(2)
            if sr['pol_choice'] in ["TE", "Both (TE & TM)"]:
                fig_c1, ax_c1 = plt.subplots(figsize=(6, 4))
                cp1 = ax_c1.contourf(sr['vec1'], sr['vec2'], sr['gamma_core_te'], levels=10, cmap='jet')
                fig_c1.colorbar(cp1, ax=ax_c1, label='Core Confinement TE [%]')
                ax_c1.set_xlabel(sr['param1_name']); ax_c1.set_ylabel(sr['param2_name']); ax_c1.set_title("TE Core Confinement Γ_Core (%)")
                col_m1.pyplot(fig_c1)

            if sr['pol_choice'] in ["TM", "Both (TE & TM)"]:
                fig_c2, ax_c2 = plt.subplots(figsize=(6, 4))
                cp2 = ax_c2.contourf(sr['vec1'], sr['vec2'], sr['gamma_core_tm'], levels=10, cmap='jet')
                fig_c2.colorbar(cp2, ax=ax_c2, label='Core Confinement TM [%]')
                ax_c2.set_xlabel(sr['param1_name']); ax_c2.set_ylabel(sr['param2_name']); ax_c2.set_title("TM Core Confinement Γ_Core (%)")
                col_m2.pyplot(fig_c2)

        with tab2d_2:
            col_a1, col_a2 = st.columns(2)
            if sr['pol_choice'] in ["TE", "Both (TE & TM)"]:
                fig_a1, ax_a1 = plt.subplots(figsize=(6, 4))
                cp_a1 = ax_a1.contourf(sr['vec1'], sr['vec2'], sr['gamma_air_te'], levels=10, cmap='jet')
                fig_a1.colorbar(cp_a1, ax=ax_a1, label='Air Confinement TE [%]')
                ax_a1.set_xlabel(sr['param1_name']); ax_a1.set_ylabel(sr['param2_name']); ax_a1.set_title("TE Air Confinement Γ_Air (%)")
                col_a1.pyplot(fig_a1)

            if sr['pol_choice'] in ["TM", "Both (TE & TM)"]:
                fig_a2, ax_a2 = plt.subplots(figsize=(6, 4))
                cp_a2 = ax_a2.contourf(sr['vec1'], sr['vec2'], sr['gamma_air_tm'], levels=10, cmap='jet')
                fig_a2.colorbar(cp_a2, ax=ax_a2, label='Air Confinement TM [%]')
                ax_a2.set_xlabel(sr['param1_name']); ax_a2.set_ylabel(sr['param2_name']); ax_a2.set_title("TM Air Confinement Γ_Air (%)")
                col_a2.pyplot(fig_a2)

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
