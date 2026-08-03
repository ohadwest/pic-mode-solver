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

# --- GOOGLE ANALYTICS ---
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
st.markdown("### Integrated Optics Mode Solver with TE/TM Polarization Selector")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🧪 Material & Polarization")
core_material = st.sidebar.selectbox(
    "Core Material",
    options=["Si3N4 (Stoichiometric)", "SiN (Low Stress)", "Al2O3 (Alumina)", "Si (Silicon)"],
    index=0
)

pol_choice = st.sidebar.selectbox(
    "Polarization Mode",
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
            with st.spinner("Calculating modes..."):
                pol_str = "TE" if pol_choice=="TE" else ("TM" if pol_choice=="TM" else "Both")
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_str)
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
                ax_te.set_title(f"Quasi-TE Fundamental Mode (n_eff={r['neff_te']:.4f})")
                col_p1.pyplot(fig_te)

            if 'phi_tm' in r:
                fig_tm, ax_tm = plt.subplots(figsize=(6, 4))
                im_tm = ax_tm.imshow(r['phi_tm'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                ax_tm.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                fig_tm.colorbar(im_tm, ax=ax_tm, label='TM Field (Ey)')
                ax_tm.set_title(f"Quasi-TM Fundamental Mode (n_eff={r['neff_tm']:.4f})")
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

            pol_str = "TE" if pol_choice=="TE" else ("TM" if pol_choice=="TM" else "Both")
            res_1d = run_1d_sweep(axis_1d, vec_1d, fixed_dict_1d, res_mode, core_material, pol_str, progress_callback=update_prog_1d)
            status_txt_1d.success("✅ 1D Sweep completed!")
            time.sleep(0.5); status_txt_1d.empty(); prog_bar_1d.empty()
            st.session_state['sweep_1d_results'] = res_1d

        s1 = st.session_state['sweep_1d_results']

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Effective Index (n_eff)", "🎯 Confinement Factor (Γ)", "⚡ Group Index (n_g)", "🌊 Dispersion (D)"])

        with tab1:
            fig_n, ax_n = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both"]: ax_n.plot(s1['param_vec'], s1['neff_te'], 'bo-', lw=2, label='Quasi-TE')
            if s1['pol_choice'] in ["TM", "Both"]: ax_n.plot(s1['param_vec'], s1['neff_tm'], 'rs-', lw=2, label='Quasi-TM')
            ax_n.grid(True); ax_n.legend(); ax_n.set_xlabel(s1['param_name']); ax_n.set_ylabel('Effective Index (n_eff)')
            st.pyplot(fig_n)

        with tab2:
            fig_c, ax_c = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both"]:
                ax_c.plot(s1['param_vec'], s1['gamma_core_te'], 'b-o', lw=2, label='Γ_Core (TE)')
                ax_c.plot(s1['param_vec'], s1['gamma_air_te'], 'b--^', lw=1.5, label='Γ_Air (TE)')
            if s1['pol_choice'] in ["TM", "Both"]:
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
            else: st.info("Group Index n_g is available when scanning Wavelength.")

        with tab4:
            if 'D_te' in s1 or 'D_tm' in s1:
                fig_d, ax_d = plt.subplots(figsize=(7, 4))
                if 'D_te' in s1: ax_d.plot(s1['param_vec'], s1['D_te'], 'bo-', lw=2, label='Dispersion D (TE)')
                if 'D_tm' in s1: ax_d.plot(s1['param_vec'], s1['D_tm'], 'rs-', lw=2, label='Dispersion D (TM)')
                ax_d.grid(True); ax_d.legend(); ax_d.set_xlabel(s1['param_name']); ax_d.set_ylabel('D [ps/(nm·km)]')
                st.pyplot(fig_d)
            else: st.info("Dispersion D is available when scanning Wavelength.")

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
    str_vec_y = st.sidebar.text_input(
