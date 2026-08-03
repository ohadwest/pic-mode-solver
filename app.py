import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import time
import os
import streamlit.components.v1 as components
from mode_engine import run_single_point, run_1d_sweep, run_2d_universal_sweep

st.set_page_config(
    page_title="Universal Waveguide Confinement & Mode Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.markdown("### Integrated Optics Multi-Mode Solver with Range Controls & Sample Field Displays")

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

# --- HELPER FUNCTION FOR FIELD DISPLAY ---
def render_sample_3_fields(sample_points_dict):
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    tags = ["Min", "Mid", "Max"]
    
    for idx, tag in enumerate(tags):
        if tag in sample_points_dict:
            s_data = sample_points_dict[tag]
            with cols[idx]:
                st.markdown(f"#### 📍 {tag} Sample Point")
                if 'val' in s_data: st.caption(f"Parameter Value: **{s_data['val']:.3f}**")
                elif 'p1' in s_data: st.caption(f"({s_data['p1']:.2f}, {s_data['p2']:.2f})")
                
                sp_r = s_data['res']
                
                # Show TE0 if available
                if len(sp_r['te_modes']) > 0:
                    m0 = sp_r['te_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    ax.plot([sp_r['x_min'], sp_r['x_max'], sp_r['x_max'], sp_r['x_min'], sp_r['x_min']], [sp_r['y_min'], sp_r['y_min'], sp_r['y_max'], sp_r['y_max'], sp_r['y_min']], 'w--', lw=1.2)
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TE0 Field (n_eff={m0['neff']:.4f})")
                    st.pyplot(fig)
                
                # Show TM0 if available
                if len(sp_r['tm_modes']) > 0:
                    m0 = sp_r['tm_modes'][0]
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(m0['field'].T, origin='lower', extent=[sp_r['xc'][0], sp_r['xc'][-1], sp_r['yc'][0], sp_r['yc'][-1]], cmap='jet', aspect='auto')
                    ax.plot([sp_r['x_min'], sp_r['x_max'], sp_r['x_max'], sp_r['x_min'], sp_r['x_min']], [sp_r['y_min'], sp_r['y_min'], sp_r['y_max'], sp_r['y_max'], sp_r['y_min']], 'w--', lw=1.2)
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"TM0 Field (n_eff={m0['neff']:.4f})")
                    st.pyplot(fig)

# --- MODE 1: SINGLE POINT ANALYSIS ---
if analysis_type == "Single Point Analysis":
    st.sidebar.header("🛠️ Geometry & Wavelength")
    w_core = st.sidebar.number_input("Waveguide Width [μm]", value=1.5, step=0.1)
    h_core = st.sidebar.number_input("Waveguide Height [μm]", value=0.4, step=0.05)
    lam_um = st.sidebar.number_input("Wavelength [μm]", value=1.55, step=0.01)
    top_ox = st.sidebar.number_input("Oxide Top Thickness [μm]", value=0.1, step=0.05)
    bottom_ox = st.sidebar.number_input("Oxide Bottom Thickness (BOX) [μm]", value=4.0, step=0.5)

    run_sp_btn = st.sidebar.button("🚀 Calculate Single Point", type="primary", use_container_width=True)

    if run_sp_btn or 'sp_results' in st.session_state:
        if run_sp_btn:
            with st.spinner("Searching and solving for all bound modes..."):
                res = run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice)
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
            
        if len(tab_list) > 0:
            tabs = st.tabs(tab_list)
            tab_idx = 0
            
            for m in r['te_modes']:
                with tabs[tab_idx]:
                    st.subheader(f"Quasi-TE Mode TE_{m['mode_num']} (n_eff = {m['neff']:.5f})")
                    col_info, col_img = st.columns([1, 2])
                    with col_info:
                        st.markdown(f"""
                        * **Effective Index ($n_{{\\text{{eff}}}}$):** {m['neff']:.5f}
                        * **Core Confinement ($\Gamma_{{\\text{{Core}}}}$):** {m['gamma_core']:.2f}%
                        * **Air Confinement ($\Gamma_{{\\text{{Air}}}}$):** {m['gamma_air']:.2f}%
                        * **Effective Area ($A_{{\\text{{eff}}}}$):** {m['a_eff']:.2f} μm²
                        """)
                    with col_img:
                        fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
                        im_m = ax_m.imshow(m['field'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                        ax_m.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                        fig_m.colorbar(im_m, ax=ax_m, label='Field (Ex)')
                        st.pyplot(fig_m)
                tab_idx += 1
                
            for m in r['tm_modes']:
                with tabs[tab_idx]:
                    st.subheader(f"Quasi-TM Mode TM_{m['mode_num']} (n_eff = {m['neff']:.5f})")
                    col_info, col_img = st.columns([1, 2])
                    with col_info:
                        st.markdown(f"""
                        * **Effective Index ($n_{{\\text{{eff}}}}$):** {m['neff']:.5f}
                        * **Core Confinement ($\Gamma_{{\\text{{Core}}}}$):** {m['gamma_core']:.2f}%
                        * **Air Confinement ($\Gamma_{{\\text{{Air}}}}$):** {m['gamma_air']:.2f}%
                        * **Effective Area ($A_{{\\text{{eff}}}}$):** {m['a_eff']:.2f} μm²
                        """)
                    with col_img:
                        fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
                        im_m = ax_m.imshow(m['field'].T, origin='lower', extent=[r['xc'][0], r['xc'][-1], r['yc'][0], r['yc'][-1]], cmap='jet', aspect='auto')
                        ax_m.plot([r['x_min'], r['x_max'], r['x_max'], r['x_min'], r['x_min']], [r['y_min'], r['y_min'], r['y_max'], r['y_max'], r['y_min']], 'w--', lw=1.5)
                        fig_m.colorbar(im_m, ax=ax_m, label='Field (Ey)')
                        st.pyplot(fig_m)
                tab_idx += 1

# --- MODE 2: 1D PARAMETRIC SWEEP ---
elif analysis_type == "1D Parametric Sweep":
    st.sidebar.header("🎯 1D Scan Parameter Controls")
    param_options = ["Wavelength", "Waveguide Width", "Waveguide Height", "Oxide Top Thickness"]
    axis_1d = st.sidebar.selectbox("Scanned Parameter", options=param_options, index=0)

    # Clean Min / Max / Points Inputs
    def_min = 1.50 if axis_1d == "Wavelength" else (0.6 if axis_1d == "Waveguide Width" else 0.2)
    def_max = 1.60 if axis_1d == "Wavelength" else (1.8 if axis_1d == "Waveguide Width" else 0.6)
    def_pts = 11

    c_min, c_max, c_pts = st.sidebar.columns(3)
    val_min = c_min.number_input("Min", value=def_min, step=0.05)
    val_max = c_max.number_input("Max", value=def_max, step=0.05)
    num_pts = c_pts.number_input("Points", value=def_pts, min_value=3, max_value=51, step=2)

    rem_params_1d = [p for p in param_options if p != axis_1d]
    fixed_dict_1d = {}
    st.sidebar.markdown("**Fixed Parameters:**")
    for p in rem_params_1d:
        def_val = 1.0 if p=="Waveguide Width" else (0.4 if p=="Waveguide Height" else (1.55 if p=="Wavelength" else 0.1))
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
                status_txt_1d.markdown(f"⏳ **Running 1D Sweep Point {curr}/{tot} ({pct}%)...**")

            res_1d = run_1d_sweep(axis_1d, vec_1d, fixed_dict_1d, res_mode, core_material, pol_choice, progress_callback=update_prog_1d)
            status_txt_1d.success("✅ 1D Sweep completed!")
            time.sleep(0.5); status_txt_1d.empty(); prog_bar_1d.empty()
            st.session_state['sweep_1d_results'] = res_1d

        s1 = st.session_state['sweep_1d_results']

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Effective Index (n_eff)",
            "🎯 Confinement Factor (Γ)",
            "🖼️ Sample Field Profiles (Min/Mid/Max)",
            "⚡ Group Index (n_g)",
            "🌊 Dispersion (D)",
            "📐 Effective Area (A_eff)"
        ])

        styles = ['o-', 's--', '^:']

        with tab1:
            fig_n, ax_n = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['te'][m]['neff'])):
                        ax_n.plot(s1['param_vec'], s1['te'][m]['neff'], f'b{styles[m]}', lw=2, label=f'TE{m}')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['tm'][m]['neff'])):
                        ax_n.plot(s1['param_vec'], s1['tm'][m]['neff'], f'r{styles[m]}', lw=2, label=f'TM{m}')
            ax_n.grid(True); ax_n.legend(); ax_n.set_xlabel(s1['param_name']); ax_n.set_ylabel('Effective Index (n_eff)')
            st.pyplot(fig_n)

        with tab2:
            fig_c, ax_c = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['te'][m]['gamma_core'])):
                        ax_c.plot(s1['param_vec'], s1['te'][m]['gamma_core'], f'b{styles[m]}', lw=2, label=f'Γ_Core (TE{m})')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['tm'][m]['gamma_core'])):
                        ax_c.plot(s1['param_vec'], s1['tm'][m]['gamma_core'], f'r{styles[m]}', lw=2, label=f'Γ_Core (TM{m})')
            ax_c.grid(True); ax_c.legend(); ax_c.set_xlabel(s1['param_name']); ax_c.set_ylabel('Confinement Factor [%]')
            st.pyplot(fig_c)

        with tab3:
            render_sample_3_fields(s1['sample_points'])

        with tab4:
            fig_ng, ax_ng = plt.subplots(figsize=(7, 4))
            has_ng = False
            for pol_k, col_c in [('te', 'b'), ('tm', 'r')]:
                for m in range(3):
                    if 'ng' in s1[pol_k][m] and not np.all(np.isnan(s1[pol_k][m]['ng'])):
                        ax_ng.plot(s1['param_vec'], s1[pol_k][m]['ng'], f'{col_c}{styles[m]}', lw=2, label=f'n_g ({pol_k.upper()}{m})')
                        has_ng = True
            if has_ng:
                ax_ng.grid(True); ax_ng.legend(); ax_ng.set_xlabel(s1['param_name']); ax_ng.set_ylabel('Group Index (n_g)')
                st.pyplot(fig_ng)
            else: st.info("Group Index n_g is calculated when scanning Wavelength.")

        with tab5:
            fig_d, ax_d = plt.subplots(figsize=(7, 4))
            has_d = False
            for pol_k, col_c in [('te', 'b'), ('tm', 'r')]:
                for m in range(3):
                    if 'D' in s1[pol_k][m] and not np.all(np.isnan(s1[pol_k][m]['D'])):
                        ax_d.plot(s1['param_vec'], s1[pol_k][m]['D'], f'{col_c}{styles[m]}', lw=2, label=f'Dispersion D ({pol_k.upper()}{m})')
                        has_d = True
            if has_d:
                ax_d.grid(True); ax_d.legend(); ax_d.set_xlabel(s1['param_name']); ax_d.set_ylabel('D [ps/(nm·km)]')
                st.pyplot(fig_d)
            else: st.info("Dispersion D is calculated when scanning Wavelength.")

        with tab6:
            fig_a, ax_a = plt.subplots(figsize=(7, 4))
            if s1['pol_choice'] in ["TE", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['te'][m]['a_eff'])):
                        ax_a.plot(s1['param_vec'], s1['te'][m]['a_eff'], f'b{styles[m]}', lw=2, label=f'A_eff (TE{m})')
            if s1['pol_choice'] in ["TM", "Both (TE & TM)"]:
                for m in range(3):
                    if not np.all(np.isnan(s1['tm'][m]['a_eff'])):
                        ax_a.plot(s1['param_vec'], s1['tm'][m]['a_eff'], f'r{styles[m]}', lw=2, label=f'A_eff (TM{m})')
            ax_a.grid(True); ax_a.legend(); ax_a.set_xlabel(s1['param_name']); ax_a.set_ylabel('Effective Area A_eff [μm²]')
            st.pyplot(fig_a)

# --- MODE 3: 2D UNIVERSAL PARAMETRIC SWEEP ---
else:
    st.sidebar.header("🎯 2D Scan Axes Controls")
    param_options = ["Waveguide Width", "Waveguide Height", "Wavelength", "Oxide Top Thickness"]
    axis_x = st.sidebar.selectbox("First Scan Axis (X)", options=param_options, index=2)
    axis_y = st.sidebar.selectbox("Second Scan Axis (Y)", options=[p for p in param_options if p != axis_x], index=0)

    st.sidebar.markdown(f"**Axis X ({axis_x}) Range:**")
    cx_min, cx_max, cx_pts = st.sidebar.columns(3)
    vx_min = cx_min.number_input("Min X", value=1.50 if axis_x=="Wavelength" else 0.6, key="vx_min")
    vx_max = cx_max.number_input("Max X", value=1.60 if axis_x=="Wavelength" else 1.8, key="vx_max")
    nx_pts = cx_pts.number_input("Pts X", value=7, min_value=3, max_value=21, key="nx_pts")

    st.sidebar.markdown(f"**Axis Y ({axis_y}) Range:**")
    cy_min, cy_max, cy_pts = st.sidebar.columns(3)
    vy_min = cy_min.number_input("Min Y", value=0.2 if axis_y=="Waveguide Height" else 0.6, key="vy_min")
    vy_max = cy_max.number_input("Max Y", value=0.6 if axis_y=="Waveguide Height" else 1.8, key="vy_max")
    ny_pts = cy_pts.number_input("Pts Y", value=5, min_value=3, max_value=21, key="ny_pts")

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
            vec_x = np.linspace(vx_min, vx_max, int(nx_pts))
            vec_y = np.linspace(vy_min, vy_max, int(ny_pts))
            
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

        tab2d_1, tab2d_2, tab2d_3 = st.tabs([
            "🗺️ Core Confinement Maps (Γ_Core)",
            "🗺️ Air Cladding Confinement Maps (Γ_Air)",
            "🖼️ Sample Field Profiles (Min/Mid/Max)"
        ])

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

        with tab2d_3:
            render_sample_3_fields(sr['sample_points'])
