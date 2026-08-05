# ==============================================================================
# File: mode_engine.py
# Version: v2.4.1 (Advanced Edition - Fixed & Metal Gap Implemented)
# Date: August 2026
# Description: Added Metal Gap parameter (distance from top of WG core to bottom of heater) and updated mesh generation.
# ==============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- REFRACTIVE INDEX DISPERSION & METAL MODELS ---

def sellmeier_sio2(lam_um):
    """Cladding: Thermal SiO2 Sellmeier Model"""
    b1, c1 = 0.6961663, 0.0684043
    b2, c2 = 0.4079426, 0.1162414
    b3, c3 = 0.8974794, 9.896161
    n_sq = 1.0 + (b1 * lam_um**2) / (lam_um**2 - c1**2) + \
                 (b2 * lam_um**2) / (lam_um**2 - c2**2) + \
                 (b3 * lam_um**2) / (lam_um**2 - c3**2)
    return np.sqrt(n_sq)

def n_sin_stoch(lam_um):
    """Stoichiometric Si3N4 (Cauchy)"""
    return 1.981800 + (1.407700e-02 / (lam_um**2))

def n_sin_lowstress(lam_um):
    """Low-Stress SiN (Cauchy)"""
    return 2.087000 + (3.109100e-02 / (lam_um**2))

def n_al2o3(lam_um):
    """Alumina (Al2O3) Core - ALUVIA PDK Sellmeier"""
    eps_inf, A, E, P = 1.0, 1.912, 0.09566, 0.00306
    n_sq = eps_inf + (A * lam_um**2) / (lam_um**2 - E**2) - P * lam_um**2
    return np.sqrt(np.maximum(n_sq, 1.0))

def n_silicon(lam_um):
    """Crystalline Silicon (Malitson Sellmeier)"""
    n_sq = 1.0 + (10.6684293 * lam_um**2) / (lam_um**2 - 0.301516485**2) + \
                 (0.0030434748 * lam_um**2) / (lam_um**2 - 1.13475115**2) + \
                 (1.54133408 * lam_um**2) / (lam_um**2 - 1104.0**2)
    return np.sqrt(np.maximum(n_sq, 1.0))

def get_core_index(lam_um, material_name):
    if material_name == "Si3N4 (Stoichiometric)": return n_sin_stoch(lam_um)
    elif material_name == "SiN (Low Stress)": return n_sin_lowstress(lam_um)
    elif material_name == "Al2O3 (Alumina)": return n_al2o3(lam_um)
    elif material_name == "Si (Silicon)": return n_silicon(lam_um)
    else: return n_sin_stoch(lam_um)

def get_metal_complex_index(metal_type, lam_um=1.55):
    """Returns complex refractive index n + i*k for metals at NIR/1.55um"""
    if "Al" in metal_type:
        n_m, k_m = 1.44, 16.00
    elif "Au" in metal_type:
        n_m, k_m = 0.55, 11.50
    elif "Pt" in metal_type:
        n_m, k_m = 4.00, 7.00
    else:
        n_m, k_m = 1.44, 16.00
    return complex(n_m, k_m)

# --- TRAPEZOIDAL, RIB, BENDED & METAL MESH GENERATION ---

def build_advanced_mesh(w_core, h_core, bottom_ox, top_ox, side_margin, dx, dy, n_core, n_clad, 
                        sidewall_angle_deg=90.0, ring_radius_um=0.0,
                        wg_type="Strip", h_slab=0.0, w_slab=0.0,
                        include_metal=False, metal_type="Al (Aluminum)", metal_thick_um=0.10, metal_width_um=2.0, metal_offset_um=0.0, metal_gap_um=1.0):
    
    angle_rad = np.radians(np.clip(sidewall_angle_deg, 1.0, 90.0))
    w_bottom = w_core + (2.0 * h_core / np.tan(angle_rad)) if sidewall_angle_deg < 89.9 else w_core
    
    max_w = max(w_core, w_bottom, w_slab if wg_type=="Rib" else 0.0, abs(metal_offset_um) + metal_width_um/2.0 if include_metal else 0.0)
    
    y_core_top = bottom_ox + (h_slab if wg_type=="Rib" else 0.0) + h_core
    y_metal_max = (y_core_top + metal_gap_um + metal_thick_um) if include_metal else y_core_top
    
    total_width = max_w + 2 * side_margin
    total_height = max(y_metal_max, y_core_top + top_ox) + 1.5
    
    nx = int(np.round(total_width / dx)) + 1
    ny = int(np.round(total_height / dy)) + 1
    
    x = np.linspace(-total_width / 2.0, total_width / 2.0, nx)
    y = np.linspace(0, total_height, ny)
    
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    
    dtype_mesh = np.complex128 if include_metal else np.float64
    eps = np.full((len(xc), len(yc)), n_clad**2, dtype=dtype_mesh)
    
    XC, YC = np.meshgrid(xc, yc, indexing='ij')
    
    if wg_type == "Rib":
        y_slab_min = bottom_ox
        y_slab_max = bottom_ox + h_slab
        slab_mask = (YC >= y_slab_min) & (YC <= y_slab_max) & (np.abs(XC) <= (w_slab / 2.0))
        eps[slab_mask] = n_core**2
        
        y_core_min = y_slab_max
        y_core_max = y_slab_max + h_core
        in_core_height = (YC >= y_core_min) & (YC <= y_core_max)
        
        if sidewall_angle_deg >= 89.9:
            half_w_y = w_core / 2.0
        else:
            half_w_y = (w_core / 2.0) + (y_core_max - YC) / np.tan(angle_rad)
            
        core_mask_upper = in_core_height & (np.abs(XC) <= half_w_y)
        eps[core_mask_upper] = n_core**2
        
        core_mask = slab_mask | core_mask_upper
        interface_y = bottom_ox + h_slab + h_core + top_ox
        y_min, y_max = bottom_ox, y_core_max
        
    else: # Strip Waveguide
        y_min, y_max = bottom_ox, bottom_ox + h_core
        in_height = (YC >= y_min) & (YC <= y_max)
        
        if sidewall_angle_deg >= 89.9:
            half_w_y = w_core / 2.0
        else:
            half_w_y = (w_core / 2.0) + (y_max - YC) / np.tan(angle_rad)
            
        core_mask = in_height & (np.abs(XC) <= half_w_y)
        eps[core_mask] = n_core**2
        interface_y = bottom_ox + h_core + top_ox

    # Add Metal Heater Element if enabled
    metal_mask = np.zeros_like(core_mask, dtype=bool)
    if include_metal:
        n_complex_metal = get_metal_complex_index(metal_type)
        eps_metal = n_complex_metal**2
        
        y_metal_min = y_core_top + metal_gap_um
        y_metal_max = y_metal_min + metal_thick_um
        x_metal_min = metal_offset_um - (metal_width_um / 2.0)
        x_metal_max = metal_offset_um + (metal_width_um / 2.0)
        
        metal_mask = (YC >= y_metal_min) & (YC <= y_metal_max) & (XC >= x_metal_min) & (XC <= x_metal_max)
        eps[metal_mask] = eps_metal

    air_mask_1d = yc > max(interface_y, y_core_top + metal_gap_um + (metal_thick_um if include_metal else 0.0))
    eps[:, air_mask_1d] = 1.0**2  # Air n=1
    
    if ring_radius_um > 0.1:
        conformal_factor = (1.0 + XC / ring_radius_um)**2
        eps = eps * conformal_factor
    
    air_mask = np.repeat(air_mask_1d[None, :], len(xc), axis=0)
    x_min, x_max = -w_bottom / 2.0, w_bottom / 2.0
    
    return xc, yc, eps, core_mask, air_mask, metal_mask, interface_y, x_min, x_max, y_min, y_max, w_bottom

# --- 2D SVFD EIGENMODE SOLVER ---

def svmodes_2d(lam_um, guess, nmodes, dx, dy, eps_mesh, polarization='ex'):
    nx, ny = eps_mesh.shape
    k0 = 2.0 * np.pi / lam_um
    eps_padded = np.pad(eps_mesh, ((1, 1), (1, 1)), mode='edge')
    
    ep = eps_padded[1:nx+1, 1:ny+1]
    en = eps_padded[1:nx+1, 2:ny+2]
    es = eps_padded[1:nx+1, 0:ny]
    ee = eps_padded[2:nx+2, 1:ny+1]
    ew = eps_padded[0:nx,   1:ny+1]
    
    n_mat, s_mat = np.full((nx, ny), dy), np.full((nx, ny), dy)
    e_mat, w_mat = np.full((nx, ny), dx), np.full((nx, ny), dx)
    p_mat, q_mat = np.full((nx, ny), dx), np.full((nx, ny), dy)
    
    if polarization.lower() == 'ex':
        an = 2.0 / (n_mat * (n_mat + s_mat))
        as_ = 2.0 / (s_mat * (n_mat + s_mat))
        num_e = 8.0 * (p_mat * (ep - ew) + 2.0 * w_mat * ew) * ee
        den_e = (p_mat * (ep - ee) + 2.0 * e_mat * ee) * (p_mat**2 * (ep - ew) + 4.0 * w_mat**2 * ew) + \
                (p_mat * (ep - ew) + 2.0 * w_mat * ew) * (p_mat**2 * (ep - ee) + 4.0 * e_mat**2 * ee)
        ae = num_e / den_e
        num_w = 8.0 * (p_mat * (ep - ee) + 2.0 * e_mat * ee) * ew
        aw = num_w / den_e
        ap = ep * (k0**2) - an - as_ - ae * (ep / ee) - aw * (ep / ew)
    else:
        num_n = 8.0 * (q_mat * (ep - es) + 2.0 * s_mat * es) * en
        den_n = (q_mat * (ep - en) + 2.0 * n_mat * en) * (q_mat**2 * (ep - es) + 4.0 * s_mat**2 * es) + \
                (q_mat * (ep - es) + 2.0 * s_mat * es) * (q_mat**2 * (ep - en) + 4.0 * n_mat**2 * en)
        an = num_n / den_n
        as_ = 8.0 * (q_mat * (ep - en) + 2.0 * n_mat * en) * es / den_n
        ae = 2.0 / (e_mat * (e_mat + w_mat))
        aw = 2.0 / (w_mat * (e_mat + w_mat))
        ap = ep * (k0**2) - an * (ep / en) - as_ * (ep / es) - ae - aw

    N = nx * ny
    main_diag = ap.flatten('F')
    ae_diag = ae.flatten('F')[:-1]
    aw_diag = aw.flatten('F')[1:]
    an_diag = an.flatten('F')[:-nx]
    as_diag = as_.flatten('F')[nx:]
    
    A = sp.diags([main_diag, ae_diag, aw_diag, an_diag, as_diag], [0, 1, -1, nx, -nx], shape=(N, N), format='csc')
    shift = (2.0 * np.pi * guess / lam_um)**2
    
    vals, vecs = spla.eigs(A, k=nmodes, sigma=shift, which='LM')
    
    neff_vals = (lam_um / (2.0 * np.pi)) * np.sqrt(vals)
    sorted_indices = np.argsort(np.real(neff_vals))[::-1]
    neff_vals = neff_vals[sorted_indices]
    
    phi_modes = np.zeros((nx, ny, nmodes), dtype=np.complex128 if np.iscomplexobj(eps_mesh) else np.float64)
    for idx in range(nmodes):
        s_idx = sorted_indices[idx]
        mode_2d = vecs[:, s_idx].reshape((nx, ny), order='F')
        max_abs = np.max(np.abs(mode_2d))
        if max_abs > 0: mode_2d /= max_abs
        phi_modes[:, :, idx] = mode_2d
        
    return phi_modes, neff_vals

# --- HELPER: BENDING LOSS & CONVERGENCE SOLVER ---

def calc_bending_loss_methods(neff, n_clad, lam_um, ring_radius_um, xc, yc, field_2d, dx, dy):
    neff_real = np.real(neff)
    if ring_radius_um <= 0.1:
        return {'loss_m1': 0.0, 'loss_m3': 0.0, 'r_min_um': 0.0, 'x_rad_um': 0.0, 'converged': True}

    k0 = 2.0 * np.pi / lam_um
    delta_n2 = max(neff_real**2 - n_clad**2, 1e-4)

    r_min_um = (3.0 * lam_um) / (4.0 * np.pi * (delta_n2**1.5))
    arg_m3 = (2.0 / 3.0) * k0 * ring_radius_um * (delta_n2**1.5) / (neff_real**2)
    loss_m3 = 500.0 * (lam_um / ring_radius_um) * np.exp(-arg_m3)

    x_rad = ring_radius_um * ((neff_real / n_clad) - 1.0) if neff_real > n_clad else 0.0
    
    XC, _ = np.meshgrid(xc, yc, indexing='ij')
    caustic_mask = XC >= x_rad
    
    f_abs2 = np.abs(field_2d)**2
    p_tot = np.sum(f_abs2) * dx * dy
    p_caustic = np.sum(f_abs2[caustic_mask]) * dx * dy if p_tot > 0 else 0.0
    tail_fraction = p_caustic / p_tot if p_tot > 0 else 0.0
    
    gamma = k0 * np.sqrt(delta_n2)
    loss_m1 = 4.343 * 2.0 * gamma * tail_fraction * 1e4

    rel_diff = abs(loss_m1 - loss_m3) / max(loss_m1, loss_m3, 1e-3)
    converged = bool(rel_diff < 0.35 or (loss_m1 < 0.1 and loss_m3 < 0.1))

    return {
        'loss_m1': float(loss_m1),
        'loss_m3': float(loss_m3),
        'r_min_um': float(r_min_um),
        'x_rad_um': float(x_rad),
        'converged': converged
    }

def calc_fwhm(vec, profile):
    p_real = np.abs(profile)
    half_max = np.max(p_real) / 2.0
    above = np.where(p_real >= half_max)[0]
    if len(above) >= 2:
        return float(vec[above[-1]] - vec[above[0]])
    return 0.0

# --- SINGLE POINT SOLVER ---

def run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice="Both (TE & TM)", 
                     search_higher_modes=False, sidewall_angle_deg=90.0, ring_radius_um=0.0, 
                     wg_type="Strip", h_slab=0.0, w_slab=0.0,
                     include_metal=False, metal_type="Al (Aluminum)", metal_thick_um=0.10, metal_width_um=2.0, metal_offset_um=0.0, metal_gap_um=1.0):
    
    dx = dy = 0.005 if "hr" in res_mode else (0.01 if "mr" in res_mode else 0.02)
    n_core = get_core_index(lam_um, core_material)
    n_clad = sellmeier_sio2(lam_um)
    
    xc, yc, eps_mesh, core_mask, air_mask, metal_mask, interface_y, x_min, x_max, y_min, y_max, w_bottom = build_advanced_mesh(
        w_core, h_core, bottom_ox, top_ox, 2.0, dx, dy, n_core, n_clad, sidewall_angle_deg, ring_radius_um, 
        wg_type, h_slab, w_slab, include_metal, metal_type, metal_thick_um, metal_width_um, metal_offset_um, metal_gap_um
    )
    
    res = {
        'xc': xc, 'yc': yc, 'eps_mesh': eps_mesh, 'core_mask': core_mask, 'metal_mask': metal_mask,
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'w_top': w_core, 'w_bottom': w_bottom, 'sidewall_angle_deg': sidewall_angle_deg,
        'ring_radius_um': ring_radius_um, 'wg_type': wg_type, 'h_slab': h_slab, 'w_slab': w_slab,
        'include_metal': include_metal, 'metal_type': metal_type, 'metal_thick_um': metal_thick_um,
        'metal_width_um': metal_width_um, 'metal_offset_um': metal_offset_um, 'metal_gap_um': metal_gap_um,
        'interface_y': interface_y, 'lam_um': lam_um, 'pol_choice': pol_choice,
        'te_modes': [], 'tm_modes': []
    }
    
    max_search_modes = 8 if search_higher_modes else 1
    
    mid_y_idx = np.argmin(np.abs(yc - (bottom_ox + (h_slab + h_core) / 2.0)))
    mid_x_idx = len(xc) // 2
    
    if pol_choice in ["TE", "Both (TE & TM)"]:
        guess_te = n_core - 0.01
        phi_te, neff_te = svmodes_2d(lam_um, guess_te, max_search_modes, dx, dy, eps_mesh, 'ex')
        for idx in range(max_search_modes):
            n_val = neff_te[idx]
            n_real = np.real(n_val)
            n_imag = np.imag(n_val)
            p_m = phi_te[:, :, idx]
            
            p_abs2 = np.abs(p_m)**2
            tot_p = np.sum(p_abs2)
            g_core = (np.sum(p_abs2[core_mask]) / tot_p) * 100.0 if tot_p > 0 else 0.0
            g_air = (np.sum(p_abs2[air_mask]) / tot_p) * 100.0 if tot_p > 0 else 0.0
            a_eff = ((tot_p * dx * dy)**2) / (np.sum(p_abs2**2) * dx * dy) if np.sum(p_abs2**2) > 0 else 0.0
            
            fwhm_x = calc_fwhm(xc, p_abs2[:, mid_y_idx])
            fwhm_y = calc_fwhm(yc, p_abs2[mid_x_idx, :])
            
            metal_loss_db_cm = 4.343 * (4.0 * np.pi / (lam_um * 1e-4)) * abs(n_imag) if include_metal else 0.0
            bend_info = calc_bending_loss_methods(n_val, n_clad, lam_um, ring_radius_um, xc, yc, p_abs2, dx, dy)
            
            if n_real > (n_clad + 0.001) and g_core > 5.0:
                res['te_modes'].append({
                    'mode_num': len(res['te_modes']),
                    'neff': n_real, 'neff_complex': n_val,
                    'metal_loss_db_cm': metal_loss_db_cm,
                    'field': p_abs2, 'field_complex': p_m,
                    'gamma_core': g_core, 'gamma_air': g_air,
                    'a_eff': a_eff, 'mfd': 2.0 * np.sqrt(a_eff / np.pi) if a_eff > 0 else 0.0,
                    'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y,
                    'bend_info': bend_info,
                    'cut_x': p_abs2[:, mid_y_idx], 'cut_y': p_abs2[mid_x_idx, :]
                })
                
    if pol_choice in ["TM", "Both (TE & TM)"]:
        guess_tm = n_core - 0.02
        phi_tm, neff_tm = svmodes_2d(lam_um, guess_tm, max_search_modes, dx, dy, eps_mesh, 'ey')
        for idx in range(max_search_modes):
            n_val = neff_tm[idx]
            n_real = np.real(n_val)
            n_imag = np.imag(n_val)
            p_m = phi_tm[:, :, idx]
            
            p_abs2 = np.abs(p_m)**2
            tot_p = np.sum(p_abs2)
            g_core = (np.sum(p_abs2[core_mask]) / tot_p) * 100.0 if tot_p > 0 else 0.0
            g_air = (np.sum(p_abs2[air_mask]) / tot_p) * 100.0 if tot_p > 0 else 0.0
            a_eff = ((tot_p * dx * dy)**2) / (np.sum(p_abs2**2) * dx * dy) if np.sum(p_abs2**2) > 0 else 0.0
            
            fwhm_x = calc_fwhm(xc, p_abs2[:, mid_y_idx])
            fwhm_y = calc_fwhm(yc, p_abs2[mid_x_idx, :])
            
            metal_loss_db_cm = 4.343 * (4.0 * np.pi / (lam_um * 1e-4)) * abs(n_imag) if include_metal else 0.0
            bend_info = calc_bending_loss_methods(n_val, n_clad, lam_um, ring_radius_um, xc, yc, p_abs2, dx, dy)
            
            if n_real > (n_clad + 0.001) and g_core > 5.0:
                res['tm_modes'].append({
                    'mode_num': len(res['tm_modes']),
                    'neff': n_real, 'neff_complex': n_val,
                    'metal_loss_db_cm': metal_loss_db_cm,
                    'field': p_abs2, 'field_complex': p_m,
                    'gamma_core': g_core, 'gamma_air': g_air,
                    'a_eff': a_eff, 'mfd': 2.0 * np.sqrt(a_eff / np.pi) if a_eff > 0 else 0.0,
                    'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y,
                    'bend_info': bend_info,
                    'cut_x': p_abs2[:, mid_y_idx], 'cut_y': p_abs2[mid_x_idx, :]
                })
                
    return res

# --- FAST 1D SWEEP ---

def run_1d_sweep(param_name, param_vec, fixed_params, res_mode, core_material, pol_choice="Both (TE & TM)", progress_callback=None):
    n_pts = len(param_vec)
    
    res = {
        'param_name': param_name, 'param_vec': param_vec, 'pol_choice': pol_choice,
        'sample_points': {},
        'neff_te': np.full(n_pts, np.nan), 'neff_tm': np.full(n_pts, np.nan),
        'metal_loss_te': np.full(n_pts, np.nan), 'metal_loss_tm': np.full(n_pts, np.nan),
        'loss_m1_te': np.full(n_pts, np.nan), 'loss_m3_te': np.full(n_pts, np.nan),
        'loss_m1_tm': np.full(n_pts, np.nan), 'loss_m3_tm': np.full(n_pts, np.nan),
        'r_min_te': np.full(n_pts, np.nan), 'r_min_tm': np.full(n_pts, np.nan),
        'n_core_vec': np.full(n_pts, np.nan), 'n_clad_vec': np.full(n_pts, np.nan),
        'gamma_core_te': np.full(n_pts, np.nan), 'gamma_core_tm': np.full(n_pts, np.nan),
        'gamma_air_te': np.full(n_pts, np.nan), 'gamma_air_tm': np.full(n_pts, np.nan),
        'a_eff_te': np.full(n_pts, np.nan), 'a_eff_tm': np.full(n_pts, np.nan)
    }
    
    sample_indices = [0, n_pts // 2, n_pts - 1]
    
    for i, val in enumerate(param_vec):
        if progress_callback: progress_callback(i + 1, n_pts)
        p_dict = fixed_params.copy()
        p_dict[param_name] = val
        
        lam_curr = p_dict['Wavelength']
        sidewall_angle = p_dict.get('Sidewall Angle', 90.0)
        ring_radius = p_dict.get('Ring Radius', 0.0)
        wg_type = p_dict.get('Profile Type', 'Strip')
        h_slab = p_dict.get('Slab Height', 0.0)
        w_slab = p_dict.get('Slab Width', 0.0)
        inc_metal = p_dict.get('Include Metal Heater', False)
        m_type = p_dict.get('Metal Type', 'Al (Aluminum)')
        m_thick = p_dict.get('Metal Thickness', 0.10)
        m_width = p_dict.get('Metal Width', 2.0)
        m_offset = p_dict.get('Metal Offset', 0.0)
        m_gap = p_dict.get('Metal Gap', 1.0)
        
        res['n_core_vec'][i] = get_core_index(lam_curr, core_material)
        res['n_clad_vec'][i] = sellmeier_sio2(lam_curr)
        
        sp_res = run_single_point(
            p_dict['Waveguide Width'], p_dict['Waveguide Height'],
            p_dict['Oxide Bottom Thickness'], p_dict['Oxide Top Thickness'],
            lam_curr, res_mode, core_material, pol_choice,
            search_higher_modes=False, sidewall_angle_deg=sidewall_angle, ring_radius_um=ring_radius,
            wg_type=wg_type, h_slab=h_slab, w_slab=w_slab,
            include_metal=inc_metal, metal_type=m_type, metal_thick_um=m_thick, metal_width_um=m_width, metal_offset_um=m_offset, metal_gap_um=m_gap
        )
        
        if i in sample_indices:
            tag = "Min" if i == 0 else ("Mid" if i == sample_indices[1] else "Max")
            res['sample_points'][tag] = {'val': val, 'res': sp_res}
            
        if len(sp_res['te_modes']) > 0:
            m0 = sp_res['te_modes'][0]
            res['neff_te'][i] = m0['neff']
            res['metal_loss_te'][i] = m0['metal_loss_db_cm']
            res['gamma_core_te'][i] = m0['gamma_core']
            res['gamma_air_te'][i] = m0['gamma_air']
            res['a_eff_te'][i] = m0['a_eff']
            res['loss_m1_te'][i] = m0['bend_info']['loss_m1']
            res['loss_m3_te'][i] = m0['bend_info']['loss_m3']
            res['r_min_te'][i] = m0['bend_info']['r_min_um']
            
        if len(sp_res['tm_modes']) > 0:
            m0 = sp_res['tm_modes'][0]
            res['neff_tm'][i] = m0['neff']
            res['metal_loss_tm'][i] = m0['metal_loss_db_cm']
            res['gamma_core_tm'][i] = m0['gamma_core']
            res['gamma_air_tm'][i] = m0['gamma_air']
            res['a_eff_tm'][i] = m0['a_eff']
            res['loss_m1_tm'][i] = m0['bend_info']['loss_m1']
            res['loss_m3_tm'][i] = m0['bend_info']['loss_m3']
            res['r_min_tm'][i] = m0['bend_info']['r_min_um']

    if param_name == "Wavelength" and n_pts >= 3:
        dlam = (param_vec[-1] - param_vec[0]) / (n_pts - 1)
        c_speed = 299792458.0
        
        if pol_choice in ["TE", "Both (TE & TM)"] and not np.all(np.isnan(res['neff_te'])):
            dneff = np.gradient(res['neff_te'], dlam)
            d2neff = np.gradient(dneff, dlam)
            res['ng_te'] = res['neff_te'] - param_vec * dneff
            res['D_te'] = -(param_vec * 1e-6 / c_speed) * (d2neff * 1e12) * 1e-3
            
        if pol_choice in ["TM", "Both (TE & TM)"] and not np.all(np.isnan(res['neff_tm'])):
            dneff = np.gradient(res['neff_tm'], dlam)
            d2neff = np.gradient(dneff, dlam)
            res['ng_tm'] = res['neff_tm'] - param_vec * dneff
            res['D_tm'] = -(param_vec * 1e-6 / c_speed) * (d2neff * 1e12) * 1e-3

    return res

# --- FAST 2D SWEEP ---

def run_2d_universal_sweep(param1_name, vec1, param2_name, vec2, fixed_params, res_mode, core_material, pol_choice="Both (TE & TM)", progress_callback=None):
    n1, n2 = len(vec1), len(vec2)
    total_sims = n1 * n2
    
    res = {
        'vec1': vec1, 'vec2': vec2,
        'param1_name': param1_name, 'param2_name': param2_name, 'pol_choice': pol_choice,
        'sample_points': {},
        'neff_te': np.zeros((n2, n1)), 'neff_tm': np.zeros((n2, n1)),
        'gamma_core_te': np.zeros((n2, n1)), 'gamma_core_tm': np.zeros((n2, n1)),
        'gamma_air_te': np.zeros((n2, n1)), 'gamma_air_tm': np.zeros((n2, n1))
    }
    
    mid_j, mid_i = n2 // 2, n1 // 2
    
    count = 0
    for j, val2 in enumerate(vec2):
        for i, val1 in enumerate(vec1):
            count += 1
            if progress_callback: progress_callback(count, total_sims)
            
            p_dict = fixed_params.copy()
            p_dict[param1_name] = val1
            p_dict[param2_name] = val2
            
            sidewall_angle = p_dict.get('Sidewall Angle', 90.0)
            ring_radius = p_dict.get('Ring Radius', 0.0)
            wg_type = p_dict.get('Profile Type', 'Strip')
            h_slab = p_dict.get('Slab Height', 0.0)
            w_slab = p_dict.get('Slab Width', 0.0)
            inc_metal = p_dict.get('Include Metal Heater', False)
            m_type = p_dict.get('Metal Type', 'Al (Aluminum)')
            m_thick = p_dict.get('Metal Thickness', 0.10)
            m_width = p_dict.get('Metal Width', 2.0)
            m_offset = p_dict.get('Metal Offset', 0.0)
            m_gap = p_dict.get('Metal Gap', 1.0)
            
            sp_res = run_single_point(
                p_dict['Waveguide Width'], p_dict['Waveguide Height'],
                p_dict['Oxide Bottom Thickness'], p_dict['Oxide Top Thickness'],
                p_dict['Wavelength'], res_mode, core_material, pol_choice,
                search_higher_modes=False, sidewall_angle_deg=sidewall_angle, ring_radius_um=ring_radius,
                wg_type=wg_type, h_slab=h_slab, w_slab=w_slab,
                include_metal=inc_metal, metal_type=m_type, metal_thick_um=m_thick, metal_width_um=m_width, metal_offset_um=m_offset, metal_gap_um=m_gap
            )
            
            if (j == 0 and i == 0):
                res['sample_points']['Min'] = {'p1': val1, 'p2': val2, 'res': sp_res}
            elif (j == mid_j and i == mid_i):
                res['sample_points']['Mid'] = {'p1': val1, 'p2': val2, 'res': sp_res}
            elif (j == n2 - 1 and i == n1 - 1):
                res['sample_points']['Max'] = {'p1': val1, 'p2': val2, 'res': sp_res}
            
            if len(sp_res['te_modes']) > 0:
                res['neff_te'][j, i] = sp_res['te_modes'][0]['neff']
                res['gamma_core_te'][j, i] = sp_res['te_modes'][0]['gamma_core']
                res['gamma_air_te'][j, i] = sp_res['te_modes'][0]['gamma_air']
                
            if len(sp_res['tm_modes']) > 0:
                res['neff_tm'][j, i] = sp_res['tm_modes'][0]['neff']
                res['gamma_core_tm'][j, i] = sp_res['tm_modes'][0]['gamma_core']
                res['gamma_air_tm'][j, i] = sp_res['tm_modes'][0]['gamma_air']
                
    return res
