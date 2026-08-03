import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- REFRACTIVE INDEX DISPERSION MODELS ---

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

# --- MESH GENERATION ---

def build_advanced_mesh(w_core, h_core, bottom_ox, top_ox, side_margin, dx, dy, n_core, n_clad):
    total_width = w_core + 2 * side_margin
    total_height = bottom_ox + h_core + top_ox + 1.5
    
    nx = int(np.round(total_width / dx)) + 1
    ny = int(np.round(total_height / dy)) + 1
    
    x = np.linspace(-total_width / 2.0, total_width / 2.0, nx)
    y = np.linspace(0, total_height, ny)
    
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    
    eps = np.full((len(xc), len(yc)), n_clad**2)
    
    x_min, x_max = -w_core / 2.0, w_core / 2.0
    y_min, y_max = bottom_ox, bottom_ox + h_core
    core_mask = (xc[:, None] >= x_min) & (xc[:, None] <= x_max) & \
                (yc[None, :] >= y_min) & (yc[None, :] <= y_max)
    eps[core_mask] = n_core**2
    
    interface_y = bottom_ox + h_core + top_ox
    air_mask_1d = yc > interface_y
    eps[:, air_mask_1d] = 1.0**2  # Air n=1
    
    air_mask = np.repeat(air_mask_1d[None, :], len(xc), axis=0)
    
    return xc, yc, eps, core_mask, air_mask, interface_y, x_min, x_max, y_min, y_max

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
    
    neff_vals = (lam_um / (2.0 * np.pi)) * np.sqrt(np.real(vals))
    sorted_indices = np.argsort(neff_vals)[::-1]
    neff_vals = neff_vals[sorted_indices]
    
    phi_modes = np.zeros((nx, ny, nmodes))
    for idx in range(nmodes):
        s_idx = sorted_indices[idx]
        mode_2d = np.real(vecs[:, s_idx]).reshape((nx, ny), order='F')
        if np.sum(mode_2d) < 0: mode_2d = -mode_2d
        max_abs = np.max(np.abs(mode_2d))
        if max_abs > 0: mode_2d /= max_abs
        phi_modes[:, :, idx] = mode_2d
        
    return phi_modes, neff_vals

# --- HELPER: FWHM CALCULATION ---
def calc_fwhm(vec, profile):
    half_max = np.max(profile) / 2.0
    above = np.where(profile >= half_max)[0]
    if len(above) >= 2:
        return float(vec[above[-1]] - vec[above[0]])
    return 0.0

# --- SINGLE POINT SOLVER ---
def run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material, pol_choice="Both (TE & TM)", search_higher_modes=False):
    dx = dy = 0.005 if "hr" in res_mode else (0.01 if "mr" in res_mode else 0.02)
    n_core = get_core_index(lam_um, core_material)
    n_clad = sellmeier_sio2(lam_um)
    
    xc, yc, eps_mesh, core_mask, air_mask, interface_y, x_min, x_max, y_min, y_max = build_advanced_mesh(
        w_core, h_core, bottom_ox, top_ox, 2.0, dx, dy, n_core, n_clad
    )
    
    res = {
        'xc': xc, 'yc': yc, 'eps_mesh': eps_mesh,
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'interface_y': interface_y, 'lam_um': lam_um, 'pol_choice': pol_choice,
        'te_modes': [], 'tm_modes': []
    }
    
    max_search_modes = 8 if search_higher_modes else 1
    
    mid_y_idx = np.argmin(np.abs(yc - (bottom_ox + h_core / 2.0)))
    mid_x_idx = len(xc) // 2
    
    if pol_choice in ["TE", "Both (TE & TM)"]:
        guess_te = n_core - 0.01
        phi_te, neff_te = svmodes_2d(lam_um, guess_te, max_search_modes, dx, dy, eps_mesh, 'ex')
        for idx in range(max_search_modes):
            n_val = neff_te[idx]
            p_m = phi_te[:, :, idx]
            tot_p = np.sum(p_m**2)
            g_core = (np.sum(p_m[core_mask]**2) / tot_p) * 100.0 if tot_p > 0 else 0.0
            g_air = (np.sum(p_m[air_mask]**2) / tot_p) * 100.0 if tot_p > 0 else 0.0
            a_eff = ((tot_p * dx * dy)**2) / (np.sum(p_m**4) * dx * dy) if np.sum(p_m**4) > 0 else 0.0
            
            fwhm_x = calc_fwhm(xc, p_m[:, mid_y_idx]**2)
            fwhm_y = calc_fwhm(yc, p_m[mid_x_idx, :]**2)
            
            if n_val > (n_clad + 0.001) and g_core > 5.0:
                res['te_modes'].append({
                    'mode_num': len(res['te_modes']),
                    'neff': n_val, 'field': p_m,
                    'gamma_core': g_core, 'gamma_air': g_air,
                    'a_eff': a_eff, 'mfd': 2.0 * np.sqrt(a_eff / np.pi) if a_eff > 0 else 0.0,
                    'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y,
                    'cut_x': p_m[:, mid_y_idx], 'cut_y': p_m[mid_x_idx, :]
                })
                
    if pol_choice in ["TM", "Both (TE & TM)"]:
        guess_tm = n_core - 0.02
        phi_tm, neff_tm = svmodes_2d(lam_um, guess_tm, max_search_modes, dx, dy, eps_mesh, 'ey')
        for idx in range(max_search_modes):
            n_val = neff_tm[idx]
            p_m = phi_tm[:, :, idx]
            tot_p = np.sum(p_m**2)
            g_core = (np.sum(p_m[core_mask]**2) / tot_p) * 100.0 if tot_p > 0 else 0.0
            g_air = (np.sum(p_m[air_mask]**2) / tot_p) * 100.0 if tot_p > 0 else 0.0
            a_eff = ((tot_p * dx * dy)**2) / (np.sum(p_m**4) * dx * dy) if np.sum(p_m**4) > 0 else 0.0
            
            fwhm_x = calc_fwhm(xc, p_m[:, mid_y_idx]**2)
            fwhm_y = calc_fwhm(yc, p_m[mid_x_idx, :]**2)
            
            if n_val > (n_clad + 0.001) and g_core > 5.0:
                res['tm_modes'].append({
                    'mode_num': len(res['tm_modes']),
                    'neff': n_val, 'field': p_m,
                    'gamma_core': g_core, 'gamma_air': g_air,
                    'a_eff': a_eff, 'mfd': 2.0 * np.sqrt(a_eff / np.pi) if a_eff > 0 else 0.0,
                    'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y,
                    'cut_x': p_m[:, mid_y_idx], 'cut_y': p_m[mid_x_idx, :]
                })
                
    return res

# --- FAST 1D SWEEP (FUNDAMENTAL MODE ONLY) ---
def run_1d_sweep(param_name, param_vec, fixed_params, res_mode, core_material, pol_choice="Both (TE & TM)", progress_callback=None):
    n_pts = len(param_vec)
    
    res = {
        'param_name': param_name, 'param_vec': param_vec, 'pol_choice': pol_choice,
        'sample_points': {},
        'neff_te': np.full(n_pts, np.nan), 'neff_tm': np.full(n_pts, np.nan),
        'gamma_core_te': np.full(n_pts, np.nan), 'gamma_core_tm': np.full(n_pts, np.nan),
        'gamma_air_te': np.full(n_pts, np.nan), 'gamma_air_tm': np.full(n_pts, np.nan),
        'a_eff_te': np.full(n_pts, np.nan), 'a_eff_tm': np.full(n_pts, np.nan)
    }
    
    sample_indices = [0, n_pts // 2, n_pts - 1]
    
    for i, val in enumerate(param_vec):
        if progress_callback: progress_callback(i + 1, n_pts)
        p_dict = fixed_params.copy()
        p_dict[param_name] = val
        
        # Fast execution: search_higher_modes = False
        sp_res = run_single_point(
            p_dict['Waveguide Width'], p_dict['Waveguide Height'],
            p_dict['Oxide Bottom Thickness'], p_dict['Oxide Top Thickness'],
            p_dict['Wavelength'], res_mode, core_material, pol_choice,
            search_higher_modes=False
        )
        
        if i in sample_indices:
            tag = "Min" if i == 0 else ("Mid" if i == sample_indices[1] else "Max")
            res['sample_points'][tag] = {'val': val, 'res': sp_res}
            
        if len(sp_res['te_modes']) > 0:
            m0 = sp_res['te_modes'][0]
            res['neff_te'][i] = m0['neff']
            res['gamma_core_te'][i] = m0['gamma_core']
            res['gamma_air_te'][i] = m0['gamma_air']
            res['a_eff_te'][i] = m0['a_eff']
            
        if len(sp_res['tm_modes']) > 0:
            m0 = sp_res['tm_modes'][0]
            res['neff_tm'][i] = m0['neff']
            res['gamma_core_tm'][i] = m0['gamma_core']
            res['gamma_air_tm'][i] = m0['gamma_air']
            res['a_eff_tm'][i] = m0['a_eff']

    # Dispersion & ng calculation for Wavelength sweep
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

# --- FAST 2D SWEEP (FUNDAMENTAL MODE ONLY) ---
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
            
            # Fast execution: search_higher_modes = False
            sp_res = run_single_point(
                p_dict['Waveguide Width'], p_dict['Waveguide Height'],
                p_dict['Oxide Bottom Thickness'], p_dict['Oxide Top Thickness'],
                p_dict['Wavelength'], res_mode, core_material, pol_choice,
                search_higher_modes=False
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
