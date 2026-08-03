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
    eps_inf = 1.0
    A = 1.912
    E = 0.09566
    P = 0.00306
    n_sq = eps_inf + (A * lam_um**2) / (lam_um**2 - E**2) - P * lam_um**2
    return np.sqrt(np.maximum(n_sq, 1.0))

def n_silicon(lam_um):
    """Crystalline Silicon (Malitson Sellmeier)"""
    n_sq = 1.0 + (10.6684293 * lam_um**2) / (lam_um**2 - 0.301516485**2) + \
                 (0.0030434748 * lam_um**2) / (lam_um**2 - 1.13475115**2) + \
                 (1.54133408 * lam_um**2) / (lam_um**2 - 1104.0**2)
    return np.sqrt(np.maximum(n_sq, 1.0))

def get_core_index(lam_um, material_name):
    if material_name == "Si3N4 (Stoichiometric)":
        return n_sin_stoch(lam_um)
    elif material_name == "SiN (Low Stress)":
        return n_sin_lowstress(lam_um)
    elif material_name == "Al2O3 (Alumina)":
        return n_al2o3(lam_um)
    elif material_name == "Si (Silicon)":
        return n_silicon(lam_um)
    else:
        return n_sin_stoch(lam_um)

# --- MESH GENERATION & INTEGRATION MASKS ---

def build_advanced_mesh(w_core, h_core, bottom_ox, top_ox, side_margin, dx, dy, n_core, n_clad):
    total_width = w_core + 2 * side_margin
    total_height = bottom_ox + h_core + top_ox + 1.5  # Extra space for air cladding
    
    nx = int(np.round(total_width / dx)) + 1
    ny = int(np.round(total_height / dy)) + 1
    
    x = np.linspace(-total_width / 2.0, total_width / 2.0, nx)
    y = np.linspace(0, total_height, ny)
    
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    
    eps = np.full((len(xc), len(yc)), n_clad**2)
    
    # Core Region
    x_min, x_max = -w_core / 2.0, w_core / 2.0
    y_min, y_max = bottom_ox, bottom_ox + h_core
    core_mask = (xc[:, None] >= x_min) & (xc[:, None] <= x_max) & \
                (yc[None, :] >= y_min) & (yc[None, :] <= y_max)
    eps[core_mask] = n_core**2
    
    # Air Cladding Boundary (Above Oxide Top)
    interface_y = bottom_ox + h_core + top_ox
    air_mask = yc[None, :] > interface_y
    eps[np.repeat(air_mask, len(xc), axis=0)] = 1.0**2  # Air n=1
    
    return xc, yc, eps, core_mask, air_mask, interface_y, x_min, x_max, y_min, y_max

# --- RIGOROUS 2D SVFD EIGENMODE SOLVER (TE & TM) ---

def svmodes_2d(lam_um, guess, nmodes, dx, dy, eps_mesh, polarization='ex'):
    nx, ny = eps_mesh.shape
    k0 = 2.0 * np.pi / lam_um
    eps_padded = np.pad(eps_mesh, ((1, 1), (1, 1)), mode='edge')
    
    ep = eps_padded[1:nx+1, 1:ny+1]
    en = eps_padded[1:nx+1, 2:ny+2]
    es = eps_padded[1:nx+1, 0:ny]
    ee = eps_padded[2:nx+2, 1:ny+1]
    ew = eps_padded[0:nx,   1:ny+1]
    
    n_mat = np.full((nx, ny), dy)
    s_mat = np.full((nx, ny), dy)
    e_mat = np.full((nx, ny), dx)
    w_mat = np.full((nx, ny), dx)
    p_mat = np.full((nx, ny), dx)
    q_mat = np.full((nx, ny), dy)
    
    if polarization.lower() == 'ex':
        # Quasi-TE Mode Solver (Ex main component)
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
        # Quasi-TM Mode Solver (Ey main component)
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
        if np.sum(mode_2d) < 0:
            mode_2d = -mode_2d
        max_abs = np.max(np.abs(mode_2d))
        if max_abs > 0:
            mode_2d /= max_abs
        phi_modes[:, :, idx] = mode_2d
        
    return phi_modes, neff_vals

# --- SIMULATION ENGINES ---

def run_single_point(w_core, h_core, bottom_ox, top_ox, lam_um, res_mode, core_material):
    dx = dy = 0.005 if "hr" in res_mode else (0.01 if "mr" in res_mode else 0.02)
    n_core = get_core_index(lam_um, core_material)
    n_clad = sellmeier_sio2(lam_um)
    
    xc, yc, eps_mesh, core_mask, air_mask_1d, interface_y, x_min, x_max, y_min, y_max = build_advanced_mesh(
        w_core, h_core, bottom_ox, top_ox, 2.0, dx, dy, n_core, n_clad
    )
    
    air_mask = np.repeat(air_mask_1d, len(xc), axis=0)
    
    # TE Mode Calculation
    guess_te = (n_core + n_clad) / 2.0
    phi_te, neff_te = svmodes_2d(lam_um, guess_te, 1, dx, dy, eps_mesh, 'ex')
    p_te = phi_te[:, :, 0]
    total_p_te = np.sum(p_te**2)
    gamma_core_te = (np.sum(p_te[core_mask]**2) / total_p_te) * 100.0 if total_p_te > 0 else 0.0
    gamma_air_te = (np.sum(p_te[air_mask]**2) / total_p_te) * 100.0 if total_p_te > 0 else 0.0
    
    # TM Mode Calculation (Independent Ey Solver)
    guess_tm = (n_core + 2*n_clad) / 3.0  # Slightly lower guess for TM fundamental
    phi_tm, neff_tm = svmodes_2d(lam_um, guess_tm, 1, dx, dy, eps_mesh, 'ey')
    p_tm = phi_tm[:, :, 0]
    total_p_tm = np.sum(p_tm**2)
    gamma_core_tm = (np.sum(p_tm[core_mask]**2) / total_p_tm) * 100.0 if total_p_tm > 0 else 0.0
    gamma_air_tm = (np.sum(p_tm[air_mask]**2) / total_p_tm) * 100.0 if total_p_tm > 0 else 0.0
    
    return {
        'xc': xc, 'yc': yc, 'eps_mesh': eps_mesh,
        'phi_te': p_te, 'phi_tm': p_tm,
        'neff_te': neff_te[0], 'neff_tm': neff_tm[0],
        'gamma_core_te': gamma_core_te, 'gamma_core_tm': gamma_core_tm,
        'gamma_air_te': gamma_air_te, 'gamma_air_tm': gamma_air_tm,
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'interface_y': interface_y, 'lam_um': lam_um
    }

def run_1d_sweep(param_name, param_vec, fixed_params, res_mode, core_material, progress_callback=None):
    n_pts = len(param_vec)
    neff_te_vec = np.zeros(n_pts)
    neff_tm_vec = np.zeros(n_pts)
    gamma_core_te_vec = np.zeros(n_pts)
    gamma_core_tm_vec = np.zeros(n_pts)
    gamma_air_te_vec = np.zeros(n_pts)
    gamma_air_tm_vec = np.zeros(n_pts)
    
    for i, val in enumerate(param_vec):
        if progress_callback:
            progress_callback(i + 1, n_pts)
            
        p_dict = fixed_params.copy()
        p_dict[param_name] = val
        
        w_c = p_dict['Waveguide Width']
        h_c = p_dict['Waveguide Height']
        lam = p_dict['Wavelength']
        t_ox = p_dict['Oxide Top Thickness']
        b_ox = p_dict['Oxide Bottom Thickness']
        
        res = run_single_point(w_c, h_c, b_ox, t_ox, lam, res_mode, core_material)
        
        neff_te_vec[i] = res['neff_te']
        neff_tm_vec[i] = res['neff_tm']
        gamma_core_te_vec[i] = res['gamma_core_te']
        gamma_core_tm_vec[i] = res['gamma_core_tm']
        gamma_air_te_vec[i] = res['gamma_air_te']
        gamma_air_tm_vec[i] = res['gamma_air_tm']
        
    return {
        'param_name': param_name, 'param_vec': param_vec,
        'neff_te_vec': neff_te_vec, 'neff_tm_vec': neff_tm_vec,
        'gamma_core_te_vec': gamma_core_te_vec, 'gamma_core_tm_vec': gamma_core_tm_vec,
        'gamma_air_te_vec': gamma_air_te_vec, 'gamma_air_tm_vec': gamma_air_tm_vec
    }

def run_2d_universal_sweep(param1_name, vec1, param2_name, vec2, fixed_params, res_mode, core_material, progress_callback=None):
    n1, n2 = len(vec1), len(vec2)
    total_sims = n1 * n2
    
    neff_te_mat = np.zeros((n2, n1))
    neff_tm_mat = np.zeros((n2, n1))
    gamma_core_te_mat = np.zeros((n2, n1))
    gamma_core_tm_mat = np.zeros((n2, n1))
    gamma_air_te_mat = np.zeros((n2, n1))
    gamma_air_tm_mat = np.zeros((n2, n1))
    
    count = 0
    for j, val2 in enumerate(vec2):
        for i, val1 in enumerate(vec1):
            count += 1
            if progress_callback:
                progress_callback(count, total_sims)
                
            p_dict = fixed_params.copy()
            p_dict[param1_name] = val1
            p_dict[param2_name] = val2
            
            w_c = p_dict['Waveguide Width']
            h_c = p_dict['Waveguide Height']
            lam = p_dict['Wavelength']
            t_ox = p_dict['Oxide Top Thickness']
            b_ox = p_dict['Oxide Bottom Thickness']
            
            res = run_single_point(w_c, h_c, b_ox, t_ox, lam, res_mode, core_material)
            
            neff_te_mat[j, i] = res['neff_te']
            neff_tm_mat[j, i] = res['neff_tm']
            gamma_core_te_mat[j, i] = res['gamma_core_te']
            gamma_core_tm_mat[j, i] = res['gamma_core_tm']
            gamma_air_te_mat[j, i] = res['gamma_air_te']
            gamma_air_tm_mat[j, i] = res['gamma_air_tm']
            
    return {
        'vec1': vec1, 'vec2': vec2,
        'param1_name': param1_name, 'param2_name': param2_name,
        'gamma_core_te_mat': gamma_core_te_mat, 'gamma_core_tm_mat': gamma_core_tm_mat,
        'gamma_air_te_mat': gamma_air_te_mat, 'gamma_air_tm_mat': gamma_air_tm_mat,
        'neff_te_mat': neff_te_mat, 'neff_tm_mat': neff_tm_mat
    }
