# clean_gpm_bvar_trends/simulation_smoothing.py

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpyro # Retaining if MCMC object passed needs it for type or methods
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import xarray as xr 
import arviz as az

# Local imports from the new refactored structure
from .common_types import EnhancedBVARParams # Used by _extract_parameters_for_ss_builder
from .gpm_model_parser import ReducedModel, ReducedExpression # Crucial for reconstruction
from .state_space_builder import StateSpaceBuilder           # For evaluating expressions
# ParameterContract not directly used here, but ss_builder uses it internally
from .constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER
from .Kalman_filter_jax import KalmanFilter # Note the leading dot for relative import

# --- stationary_prior_jax_simplified import ---
try:
    from .stationary_prior_jax_simplified import make_stationary_var_transformation_jax, _JITTER as _SP_JITTER
except ImportError:
    print("CRITICAL ERROR: stationary_prior_jax_simplified.py not found or cannot be imported.")
    make_stationary_var_transformation_jax = None
    _SP_JITTER = 1e-8 # Fallback, though functionality will be impaired


# --- Jarocinski Durbin & Koopman Simulation Smoother Components ---

def jarocinski_corrected_simulation_smoother(
    y: jnp.ndarray, 
    F: jnp.ndarray, 
    R_ss: jnp.ndarray, # Shock impact matrix (n_states x n_shocks_actual)
    C: jnp.ndarray, 
    H: jnp.ndarray,    # Observation noise covariance
    init_mean: jnp.ndarray, 
    init_cov: jnp.ndarray,
    key: jax.Array  # Changed from jnp.ndarray for PRNGKey
) -> jax.Array:       # Changed from jnp.ndarray
    """
    Jarocinski (2015) corrected Durbin & Koopman simulation smoother.
    Assumes R_ss is the shock impact matrix such that state noise cov Q = R_ss @ R_ss.T
    (if R_ss itself is Cholesky of Q) or Q = R_ss @ Sigma_eta_diag @ R_ss.T (if R_ss selects shocks).
    For this smoother, typically R_ss is such that R_ss @ N(0,I) generates the state noise.
    So, if Q is state noise covariance, R_ss would be its Cholesky factor.
    """
    T_local, _ = y.shape # n_obs not directly used here but in helpers
    state_dim_local = F.shape[0]
    
    if T_local == 0:
        return jnp.empty((0, state_dim_local), dtype=_DEFAULT_DTYPE)
    
    key_local, step1_key_local = random.split(key)
    zero_init_mean_local = jnp.zeros_like(init_mean)
    
    alpha_plus, y_plus = simulate_forward_with_zero_mean(
        F, R_ss, C, H, zero_init_mean_local, init_cov, T_local, step1_key_local
    )
    
    y_star = y - y_plus
    
    alpha_hat_star = compute_smoothed_expectation(
        y_star, F, R_ss, C, H, init_mean, init_cov
    )
    
    alpha_tilde = alpha_hat_star + alpha_plus
    return alpha_tilde


def simulate_forward_with_zero_mean(
    F: jnp.ndarray, 
    R_ss: jnp.ndarray, # Shock impact matrix (n_states x n_shocks_actual)
    C: jnp.ndarray, 
    H: jnp.ndarray,
    zero_init_mean: jnp.ndarray, 
    init_cov: jnp.ndarray, 
    T_sim: int, 
    key_sim: jax.Array # Changed from jnp.ndarray
) -> Tuple[jax.Array, jax.Array]: # Changed from jnp.ndarray
    """
    Forward simulation with zero initial mean, using R_ss for shocks.
    """
    state_dim_fwd = F.shape[0]
    n_obs_fwd = C.shape[0]
    n_actual_shocks = R_ss.shape[1] if R_ss.ndim == 2 and R_ss.shape[0] == state_dim_fwd else 0

    alpha_p = jnp.zeros((T_sim, state_dim_fwd), dtype=_DEFAULT_DTYPE)
    y_p = jnp.zeros((T_sim, n_obs_fwd), dtype=_DEFAULT_DTYPE)
    
    current_key, init_key_fwd = random.split(key_sim)
    init_cov_reg_fwd = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim_fwd, dtype=_DEFAULT_DTYPE)
    
    try:
        alpha_0_fwd = random.multivariate_normal(init_key_fwd, zero_init_mean, init_cov_reg_fwd, dtype=_DEFAULT_DTYPE)
        if not jnp.all(jnp.isfinite(alpha_0_fwd)): alpha_0_fwd = zero_init_mean
    except Exception: # Fallback if MVN fails (e.g., cov not PSD despite jitter)
        alpha_0_fwd = zero_init_mean
    
    current_s = alpha_0_fwd
    
    for t_idx in range(T_sim):
        alpha_p = alpha_p.at[t_idx].set(current_s)
        current_key, obs_key_fwd = random.split(current_key)
        obs_m = C @ current_s
        
        # Add jitter to H for sampling robustness
        H_reg_fwd = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_obs_fwd, dtype=_DEFAULT_DTYPE)
        try:
            y_t_fwd = random.multivariate_normal(obs_key_fwd, obs_m, H_reg_fwd, dtype=_DEFAULT_DTYPE)
            if not jnp.all(jnp.isfinite(y_t_fwd)): y_t_fwd = obs_m
        except Exception: y_t_fwd = obs_m
        y_p = y_p.at[t_idx].set(y_t_fwd)
        
        if t_idx < T_sim - 1:
            current_key, state_key_fwd = random.split(current_key)
            state_noise_term = jnp.zeros(state_dim_fwd, dtype=_DEFAULT_DTYPE)
            if n_actual_shocks > 0:
                try:
                    eta_t_fwd = random.normal(state_key_fwd, shape=(n_actual_shocks,), dtype=_DEFAULT_DTYPE)
                    state_noise_term = R_ss @ eta_t_fwd
                    if not jnp.all(jnp.isfinite(state_noise_term)): state_noise_term = jnp.zeros_like(state_noise_term)
                except Exception: pass # state_noise_term remains zeros
            current_s = F @ current_s + state_noise_term
    return alpha_p, y_p


def compute_smoothed_expectation(
    y_star: jnp.ndarray, 
    F: jnp.ndarray, 
    R_ss: jnp.ndarray, # Shock impact matrix
    C: jnp.ndarray, 
    H: jnp.ndarray,
    init_mean: jnp.ndarray, 
    init_cov: jnp.ndarray
) -> jax.Array: # Changed from jnp.ndarray
    """
    Compute E(α|y*) using Kalman filter/smoother with original initial conditions.
    """
    T_exp, n_obs_exp = y_star.shape
    state_dim_exp = F.shape[0]
    
    if KalmanFilter is None:
        print("ERROR: KalmanFilter not imported/available in compute_smoothed_expectation.")
        return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE) # Or raise error

    kf_instance = KalmanFilter(T=F, R=R_ss, C=C, H=H, init_x=init_mean, init_P=init_cov)
    
    valid_obs_idx_static = jnp.arange(n_obs_exp, dtype=jnp.int32)
    I_obs_static = jnp.eye(n_obs_exp, dtype=_DEFAULT_DTYPE)
    
    try:
        filter_results = kf_instance.filter(
            y_star, 
            static_valid_obs_idx=valid_obs_idx_static,
            static_n_obs_actual=n_obs_exp,
            static_C_obs=C, 
            static_H_obs=H, 
            static_I_obs=I_obs_static
        )
        smoothed_means, _ = kf_instance.smooth( 
            y_star,
            filter_results=filter_results,
            static_valid_obs_idx=valid_obs_idx_static, # Pass if smoother needs them for internal filter re-run or checks
            static_n_obs_actual=n_obs_exp,
            static_C_obs_for_filter=C, 
            static_H_obs_for_filter=H,
            static_I_obs_for_filter=I_obs_static
        )
        if not jnp.all(jnp.isfinite(smoothed_means)):
            # print("Warning: Smoothed expectation resulted in non-finite values.")
            return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE)
        return smoothed_means
    except Exception as e_smooth_exp:
        # print(f"Error during smoothed expectation computation: {e_smooth_exp}")
        return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE)


# --- MCMC Helper/Interface functions (from your original simulation_smoothing.py) ---
def _identify_required_sites(mcmc_samples: Dict, gpm_model_struct: ReducedModel) -> list:
    # (As provided before, ensure names match what gpm_numpyro_models.py samples)
    required = []; 
    if hasattr(gpm_model_struct, 'trend_shocks'):
        for shock_bn in gpm_model_struct.trend_shocks:
            mcmc_sn = f"sigma_{shock_bn}"; 
            if mcmc_sn in mcmc_samples: required.append(mcmc_sn)
    if hasattr(gpm_model_struct, 'stationary_shocks'):
        for shock_bn in gpm_model_struct.stationary_shocks:
            mcmc_sn = f"sigma_{shock_bn}"; 
            if mcmc_sn in mcmc_samples: required.append(mcmc_sn)
    if hasattr(gpm_model_struct, 'var_prior_setup') and gpm_model_struct.var_prior_setup:
        if "A_transformed" in mcmc_samples: required.append("A_transformed")
        if "Omega_u_chol" in mcmc_samples: required.append("Omega_u_chol")
    if hasattr(gpm_model_struct, 'parameters'):
        for param_n in gpm_model_struct.parameters:
            if param_n in mcmc_samples: required.append(param_n)
    if "init_mean_full" in mcmc_samples: required.append("init_mean_full")
    if not required and mcmc_samples: return [list(mcmc_samples.keys())[0]] # Fallback
    if not required: return ["DUMMY_FALLBACK_SITE"] # Prevent crash on empty
    return list(set(required))

def _extract_parameters_for_ss_builder( 
    mcmc_samples_dict: Dict,          
    mcmc_draw_idx: int, 
    gpm_model_struct: ReducedModel,   
    ) -> EnhancedBVARParams:
    
    struct_p_draw = {}
    if hasattr(gpm_model_struct, 'parameters'):
        for p_n in gpm_model_struct.parameters:
            if p_n in mcmc_samples_dict and mcmc_samples_dict[p_n].shape[0] > mcmc_draw_idx:
                struct_p_draw[p_n] = mcmc_samples_dict[p_n][mcmc_draw_idx]

    trend_s_vals = []
    if hasattr(gpm_model_struct, 'trend_shocks'):
        for s_bn in gpm_model_struct.trend_shocks:
            m_sn = f"sigma_{s_bn}"
            if m_sn in mcmc_samples_dict and mcmc_samples_dict[m_sn].shape[0] > mcmc_draw_idx:
                trend_s_vals.append(mcmc_samples_dict[m_sn][mcmc_draw_idx])
            else: trend_s_vals.append(0.01) 
    n_core_t = len(gpm_model_struct.core_variables) - len(gpm_model_struct.stationary_variables)
    
    if len(trend_s_vals) < n_core_t : trend_s_vals.extend([0.01]*(n_core_t - len(trend_s_vals)))
    elif len(trend_s_vals) > n_core_t : trend_s_vals = trend_s_vals[:n_core_t]
    Sigma_eta_d = jnp.diag(jnp.array(trend_s_vals, dtype=_DEFAULT_DTYPE)**2) if trend_s_vals else jnp.eye(max(0,n_core_t)) * 0.001

    A_trans_d = mcmc_samples_dict.get("A_transformed")
    A_trans_d = A_trans_d[mcmc_draw_idx] if A_trans_d is not None and A_trans_d.shape[0] > mcmc_draw_idx else None
    if A_trans_d is None: 
        n_s = len(gpm_model_struct.stationary_variables)
        vo = gpm_model_struct.var_prior_setup.var_order if gpm_model_struct.var_prior_setup else 1
        A_trans_d = jnp.zeros((vo, n_s, n_s))

    Omega_uc_d = mcmc_samples_dict.get("Omega_u_chol")
    Sigma_u_d = None; n_s_vars = len(gpm_model_struct.stationary_variables)
    if n_s_vars > 0:
        stat_s_vals = []
        if hasattr(gpm_model_struct, 'stationary_shocks'):
            for s_bn in gpm_model_struct.stationary_shocks: # Using GPM defined shock names
                m_sn = f"sigma_{s_bn}"
                if m_sn in mcmc_samples_dict and mcmc_samples_dict[m_sn].shape[0] > mcmc_draw_idx:
                    stat_s_vals.append(mcmc_samples_dict[m_sn][mcmc_draw_idx])
                else: stat_s_vals.append(0.01)
        if len(stat_s_vals) < n_s_vars: stat_s_vals.extend([0.01]*(n_s_vars - len(stat_s_vals)))
        elif len(stat_s_vals) > n_s_vars: stat_s_vals = stat_s_vals[:n_s_vars]
        
        s_u_v = jnp.array(stat_s_vals, dtype=_DEFAULT_DTYPE)
        Om_uc = Omega_uc_d[mcmc_draw_idx] if Omega_uc_d is not None and Omega_uc_d.shape[0] > mcmc_draw_idx else None
        if Om_uc is not None and Om_uc.shape == (n_s_vars, n_s_vars):
            Sigma_u_d = jnp.diag(s_u_v) @ Om_uc @ Om_uc.T @ jnp.diag(s_u_v)
            Sigma_u_d = (Sigma_u_d + Sigma_u_d.T) / 2.0 + _JITTER * jnp.eye(n_s_vars)
        else: Sigma_u_d = jnp.diag(s_u_v**2) + _JITTER * jnp.eye(n_s_vars)
    else: Sigma_u_d = jnp.empty((0,0))
    
    return EnhancedBVARParams(A=A_trans_d, Sigma_u=Sigma_u_d, Sigma_eta=Sigma_eta_d,
                              structural_params=struct_p_draw, Sigma_eps=None)


def _extract_initial_mean(mcmc_samples_dict: Dict, mcmc_draw_idx: int, state_dim: int) -> jnp.ndarray: 
    if "init_mean_full" in mcmc_samples_dict and mcmc_samples_dict["init_mean_full"].shape[0] > mcmc_draw_idx:
        init_mean = mcmc_samples_dict["init_mean_full"][mcmc_draw_idx]
        return jnp.where(jnp.isfinite(init_mean), init_mean, jnp.zeros_like(init_mean))
    return jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)

def _create_reasonable_initial_covariance(state_dim: int, n_dynamic_trends_in_state: int) -> jnp.ndarray:
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 10.0 
    if state_dim > n_dynamic_trends_in_state: 
        init_cov = init_cov.at[n_dynamic_trends_in_state:, n_dynamic_trends_in_state:].set(
            jnp.eye(state_dim - n_dynamic_trends_in_state, dtype=_DEFAULT_DTYPE) * 1.0 
        )
    return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

# --- HDI Computation ---
def _compute_and_format_hdi_az(draws: jnp.ndarray, hdi_prob: float = 0.9) -> Dict[str, np.ndarray]:
    if not isinstance(draws, np.ndarray): 
        draws_np = np.asarray(draws)
    else: 
        draws_np = draws
    
    if draws_np.ndim < 1 or draws_np.shape[0] < 2 :
        nan_shape = draws_np.shape[1:] if draws_np.ndim > 1 else (1,) * max(1, draws_np.ndim-1)
        if not nan_shape : 
            nan_shape = (1,) 
        return {'low': np.full(nan_shape, np.nan, dtype=_DEFAULT_DTYPE),'high': np.full(nan_shape, np.nan, dtype=_DEFAULT_DTYPE)}

    original_shape_after_draws = draws_np.shape[1:]
    num_elements_per_draw = int(np.prod(original_shape_after_draws)) if original_shape_after_draws else 0
    
    if num_elements_per_draw == 0: 
         return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE),
                'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}
    try:
        draws_reshaped = draws_np.reshape(draws_np.shape[0], num_elements_per_draw)
        hdi_output_reshaped = az.hdi(draws_reshaped, hdi_prob=hdi_prob, axis=0)

        hdi_for_reshape = hdi_output_reshaped.T if hdi_output_reshaped.shape == (num_elements_per_draw, 2) else hdi_output_reshaped
        
        if hdi_for_reshape.shape != (2, num_elements_per_draw): 
            raise ValueError("HDI shape mismatch after potential transpose.")
        
        final_hdi_shape = (2,) + original_shape_after_draws
        hdi_full_shape = hdi_for_reshape.reshape(final_hdi_shape)
        low = np.asarray(hdi_full_shape[0, ...]); high = np.asarray(hdi_full_shape[1, ...])
        if np.any(np.isnan(low)) or np.any(np.isnan(high)): pass 
        return {'low': low, 'high': high}
    
    except Exception as e:
        # print(f"Error during ArviZ HDI computation: {e}")
        return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE), 'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}
    

def debug_specific_mcmc_draws(mcmc_results, draw_indices):
    """Debug specific MCMC draws that will be used by the smoother"""
    print(f"\n=== DEBUGGING SPECIFIC MCMC DRAWS ===")
    
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    
    key_params = ['sigma_shk_cycle_y_us', 'sigma_shk_trend_y_us', 'init_mean_full'] # Example keys
    
    for i, mcmc_idx in enumerate(draw_indices[:5]):  # Check first 5
        print(f"\nDraw {i} (MCMC index {mcmc_idx}):")
        
        for param_name in key_params:
            if param_name in mcmc_samples:
                param_array = mcmc_samples[param_name]
                if mcmc_idx < param_array.shape[0]: # Check bounds
                    if param_name == 'init_mean_full':
                        print(f"  {param_name}[0]: {param_array[mcmc_idx][0]:.6f}")
                    else:
                        print(f"  {param_name}: {param_array[mcmc_idx]:.6f}")
                else:
                    print(f"  {param_name}: Index out of bounds for this draw.")
            else:
                print(f"  {param_name}: Not found in MCMC samples.")
    
    print("=== END DEBUGGING DRAWS ===\n")


def extract_reconstructed_components_fixed(
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray, 
    gpm_model: ReducedModel, # MODIFIED: Added gpm_model here
    ss_builder: StateSpaceBuilder,
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None,
    use_gamma_init_for_smoother: bool = True,  
    gamma_init_scaling_for_smoother: float = 1.0 
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
    """
    FIXED VERSION: 
    1. Uses draw-specific gamma-based P0 initialization
    2. Correctly maps variables using ss_builder.core_var_map
    3. _extract_gamma_matrices_for_draw uses gpm_model for shock names
    """
    print(f"\n=== FIXED SIMULATION SMOOTHER ===")
    print(f"Use gamma-based P0: {use_gamma_init_for_smoother}")
    print(f"Gamma scaling: {gamma_init_scaling_for_smoother}")
    
    if rng_key_smooth is None: 
        rng_key_smooth = random.PRNGKey(0)

    mcmc_samples = mcmc_output.get_samples()
    if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
        print("Warning: No MCMC samples available")
        return jnp.empty((0,0,0)), jnp.empty((0,0,0)), {'trends':[], 'stationary':[]}

    T_data, _ = y_data.shape
    
    first_param_key = list(mcmc_samples.keys())[0]
    total_posterior_draws = mcmc_samples[first_param_key].shape[0]
    actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

    if actual_num_smooth_draws <= 0:
        return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
               jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
               {'trends': list(gpm_model.gpm_trend_variables_original), 
                'stationary': list(gpm_model.gpm_stationary_variables_original)}

    if actual_num_smooth_draws == total_posterior_draws:
        draw_indices = np.arange(total_posterior_draws)
    else:
        draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
    print(f"Processing {actual_num_smooth_draws} draws from indices: {draw_indices[:5]}{'...' if len(draw_indices) > 5 else ''}")
    
    output_trend_draws_list = []
    output_stationary_draws_list = []
    successful_draws = 0

    for i_loop, mcmc_draw_idx in enumerate(draw_indices):
        rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
        current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
        init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
        if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
            init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]
        else:
            init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
        if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
            gamma_list_for_this_draw = _extract_gamma_matrices_for_draw(
                mcmc_samples, mcmc_draw_idx, current_builder_params, 
                gpm_model, # MODIFIED: Pass gpm_model
                ss_builder.n_stationary, ss_builder.var_order
            )
            
            if gamma_list_for_this_draw is not None:
                if i_loop < 2 : print(f"  Draw {i_loop}: Using draw-specific gamma-based P0") # Print for first few draws
                init_cov_for_smoother = _build_gamma_based_p0_for_smoother(
                    ss_builder.state_dim, 
                    ss_builder.n_dynamic_trends,
                    gamma_list_for_this_draw,
                    ss_builder.n_stationary, 
                    ss_builder.var_order,
                    gamma_init_scaling_for_smoother,
                    gpm_model,
                    ss_builder
                )
            else:
                if i_loop < 2 : print(f"  Draw {i_loop}: Gamma matrices unavailable, using standard P0")
                init_cov_for_smoother = _create_standard_p0_for_smoother(
                    ss_builder.state_dim, ss_builder.n_dynamic_trends
                )
        else:
            if i_loop < 2 : print(f"  Draw {i_loop}: Using standard P0 (gamma disabled or no VAR)")
            init_cov_for_smoother = _create_standard_p0_for_smoother(
                ss_builder.state_dim, ss_builder.n_dynamic_trends
            )

        Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        try: 
            R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
        except: 
            R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

        matrices_finite = (
            jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and 
            jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and 
            jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))
        )
        
        if not matrices_finite:
            if i_loop < 2: print(f"  Draw {i_loop}: Skipping due to non-finite matrices")
            continue
            
        try:
            core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_sm_draw, C_draw, H_draw, 
                init_mean_for_smoother, init_cov_for_smoother, sim_key
            )
            
            if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
                if i_loop < 2: print(f"  Draw {i_loop}: Skipping due to non-finite smoother output")
                continue
                
        except Exception as e:
            if i_loop < 2: print(f"  Draw {i_loop}: Smoother failed: {e}")
            continue

        current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        if i_loop < 1: print(f"  Draw {i_loop}: Mapping variables using ss_builder.core_var_map")
        for var_name, state_idx in ss_builder.core_var_map.items():
            if state_idx < core_states_smoothed_draw.shape[1]:
                current_draw_core_state_values_ts[var_name] = core_states_smoothed_draw[:, state_idx]
                if i_loop < 1:  
                    mean_val = jnp.mean(current_draw_core_state_values_ts[var_name])
                    print(f"    Mapped {var_name} from state[{state_idx}] -> mean={mean_val:.4f}")

        reconstructed_trends_this_draw = jnp.full(
            (T_data, len(gpm_model.gpm_trend_variables_original)), 
            jnp.nan, dtype=_DEFAULT_DTYPE
        )
        
        for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
            if orig_trend_name in current_draw_core_state_values_ts:
                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(
                    current_draw_core_state_values_ts[orig_trend_name]
                )
                if i_loop < 1: print(f"    {orig_trend_name}: Using core variable directly")
                    
            elif orig_trend_name in gpm_model.non_core_trend_definitions:
                expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                reconstructed_value = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                const_val = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
                if const_val != 0: reconstructed_value += const_val
                
                for var_key, coeff_str in expr_def.terms.items():
                    term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                    coeff_val = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                    
                    if term_lag == 0:  
                        if term_var_name in current_draw_core_state_values_ts:
                            reconstructed_value += coeff_val * current_draw_core_state_values_ts[term_var_name]
                        elif term_var_name in current_builder_params:
                            reconstructed_value += coeff_val * current_builder_params[term_var_name]
                
                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(reconstructed_value)
                if i_loop < 1: print(f"    {orig_trend_name}: Reconstructed from definition -> mean={jnp.mean(reconstructed_value):.4f}")
                    
        output_trend_draws_list.append(reconstructed_trends_this_draw)

        reconstructed_stationary_this_draw = jnp.full(
            (T_data, len(gpm_model.gpm_stationary_variables_original)), 
            jnp.nan, dtype=_DEFAULT_DTYPE
        )
        for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
            if orig_stat_name in current_draw_core_state_values_ts:
                reconstructed_stationary_this_draw = reconstructed_stationary_this_draw.at[:, i_orig_stat].set(
                    current_draw_core_state_values_ts[orig_stat_name]
                )
        output_stationary_draws_list.append(reconstructed_stationary_this_draw)
        successful_draws += 1

    print(f"Successfully processed {successful_draws}/{len(draw_indices)} draws")

    if not output_trend_draws_list:
        print("ERROR: No valid simulation smoother draws!")
        # Return empty arrays matching expected dimensions if possible
        num_orig_trends = len(gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
        return jnp.empty((0, T_data, num_orig_trends if num_orig_trends > 0 else 0)), \
               jnp.empty((0, T_data, num_orig_stat if num_orig_stat > 0 else 0)), \
               {'trends': list(gpm_model.gpm_trend_variables_original), 
                'stationary': list(gpm_model.gpm_stationary_variables_original)}

    final_reconstructed_trends = jnp.stack(output_trend_draws_list)
    final_reconstructed_stationary = jnp.stack(output_stationary_draws_list)
    
    component_names = {
        'trends': list(gpm_model.gpm_trend_variables_original),
        'stationary': list(gpm_model.gpm_stationary_variables_original)
    }
    
    print(f"Final output shapes:")
    print(f"  Trends: {final_reconstructed_trends.shape}")
    print(f"  Stationary: {final_reconstructed_stationary.shape}")
    print(f"=== END FIXED SIMULATION SMOOTHER ===\n")
    
    return final_reconstructed_trends, final_reconstructed_stationary, component_names


def _extract_gamma_matrices_for_draw(
    mcmc_samples: Dict[str, jnp.ndarray], 
    mcmc_draw_idx: int,
    current_builder_params: Dict[str, Any],
    gpm_model: ReducedModel, # MODIFIED: Added gpm_model
    n_stationary: int, 
    var_order: int
) -> Optional[List[jnp.ndarray]]:
    """
    Extract gamma matrices for a specific MCMC draw by re-running the VAR transformation.
    This now correctly uses actual stationary shock names from gpm_model.
    """
    if n_stationary == 0 or var_order == 0:
        return None
        
    A_transformed = current_builder_params.get("_var_coefficients")
    Sigma_u = current_builder_params.get("_var_innovation_cov_full")
    
    if Sigma_u is None and gpm_model.stationary_shocks: # Try to reconstruct Sigma_u
        stat_shock_stds_values = []
        all_shocks_found_in_builder_params = True
        for shock_builder_name in gpm_model.stationary_shocks: # Use actual shock names
            if shock_builder_name in current_builder_params:
                # Ensure value is scalar float
                val = current_builder_params[shock_builder_name]
                if hasattr(val, 'item'): val = val.item() # Extract from 0-dim array if needed
                stat_shock_stds_values.append(float(val))

            else:
                # print(f"    Warning: Stationary shock std '{shock_builder_name}' not found in current_builder_params for draw {mcmc_draw_idx}.")
                all_shocks_found_in_builder_params = False
                break
        
        if all_shocks_found_in_builder_params and len(stat_shock_stds_values) == n_stationary:
            sigma_u_vec = jnp.array(stat_shock_stds_values, dtype=_DEFAULT_DTYPE)
            Omega_u_chol = current_builder_params.get("_var_innovation_corr_chol")
            if Omega_u_chol is not None and Omega_u_chol.shape == (n_stationary, n_stationary):
                Sigma_u = jnp.diag(sigma_u_vec) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u_vec)
                Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _SP_JITTER * jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE)
            else:
                # print(f"    Warning: _var_innovation_corr_chol not suitable for Sigma_u reconstruction for draw {mcmc_draw_idx}.")
                Sigma_u = None # Mark as not reconstructable if Omega_u_chol is bad
        else:
            # print(f"    Warning: Could not gather all stationary shock stds for Sigma_u reconstruction for draw {mcmc_draw_idx}.")
            Sigma_u = None # Mark as not reconstructable

    if A_transformed is None or Sigma_u is None:
        # print(f"    Warning: VAR parameters (A_transformed or Sigma_u) not available for draw {mcmc_draw_idx}")
        return None
    
    if not (isinstance(A_transformed, (np.ndarray, jnp.ndarray)) and A_transformed.shape == (var_order, n_stationary, n_stationary)):
        # print(f"    Warning: A_transformed shape mismatch for draw {mcmc_draw_idx}: {A_transformed.shape if hasattr(A_transformed, 'shape') else 'N/A'}")
        return None
        
    if not (isinstance(Sigma_u, (np.ndarray, jnp.ndarray)) and Sigma_u.shape == (n_stationary, n_stationary)):
        # print(f"    Warning: Sigma_u shape mismatch for draw {mcmc_draw_idx}: {Sigma_u.shape if hasattr(Sigma_u, 'shape') else 'N/A'}")
        return None
    
    try:
        if make_stationary_var_transformation_jax is not None:
            A_raw_list = [A_transformed[lag] for lag in range(var_order)]
            _, gamma_list = make_stationary_var_transformation_jax(
                Sigma_u, A_raw_list, n_stationary, var_order
            )
            
            if gamma_list and len(gamma_list) == var_order:
                valid_gammas = all(
                    g is not None and 
                    g.shape == (n_stationary, n_stationary) and 
                    jnp.all(jnp.isfinite(g))
                    for g in gamma_list
                )
                if valid_gammas:
                    # print(f"    Draw {mcmc_draw_idx}: Successfully extracted {len(gamma_list)} gamma matrices")
                    return gamma_list
                # else: print(f"    Warning: Invalid gamma matrices for draw {mcmc_draw_idx}")
            # else: print(f"    Warning: Gamma list wrong length for draw {mcmc_draw_idx}: expected {var_order}, got {len(gamma_list) if gamma_list else 'None'}")
                
    except Exception as e:
        # print(f"    Warning: Failed to extract gamma matrices for draw {mcmc_draw_idx}: {e}")
        pass # Fall through to return None
    
    return None


# def _build_gamma_based_p0_for_smoother(
#     state_dim: int,
#     n_dynamic_trends: int, 
#     gamma_list: List[jnp.ndarray],
#     n_stationary: int,
#     var_order: int,
#     gamma_scaling: float,
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder
# ) -> jnp.ndarray:
#     """
#     Build gamma-based P0 for smoother using the same logic as in gpm_numpyro_models.py.
#     """
#     # print(f"    Building gamma-based P0: state_dim={state_dim}, n_dynamic_trends={n_dynamic_trends}")
#     init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
#     if n_dynamic_trends > 0:
#         init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
#             jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1e4
#         )
#         # print(f"    Set diffuse prior for {n_dynamic_trends} dynamic trends")
    
#     var_start_idx = n_dynamic_trends
#     var_state_total_dim = n_stationary * var_order
    
#     if n_stationary > 0 and var_order > 0 and gamma_list:
#         # print(f"    Building VAR block: start_idx={var_start_idx}, total_dim={var_state_total_dim}")
#         var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
#         g0 = gamma_list[0] 
        
#         for r_idx in range(var_order):
#             for c_idx in range(var_order):
#                 lag_d = abs(r_idx - c_idx)
                
#                 if lag_d < len(gamma_list) and gamma_list[lag_d] is not None:
#                     blk_unscaled = gamma_list[lag_d]
#                 else:
#                     blk_unscaled = g0 * (0.5**lag_d)
                
#                 curr_blk = blk_unscaled * gamma_scaling
#                 if r_idx > c_idx: curr_blk = curr_blk.T
                
#                 r_s, r_e = r_idx * n_stationary, (r_idx + 1) * n_stationary
#                 c_s, c_e = c_idx * n_stationary, (c_idx + 1) * n_stationary
                
#                 if r_e <= var_state_total_dim and c_e <= var_state_total_dim:
#                     var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
        
#         if var_start_idx + var_state_total_dim <= state_dim:
#             init_cov = init_cov.at[
#                 var_start_idx:var_start_idx + var_state_total_dim,
#                 var_start_idx:var_start_idx + var_state_total_dim
#             ].set(var_block_cov)
#             # print(f"    Successfully built VAR covariance block")
#         # else: print(f"    Warning: VAR block dimensions don't fit in state vector")
    
#     init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
#     try:
#         jnp.linalg.cholesky(init_cov)
#         # print(f"    ✓ Gamma-based P0 is positive definite")
#     except Exception as e:
#         # print(f"    Warning: Gamma-based P0 not PSD, adding more jitter: {e}")
#         init_cov = init_cov + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
#     return init_cov

# def _create_standard_p0_for_smoother(state_dim: int, n_dynamic_trends: int) -> jnp.ndarray:
#     init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e4
#     if state_dim > n_dynamic_trends:
#         init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
#             jnp.eye(state_dim - n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1.0
#         )
#     return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)




# This function is called if use_gamma_init_for_smoother is True
def _build_gamma_based_p0_for_smoother(
    state_dim: int, n_dynamic_trends: int, gamma_list: List[jnp.ndarray],
    n_stationary: int, var_order: int, gamma_scaling: float,
    gpm_model: ReducedModel, ss_builder: StateSpaceBuilder, 
    context: str = "mcmc" # Default to MCMC context for smoother
) -> jnp.ndarray:
    """
    Build gamma-based P0 for smoother, context-aware.
    Trends variance: 1e6 (MCMC context)
    VAR fallback variance: 4.0 (MCMC context)
    """
    # Context: Post-MCMC Smoothing (defaults to "mcmc" scales)
    trend_var_scale_smoother = 1e6 if context == "mcmc" else 1e4 
    var_fallback_scale_smoother = 4.0 if context == "mcmc" else 1.0
    print(f"  P0 (Smoother, Gamma-Based, Context: {context}): Using trend scale = {trend_var_scale_smoother}")

    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    if n_dynamic_trends > 0:
        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
            jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * trend_var_scale_smoother
        )
    
    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order
    
    gamma_list_is_valid_smooth = (
        gamma_list and
        len(gamma_list) == var_order and
        gamma_list[0] is not None and
        hasattr(gamma_list[0], 'shape') and
        gamma_list[0].shape == (n_stationary, n_stationary) and
        all(
            g is not None and hasattr(g, 'shape') and g.shape == (n_stationary, n_stationary) and jnp.all(jnp.isfinite(g))
            for g in gamma_list
        )
    )

    if n_stationary > 0 and var_order > 0 and gamma_list_is_valid_smooth:
        print(f"  P0 (Smoother, Gamma-Based): Building VAR block using gamma matrices (scaling: {gamma_scaling}).")
        var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
        g0 = gamma_list[0] 
        
        for r_idx in range(var_order):
            for c_idx in range(var_order):
                lag_d = abs(r_idx - c_idx)
                blk_unscaled = gamma_list[lag_d] # Validity checked by gamma_list_is_valid_smooth
                
                curr_blk = blk_unscaled * gamma_scaling
                if r_idx > c_idx: curr_blk = curr_blk.T
                
                r_s, r_e = r_idx * n_stationary, (r_idx + 1) * n_stationary
                c_s, c_e = c_idx * n_stationary, (c_idx + 1) * n_stationary
                
                if r_e <= var_state_total_dim and c_e <= var_state_total_dim:
                    var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
        
        if var_start_idx + var_state_total_dim <= state_dim:
            init_cov = init_cov.at[
                var_start_idx:var_start_idx + var_state_total_dim,
                var_start_idx:var_start_idx + var_state_total_dim
            ].set(var_block_cov)
        else: # Should not happen
            print(f"    Warning: P0 (Smoother, Gamma-Based) VAR block dimension issue. Applying fallback.")
            init_cov = init_cov.at[var_start_idx:var_start_idx+var_state_total_dim, 
                                   var_start_idx:var_start_idx+var_state_total_dim].set(
                                       jnp.eye(var_state_total_dim,dtype=_DEFAULT_DTYPE)*var_fallback_scale_smoother)
    elif var_state_total_dim > 0: # VAR part exists but gamma_list was not valid
        print(f"  P0 (Smoother, Gamma-Based): Gamma list not suitable. Using VAR fallback scale {var_fallback_scale_smoother}.")
        init_cov = init_cov.at[var_start_idx:var_start_idx+var_state_total_dim, 
                              var_start_idx:var_start_idx+var_state_total_dim].set(
                                  jnp.eye(var_state_total_dim,dtype=_DEFAULT_DTYPE)*var_fallback_scale_smoother)
    
    # Jitter consistent with MCMC context typically
    regularization = _KF_JITTER * (10 if context == "mcmc" else 1)
    init_cov = (init_cov + init_cov.T) / 2.0 + regularization * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    try: jnp.linalg.cholesky(init_cov)
    except Exception: init_cov = init_cov + regularization * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) 
    
    print(f"  P0 (Smoother, Gamma-Based): Constructed. Diag min/max: {jnp.min(jnp.diag(init_cov)):.2e}, {jnp.max(jnp.diag(init_cov)):.2e}")
    return init_cov

# This function is called if use_gamma_init_for_smoother is False
def _create_standard_p0_for_smoother(state_dim: int, n_dynamic_trends: int, context: str = "mcmc") -> jnp.ndarray:
    """
    Standard initial covariance for smoother, context-aware.
    Trends variance: 1e6 (MCMC context)
    VAR fallback variance: 4.0 (MCMC context)
    """
    # Context: Post-MCMC Smoothing (defaults to "mcmc" scales)
    trend_scale = 1e6 if context == "mcmc" else 1e4
    var_scale = 4.0 if context == "mcmc" else 1.0
    print(f"  P0 (Smoother, Standard, Context: {context}): Using trend scale = {trend_scale}, VAR scale = {var_scale}")

    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * trend_scale
    if state_dim > n_dynamic_trends:
        init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
            jnp.eye(state_dim - n_dynamic_trends, dtype=_DEFAULT_DTYPE) * var_scale
        )
    
    regularization = _KF_JITTER * (10 if context == "mcmc" else 1)
    init_cov = (init_cov + init_cov.T) / 2.0 + regularization * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    print(f"  P0 (Smoother, Standard): Constructed. Diag min/max: {jnp.min(jnp.diag(init_cov)):.2e}, {jnp.max(jnp.diag(init_cov)):.2e}")
    return init_cov


def debug_non_core_trend_reconstruction(
    gpm_model: ReducedModel,
    core_states_smoothed_draw: jnp.ndarray,
    current_builder_params: Dict[str, Any],
    ss_builder: StateSpaceBuilder,
    T_data: int
):
    """
    Debug function to trace non-core trend reconstruction
    """
    print("\n=== DEBUGGING NON-CORE TREND RECONSTRUCTION ===")
    print(f"Non-core trend definitions: {list(gpm_model.non_core_trend_definitions.keys())}")
    
    for name, expr_def in gpm_model.non_core_trend_definitions.items():
        print(f"\nNon-core trend: {name}")
        print(f"  Terms: {expr_def.terms}")
        print(f"  Constant: {expr_def.constant_str}")
    
    print(f"\nCore variables: {gpm_model.core_variables}")
    print(f"Core var map: {ss_builder.core_var_map}")
    
    current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
    for var_name, state_idx in ss_builder.core_var_map.items():
        if state_idx < core_states_smoothed_draw.shape[1]:
            current_draw_core_state_values_ts[var_name] = core_states_smoothed_draw[:, state_idx]
            print(f"  Mapped core variable {var_name} from state index {state_idx}, mean val: {jnp.mean(current_draw_core_state_values_ts[var_name]):.4f}")
    
    print(f"\nAvailable core state values for reconstruction: {list(current_draw_core_state_values_ts.keys())}")
    
    for orig_trend_name in gpm_model.gpm_trend_variables_original:
        if orig_trend_name in gpm_model.non_core_trend_definitions:
            print(f"\n--- Reconstructing non-core trend: {orig_trend_name} ---")
            expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
            const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
            print(f"  Constant term: {const_val_numeric}")
            reconstructed_value = jnp.full(T_data, const_val_numeric, dtype=_DEFAULT_DTYPE)
            
            for var_key, coeff_str in expr_def.terms.items():
                term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                print(f"  Term: {coeff_str} * {var_key} (var={term_var_name}, lag={term_lag}), Coeff val: {coeff_numeric}")
                
                if term_lag == 0:
                    if term_var_name in current_draw_core_state_values_ts:
                        term_contribution = coeff_numeric * current_draw_core_state_values_ts[term_var_name]
                        reconstructed_value += term_contribution
                        print(f"    ✓ Added core state {term_var_name}. Mean contribution: {jnp.mean(term_contribution):.4f}")
                    elif term_var_name in current_builder_params:
                        term_contribution = coeff_numeric * current_builder_params[term_var_name]
                        reconstructed_value += term_contribution
                        print(f"    ✓ Added parameter {term_var_name}: {current_builder_params[term_var_name]}")
                    else:
                        print(f"    ✗ Could not find {term_var_name} in core states or parameters!")
            print(f"  Final reconstructed {orig_trend_name}: mean={jnp.mean(reconstructed_value):.4f}, std={jnp.std(reconstructed_value):.4f}")
    print("=== END NON-CORE TREND DEBUG ===\n")