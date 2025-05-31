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
from common_types import EnhancedBVARParams # Used by _extract_parameters_for_ss_builder
from gpm_model_parser import ReducedModel, ReducedExpression # Crucial for reconstruction
from state_space_builder import StateSpaceBuilder           # For evaluating expressions
# ParameterContract not directly used here, but ss_builder uses it internally
from constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER

from Kalman_filter_jax import KalmanFilter # Note the leading dot for relative import

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

    # KalmanFilter class expects R to be the Cholesky of Q, or a matrix such that Q = R @ R.T
    # Here, R_ss is the shock impact matrix. If KF expects Cholesky(Q), then R_for_KF = R_ss
    # assuming R_ss already is L where Q = L L.T.
    # If R_ss maps N(0,I) shocks, then Q = R_ss @ R_ss.T.
    # Let's assume your KalmanFilter's "R" argument is indeed this shock impact matrix R_ss.
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
        smoothed_means, _ = kf_instance.smooth( # Assuming smoother also takes these static args if needed
            y_star,
            filter_results=filter_results,
            static_valid_obs_idx=valid_obs_idx_static,
            static_n_obs_actual=n_obs_exp,
            static_C_obs_for_filter=C, # Pass if smoother re-runs filter or needs C,H
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

def _extract_parameters_for_ss_builder( # This function might be redundant now
    mcmc_samples_dict: Dict,           # The ss_builder itself has _extract_params_from_mcmc_draw
    mcmc_draw_idx: int, 
    gpm_model_struct: ReducedModel,    # Used to reconstruct EnhancedBVARParams
    # ss_builder_instance: StateSpaceBuilder # Not needed if reconstructing EnhancedBVARParams here
    ) -> EnhancedBVARParams:
    # This reconstructs EnhancedBVARParams from raw MCMC samples for a given draw.
    # This is what StateSpaceBuilder.build_state_space_from_enhanced_bvar would expect.
    # gpm_numpyro_models.py already forms this object internally for each draw.
    # This helper is useful if you only have raw MCMC samples and need to create an EnhancedBVARParams.
    
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
            else: trend_s_vals.append(0.01) # Default if missing
    n_core_t = len(gpm_model_struct.core_variables) - len(gpm_model_struct.stationary_variables)
    # Ensure trend_s_vals matches dimension for Sigma_eta diagonal
    if len(trend_s_vals) < n_core_t : trend_s_vals.extend([0.01]*(n_core_t - len(trend_s_vals)))
    elif len(trend_s_vals) > n_core_t : trend_s_vals = trend_s_vals[:n_core_t]
    Sigma_eta_d = jnp.diag(jnp.array(trend_s_vals, dtype=_DEFAULT_DTYPE)**2) if trend_s_vals else jnp.eye(max(0,n_core_t)) * 0.001

    A_trans_d = mcmc_samples_dict.get("A_transformed")
    A_trans_d = A_trans_d[mcmc_draw_idx] if A_trans_d is not None and A_trans_d.shape[0] > mcmc_draw_idx else None
    if A_trans_d is None: # Fallback shape
        n_s = len(gpm_model_struct.stationary_variables)
        vo = gpm_model_struct.var_prior_setup.var_order if gpm_model_struct.var_prior_setup else 1
        A_trans_d = jnp.zeros((vo, n_s, n_s))

    Omega_uc_d = mcmc_samples_dict.get("Omega_u_chol")
    Sigma_u_d = None; n_s_vars = len(gpm_model_struct.stationary_variables)
    if n_s_vars > 0:
        stat_s_vals = []
        if hasattr(gpm_model_struct, 'stationary_shocks'):
            for s_bn in gpm_model_struct.stationary_shocks:
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


def _extract_initial_mean(mcmc_samples_dict: Dict, mcmc_draw_idx: int, state_dim: int) -> jnp.ndarray: # As before
    if "init_mean_full" in mcmc_samples_dict and mcmc_samples_dict["init_mean_full"].shape[0] > mcmc_draw_idx:
        init_mean = mcmc_samples_dict["init_mean_full"][mcmc_draw_idx]
        return jnp.where(jnp.isfinite(init_mean), init_mean, jnp.zeros_like(init_mean))
    return jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)

def _create_reasonable_initial_covariance(state_dim: int, n_dynamic_trends_in_state: int) -> jnp.ndarray:
    # n_dynamic_trends_in_state is the count of the first block of the state vector (non-VAR core trends)
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 10.0 # Default for all
    if state_dim > n_dynamic_trends_in_state: # If there are other states (like VAR states)
        init_cov = init_cov.at[n_dynamic_trends_in_state:, n_dynamic_trends_in_state:].set(
            jnp.eye(state_dim - n_dynamic_trends_in_state, dtype=_DEFAULT_DTYPE) * 1.0 # More informative for VAR/other
        )
    return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

# --- HDI Computation ---


def _compute_and_format_hdi_az(draws: jnp.ndarray, hdi_prob: float = 0.9) -> Dict[str, np.ndarray]:
    # (As provided before, unchanged)
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
        if np.any(np.isnan(low)) or np.any(np.isnan(high)): pass # print(f"Warning: HDI contains NaN values.")
        return {'low': low, 'high': high}
    
    except Exception as e:
        # print(f"Error during ArviZ HDI computation: {e}")
        return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE), 'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}
    

def debug_specific_mcmc_draws(mcmc_results, draw_indices):
    """Debug specific MCMC draws that will be used by the smoother"""
    print(f"\n=== DEBUGGING SPECIFIC MCMC DRAWS ===")
    
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    
    key_params = ['sigma_shk_cycle_y_us', 'sigma_shk_trend_y_us', 'init_mean_full']
    
    for i, mcmc_idx in enumerate(draw_indices[:5]):  # Check first 5
        print(f"\nDraw {i} (MCMC index {mcmc_idx}):")
        
        for param_name in key_params:
            if param_name in mcmc_samples:
                param_array = mcmc_samples[param_name]
                if param_name == 'init_mean_full':
                    print(f"  {param_name}[0]: {param_array[mcmc_idx][0]:.6f}")
                else:
                    print(f"  {param_name}: {param_array[mcmc_idx]:.6f}")
    
    print("=== END DEBUGGING DRAWS ===\n")

# # def extract_reconstructed_components(
# #     mcmc_output: numpyro.infer.MCMC,
# #     y_data: jnp.ndarray, 
# #     gpm_model: ReducedModel,
# #     ss_builder: StateSpaceBuilder,
# #     num_smooth_draws: int = 100,
# #     rng_key_smooth: Optional[jax.Array] = None
# # ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
# #     """
# #     FIXED VERSION: Properly extracts parameters for each MCMC draw
# #     """
# #     print(f"\n=== SIMULATION SMOOTHER DEBUG ===")
    
# #     if rng_key_smooth is None: 
# #         rng_key_smooth = random.PRNGKey(0)

# #     mcmc_samples = mcmc_output.get_samples()
# #     if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
# #         print("Warning: No MCMC samples available")
# #         return jnp.empty((0,0,0)), jnp.empty((0,0,0)), {'trends':[], 'stationary':[]}

# #     T_data, _ = y_data.shape
    
# #     first_param_key = list(mcmc_samples.keys())[0]
# #     total_posterior_draws = mcmc_samples[first_param_key].shape[0]
# #     actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

# #     print(f"Total MCMC draws: {total_posterior_draws}")
# #     print(f"Requested smooth draws: {num_smooth_draws}")
# #     print(f"Actual smooth draws: {actual_num_smooth_draws}")

# #     if actual_num_smooth_draws <= 0:
# #         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
# #                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
# #                {'trends': list(gpm_model.gpm_trend_variables_original), 
# #                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

# #     # CRITICAL FIX: Use different draws, not just evenly spaced
# #     if actual_num_smooth_draws == total_posterior_draws:
# #         draw_indices = np.arange(total_posterior_draws)
# #     else:
# #         draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
# #     print(f"Using draw indices: {draw_indices[:10]}{'...' if len(draw_indices) > 10 else ''}")
    
# #     num_dynamic_trends_in_state = ss_builder.n_core - ss_builder.n_stationary
# #     num_stat_vars_in_state_block = ss_builder.n_stationary

# #     output_trend_draws_list_all_orig_trends = []
# #     output_stationary_draws_list_all_orig_stat = []

# #     # Track parameter variation for debugging
# #     param_tracking = {}

# #     for i_loop, mcmc_draw_idx in enumerate(draw_indices):
# #         # CRITICAL FIX: Ensure unique random key for each draw
# #         rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
# #         # CRITICAL FIX: Extract parameters for THIS specific draw
# #         current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
# #         # DEBUG: Track parameter values for first few draws
# #         if i_loop < 5:
# #             print(f"\n=== SMOOTHER DRAW {i_loop} (MCMC index {mcmc_draw_idx}) ===")
# #             print(f"\nDraw {i_loop} (MCMC index {mcmc_draw_idx}):")
# #             key_params_to_check = ['shk_cycle_y_us', 'shk_trend_y_us']

# #             for param_key in key_params_to_check:
# #                 if param_key in current_builder_params:
# #                     val = current_builder_params[param_key]
# #                     if hasattr(val, 'item'):
# #                         print(f"  {param_key}: {val.item():.6f}")
# #                     else:
# #                         print(f"  {param_key}: {val}")

# #             for key, val in current_builder_params.items():
# #                 if hasattr(val, 'item') and 'shk_cycle_y_us' in key:
# #                     print(f"  {key}: {val.item():.6f}")
# #                     if key not in param_tracking:
# #                         param_tracking[key] = []
# #                     param_tracking[key].append(val.item())
        
# #         # CRITICAL FIX: Build SS matrices using THIS draw's parameters
# #         F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
# #         # ADD THIS DEBUG CODE TOO:
# #         if i_loop < 3:
# #             print(f"  Q_draw[0,0]: {Q_draw[0,0]:.6f}")
# #             print(f"  F_draw[0,0]: {F_draw[0,0]:.6f}")

# #         # Check init_mean if available
# #         init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
# #         if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
# #             init_mean_this_draw = init_mean_mcmc_val[mcmc_draw_idx]
# #             print(f"  init_mean_full[0]: {init_mean_this_draw[0]:.6f}")
        
# #         print(f"=== END DRAW {i_loop} ===")

# #         # CRITICAL FIX: Get P0 for THIS specific draw
# #         init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
# #         if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
# #             init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]  # Use THIS draw's initial mean
# #         else:
# #             init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
            
# #         # Use reasonable initial covariance
# #         init_cov_for_smoother = _create_reasonable_initial_covariance(ss_builder.state_dim, num_dynamic_trends_in_state)

# #         # Regularize matrices
# #         Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
# #         try: 
# #             R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
# #         except: 
# #             R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

# #         # Check matrices are finite
# #         if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and \
# #                 jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and \
# #                 jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
# #             print(f"Warning: Non-finite matrices for draw {mcmc_draw_idx}. Skipping.")
# #             continue
            
# #         # Run simulation smoother with THIS draw's parameters
# #         try:
# #             core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
# #                 y_data, F_draw, R_sm_draw, C_draw, H_draw, init_mean_for_smoother, init_cov_for_smoother, sim_key
# #             )
# #             if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
# #                 print(f"Warning: Non-finite smoother output for draw {mcmc_draw_idx}. Skipping.")
# #                 continue
# #         except Exception as e:
# #             print(f"Warning: Smoother failed for draw {mcmc_draw_idx}: {e}. Skipping.")
# #             continue

# #         # [Rest of reconstruction code remains the same...]
# #         # Reconstruct original variables from core states
# #         current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        
# #         # Dynamic trends
# #         dynamic_trend_count = 0
# #         for core_var_name in gpm_model.core_variables:
# #             if core_var_name not in gpm_model.stationary_variables:
# #                 state_vector_idx = dynamic_trend_count
# #                 if state_vector_idx < num_dynamic_trends_in_state:
# #                     current_draw_core_state_values_ts[core_var_name] = core_states_smoothed_draw[:, state_vector_idx]
# #                 dynamic_trend_count += 1
        
# #         # Stationary variables (current period)
# #         var_block_start_idx_in_state = num_dynamic_trends_in_state
# #         for i_stat_var, stat_var_name in enumerate(gpm_model.stationary_variables):
# #             state_vector_idx = var_block_start_idx_in_state + i_stat_var
# #             if state_vector_idx < ss_builder.state_dim:
# #                 current_draw_core_state_values_ts[stat_var_name] = core_states_smoothed_draw[:, state_vector_idx]
        
# #         # Reconstruct original trends
# #         reconstructed_trends_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
# #         for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
# #             if orig_trend_name in gpm_model.core_variables and orig_trend_name not in gpm_model.stationary_variables:
# #                 if orig_trend_name in current_draw_core_state_values_ts:
# #                     reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(
# #                         current_draw_core_state_values_ts[orig_trend_name]
# #                     )
# #             elif orig_trend_name in gpm_model.non_core_trend_definitions:
# #                 expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
# #                 reconstructed_value_for_orig_trend_t = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
# #                 const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
# #                 reconstructed_value_for_orig_trend_t += const_val_numeric

# #                 for var_key_in_def, coeff_expr_str_in_def in expr_def.terms.items():
# #                     term_var_name, term_lag = ss_builder._parse_variable_key(var_key_in_def)
# #                     coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_expr_str_in_def, current_builder_params)
                    
# #                     if term_lag == 0:
# #                         if term_var_name in current_draw_core_state_values_ts:
# #                             reconstructed_value_for_orig_trend_t += coeff_numeric * current_draw_core_state_values_ts[term_var_name]
# #                         elif term_var_name in current_builder_params:
# #                             reconstructed_value_for_orig_trend_t += coeff_numeric * current_builder_params[term_var_name]
                            
# #                 reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(reconstructed_value_for_orig_trend_t)
                
# #         output_trend_draws_list_all_orig_trends.append(reconstructed_trends_this_mcmc_draw)

# #         # Reconstruct original stationary variables
# #         reconstructed_stationary_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
# #         for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
# #             if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
# #                 reconstructed_stationary_this_mcmc_draw = reconstructed_stationary_this_mcmc_draw.at[:, i_orig_stat].set(
# #                     current_draw_core_state_values_ts[orig_stat_name]
# #                 )
# #         output_stationary_draws_list_all_orig_stat.append(reconstructed_stationary_this_mcmc_draw)

# #     # Check if we got any valid draws
# #     if not output_trend_draws_list_all_orig_trends:
# #         print("ERROR: No valid simulation smoother draws!")
# #         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
# #                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
# #                {'trends': list(gpm_model.gpm_trend_variables_original), 
# #                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

# #     # Debug: Check parameter variation
# #     print(f"\nParameter variation check:")
# #     for key, values in param_tracking.items():
# #         if len(values) > 1:
# #             print(f"  {key}: std = {np.std(values):.6f}, range = [{min(values):.6f}, {max(values):.6f}]")

# #     final_reconstructed_trends = jnp.stack(output_trend_draws_list_all_orig_trends)
# #     final_reconstructed_stationary = jnp.stack(output_stationary_draws_list_all_orig_stat)
    
# #     print(f"Final output shapes:")
# #     print(f"  Trends: {final_reconstructed_trends.shape}")
# #     print(f"  Stationary: {final_reconstructed_stationary.shape}")
# #     print(f"=== END SIMULATION SMOOTHER DEBUG ===\n")
    
# #     component_names = {
# #         'trends': list(gpm_model.gpm_trend_variables_original),
# #         'stationary': list(gpm_model.gpm_stationary_variables_original)
# #     }
    
# #     return final_reconstructed_trends, final_reconstructed_stationary, component_names


# # def extract_reconstructed_components(
# #     mcmc_output: numpyro.infer.MCMC,
# #     y_data: jnp.ndarray, 
# #     gpm_model: ReducedModel,
# #     ss_builder: StateSpaceBuilder,
# #     num_smooth_draws: int = 100,
# #     rng_key_smooth: Optional[jax.Array] = None,
# #     use_gamma_init_for_smoother: bool = True,  # NEW: Control gamma P0 usage
# #     gamma_init_scaling_for_smoother: float = 1.0  # NEW: Scaling for gamma P0
# # ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
# #     """
# #     FIXED VERSION: Properly uses draw-specific P0 based on gamma matrices from VAR transformation
# #     """
# #     print(f"\n=== SIMULATION SMOOTHER WITH PROPER P0 HANDLING ===")
    
# #     if rng_key_smooth is None: 
# #         rng_key_smooth = random.PRNGKey(0)

# #     mcmc_samples = mcmc_output.get_samples()
# #     if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
# #         print("Warning: No MCMC samples available")
# #         return jnp.empty((0,0,0)), jnp.empty((0,0,0)), {'trends':[], 'stationary':[]}

# #     T_data, _ = y_data.shape
    
# #     first_param_key = list(mcmc_samples.keys())[0]
# #     total_posterior_draws = mcmc_samples[first_param_key].shape[0]
# #     actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

# #     print(f"Total MCMC draws: {total_posterior_draws}")
# #     print(f"Actual smooth draws: {actual_num_smooth_draws}")
# #     print(f"Use gamma-based P0: {use_gamma_init_for_smoother}")

# #     if actual_num_smooth_draws <= 0:
# #         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
# #                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
# #                {'trends': list(gpm_model.gpm_trend_variables_original), 
# #                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

# #     # Use different draws, not just evenly spaced
# #     if actual_num_smooth_draws == total_posterior_draws:
# #         draw_indices = np.arange(total_posterior_draws)
# #     else:
# #         draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
# #     num_dynamic_trends_in_state = ss_builder.n_core - ss_builder.n_stationary
# #     num_stat_vars_in_state_block = ss_builder.n_stationary

# #     output_trend_draws_list_all_orig_trends = []
# #     output_stationary_draws_list_all_orig_stat = []

# #     for i_loop, mcmc_draw_idx in enumerate(draw_indices):
# #         rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
# #         # Extract parameters for THIS specific draw
# #         current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
# #         # Build SS matrices using THIS draw's parameters
# #         F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
# #         # CRITICAL FIX: Get proper P0 for THIS specific draw
# #         init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
# #         if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
# #             init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]
# #         else:
# #             init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
# #         # CRITICAL FIX: Build draw-specific initial covariance
# #         if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
# #             # Get gamma matrices for this specific draw
# #             gamma_list_for_this_draw = _extract_gamma_matrices_for_draw(
# #                 mcmc_samples, mcmc_draw_idx, current_builder_params, 
# #                 ss_builder.n_stationary, ss_builder.var_order
# #             )
            
# #             if gamma_list_for_this_draw is not None:
# #                 print(f"  Draw {i_loop}: Using gamma-based P0 with {len(gamma_list_for_this_draw)} gamma matrices")
# #                 init_cov_for_smoother = _build_gamma_based_p0_for_smoother(
# #                     ss_builder.state_dim, 
# #                     num_dynamic_trends_in_state,
# #                     gamma_list_for_this_draw,
# #                     ss_builder.n_stationary, 
# #                     ss_builder.var_order,
# #                     gamma_init_scaling_for_smoother,
# #                     gpm_model,
# #                     ss_builder
# #                 )
# #             else:
# #                 print(f"  Draw {i_loop}: Gamma matrices not available, using standard P0")
# #                 init_cov_for_smoother = _create_standard_p0_for_smoother(
# #                     ss_builder.state_dim, num_dynamic_trends_in_state
# #                 )
# #         else:
# #             # Use standard P0
# #             init_cov_for_smoother = _create_standard_p0_for_smoother(
# #                 ss_builder.state_dim, num_dynamic_trends_in_state
# #             )

# #         # Regularize matrices
# #         Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
# #         try: 
# #             R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
# #         except: 
# #             R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

# #         # Check matrices are finite
# #         if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and \
# #                 jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and \
# #                 jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
# #             print(f"Warning: Non-finite matrices for draw {mcmc_draw_idx}. Skipping.")
# #             continue
            
# #         # Run simulation smoother with THIS draw's parameters AND P0
# #         try:
# #             core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
# #                 y_data, F_draw, R_sm_draw, C_draw, H_draw, init_mean_for_smoother, init_cov_for_smoother, sim_key
# #             )
# #             if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
# #                 print(f"Warning: Non-finite smoother output for draw {mcmc_draw_idx}. Skipping.")
# #                 continue
# #         except Exception as e:
# #             print(f"Warning: Smoother failed for draw {mcmc_draw_idx}: {e}. Skipping.")
# #             continue

# #         # [Rest of reconstruction code remains the same...]
# #         # Reconstruct original variables from core states
# #         current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        
# #         # Dynamic trends
# #         dynamic_trend_count = 0
# #         for core_var_name in gpm_model.core_variables:
# #             if core_var_name not in gpm_model.stationary_variables:
# #                 state_vector_idx = dynamic_trend_count
# #                 if state_vector_idx < num_dynamic_trends_in_state:
# #                     current_draw_core_state_values_ts[core_var_name] = core_states_smoothed_draw[:, state_vector_idx]
# #                 dynamic_trend_count += 1
        
# #         # Stationary variables (current period)
# #         var_block_start_idx_in_state = num_dynamic_trends_in_state
# #         for i_stat_var, stat_var_name in enumerate(gpm_model.stationary_variables):
# #             state_vector_idx = var_block_start_idx_in_state + i_stat_var
# #             if state_vector_idx < ss_builder.state_dim:
# #                 current_draw_core_state_values_ts[stat_var_name] = core_states_smoothed_draw[:, state_vector_idx]
        
# #         # Reconstruct original trends
# #         reconstructed_trends_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
# #         for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
# #             if orig_trend_name in gpm_model.core_variables and orig_trend_name not in gpm_model.stationary_variables:
# #                 if orig_trend_name in current_draw_core_state_values_ts:
# #                     reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(
# #                         current_draw_core_state_values_ts[orig_trend_name]
# #                     )
# #             elif orig_trend_name in gpm_model.non_core_trend_definitions:
# #                 expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
# #                 reconstructed_value_for_orig_trend_t = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
# #                 const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
# #                 reconstructed_value_for_orig_trend_t += const_val_numeric

# #                 for var_key_in_def, coeff_expr_str_in_def in expr_def.terms.items():
# #                     term_var_name, term_lag = ss_builder._parse_variable_key(var_key_in_def)
# #                     coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_expr_str_in_def, current_builder_params)
                    
# #                     if term_lag == 0:
# #                         if term_var_name in current_draw_core_state_values_ts:
# #                             reconstructed_value_for_orig_trend_t += coeff_numeric * current_draw_core_state_values_ts[term_var_name]
# #                         elif term_var_name in current_builder_params:
# #                             reconstructed_value_for_orig_trend_t += coeff_numeric * current_builder_params[term_var_name]
                            
# #                 reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(reconstructed_value_for_orig_trend_t)
                
# #         output_trend_draws_list_all_orig_trends.append(reconstructed_trends_this_mcmc_draw)

# #         # Reconstruct original stationary variables
# #         reconstructed_stationary_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
# #         for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
# #             if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
# #                 reconstructed_stationary_this_mcmc_draw = reconstructed_stationary_this_mcmc_draw.at[:, i_orig_stat].set(
# #                     current_draw_core_state_values_ts[orig_stat_name]
# #                 )
# #         output_stationary_draws_list_all_orig_stat.append(reconstructed_stationary_this_mcmc_draw)

# #     # Check if we got any valid draws
# #     if not output_trend_draws_list_all_orig_trends:
# #         print("ERROR: No valid simulation smoother draws!")
# #         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
# #                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
# #                {'trends': list(gpm_model.gpm_trend_variables_original), 
# #                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

# #     final_reconstructed_trends = jnp.stack(output_trend_draws_list_all_orig_trends)
# #     final_reconstructed_stationary = jnp.stack(output_stationary_draws_list_all_orig_stat)
    
# #     component_names = {
# #         'trends': list(gpm_model.gpm_trend_variables_original),
# #         'stationary': list(gpm_model.gpm_stationary_variables_original)
# #     }
    
# #     print(f"Final output shapes:")
# #     print(f"  Trends: {final_reconstructed_trends.shape}")
# #     print(f"  Stationary: {final_reconstructed_stationary.shape}")
# #     print(f"=== END SIMULATION SMOOTHER WITH PROPER P0 ===\n")
    
# #     return final_reconstructed_trends, final_reconstructed_stationary, component_names



# def extract_reconstructed_components(
#     mcmc_output: numpyro.infer.MCMC,
#     y_data: jnp.ndarray, 
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder,
#     num_smooth_draws: int = 100,
#     rng_key_smooth: Optional[jax.Array] = None,
#     use_gamma_init_for_smoother: bool = True,
#     gamma_init_scaling_for_smoother: float = 1.0
# ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
#     """
#     FIXED VERSION: Uses correct variable-to-state mapping from ss_builder.core_var_map
#     """
#     print(f"\n=== SIMULATION SMOOTHER WITH FIXED MAPPING ===")
    
#     if rng_key_smooth is None: 
#         rng_key_smooth = random.PRNGKey(0)

#     mcmc_samples = mcmc_output.get_samples()
#     if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
#         print("Warning: No MCMC samples available")
#         return jnp.empty((0,0,0)), jnp.empty((0,0,0)), {'trends':[], 'stationary':[]}

#     T_data, _ = y_data.shape
    
#     first_param_key = list(mcmc_samples.keys())[0]
#     total_posterior_draws = mcmc_samples[first_param_key].shape[0]
#     actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

#     if actual_num_smooth_draws <= 0:
#         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
#                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
#                {'trends': list(gpm_model.gpm_trend_variables_original), 
#                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

#     # Use different draws
#     if actual_num_smooth_draws == total_posterior_draws:
#         draw_indices = np.arange(total_posterior_draws)
#     else:
#         draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
#     output_trend_draws_list_all_orig_trends = []
#     output_stationary_draws_list_all_orig_stat = []

#     for i_loop, mcmc_draw_idx in enumerate(draw_indices):
#         rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
#         # Extract parameters for THIS specific draw
#         current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
#         # Build SS matrices using THIS draw's parameters
#         F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
#         # Get proper P0 for THIS specific draw
#         init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
#         if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
#             init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]
#         else:
#             init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
#         # Build draw-specific initial covariance
#         if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
#             gamma_list_for_this_draw = _extract_gamma_matrices_for_draw(
#                 mcmc_samples, mcmc_draw_idx, current_builder_params, 
#                 ss_builder.n_stationary, ss_builder.var_order
#             )
            
#             if gamma_list_for_this_draw is not None:
#                 init_cov_for_smoother = _build_gamma_based_p0_for_smoother(
#                     ss_builder.state_dim, 
#                     ss_builder.n_dynamic_trends,
#                     gamma_list_for_this_draw,
#                     ss_builder.n_stationary, 
#                     ss_builder.var_order,
#                     gamma_init_scaling_for_smoother,
#                     gpm_model,
#                     ss_builder
#                 )
#             else:
#                 init_cov_for_smoother = _create_standard_p0_for_smoother(
#                     ss_builder.state_dim, ss_builder.n_dynamic_trends
#                 )
#         else:
#             init_cov_for_smoother = _create_standard_p0_for_smoother(
#                 ss_builder.state_dim, ss_builder.n_dynamic_trends
#             )

#         # Regularize matrices
#         Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
#         try: 
#             R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
#         except: 
#             R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

#         # Check matrices are finite
#         if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and \
#                 jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and \
#                 jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
#             print(f"Warning: Non-finite matrices for draw {mcmc_draw_idx}. Skipping.")
#             continue
            
#         # Run simulation smoother
#         try:
#             core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
#                 y_data, F_draw, R_sm_draw, C_draw, H_draw, init_mean_for_smoother, init_cov_for_smoother, sim_key
#             )
#             if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
#                 print(f"Warning: Non-finite smoother output for draw {mcmc_draw_idx}. Skipping.")
#                 continue
#         except Exception as e:
#             print(f"Warning: Smoother failed for draw {mcmc_draw_idx}: {e}. Skipping.")
#             continue

#         # *** CRITICAL FIX: Use ss_builder.core_var_map for correct mapping ***
#         current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        
#         # Map ALL core variables using the actual mapping from ss_builder
#         for var_name, state_idx in ss_builder.core_var_map.items():
#             if state_idx < core_states_smoothed_draw.shape[1]:
#                 current_draw_core_state_values_ts[var_name] = core_states_smoothed_draw[:, state_idx]
#                 if i_loop == 0:  # Debug first draw
#                     print(f"  Mapped {var_name} from state index {state_idx}")

#         # Debug: Print what we have available
#         if i_loop == 0:
#             print(f"Available core state variables: {list(current_draw_core_state_values_ts.keys())}")

#         # Reconstruct original trends
#         reconstructed_trends_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        
#         for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
#             if orig_trend_name in current_draw_core_state_values_ts:
#                 # It's a core variable - use directly
#                 reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(
#                     current_draw_core_state_values_ts[orig_trend_name]
#                 )
#                 if i_loop == 0:
#                     print(f"  {orig_trend_name}: Using core variable directly")
                    
#             elif orig_trend_name in gpm_model.non_core_trend_definitions:
#                 # It's a non-core trend - reconstruct from definition
#                 expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                
#                 # Start with constant (which is 0 in your case)
#                 reconstructed_value = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
#                 # Add constant term
#                 const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
#                 if const_val_numeric != 0:
#                     reconstructed_value += const_val_numeric
                
#                 if i_loop == 0:
#                     print(f"  {orig_trend_name}: Reconstructing from definition")
#                     print(f"    Constant: {const_val_numeric}")
                
#                 # Add variable terms
#                 for var_key, coeff_str in expr_def.terms.items():
#                     term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
#                     coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                    
#                     if i_loop == 0:
#                         print(f"    Term: {coeff_str} * {var_key} = {coeff_numeric} * {term_var_name}")
                    
#                     if term_lag == 0:  # Only current period terms
#                         if term_var_name in current_draw_core_state_values_ts:
#                             term_contribution = coeff_numeric * current_draw_core_state_values_ts[term_var_name]
#                             reconstructed_value += term_contribution
#                             if i_loop == 0:
#                                 print(f"      ✓ Added {term_var_name}: mean contribution = {jnp.mean(term_contribution):.4f}")
#                         elif term_var_name in current_builder_params:
#                             # Parameter used as a variable
#                             term_contribution = coeff_numeric * current_builder_params[term_var_name]
#                             reconstructed_value += term_contribution
#                             if i_loop == 0:
#                                 print(f"      ✓ Added parameter {term_var_name}: {current_builder_params[term_var_name]}")
#                         else:
#                             if i_loop == 0:
#                                 print(f"      ✗ Could not find {term_var_name}")
                
#                 reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(reconstructed_value)
                
#                 if i_loop == 0:
#                     print(f"    Final {orig_trend_name}: mean = {jnp.mean(reconstructed_value):.4f}, std = {jnp.std(reconstructed_value):.4f}")
#             else:
#                 if i_loop == 0:
#                     print(f"  {orig_trend_name}: ✗ Not found in core variables or non-core definitions!")
                
#         output_trend_draws_list_all_orig_trends.append(reconstructed_trends_this_mcmc_draw)

#         # Reconstruct original stationary variables
#         reconstructed_stationary_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
#         for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
#             if orig_stat_name in current_draw_core_state_values_ts:
#                 reconstructed_stationary_this_mcmc_draw = reconstructed_stationary_this_mcmc_draw.at[:, i_orig_stat].set(
#                     current_draw_core_state_values_ts[orig_stat_name]
#                 )
#         output_stationary_draws_list_all_orig_stat.append(reconstructed_stationary_this_mcmc_draw)

#     if not output_trend_draws_list_all_orig_trends:
#         print("ERROR: No valid simulation smoother draws!")
#         return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
#                jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
#                {'trends': list(gpm_model.gpm_trend_variables_original), 
#                 'stationary': list(gpm_model.gpm_stationary_variables_original)}

#     final_reconstructed_trends = jnp.stack(output_trend_draws_list_all_orig_trends)
#     final_reconstructed_stationary = jnp.stack(output_stationary_draws_list_all_orig_stat)
    
#     component_names = {
#         'trends': list(gpm_model.gpm_trend_variables_original),
#         'stationary': list(gpm_model.gpm_stationary_variables_original)
#     }
    
#     print(f"Final output shapes:")
#     print(f"  Trends: {final_reconstructed_trends.shape}")
#     print(f"  Stationary: {final_reconstructed_stationary.shape}")
#     print(f"=== END FIXED SIMULATION SMOOTHER ===\n")
    
#     return final_reconstructed_trends, final_reconstructed_stationary, component_names


def extract_reconstructed_components_fixed(
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray, 
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None,
    use_gamma_init_for_smoother: bool = True,  # NEW: Enable gamma-based P0
    gamma_init_scaling_for_smoother: float = 1.0  # NEW: Scaling factor
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
    """
    FIXED VERSION: 
    1. Uses draw-specific gamma-based P0 initialization
    2. Correctly maps variables using ss_builder.core_var_map
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

    # Use different draws for variety
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
        
        # CRITICAL FIX 1: Extract parameters for THIS specific draw
        current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
        # Build SS matrices using THIS draw's parameters
        F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
        # Get initial mean for THIS specific draw
        init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
        if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
            init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]
        else:
            init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
        # CRITICAL FIX 1: Build draw-specific gamma-based P0
        if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
            # Extract gamma matrices for this specific draw
            gamma_list_for_this_draw = _extract_gamma_matrices_for_draw(
                mcmc_samples, mcmc_draw_idx, current_builder_params, 
                ss_builder.n_stationary, ss_builder.var_order
            )
            
            if gamma_list_for_this_draw is not None:
                print(f"  Draw {i_loop}: Using draw-specific gamma-based P0")
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
                print(f"  Draw {i_loop}: Gamma matrices unavailable, using standard P0")
                init_cov_for_smoother = _create_standard_p0_for_smoother(
                    ss_builder.state_dim, ss_builder.n_dynamic_trends
                )
        else:
            print(f"  Draw {i_loop}: Using standard P0 (gamma disabled or no VAR)")
            init_cov_for_smoother = _create_standard_p0_for_smoother(
                ss_builder.state_dim, ss_builder.n_dynamic_trends
            )

        # Regularize matrices for numerical stability
        Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        try: 
            R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
        except: 
            R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

        # Check all matrices are finite
        matrices_finite = (
            jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and 
            jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and 
            jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))
        )
        
        if not matrices_finite:
            print(f"  Draw {i_loop}: Skipping due to non-finite matrices")
            continue
            
        # Run simulation smoother with proper P0
        try:
            core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_sm_draw, C_draw, H_draw, 
                init_mean_for_smoother, init_cov_for_smoother, sim_key
            )
            
            if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
                print(f"  Draw {i_loop}: Skipping due to non-finite smoother output")
                continue
                
        except Exception as e:
            print(f"  Draw {i_loop}: Smoother failed: {e}")
            continue

        # CRITICAL FIX 2: Use correct variable mapping from ss_builder.core_var_map
        current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        
        print(f"  Draw {i_loop}: Mapping variables using ss_builder.core_var_map")
        for var_name, state_idx in ss_builder.core_var_map.items():
            if state_idx < core_states_smoothed_draw.shape[1]:
                current_draw_core_state_values_ts[var_name] = core_states_smoothed_draw[:, state_idx]
                if i_loop == 0:  # Debug output for first draw
                    mean_val = jnp.mean(current_draw_core_state_values_ts[var_name])
                    print(f"    Mapped {var_name} from state[{state_idx}] -> mean={mean_val:.4f}")

        # Reconstruct original trend variables
        reconstructed_trends_this_draw = jnp.full(
            (T_data, len(gpm_model.gpm_trend_variables_original)), 
            jnp.nan, dtype=_DEFAULT_DTYPE
        )
        
        for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
            if orig_trend_name in current_draw_core_state_values_ts:
                # Direct core variable
                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(
                    current_draw_core_state_values_ts[orig_trend_name]
                )
                if i_loop == 0:
                    print(f"    {orig_trend_name}: Using core variable directly")
                    
            elif orig_trend_name in gpm_model.non_core_trend_definitions:
                # Reconstruct from definition
                expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                reconstructed_value = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
                # Add constant term
                const_val = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
                if const_val != 0:
                    reconstructed_value += const_val
                
                # Add variable terms
                for var_key, coeff_str in expr_def.terms.items():
                    term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                    coeff_val = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                    
                    if term_lag == 0:  # Current period only
                        if term_var_name in current_draw_core_state_values_ts:
                            reconstructed_value += coeff_val * current_draw_core_state_values_ts[term_var_name]
                        elif term_var_name in current_builder_params:
                            reconstructed_value += coeff_val * current_builder_params[term_var_name]
                
                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(reconstructed_value)
                
                if i_loop == 0:
                    print(f"    {orig_trend_name}: Reconstructed from definition -> mean={jnp.mean(reconstructed_value):.4f}")
                    
        output_trend_draws_list.append(reconstructed_trends_this_draw)

        # Reconstruct original stationary variables
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

    # Check if we got any valid draws
    if not output_trend_draws_list:
        print("ERROR: No valid simulation smoother draws!")
        return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
               jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
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



# def _extract_gamma_matrices_for_draw(
#     mcmc_samples: Dict[str, jnp.ndarray], 
#     mcmc_draw_idx: int,
#     current_builder_params: Dict[str, Any],
#     n_stationary: int, 
#     var_order: int
# ) -> Optional[List[jnp.ndarray]]:
#     """
#     Extract gamma matrices for a specific MCMC draw by re-running the VAR transformation
#     """
#     if n_stationary == 0 or var_order == 0:
#         return None
        
#     # Try to get VAR coefficients and innovation covariance for this draw
#     A_transformed = current_builder_params.get("_var_coefficients")
#     Sigma_u = current_builder_params.get("_var_innovation_cov_full") 
    
#     if A_transformed is None or Sigma_u is None:
#         print(f"    Warning: VAR parameters not available for draw {mcmc_draw_idx}")
#         return None
    
#     if A_transformed.shape != (var_order, n_stationary, n_stationary):
#         print(f"    Warning: A_transformed shape mismatch for draw {mcmc_draw_idx}")
#         return None
        
#     if Sigma_u.shape != (n_stationary, n_stationary):
#         print(f"    Warning: Sigma_u shape mismatch for draw {mcmc_draw_idx}")
#         return None
    
#     # Re-run the stationary transformation to get gamma matrices
#     try:
#         if make_stationary_var_transformation_jax is not None:
#             # Convert A_transformed back to "raw" form for the transformation
#             A_raw_list = [A_transformed[lag] for lag in range(var_order)]
            
#             # Run the transformation to get phi and gamma lists
#             phi_list, gamma_list = make_stationary_var_transformation_jax(
#                 Sigma_u, A_raw_list, n_stationary, var_order
#             )
            
#             # Validate gamma list
#             if gamma_list and len(gamma_list) == var_order:
#                 valid_gammas = all(
#                     g is not None and g.shape == (n_stationary, n_stationary) and jnp.all(jnp.isfinite(g))
#                     for g in gamma_list
#                 )
#                 if valid_gammas:
#                     return gamma_list
#                 else:
#                     print(f"    Warning: Invalid gamma matrices for draw {mcmc_draw_idx}")
#             else:
#                 print(f"    Warning: Gamma list wrong length for draw {mcmc_draw_idx}")
                
#     except Exception as e:
#         print(f"    Warning: Failed to extract gamma matrices for draw {mcmc_draw_idx}: {e}")
    
#     return None


def _extract_gamma_matrices_for_draw(
    mcmc_samples: Dict[str, jnp.ndarray], 
    mcmc_draw_idx: int,
    current_builder_params: Dict[str, Any],
    n_stationary: int, 
    var_order: int
) -> Optional[List[jnp.ndarray]]:
    """
    Extract gamma matrices for a specific MCMC draw by re-running the VAR transformation.
    
    This is critical for proper P0 initialization in the simulation smoother.
    The gamma matrices contain the exact theoretical unconditional autocovariances.
    """
    if n_stationary == 0 or var_order == 0:
        return None
        
    # Get VAR coefficients and innovation covariance for this specific draw
    A_transformed = current_builder_params.get("_var_coefficients")
    
    # Try multiple ways to get Sigma_u for this draw
    Sigma_u = current_builder_params.get("_var_innovation_cov_full")
    if Sigma_u is None:
        # Reconstruct from individual shock std devs and correlation matrix
        stat_shock_stds = []
        for i in range(n_stationary):
            # Look for individual shock standard deviations
            shock_name = f"shk_stat{i+1}"  # Adjust naming as needed
            if shock_name in current_builder_params:
                stat_shock_stds.append(current_builder_params[shock_name])
        
        if len(stat_shock_stds) == n_stationary:
            sigma_u_vec = jnp.array(stat_shock_stds, dtype=_DEFAULT_DTYPE)
            Omega_u_chol = current_builder_params.get("_var_innovation_corr_chol")
            if Omega_u_chol is not None and Omega_u_chol.shape == (n_stationary, n_stationary):
                Sigma_u = jnp.diag(sigma_u_vec) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u_vec)
                Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _SP_JITTER * jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE)
    
    if A_transformed is None or Sigma_u is None:
        print(f"    Warning: VAR parameters not available for draw {mcmc_draw_idx}")
        return None
    
    if A_transformed.shape != (var_order, n_stationary, n_stationary):
        print(f"    Warning: A_transformed shape mismatch for draw {mcmc_draw_idx}: {A_transformed.shape}")
        return None
        
    if Sigma_u.shape != (n_stationary, n_stationary):
        print(f"    Warning: Sigma_u shape mismatch for draw {mcmc_draw_idx}: {Sigma_u.shape}")
        return None
    
    # Re-run the stationary transformation to get gamma matrices
    try:
        if make_stationary_var_transformation_jax is not None:
            # Convert A_transformed back to "raw" form for the transformation
            A_raw_list = [A_transformed[lag] for lag in range(var_order)]
            
            # Run the transformation to get phi and gamma lists
            phi_list, gamma_list = make_stationary_var_transformation_jax(
                Sigma_u, A_raw_list, n_stationary, var_order
            )
            
            # Validate gamma list
            if gamma_list and len(gamma_list) == var_order:
                valid_gammas = all(
                    g is not None and 
                    g.shape == (n_stationary, n_stationary) and 
                    jnp.all(jnp.isfinite(g))
                    for g in gamma_list
                )
                if valid_gammas:
                    print(f"    Draw {mcmc_draw_idx}: Successfully extracted {len(gamma_list)} gamma matrices")
                    return gamma_list
                else:
                    print(f"    Warning: Invalid gamma matrices for draw {mcmc_draw_idx}")
            else:
                print(f"    Warning: Gamma list wrong length for draw {mcmc_draw_idx}: expected {var_order}, got {len(gamma_list) if gamma_list else 'None'}")
                
    except Exception as e:
        print(f"    Warning: Failed to extract gamma matrices for draw {mcmc_draw_idx}: {e}")
    
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
#     Build gamma-based P0 for smoother using the same logic as in gpm_numpyro_models.py
#     """
#     # Start with basic structure
#     init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
#     # Large variance for dynamic trends (diffuse prior)
#     if n_dynamic_trends > 0:
#         init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
#             jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1e4
#         )
    
#     # VAR block using gamma matrices
#     var_start_idx = n_dynamic_trends
#     var_state_total_dim = n_stationary * var_order
    
#     if n_stationary > 0 and var_order > 0 and gamma_list:
#         var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
#         g0 = gamma_list[0]  # This should be Gamma_0
        
#         for r_idx in range(var_order):
#             for c_idx in range(var_order):
#                 lag_d = abs(r_idx - c_idx)
                
#                 # Get gamma matrix for this lag difference
#                 if lag_d < len(gamma_list) and gamma_list[lag_d] is not None:
#                     blk_unscaled = gamma_list[lag_d]
#                 else:
#                     # Fallback: exponential decay
#                     blk_unscaled = g0 * (0.5**lag_d)
                
#                 curr_blk = blk_unscaled * gamma_scaling
#                 if r_idx > c_idx:
#                     curr_blk = curr_blk.T
                
#                 # Insert block into VAR covariance
#                 r_s, r_e = r_idx * n_stationary, (r_idx + 1) * n_stationary
#                 c_s, c_e = c_idx * n_stationary, (c_idx + 1) * n_stationary
                
#                 if r_e <= var_state_total_dim and c_e <= var_state_total_dim:
#                     var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
        
#         # Insert VAR block into full covariance matrix
#         if var_start_idx + var_state_total_dim <= state_dim:
#             init_cov = init_cov.at[
#                 var_start_idx:var_start_idx + var_state_total_dim,
#                 var_start_idx:var_start_idx + var_state_total_dim
#             ].set(var_block_cov)
    
#     # Ensure positive definite
#     init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
#     return init_cov


def _build_gamma_based_p0_for_smoother(
    state_dim: int,
    n_dynamic_trends: int, 
    gamma_list: List[jnp.ndarray],
    n_stationary: int,
    var_order: int,
    gamma_scaling: float,
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder
) -> jnp.ndarray:
    """
    Build gamma-based P0 for smoother using the same logic as in gpm_numpyro_models.py.
    
    This ensures theoretical consistency between the NumPyro model and simulation smoother.
    Uses the actual unconditional autocovariances from the VAR system.
    """
    print(f"    Building gamma-based P0: state_dim={state_dim}, n_dynamic_trends={n_dynamic_trends}")
    
    # Start with identity matrix
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Large variance for dynamic trends (diffuse prior)
    if n_dynamic_trends > 0:
        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
            jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1e4
        )
        print(f"    Set diffuse prior for {n_dynamic_trends} dynamic trends")
    
    # VAR block using gamma matrices (the critical fix)
    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order
    
    if n_stationary > 0 and var_order > 0 and gamma_list:
        print(f"    Building VAR block: start_idx={var_start_idx}, total_dim={var_state_total_dim}")
        
        var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
        g0 = gamma_list[0]  # This is Gamma_0 - the key theoretical matrix
        
        for r_idx in range(var_order):
            for c_idx in range(var_order):
                lag_d = abs(r_idx - c_idx)
                
                # Get gamma matrix for this lag difference
                if lag_d < len(gamma_list) and gamma_list[lag_d] is not None:
                    blk_unscaled = gamma_list[lag_d]
                else:
                    # Fallback: exponential decay from Gamma_0
                    blk_unscaled = g0 * (0.5**lag_d)
                
                curr_blk = blk_unscaled * gamma_scaling
                
                # For non-diagonal blocks in the Toeplitz structure
                if r_idx > c_idx:
                    curr_blk = curr_blk.T
                
                # Insert block into VAR covariance matrix
                r_s, r_e = r_idx * n_stationary, (r_idx + 1) * n_stationary
                c_s, c_e = c_idx * n_stationary, (c_idx + 1) * n_stationary
                
                if r_e <= var_state_total_dim and c_e <= var_state_total_dim:
                    var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
        
        # Insert VAR block into full covariance matrix
        if var_start_idx + var_state_total_dim <= state_dim:
            init_cov = init_cov.at[
                var_start_idx:var_start_idx + var_state_total_dim,
                var_start_idx:var_start_idx + var_state_total_dim
            ].set(var_block_cov)
            print(f"    Successfully built VAR covariance block")
        else:
            print(f"    Warning: VAR block dimensions don't fit in state vector")
    
    # Ensure positive definite and symmetric
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Validate the matrix
    try:
        jnp.linalg.cholesky(init_cov)
        print(f"    ✓ Gamma-based P0 is positive definite")
    except Exception as e:
        print(f"    Warning: Gamma-based P0 not PSD, adding more jitter: {e}")
        init_cov = init_cov + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov



# def _create_standard_p0_for_smoother(state_dim: int, n_dynamic_trends: int) -> jnp.ndarray:
#     """
#     Create standard P0 for smoother (fallback when gamma-based P0 is not available)
#     """
#     init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e4
    
#     # More informative prior for non-trend states (VAR states)
#     if state_dim > n_dynamic_trends:
#         init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
#             jnp.eye(state_dim - n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1.0
#         )
    
#     return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


def _create_standard_p0_for_smoother(state_dim: int, n_dynamic_trends: int) -> jnp.ndarray:
    """
    Create standard P0 for smoother (fallback when gamma-based P0 is not available).
    """
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e4
    
    # More informative prior for non-trend states (VAR states)
    if state_dim > n_dynamic_trends:
        init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
            jnp.eye(state_dim - n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1.0
        )
    
    return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


#Ummm
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
    
    # Check what's in the non_core_trend_definitions
    print(f"Non-core trend definitions: {list(gpm_model.non_core_trend_definitions.keys())}")
    
    for name, expr_def in gpm_model.non_core_trend_definitions.items():
        print(f"\nNon-core trend: {name}")
        print(f"  Terms: {expr_def.terms}")
        print(f"  Constant: {expr_def.constant_str}")
        print(f"  Parameters: {expr_def.parameters}")
    
    # Check core variable mapping
    print(f"\nCore variables: {gpm_model.core_variables}")
    print(f"Stationary variables: {gpm_model.stationary_variables}")
    print(f"Core var map: {ss_builder.core_var_map}")
    
    # Create mapping from core states to time series
    current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
    
    # Dynamic trends
    dynamic_trend_count = 0
    for core_var_name in gpm_model.core_variables:
        if core_var_name not in gpm_model.stationary_variables:
            state_vector_idx = dynamic_trend_count
            if state_vector_idx < core_states_smoothed_draw.shape[1]:
                current_draw_core_state_values_ts[core_var_name] = core_states_smoothed_draw[:, state_vector_idx]
                print(f"  Mapped core trend {core_var_name} to state index {state_vector_idx}")
                print(f"    Mean value: {jnp.mean(current_draw_core_state_values_ts[core_var_name]):.4f}")
            dynamic_trend_count += 1
    
    # Stationary variables (current period)
    n_dynamic_trends = len([cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables])
    var_block_start_idx_in_state = n_dynamic_trends
    for i_stat_var, stat_var_name in enumerate(gpm_model.stationary_variables):
        state_vector_idx = var_block_start_idx_in_state + i_stat_var
        if state_vector_idx < core_states_smoothed_draw.shape[1]:
            current_draw_core_state_values_ts[stat_var_name] = core_states_smoothed_draw[:, state_vector_idx]
            print(f"  Mapped stationary var {stat_var_name} to state index {state_vector_idx}")
    
    print(f"\nAvailable core state values: {list(current_draw_core_state_values_ts.keys())}")
    
    # Now try to reconstruct each non-core trend step by step
    for orig_trend_name in gpm_model.gpm_trend_variables_original:
        if orig_trend_name in gpm_model.non_core_trend_definitions:
            print(f"\n--- Reconstructing {orig_trend_name} ---")
            expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
            
            # Start with constant
            const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
            print(f"  Constant term: {const_val_numeric}")
            
            reconstructed_value = jnp.full(T_data, const_val_numeric, dtype=_DEFAULT_DTYPE)
            
            # Add each term
            for var_key, coeff_str in expr_def.terms.items():
                term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                
                print(f"  Term: {coeff_str} * {var_key} (var={term_var_name}, lag={term_lag})")
                print(f"    Coefficient value: {coeff_numeric}")
                
                if term_lag == 0:
                    if term_var_name in current_draw_core_state_values_ts:
                        term_contribution = coeff_numeric * current_draw_core_state_values_ts[term_var_name]
                        reconstructed_value += term_contribution
                        print(f"    ✓ Found {term_var_name} in core states")
                        print(f"    Mean contribution: {jnp.mean(term_contribution):.4f}")
                    elif term_var_name in current_builder_params:
                        term_contribution = coeff_numeric * current_builder_params[term_var_name]
                        reconstructed_value += term_contribution
                        print(f"    ✓ Found {term_var_name} as parameter: {current_builder_params[term_var_name]}")
                    else:
                        print(f"    ✗ Could not find {term_var_name} in core states or parameters!")
                        print(f"      Available core states: {list(current_draw_core_state_values_ts.keys())}")
                        print(f"      Available parameters: {list(current_builder_params.keys())}")
                else:
                    print(f"    ⚠ Non-zero lag ({term_lag}) in non-core definition - this may be problematic")
            
            print(f"  Final reconstructed mean: {jnp.mean(reconstructed_value):.4f}")
            print(f"  Final reconstructed std: {jnp.std(reconstructed_value):.4f}")
    
    print("=== END DEBUG ===\n")


def fixed_extract_reconstructed_components_with_debug(
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray, 
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None,
    use_gamma_init_for_smoother: bool = True,
    gamma_init_scaling_for_smoother: float = 1.0,
    debug_first_draw: bool = True  # New parameter for debugging
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]:
    """
    Enhanced version with debugging for non-core trend reconstruction
    """
    print(f"\n=== SIMULATION SMOOTHER WITH DEBUG ===")
    
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

    # Use different draws
    if actual_num_smooth_draws == total_posterior_draws:
        draw_indices = np.arange(total_posterior_draws)
    else:
        draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
    num_dynamic_trends_in_state = ss_builder.n_core - ss_builder.n_stationary
    
    output_trend_draws_list_all_orig_trends = []
    output_stationary_draws_list_all_orig_stat = []

    for i_loop, mcmc_draw_idx in enumerate(draw_indices):
        rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
        # Extract parameters for THIS specific draw
        current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
        # Build SS matrices using THIS draw's parameters
        F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
        # Get proper P0 for THIS specific draw
        init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
        if init_mean_mcmc_val is not None and mcmc_draw_idx < init_mean_mcmc_val.shape[0]:
            init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx]
        else:
            init_mean_for_smoother = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
        # Build draw-specific initial covariance (using your previous implementation)
        if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
            gamma_list_for_this_draw = _extract_gamma_matrices_for_draw(
                mcmc_samples, mcmc_draw_idx, current_builder_params, 
                ss_builder.n_stationary, ss_builder.var_order
            )
            
            if gamma_list_for_this_draw is not None:
                init_cov_for_smoother = _build_gamma_based_p0_for_smoother(
                    ss_builder.state_dim, 
                    num_dynamic_trends_in_state,
                    gamma_list_for_this_draw,
                    ss_builder.n_stationary, 
                    ss_builder.var_order,
                    gamma_init_scaling_for_smoother,
                    gpm_model,
                    ss_builder
                )
            else:
                init_cov_for_smoother = _create_standard_p0_for_smoother(
                    ss_builder.state_dim, num_dynamic_trends_in_state
                )
        else:
            init_cov_for_smoother = _create_standard_p0_for_smoother(
                ss_builder.state_dim, num_dynamic_trends_in_state
            )

        # Regularize matrices
        Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        try: 
            R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
        except: 
            R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

        # Check matrices are finite
        if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and \
                jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and \
                jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
            continue
            
        # Run simulation smoother
        try:
            core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_sm_draw, C_draw, H_draw, init_mean_for_smoother, init_cov_for_smoother, sim_key
            )
            if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): 
                continue
        except Exception as e:
            continue

        # DEBUG: For first draw, run detailed debugging
        if debug_first_draw and i_loop == 0:
            debug_non_core_trend_reconstruction(
                gpm_model, core_states_smoothed_draw, current_builder_params, 
                ss_builder, T_data
            )

        # Reconstruct original variables from core states
        current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        
        # FIXED: Properly map dynamic trends
        dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
        for i, trend_name in enumerate(dynamic_trend_names):
            if i < core_states_smoothed_draw.shape[1]:
                current_draw_core_state_values_ts[trend_name] = core_states_smoothed_draw[:, i]
        
        # FIXED: Properly map stationary variables
        var_block_start = len(dynamic_trend_names)
        for i, stat_name in enumerate(gpm_model.stationary_variables):
            state_idx = var_block_start + i
            if state_idx < core_states_smoothed_draw.shape[1]:
                current_draw_core_state_values_ts[stat_name] = core_states_smoothed_draw[:, state_idx]

        # Reconstruct original trends
        reconstructed_trends_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        
        for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
            if orig_trend_name in current_draw_core_state_values_ts:
                # It's a core trend - use directly
                reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(
                    current_draw_core_state_values_ts[orig_trend_name]
                )
            elif orig_trend_name in gpm_model.non_core_trend_definitions:
                # It's a non-core trend - reconstruct from definition
                expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                
                # FIXED: Start with zeros and add constant
                reconstructed_value = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
                # Add constant term
                const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
                if const_val_numeric != 0:
                    reconstructed_value += const_val_numeric
                
                # Add variable terms
                for var_key, coeff_str in expr_def.terms.items():
                    term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                    coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params)
                    
                    if term_lag == 0:  # Only current period terms
                        if term_var_name in current_draw_core_state_values_ts:
                            reconstructed_value += coeff_numeric * current_draw_core_state_values_ts[term_var_name]
                        elif term_var_name in current_builder_params:
                            # Parameter used as a variable
                            reconstructed_value += coeff_numeric * current_builder_params[term_var_name]
                        else:
                            print(f"Warning: Could not find {term_var_name} for {orig_trend_name}")
                
                reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(reconstructed_value)
                
        output_trend_draws_list_all_orig_trends.append(reconstructed_trends_this_mcmc_draw)

        # Reconstruct original stationary variables
        reconstructed_stationary_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
            if orig_stat_name in current_draw_core_state_values_ts:
                reconstructed_stationary_this_mcmc_draw = reconstructed_stationary_this_mcmc_draw.at[:, i_orig_stat].set(
                    current_draw_core_state_values_ts[orig_stat_name]
                )
        output_stationary_draws_list_all_orig_stat.append(reconstructed_stationary_this_mcmc_draw)

    if not output_trend_draws_list_all_orig_trends:
        print("ERROR: No valid simulation smoother draws!")
        return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
               jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
               {'trends': list(gpm_model.gpm_trend_variables_original), 
                'stationary': list(gpm_model.gpm_stationary_variables_original)}

    final_reconstructed_trends = jnp.stack(output_trend_draws_list_all_orig_trends)
    final_reconstructed_stationary = jnp.stack(output_stationary_draws_list_all_orig_stat)
    
    component_names = {
        'trends': list(gpm_model.gpm_trend_variables_original),
        'stationary': list(gpm_model.gpm_stationary_variables_original)
    }
    
    return final_reconstructed_trends, final_reconstructed_stationary, component_names

def diagnose_state_variable_mapping(gpm_model: ReducedModel, ss_builder: StateSpaceBuilder):
    """
    Diagnose how variables are mapped to state vector indices
    """
    print("\n=== STATE VARIABLE MAPPING DIAGNOSIS ===")
    
    print(f"GPM trend variables (original): {gpm_model.gpm_trend_variables_original}")
    print(f"GPM stationary variables (original): {gpm_model.gpm_stationary_variables_original}")
    print(f"GPM observed variables (original): {gpm_model.gpm_observed_variables_original}")
    
    print(f"\nCore variables identified: {gpm_model.core_variables}")
    print(f"Stationary variables: {gpm_model.stationary_variables}")
    
    print(f"\nState space builder info:")
    print(f"  State dimension: {ss_builder.state_dim}")
    print(f"  n_core: {ss_builder.n_core}")
    print(f"  n_dynamic_trends: {ss_builder.n_dynamic_trends}")
    print(f"  n_stationary: {ss_builder.n_stationary}")
    print(f"  var_order: {ss_builder.var_order}")
    
    print(f"\nCore variable mapping (ss_builder.core_var_map):")
    for var, idx in ss_builder.core_var_map.items():
        print(f"  {var} -> state index {idx}")
    
    print(f"\nStationary variable mapping (ss_builder.stat_var_map):")
    for var, idx in ss_builder.stat_var_map.items():
        print(f"  {var} -> stat index {idx}")
    
    print(f"\nObserved variable mapping (ss_builder.obs_var_map):")
    for var, idx in ss_builder.obs_var_map.items():
        print(f"  {var} -> obs index {idx}")
    
    # Check non-core trend definitions
    print(f"\nNon-core trend definitions:")
    for trend_name, expr in gpm_model.non_core_trend_definitions.items():
        print(f"  {trend_name}:")
        print(f"    Terms: {expr.terms}")
        print(f"    Constant: {expr.constant_str}")
    
    # Check measurement equations
    print(f"\nReduced measurement equations:")
    for obs_var, expr in gpm_model.reduced_measurement_equations.items():
        print(f"  {obs_var}:")
        print(f"    Terms: {expr.terms}")
        print(f"    Constant: {expr.constant_str}")
    
    print("=== END DIAGNOSIS ===\n")