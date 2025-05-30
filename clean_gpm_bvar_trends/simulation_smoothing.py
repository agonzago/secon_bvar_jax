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

# --- Main Component Extraction and Reconstruction ---
def extract_reconstructed_components(
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray, # Not strictly needed if only using states, but good for context (T_data)
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder, # Passed in, already configured with gpm_model
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]: # Added names dict
    """
    Extracts and reconstructs all original GPM trend and stationary variables
    from MCMC simulation smoother draws of the CORE states.

    Returns:
        Tuple of (reconstructed_all_trends, reconstructed_all_stationary, component_names_dict):
            - reconstructed_all_trends: (num_draws, T, num_original_trends)
            - reconstructed_all_stationary: (num_draws, T, num_original_stationary_vars_in_var_block)
            - component_names_dict: {'trends': List[str_orig_trend_names], 'stationary': List[str_orig_stat_names]}
    """
    if rng_key_smooth is None: rng_key_smooth = random.PRNGKey(0)

    mcmc_samples = mcmc_output.get_samples()
    if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
        # print("Warning: MCMC samples are empty or invalid. Cannot extract components.")
        return jnp.empty((0,0,0)), jnp.empty((0,0,0)), {'trends':[], 'stationary':[]}

    T_data, _ = y_data.shape
    
    first_param_key = list(mcmc_samples.keys())[0] # Assuming all samples have same number of draws
    total_posterior_draws = mcmc_samples[first_param_key].shape[0]
    actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

    if actual_num_smooth_draws <= 0:
        # print("Warning: No draws selected for smoothing.")
        return jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original))), \
               jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original))), \
               {'trends': list(gpm_model.gpm_trend_variables_original), 
                'stationary': list(gpm_model.gpm_stationary_variables_original)}


    draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    
    # These are the number of "slots" in the core state vector
    num_dynamic_trends_in_state = ss_builder.n_core - ss_builder.n_stationary # Trends part of core_variables
    num_stat_vars_in_state_block = ss_builder.n_stationary # Number of unique stationary variables in VAR block

    output_trend_draws_list_all_orig_trends = []
    output_stationary_draws_list_all_orig_stat = []

    for i_loop, mcmc_draw_idx in enumerate(draw_indices):
        rng_key_smooth, sim_key = random.split(rng_key_smooth)
        
        # 1. Get parameters for current MCMC draw (builder-friendly keys)
        current_builder_params = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
        
        # 2. Build SS matrices for this draw
        F_draw, Q_draw, C_draw, H_draw = ss_builder._build_matrices_internal(current_builder_params)
        
        # 3. Get P0 for this draw
        init_mean_mcmc_val = mcmc_samples.get("init_mean_full")
        init_mean_for_smoother = init_mean_mcmc_val[mcmc_draw_idx] if init_mean_mcmc_val is not None and init_mean_mcmc_val.shape[0] > mcmc_draw_idx else jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        # For init_cov, use a reasonable fixed one for the smoother
        init_cov_for_smoother = _create_reasonable_initial_covariance(ss_builder.state_dim, num_dynamic_trends_in_state)

        Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        try: R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
        except: R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))

        if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(R_sm_draw)) and \
                jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and \
                jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
            # print(f"Warning: NaN in SS matrices or P0 for draw {mcmc_draw_idx}. Skipping this smoother draw.")
            continue
        try:
            core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_sm_draw, C_draw, H_draw, init_mean_for_smoother, init_cov_for_smoother, sim_key)
            if not jnp.all(jnp.isfinite(core_states_smoothed_draw)): continue
        except Exception: continue

        # --- 4. Reconstruction from core_states_smoothed_draw ---
        # core_states_smoothed_draw is (T_data, ss_builder.state_dim)
        # State vector order: [dynamic_core_trends_t, stat_vars_t, stat_vars_t-1, ..., stat_vars_t-p+1]

        # A. Create a dictionary of time series for all CURRENT PERIOD core states from this draw
        current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {} # var_name -> (T_data,) array
        
        # Populate dynamic core trends
        dynamic_trend_count = 0
        for core_var_name in gpm_model.core_variables: # This list includes dynamic trends and stationary vars
            if core_var_name not in gpm_model.stationary_variables: # It's a dynamic trend
                # Its index in the state vector is its order among dynamic trends
                state_vector_idx = dynamic_trend_count
                if state_vector_idx < num_dynamic_trends_in_state: # Safety
                    current_draw_core_state_values_ts[core_var_name] = core_states_smoothed_draw[:, state_vector_idx]
                dynamic_trend_count += 1
        
        # Populate current period stationary (VAR) states
        var_block_start_idx_in_state = num_dynamic_trends_in_state
        for i_stat_var, stat_var_name in enumerate(gpm_model.stationary_variables):
            # This is the index for the current value (lag 0) of this stationary variable
            state_vector_idx = var_block_start_idx_in_state + i_stat_var
            if state_vector_idx < ss_builder.state_dim: # Safety
                current_draw_core_state_values_ts[stat_var_name] = core_states_smoothed_draw[:, state_vector_idx]
        
        # B. Reconstruct original GPM trend variables
        # Output array for this MCMC draw, for all original GPM trends
        reconstructed_trends_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
            if orig_trend_name in gpm_model.core_variables and orig_trend_name not in gpm_model.stationary_variables:
                # It's a dynamic core trend, directly use its smoothed path
                if orig_trend_name in current_draw_core_state_values_ts:
                    reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(
                        current_draw_core_state_values_ts[orig_trend_name]
                    )
            elif orig_trend_name in gpm_model.non_core_trend_definitions:
                # It's a non-core trend; evaluate its definition
                expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                reconstructed_value_for_orig_trend_t = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
                
                # Add constant part, evaluated with current MCMC params
                const_val_numeric = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params)
                reconstructed_value_for_orig_trend_t += const_val_numeric

                # Add terms involving core variables or parameters
                for var_key_in_def, coeff_expr_str_in_def in expr_def.terms.items():
                    term_var_name, term_lag = ss_builder._parse_variable_key(var_key_in_def)
                    coeff_numeric = ss_builder._evaluate_coefficient_expression(coeff_expr_str_in_def, current_builder_params)
                    
                    # Non-core trend definitions should only depend on CURRENT values of core states (lag=0)
                    # or parameters (which ss_builder._evaluate_coefficient_expression handles if coeff_str IS a param name).
                    if term_lag == 0:
                        if term_var_name in current_draw_core_state_values_ts: # It's a core variable
                            reconstructed_value_for_orig_trend_t += coeff_numeric * current_draw_core_state_values_ts[term_var_name]
                        elif term_var_name in current_builder_params: # It's a parameter used as a variable
                            reconstructed_value_for_orig_trend_t += coeff_numeric * current_builder_params[term_var_name]
                        # else: print(f"Warning: Term '{term_var_name}' in def of '{orig_trend_name}' not found in core states or params.")
                    # else: print(f"Warning: Lagged term '{var_key_in_def}' in non-core trend def for '{orig_trend_name}'. Convention is current core vars.")
                reconstructed_trends_this_mcmc_draw = reconstructed_trends_this_mcmc_draw.at[:, i_orig_trend].set(reconstructed_value_for_orig_trend_t)
            # else: print(f"Info: Original trend '{orig_trend_name}' not core and no definition. Remains NaN.")
        output_trend_draws_list_all_orig_trends.append(reconstructed_trends_this_mcmc_draw)

        # C. Reconstruct original GPM stationary variables
        # These are typically the same as gpm_model.stationary_variables (the VAR states' current values)
        reconstructed_stationary_this_mcmc_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
            if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
                reconstructed_stationary_this_mcmc_draw = reconstructed_stationary_this_mcmc_draw.at[:, i_orig_stat].set(
                    current_draw_core_state_values_ts[orig_stat_name]
                )
            # else: If an original stationary var is not in current_draw_core_values_ts (i.e. not a VAR state), it's an issue or needs a static def.
                # print(f"Info: Original stationary var '{orig_stat_name}' not found as a core VAR state. Remains NaN.")
        output_stationary_draws_list_all_orig_stat.append(reconstructed_stationary_this_mcmc_draw)

    final_reconstructed_trends = jnp.stack(output_trend_draws_list_all_orig_trends) if output_trend_draws_list_all_orig_trends else \
                                 jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original)), dtype=_DEFAULT_DTYPE)
    final_reconstructed_stationary = jnp.stack(output_stationary_draws_list_all_orig_stat) if output_stationary_draws_list_all_orig_stat else \
                                     jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original)), dtype=_DEFAULT_DTYPE)
    
    component_names = {
        'trends': list(gpm_model.gpm_trend_variables_original),
        'stationary': list(gpm_model.gpm_stationary_variables_original)
    }
    return final_reconstructed_trends, final_reconstructed_stationary, component_names


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
    if not isinstance(draws, np.ndarray): draws_np = np.asarray(draws)
    else: draws_np = draws
    if draws_np.ndim < 1 or draws_np.shape[0] < 2 :
        nan_shape = draws_np.shape[1:] if draws_np.ndim > 1 else (1,) * max(1, draws_np.ndim-1)
        if not nan_shape : nan_shape = (1,) 
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
        if hdi_for_reshape.shape != (2, num_elements_per_draw): raise ValueError("HDI shape mismatch after potential transpose.")
        final_hdi_shape = (2,) + original_shape_after_draws
        hdi_full_shape = hdi_for_reshape.reshape(final_hdi_shape)
        low = np.asarray(hdi_full_shape[0, ...]); high = np.asarray(hdi_full_shape[1, ...])
        if np.any(np.isnan(low)) or np.any(np.isnan(high)): pass # print(f"Warning: HDI contains NaN values.")
        return {'low': low, 'high': high}
    except Exception as e:
        # print(f"Error during ArviZ HDI computation: {e}")
        return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE), 'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}
    
