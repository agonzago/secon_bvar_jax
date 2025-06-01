# clean_gpm_bvar_trends/gpm_prior_evaluator.py
# Refactored for clarity, strict error handling, and correct Gamma_0 usage.

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import pandas as pd
import os

# Import necessary components from other modules
from integration_orchestrator import create_integration_orchestrator

# Import relevant types and constants
from common_types import EnhancedBVARParams
from gpm_model_parser import ReducedModel
from state_space_builder import StateSpaceBuilder

# Assuming KalmanFilter and simulation smoother are importable
try:
    from Kalman_filter_jax import KalmanFilter
except ImportError:
    print("CRITICAL ERROR: KalmanFilter could not be imported. Functionality will be severely limited.")
    KalmanFilter = None

try:
    from simulation_smoothing import jarocinski_corrected_simulation_smoother
except ImportError:
    print("Warning: jarocinski_corrected_simulation_smoother not found. Simulation draws will be skipped if num_sim_draws > 0.")
    jarocinski_corrected_simulation_smoother = None

try:
    from stationary_prior_jax_simplified import make_stationary_var_transformation_jax, _JITTER as _SP_JITTER
except ImportError:
    print("CRITICAL ERROR: Could not import stationary transformation (make_stationary_var_transformation_jax). VAR model evaluation will fail.")
    make_stationary_var_transformation_jax = None
    _SP_JITTER = 1e-8

try:
    from reporting_plots import (
        plot_observed_and_fitted,
        plot_estimated_components,
        plot_observed_and_trend
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Could not import plotting functions from reporting_plots.py. Plotting will be disabled.")
    PLOTTING_AVAILABLE = False

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER

def _build_initial_mean_for_test(gpm_model: ReducedModel, ss_builder: StateSpaceBuilder) -> jnp.ndarray:
    """
    Builds the initial state mean vector ($x_0$) for testing with fixed parameters.
    Uses the mean from 'initval' in the GPM for dynamic trends. Stationary variable
    means (lag 0) can also be set via 'initval'; otherwise, they default to zero.
    Lags of stationary variables default to zero mean.

    Raises:
        ValueError: If a dynamic trend lacks a valid 'initval' entry or if 'initval'
                    is malformed for a core variable.
        RuntimeError: If state vector mapping is inconsistent or leads to out-of-bounds access.
    """
    state_dim = ss_builder.state_dim
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)

    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]

    for trend_name in dynamic_trend_names:
        if trend_name not in gpm_model.initial_values:
            raise ValueError(f"Build P0 Mean Error: Dynamic trend '{trend_name}' must have an 'initval' entry.")

        var_spec = gpm_model.initial_values[trend_name]
        if not (var_spec.init_dist == 'normal_pdf' and var_spec.init_params and len(var_spec.init_params) >= 1):
            raise ValueError(f"Build P0 Mean Error: 'initval' for dynamic trend '{trend_name}' requires 'normal_pdf' with at least a mean parameter.")

        state_idx = ss_builder.core_var_map.get(trend_name)
        if state_idx is None or not (0 <= state_idx < ss_builder.n_dynamic_trends): # Dynamic trends are first
            raise RuntimeError(f"Build P0 Mean Error: Builder map index {state_idx} for dynamic trend '{trend_name}' is invalid or out of expected dynamic trend block range (0-{ss_builder.n_dynamic_trends-1}).")

        mean_val = var_spec.init_params[0]
        init_mean = init_mean.at[state_idx].set(jnp.array(mean_val, dtype=_DEFAULT_DTYPE))

    # Set means for current period (lag 0) of stationary variables if specified in initval
    var_block_start_idx = ss_builder.n_dynamic_trends # Start of the full VAR block in state vector
    for i_stat_in_block, stat_var_name in enumerate(gpm_model.stationary_variables):
        if stat_var_name in gpm_model.initial_values:
            var_spec = gpm_model.initial_values[stat_var_name]
            if var_spec.init_dist == 'normal_pdf' and var_spec.init_params and len(var_spec.init_params) >= 1:
                mean_val = var_spec.init_params[0]
                # Index for the lag 0 component of this stationary variable
                idx_for_this_stat_lag0 = var_block_start_idx + i_stat_in_block
                if not (ss_builder.n_dynamic_trends <= idx_for_this_stat_lag0 < ss_builder.n_dynamic_trends + ss_builder.n_stationary):
                    raise RuntimeError(f"Build P0 Mean Error: Calculated index {idx_for_this_stat_lag0} for stationary var '{stat_var_name}' (lag 0) is out of expected block range.")
                init_mean = init_mean.at[idx_for_this_stat_lag0].set(jnp.array(mean_val, dtype=_DEFAULT_DTYPE))
            # else: # If initval for stationary var is not normal_pdf or missing params, its mean remains 0. No error needed.
            #    print(f"Info: 'initval' for stationary variable '{stat_var_name}' not 'normal_pdf' with mean. Mean defaults to 0.")
    if not jnp.all(jnp.isfinite(init_mean)):
        raise RuntimeError("Build P0 Mean Error: Resulting initial mean vector contains non-finite values.")
    return init_mean


def _resolve_parameter_value(param_key_base: str, 
                             param_values_input: Dict[str, Any],
                             estimated_params_gpm: Dict[str, Any], 
                             is_shock_std_dev: bool = False
                             ) -> float:
    val_resolved = None
    checked_keys = []

    # Path 1: param_values_input
    checked_keys.append(param_key_base)
    if param_key_base in param_values_input:
        val_resolved = param_values_input[param_key_base]
    elif is_shock_std_dev:
        sigma_key = f"sigma_{param_key_base}"
        checked_keys.append(sigma_key)
        if sigma_key in param_values_input:
            val_resolved = param_values_input[sigma_key]

    # Path 2: GPM priors (if not found in param_values_input)
    if val_resolved is None:
        prior_spec_to_use = None
        key_in_gpm_prior = None # To know which key was actually found

        checked_keys.append(f"GPM prior for '{param_key_base}'")
        if param_key_base in estimated_params_gpm:
            prior_spec_to_use = estimated_params_gpm[param_key_base]
            key_in_gpm_prior = param_key_base
        elif is_shock_std_dev:
            sigma_key_gpm = f"sigma_{param_key_base}"
            checked_keys.append(f"GPM prior for '{sigma_key_gpm}'")
            if sigma_key_gpm in estimated_params_gpm:
                prior_spec_to_use = estimated_params_gpm[sigma_key_gpm]
                key_in_gpm_prior = sigma_key_gpm
        
        if prior_spec_to_use: # A prior was found
            if prior_spec_to_use.distribution == 'inv_gamma_pdf' and len(prior_spec_to_use.params) >= 2:
                alpha, beta = prior_spec_to_use.params
                # Ensure alpha+1 is not zero or negative if alpha <= -1
                mode = beta / (alpha + 1.0) if alpha > -1.0 else None 
                if mode is not None and mode > 0:
                    val_resolved = mode
                elif is_shock_std_dev: # If mode is non-positive for a std dev from prior
                    # This is a case where a prior exists, but it's problematic for a std dev.
                    # Per "no fallbacks", this should be an error.
                    raise ValueError(f"Parameter '{key_in_gpm_prior}': Prior '{prior_spec_to_use.distribution}' implies non-positive std dev (mode: {mode}).")
                # If not a shock_std_dev, a non-positive mode from an InvGamma might be acceptable for other params.
                # However, the current test is for a shock.

            elif prior_spec_to_use.distribution == 'normal_pdf' and len(prior_spec_to_use.params) >= 1:
                mean_val = prior_spec_to_use.params[0]
                if is_shock_std_dev:
                    val_resolved = abs(mean_val)
                    if val_resolved == 0: # Strict: if abs(mean) is 0 for a std dev, it's an issue.
                         raise ValueError(f"Parameter '{key_in_gpm_prior}': Normal prior for std dev has zero mean, leading to zero std dev.")
                else:
                    val_resolved = mean_val
            else:
                raise ValueError(f"Parameter '{key_in_gpm_prior}': Unsupported prior distribution '{prior_spec_to_use.distribution}' or insufficient params in GPM estimated_params.")
    
    # Final check: if val_resolved is still None, it means it wasn't in param_values AND no prior was found/used.
    if val_resolved is None:
        raise ValueError(f"Parameter '{param_key_base}' could not be resolved. Attempted lookups: {', '.join(checked_keys)}.")
    
    try:
        val_float = float(val_resolved)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter '{param_key_base}': Resolved value '{val_resolved}' ('{type(val_resolved).__name__}') cannot be converted to float.") from e

    if is_shock_std_dev and val_float <= 0: # This check is after potential prior-based fallbacks.
        raise RuntimeError(f"Parameter '{param_key_base}' (shock std dev): Final resolved value must be positive, got {val_float}.")
    return val_float

def _build_trend_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    """Builds Sigma_eta (trend innovation covariance) using resolved parameters."""
    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
    if not dynamic_trend_names:
        return jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    num_dynamic_trends = len(dynamic_trend_names)
    trend_sigmas_sq = jnp.zeros(num_dynamic_trends, dtype=_DEFAULT_DTYPE)

    for idx_dynamic_trend, trend_name in enumerate(dynamic_trend_names):
        associated_shock_name = next((eq.shock for eq in gpm_model.core_equations if eq.lhs == trend_name), None)
        
        if associated_shock_name:
            # Resolve the shock's standard deviation
            sigma_val = _resolve_parameter_value(
                param_key_base=associated_shock_name,
                param_values_input=param_values_input,
                estimated_params_gpm=gpm_model.estimated_params,
                is_shock_std_dev=True
            )
            trend_sigmas_sq = trend_sigmas_sq.at[idx_dynamic_trend].set(sigma_val ** 2)
        # If no shock, variance remains 0 (deterministic RW component)
        
    Sigma_eta = jnp.diag(trend_sigmas_sq)
    if not jnp.all(jnp.isfinite(Sigma_eta)):
        raise RuntimeError("Built Sigma_eta for trends contains non-finite values.")
    return Sigma_eta

def _build_var_parameters(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """
    Builds Sigma_u, A_transformed, and gamma_list = [Gamma_0, ..., Gamma_{p-1}]
    for stationary VAR components using resolved parameters.
    Gamma_0 is the calculated unconditional contemporaneous covariance.
    """
    n_stationary = len(gpm_model.stationary_variables)
    if n_stationary == 0:
        return jnp.empty((0,0), dtype=_DEFAULT_DTYPE), jnp.empty((0,0,0), dtype=_DEFAULT_DTYPE), []

    if not gpm_model.var_prior_setup or not hasattr(gpm_model.var_prior_setup, 'var_order') or gpm_model.var_prior_setup.var_order <= 0:
        raise ValueError("VAR parameters require valid 'var_prior_setup' with var_order > 0 attribute.")
    
    setup = gpm_model.var_prior_setup
    n_vars, n_lags = n_stationary, setup.var_order

    mean_diag = setup.es[0] if setup.es and len(setup.es) > 0 else 0.0
    mean_offdiag = setup.es[1] if setup.es and len(setup.es) > 1 else 0.0
    raw_A_list = []
    for _ in range(n_lags):
        A_lag = jnp.full((n_vars, n_vars), mean_offdiag, dtype=_DEFAULT_DTYPE)
        A_lag = A_lag.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(mean_diag)
        raw_A_list.append(A_lag)

    if not gpm_model.stationary_shocks or len(gpm_model.stationary_shocks) != n_vars:
        raise ValueError(f"Mismatch between #stationary_variables ({n_vars}) and #stationary_shocks ({len(gpm_model.stationary_shocks if gpm_model.stationary_shocks else 'None')}).")

    stat_sigmas_std_dev = [_resolve_parameter_value(shock, param_values_input, gpm_model.estimated_params, True) for shock in gpm_model.stationary_shocks]
    sigma_u_diag_vec = jnp.array(stat_sigmas_std_dev, dtype=_DEFAULT_DTYPE)
    
    Omega_u_chol_val = param_values_input.get("_var_innovation_corr_chol")
    if Omega_u_chol_val is not None:
        if not isinstance(Omega_u_chol_val, (jnp.ndarray, np.ndarray)) or Omega_u_chol_val.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(Omega_u_chol_val)):
            raise ValueError(f"_var_innovation_corr_chol is invalid. Shape: {Omega_u_chol_val.shape if hasattr(Omega_u_chol_val, 'shape') else 'N/A' }, Expected: ({n_vars},{n_vars}).")
        Omega_u_chol_val = jnp.asarray(Omega_u_chol_val, dtype=_DEFAULT_DTYPE) # Ensure JAX array
    else: 
        Omega_u_chol_val = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
        
    Sigma_u = jnp.diag(sigma_u_diag_vec) @ Omega_u_chol_val @ Omega_u_chol_val.T @ jnp.diag(sigma_u_diag_vec)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _SP_JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    if not jnp.all(jnp.isfinite(Sigma_u)): raise RuntimeError("Built Sigma_u (VAR innovation cov) contains non-finite values.")
    try: jnp.linalg.cholesky(Sigma_u)
    except Exception as e_chol: raise RuntimeError(f"Built Sigma_u not PSD: {e_chol}") from e_chol

    if make_stationary_var_transformation_jax is None:
        raise RuntimeError("make_stationary_var_transformation_jax is not available for VAR processing.")

    try:
        # Expects make_stationary_var_transformation_jax to return:
        # phi_list: [Phi_1, ..., Phi_p]
        # gamma_list_from_transform: [Gamma_0, Gamma_1, ..., Gamma_{p-1}] (length p)
        phi_list, gamma_list_from_transform = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
        
        if not (phi_list and len(phi_list) == n_lags and all(p_mat is not None and p_mat.shape == (n_vars,n_vars) and jnp.all(jnp.isfinite(p_mat)) for p_mat in phi_list)):
            raise RuntimeError(f"Stationary transformation returned invalid phi_list (expected len {n_lags}, got {len(phi_list) if phi_list else 'None'}, or elements invalid).")
        A_transformed = jnp.stack(phi_list)

        if not (gamma_list_from_transform and len(gamma_list_from_transform) == n_lags and
                all(g is not None and g.shape == (n_vars, n_vars) and jnp.all(jnp.isfinite(g)) for g in gamma_list_from_transform)):
            raise RuntimeError(f"Stationary transformation returned invalid gamma_list (expected len {n_lags} for [Gamma_0...Gamma_{{{n_lags-1}}}], got {len(gamma_list_from_transform) if gamma_list_from_transform else 'None'}, or elements invalid).")
        
        final_gamma_list_for_P0 = gamma_list_from_transform

    except Exception as e:
        raise RuntimeError(f"Stationary VAR transformation failed: {e}") from e

    return Sigma_u, A_transformed, final_gamma_list_for_P0

def _build_measurement_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    """Builds measurement error covariance H (Sigma_eps)."""
    n_observed = len(gpm_model.gpm_observed_variables_original)
    if n_observed == 0: return jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    
    # Placeholder: Current GPM structure does not explicitly define measurement error shocks or their priors.
    # If they were defined (e.g., "stderr_obs_VAR1"), logic similar to _build_trend_covariance
    # using _resolve_parameter_value would be used here.
    # For now, assume a small, fixed diagonal H if not specified in param_values_input.
    
    H_val = param_values_input.get("_measurement_error_cov_full") # Check if a full H is provided
    if H_val is not None:
        if not isinstance(H_val, (jnp.ndarray, np.ndarray)) or H_val.shape != (n_observed, n_observed) or not jnp.all(jnp.isfinite(H_val)):
             raise ValueError(f"Provided _measurement_error_cov_full is invalid. Shape: {H_val.shape if hasattr(H_val,'shape') else 'N/A'}, Expected: ({n_observed},{n_observed})")
        H = jnp.asarray(H_val, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_observed, dtype=_DEFAULT_DTYPE) # Ensure PSD
    else:
        # Default to a small diagonal matrix if no specific measurement error parameters are handled.
        H = jnp.eye(n_observed, dtype=_DEFAULT_DTYPE) * 1e-5 # Very small default
    
    if not jnp.all(jnp.isfinite(H)): raise RuntimeError("Built measurement covariance H contains non-finite values.")
    try: jnp.linalg.cholesky(H)
    except Exception as e_chol_h: raise RuntimeError(f"Built measurement covariance H not PSD: {e_chol_h}") from e_chol_h
    return H


def evaluate_gpm_at_parameters(gpm_file_path: str,
                                 y: jnp.ndarray,
                                 param_values: Dict[str, Any],
                                 num_sim_draws: int = 50,
                                 rng_key: jax.Array = random.PRNGKey(42),
                                 plot_results: bool = True,
                                 variable_names: Optional[List[str]] = None,
                                 use_gamma_init_for_test: bool = True,
                                 gamma_init_scaling: float = 1.0,
                                 hdi_prob: float = 0.9
                                 ) -> Dict[str, Any]:
    """
    Evaluates the GPM model's state space representation at fixed parameter values.
    Raises errors on failure.
    """
    print(f"\n--- Evaluating GPM: {gpm_file_path} at fixed parameters ---")
    if not os.path.exists(gpm_file_path): raise FileNotFoundError(f"GPM file not found: {gpm_file_path}")
    if KalmanFilter is None: raise RuntimeError("KalmanFilter is not available for evaluation.")

    orchestrator = create_integration_orchestrator(gpm_file_path, strict_validation=True)
    gpm_model = orchestrator.reduced_model
    ss_builder = orchestrator.ss_builder
    T, n_obs_data = y.shape

    print(f"  State Dim: {ss_builder.state_dim}, Dynamic Trends: {ss_builder.n_dynamic_trends}, Stat Vars: {ss_builder.n_stationary}, VAR Order: {ss_builder.var_order if ss_builder.n_stationary > 0 else 'N/A'}")

    # Resolve structural parameters
    structural_params_resolved = {}
    for param_name in gpm_model.parameters:
        structural_params_resolved[param_name] = jnp.array(
            _resolve_parameter_value(param_name, param_values, gpm_model.estimated_params, False),
            dtype=_DEFAULT_DTYPE
        )

    Sigma_eta = _build_trend_covariance(gpm_model, param_values)
    Sigma_u_innov, A_transformed_coeffs, gamma_list_for_P0 = _build_var_parameters(gpm_model, param_values)
    H_measurement_error = _build_measurement_covariance(gpm_model, param_values)

    init_mean = _build_initial_mean_for_test(gpm_model, ss_builder)
    if use_gamma_init_for_test and ss_builder.n_stationary > 0 and ss_builder.var_order > 0:
        init_cov = _build_initial_covariance_for_test(
            ss_builder.state_dim, ss_builder.n_dynamic_trends,
            gamma_list_for_P0,
            ss_builder.n_stationary, ss_builder.var_order, gamma_init_scaling
        )
    else:
        if use_gamma_init_for_test: print("  Info: Gamma P0 requested but conditions not met (no stationary vars or var_order=0). Using standard P0.")
        init_cov = jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE) * 1e6
        if ss_builder.n_dynamic_trends < ss_builder.state_dim:
            non_trend_dim = ss_builder.state_dim - ss_builder.n_dynamic_trends
            init_cov = init_cov.at[ss_builder.n_dynamic_trends:, ss_builder.n_dynamic_trends:].set(
                jnp.eye(non_trend_dim) * 1.0)
        init_cov = (init_cov + init_cov.T)/2.0 + _KF_JITTER * jnp.eye(ss_builder.state_dim)
        try: jnp.linalg.cholesky(init_cov)
        except Exception as e_chol_std: raise RuntimeError(f"Standard P0 not PSD: {e_chol_std}") from e_chol_std

    params_for_ss_builder = EnhancedBVARParams(
        A=A_transformed_coeffs, Sigma_u=Sigma_u_innov, Sigma_eta=Sigma_eta,
        structural_params=structural_params_resolved, Sigma_eps=H_measurement_error
    )
    F, Q, C, H_obs_matrix = orchestrator.build_ss_from_enhanced_bvar(params_for_ss_builder)

    # Final numerical checks for matrices from builder
    for mat, name in zip([F, Q, C, H_obs_matrix], ["F", "Q", "C", "H_obs (from builder)"]):
        if not jnp.all(jnp.isfinite(mat)): raise RuntimeError(f"Builder Matrix {name} contains non-finite values.")
    try: jnp.linalg.cholesky(Q + _KF_JITTER * jnp.eye(Q.shape[0]))
    except Exception as e: raise RuntimeError(f"Builder State Cov Q not PSD: {e}")
    if H_obs_matrix.shape[0] > 0:
        try: jnp.linalg.cholesky(H_obs_matrix + _KF_JITTER * jnp.eye(H_obs_matrix.shape[0]))
        except Exception as e: raise RuntimeError(f"Builder Measurement Cov H_obs not PSD: {e}")

    R_sim_chol_Q = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(Q.shape[0]))

    kf = KalmanFilter(T=F, R=R_sim_chol_Q, C=C, H=H_obs_matrix, init_x=init_mean, init_P=init_cov)
    valid_obs_idx = jnp.arange(n_obs_data)
    I_obs = jnp.eye(n_obs_data)

    loglik = kf.log_likelihood(y, valid_obs_idx, n_obs_data, C, H_obs_matrix, I_obs)
    if not jnp.isfinite(loglik):
        raise RuntimeError(f"Log-likelihood is non-finite ({loglik}). Check parameters or model stability.")
    print(f"  Log-likelihood: {loglik:.3f}")

    filter_results = kf.filter(y, valid_obs_idx, n_obs_data, C, H_obs_matrix, I_obs)
    # smoothed_means, smoothed_covs = kf.smooth(y, filter_results=filter_results,
    #                                           static_valid_obs_idx_for_filter=valid_obs_idx,
    #                                           static_n_obs_actual_for_filter=n_obs_data,
    #                                           static_C_obs_for_filter=C, static_H_obs_for_filter=H_obs_matrix,
    #                                           static_I_obs_for_filter=I_obs)
    smoothed_means, smoothed_covs = kf.smooth(
        y, 
        filter_results=filter_results
    )
    

    sim_draws_stacked = jnp.empty((0, T, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)
    if num_sim_draws > 0:
        if jarocinski_corrected_simulation_smoother is None:
            print("  Warning: Simulation smoother function not available. Skipping simulation draws.")
        else:
            sim_draws_list = []
            print(f"  Running simulation smoother for {num_sim_draws} draws...")
            for i in range(num_sim_draws):
                rng_key, sim_key = random.split(rng_key)
                try:
                    s_states = jarocinski_corrected_simulation_smoother(
                        y, F, R_sim_chol_Q, C, H_obs_matrix, init_mean, init_cov, sim_key)
                    if jnp.all(jnp.isfinite(s_states)): sim_draws_list.append(s_states)
                    else: print(f"  Warning: Sim draw {i+1} had non-finite values, discarding.")
                except Exception as e_sim: print(f"  Warning: Sim draw {i+1} failed: {e_sim}, discarding.")
            if sim_draws_list: sim_draws_stacked = jnp.stack(sim_draws_list)
            print(f"  Completed {sim_draws_stacked.shape[0]} simulation draws.")

    # Reconstruct original GPM variables
    reconstructed_all_trends = jnp.full((sim_draws_stacked.shape[0], T, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
    reconstructed_all_stationary = jnp.full((sim_draws_stacked.shape[0], T, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)

    if sim_draws_stacked.shape[0] > 0:
        for i_draw in range(sim_draws_stacked.shape[0]):
            core_states_draw_t = sim_draws_stacked[i_draw]
            current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
            
            # Populate with core state variable time series from the current draw
            # Dynamic trends
            for i, trend_name in enumerate([cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]):
                state_idx = ss_builder.core_var_map.get(trend_name)
                if state_idx is not None and state_idx < ss_builder.n_dynamic_trends:
                     current_draw_core_state_values_ts[trend_name] = core_states_draw_t[:, state_idx]
            
            # Stationary variables (lag 0)
            var_block_start = ss_builder.n_dynamic_trends
            for i, stat_name in enumerate(gpm_model.stationary_variables):
                state_idx = ss_builder.core_var_map.get(stat_name) # Should be var_block_start + i
                if state_idx is not None and var_block_start <= state_idx < var_block_start + ss_builder.n_stationary :
                     current_draw_core_state_values_ts[stat_name] = core_states_draw_t[:, state_idx]

            # Reconstruct original GPM trend variables
            for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
                if orig_trend_name in current_draw_core_state_values_ts:
                    reconstructed_all_trends = reconstructed_all_trends.at[i_draw, :, i_orig_trend].set(
                        current_draw_core_state_values_ts[orig_trend_name])
                elif orig_trend_name in gpm_model.non_core_trend_definitions:
                    expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
                    val_t = jnp.zeros(T, dtype=_DEFAULT_DTYPE) + ss_builder._evaluate_coefficient_expression(expr_def.constant_str, structural_params_resolved)
                    for var_key, coeff_str in expr_def.terms.items():
                        term_var_name, term_lag = ss_builder._parse_variable_key(var_key)
                        coeff_num = ss_builder._evaluate_coefficient_expression(coeff_str, structural_params_resolved)
                        if term_lag == 0: # Non-core defs usually depend on current values of core states or parameters
                            if term_var_name in current_draw_core_state_values_ts:
                                val_t += coeff_num * current_draw_core_state_values_ts[term_var_name]
                            elif term_var_name in structural_params_resolved: # Parameter used as a term
                                val_t += coeff_num * structural_params_resolved[term_var_name]
                    reconstructed_all_trends = reconstructed_all_trends.at[i_draw, :, i_orig_trend].set(val_t)

            # Reconstruct original GPM stationary variables (typically current VAR states)
            for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
                if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
                     reconstructed_all_stationary = reconstructed_all_stationary.at[i_draw, :, i_orig_stat].set(
                          current_draw_core_state_values_ts[orig_stat_name])
    
    # Smoothed means for core components (dynamic trends part and current stationary part)
    smoothed_means_core_trends = smoothed_means[:, :ss_builder.n_dynamic_trends]
    smoothed_means_core_stationary_t = smoothed_means[:, ss_builder.n_dynamic_trends : ss_builder.n_dynamic_trends + ss_builder.n_stationary]

    if plot_results and PLOTTING_AVAILABLE:
        print("  Generating plots...")
        y_np = np.asarray(y)
        obs_var_names_for_plot = variable_names if variable_names and len(variable_names) == n_obs_data else gpm_model.gpm_observed_variables_original
        
        if reconstructed_all_trends.shape[0] > 0 and reconstructed_all_stationary.shape[0] > 0:
            plot_observed_and_fitted(y_np, reconstructed_all_trends, reconstructed_all_stationary, hdi_prob, obs_var_names_for_plot)
            plot_estimated_components(reconstructed_all_trends, reconstructed_all_stationary, hdi_prob,
                                      gpm_model.gpm_trend_variables_original, gpm_model.gpm_stationary_variables_original)
            if len(obs_var_names_for_plot) == reconstructed_all_trends.shape[2]: # If counts match for direct plotting
                 plot_observed_and_trend(y_np, reconstructed_all_trends, hdi_prob, obs_var_names_for_plot)
        else: print("  Skipping plots as no valid simulation draws for reconstructed components.")

    print(f"--- Evaluation for {gpm_file_path} complete ---")
    return {
        'loglik': loglik,
        'smoothed_means_core_state': smoothed_means, 'smoothed_covs_core_state': smoothed_covs,
        'sim_draws_core_state': sim_draws_stacked,
        'reconstructed_original_trends': reconstructed_all_trends,
        'reconstructed_original_stationary': reconstructed_all_stationary,
        'smoothed_means_core_trends': smoothed_means_core_trends,
        'smoothed_means_core_stationary_t': smoothed_means_core_stationary_t,
        'F': F, 'Q': Q, 'C': C, 'H_obs': H_obs_matrix, 'P0': init_cov,
        'gpm_model': gpm_model, 'ss_builder': ss_builder,
        'params_evaluated_enhanced': params_for_ss_builder
    }

def test_gpm_prior_evaluator_module():
    """Comprehensive test function for the gpm_prior_evaluator module."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST FOR: gpm_prior_evaluator.py MODULE")
    print("="*70)

    test_dir = "temp_evaluator_test_files"
    os.makedirs(test_dir, exist_ok=True)
    data_file = os.path.join(test_dir, "test_eval_data.csv")
    gpm_file = os.path.join(test_dir, "test_eval_model.gpm")

    obs_names = ['OBS1', 'OBS2']
    np.random.seed(123)
    T_data, N_obs = 100, 2
    y_test_data = np.random.randn(T_data, N_obs) * 0.5 + np.arange(T_data)[:, None] * 0.01
    pd.DataFrame(y_test_data, columns=obs_names).to_csv(data_file, index=False)

    # GPM content MUST align with param_values for resolution, and shocks
    gpm_content_test = """
parameters kappa;
estimated_params;
    kappa, normal_pdf, 0.5, 0.1;
    stderr SHK_DT1, inv_gamma_pdf, 2.1, 0.2;  
    stderr SHK_DT2, inv_gamma_pdf, 3, 0.2; 
    stderr SHK_STAT_A, inv_gamma_pdf, 3, 0.6;   
    stderr SHK_STAT_B, inv_gamma_pdf, 3, 0.6;
end;
trends_vars DT1, DT2, NONCORE_T;
stationary_variables STAT_A, STAT_B;
trend_shocks; 
    var SHK_DT1; 
    var SHK_DT2; 
end;
shocks; 
    var SHK_STAT_A; 
    var SHK_STAT_B; 
end;
trend_model;
    DT1 = DT1(-1) + SHK_DT1;
    DT2 = DT2(-1) + SHK_DT2;
    NONCORE_T = kappa * DT1 + (1-kappa) * DT2;      
end;
varobs OBS1, OBS2;
measurement_equations;
    OBS1 = DT1 + STAT_A;
    OBS2 = NONCORE_T + STAT_B;
end;
var_prior_setup; 
    var_order=1; 
        es=0.8,0.3; 
        fs=0.1,0.1; 
        gs=1,1; 
        hs=1,1; 
        eta=2; 
end;
initval;
    DT1, normal_pdf, 0.1, 1.5;
    DT2, normal_pdf, -0.1, 1.5;
    NONCORE_T, normal_pdf, 0.0, 1.5;
end;
"""
    with open(gpm_file, "w") as f: f.write(gpm_content_test)

    # param_values for testing - provide some, let others be derived from prior
    test_params = {
        "kappa": 0.6, # Override prior mean
        "SHK_DT1": 0.05, # Override prior mode
        # SHK_DT2 will use its prior mean (abs(0.1)=0.1)
        # SHK_STAT_A will use its prior mode (0.015)
        "SHK_STAT_B": 0.25, # Override prior mean for sigma_SHK_STAT_B
        # For VAR part, _var_innovation_corr_chol can be specified if non-identity needed
        "_var_innovation_corr_chol": jnp.array([[1.0, 0.0], [0.1, jnp.sqrt(1-0.1**2)]], dtype=_DEFAULT_DTYPE)
    }
    
    all_tests_passed = True
    def run_sub_test(name, use_gamma, expected_success=True):
        nonlocal all_tests_passed
        print(f"\n--- Sub-test: {name} (Gamma P0: {use_gamma}, Expect Success: {expected_success}) ---")
        evaluation_succeeded_without_exception = False
        temp_results = None
        try:
            temp_results = evaluate_gpm_at_parameters( # Assign to temp_results
                gpm_file_path=gpm_file, y=jnp.array(y_test_data),
                param_values=test_params,
                num_sim_draws=5, plot_results=False, 
                variable_names=obs_names, 
                use_gamma_init_for_test=use_gamma,
                gamma_init_scaling=0.8
            )
            evaluation_succeeded_without_exception = True # Mark success if no exception
            if not expected_success:
                print(f"❌ Test '{name}' SUCCEEDED BUT EXPECTED FAILURE.")
                all_tests_passed = False
            else:
                print(f"✓ Test '{name}' PASSED. Loglik: {temp_results['loglik']:.2f}")
                assert jnp.isfinite(temp_results['loglik'])
                # ... other assertions ...
        except Exception as e:
            if expected_success:
                print(f"❌ Test '{name}' FAILED UNEXPECTEDLY: {type(e).__name__}: {e}")
                all_tests_passed = False
            else:
                print(f"✓ Test '{name}' FAILED AS EXPECTED: {type(e).__name__}: {e}")
        
        # Return logic based on actual success and expected success
        if expected_success:
            return temp_results if evaluation_succeeded_without_exception else None
        else: # Expected failure
            return None # Or you could return a specific marker for "expected failure occurred"

    run_sub_test("Standard P0 evaluation", use_gamma=False)
    run_sub_test("Gamma P0 evaluation", use_gamma=True)

    # Test missing initval for a dynamic trend
    gpm_missing_initval_content = gpm_content_test.replace("DT2, normal_pdf, -0.1, 1.5;", "")
    with open(gpm_file, "w") as f: f.write(gpm_missing_initval_content)
    run_sub_test("Missing initval for DT2", use_gamma=False, expected_success=False) # Expect ValueError

    # Test unresolvable parameter (structural)
    gpm_missing_kappa_prior_content = gpm_content_test.replace("kappa, normal_pdf, 0.5, 0.1;", "")
    with open(gpm_file, "w") as f: f.write(gpm_missing_kappa_prior_content)
    test_params_no_kappa = {k:v for k,v in test_params.items() if k != "kappa"}
    print(f"\n--- Sub-test: Unresolvable structural parameter 'kappa' (no value, no prior) ---")
    try:
        evaluate_gpm_at_parameters(gpm_file, jnp.array(y_test_data), test_params_no_kappa, num_sim_draws=0, plot_results=False)
        print("❌ Test 'Unresolvable kappa' SUCCEEDED BUT EXPECTED FAILURE."); all_tests_passed = False
    except ValueError as e: print(f"✓ Test 'Unresolvable kappa' FAILED AS EXPECTED: {e}")
    except Exception as e: print(f"❌ Test 'Unresolvable kappa' FAILED WITH WRONG ERROR: {type(e).__name__}: {e}"); all_tests_passed = False


    # Test unresolvable shock std dev
    #gpm_missing_shkdt1_prior_content = gpm_content_test.replace("stderr SHK_DT1, inv_gamma_pdf, 2, 0.02;", "")
    gpm_missing_shkdt1_prior_content = gpm_content_test.replace("stderr SHK_DT1, inv_gamma_pdf, 2.1, 0.2;", "")
    with open(gpm_file, "w") as f: f.write(gpm_missing_shkdt1_prior_content)
    test_params_no_shkdt1 = {k:v for k,v in test_params.items() if k not in ["SHK_DT1", "sigma_SHK_DT1"]}
    print(f"\n--- Sub-test: Unresolvable shock 'SHK_DT1' (no value, no prior) ---")
    try:
        evaluate_gpm_at_parameters(gpm_file, jnp.array(y_test_data), test_params_no_shkdt1, num_sim_draws=0, plot_results=False)
        print("❌ Test 'Unresolvable SHK_DT1' SUCCEEDED BUT EXPECTED FAILURE."); all_tests_passed = False
    except ValueError as e: print(f"✓ Test 'Unresolvable SHK_DT1' FAILED AS EXPECTED: {e}")
    except Exception as e: print(f"❌ Test 'Unresolvable SHK_DT1' FAILED WITH WRONG ERROR: {type(e).__name__}: {e}"); all_tests_passed = False

    # Cleanup
    if os.path.exists(data_file): os.remove(data_file)
    if os.path.exists(gpm_file): os.remove(gpm_file)
    if os.path.exists(test_dir): os.rmdir(test_dir) # Remove dir if empty

    print("\n" + "="*70)
    if all_tests_passed:
        print("✅ ALL gpm_prior_evaluator.py MODULE TESTS PASSED!")
    else:
        print("❌ SOME gpm_prior_evaluator.py MODULE TESTS FAILED.")
    print("="*70)
    return all_tests_passed

def _build_initial_covariance_for_test(
    state_dim: int,
    n_dynamic_trends: int,
    gamma_list: List[jnp.ndarray], # Expected: [Gamma_0, Gamma_1, ..., Gamma_{p-1}] (length p = var_order)
    n_stationary: int,
    var_order: int,
    gamma_scaling: float = 1.0
) -> jnp.ndarray:
    """
    Builds the initial state covariance matrix P0 for testing with fixed parameters.
    Uses diffuse prior for dynamic trends. For the stationary part (if var_order > 0),
    it uses the theoretical VAR unconditional autocovariances from gamma_list.
    gamma_list[0] MUST be the calculated unconditional covariance Gamma_0.

    Args:
        state_dim: Total dimension of the state vector.
        n_dynamic_trends: Number of dynamic trend components in the state vector.
        gamma_list: List of autocovariance matrices [Gamma_0, ..., Gamma_{var_order-1}].
        n_stationary: Number of stationary variables in the VAR.
        var_order: Order of the VAR model (p).
        gamma_scaling: Scaling factor to apply to the gamma matrices.

    Returns:
        The initial state covariance matrix P0.

    Raises:
        RuntimeError: If gamma_list is invalid for gamma-based P0, if matrix construction
                      fails (e.g. non-finite values, not PSD), or if dimensions mismatch.
    """
    if not isinstance(state_dim, int) or state_dim <= 0:
        raise ValueError("state_dim must be a positive integer.")
    if not isinstance(n_dynamic_trends, int) or n_dynamic_trends < 0:
        raise ValueError("n_dynamic_trends must be a non-negative integer.")
    if not isinstance(n_stationary, int) or n_stationary < 0:
        raise ValueError("n_stationary must be a non-negative integer.")
    if not isinstance(var_order, int) or var_order < 0:
        raise ValueError("var_order must be a non-negative integer.")
    if n_dynamic_trends + n_stationary * var_order != state_dim:
        raise ValueError(f"Dimension mismatch: n_dynamic_trends ({n_dynamic_trends}) + "
                         f"n_stationary ({n_stationary}) * var_order ({var_order}) "
                         f"!= state_dim ({state_dim}).")

    init_cov = jnp.zeros((state_dim, state_dim), dtype=_DEFAULT_DTYPE)

    # 1. Diffuse prior for dynamic trends (large variance)
    if n_dynamic_trends > 0:
        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
            jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * 1e6
        )

    # 2. Stationary VAR part using gamma_list
    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order

    if n_stationary > 0 and var_order > 0:
        # Validate gamma_list: must have 'var_order' elements (Gamma_0 to Gamma_{var_order-1})
        if not gamma_list or len(gamma_list) != var_order:
            raise RuntimeError(
                f"Build P0 Cov Error: Gamma-based P0 requires 'gamma_list' with length "
                f"equal to 'var_order' ({var_order}). Got length {len(gamma_list) if gamma_list is not None else 'None'}."
            )

        # Pre-allocate the block for VAR states' covariance
        var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
        
        # Populate the block Toeplitz matrix for VAR states
        for r_block_idx in range(var_order):  # Index for the block row (0 to p-1)
            for c_block_idx in range(var_order):  # Index for the block column (0 to p-1)
                # c_idx_block IS DEFINED HERE for this inner loop's scope
                
                lag_h = abs(r_block_idx - c_block_idx) # This is the lag h for Gamma_h

                # gamma_list is [Gamma_0, Gamma_1, ..., Gamma_{p-1}]
                # Access Gamma_h using gamma_list[lag_h]
                if not (0 <= lag_h < len(gamma_list)): # This check should ideally always pass given len(gamma_list) == var_order
                     raise RuntimeError(f"Build P0 Cov Error: Calculated lag difference {lag_h} is out of bounds for gamma_list (len {len(gamma_list)}). This indicates an internal logic error with var_order.")

                gamma_h_matrix = gamma_list[lag_h]
                
                # Validate the fetched Gamma_h matrix
                if gamma_h_matrix is None or \
                   gamma_h_matrix.shape != (n_stationary, n_stationary) or \
                   not jnp.all(jnp.isfinite(gamma_h_matrix)):
                    raise RuntimeError(
                        f"Build P0 Cov Error: gamma_list[{lag_h}] (Gamma_{lag_h}) is invalid. "
                        f"Shape: {gamma_h_matrix.shape if hasattr(gamma_h_matrix, 'shape') else 'None'}, "
                        f"Expected: ({n_stationary},{n_stationary}), or contains non-finite values."
                    )

                current_block_scaled = gamma_h_matrix * gamma_scaling

                # Define slice indices for inserting the current_block_scaled
                # These must use r_block_idx and c_block_idx
                row_start_slice = r_block_idx * n_stationary
                row_end_slice = (r_block_idx + 1) * n_stationary
                col_start_slice = c_block_idx * n_stationary # Uses c_idx_block
                col_end_slice = (c_block_idx + 1) * n_stationary # Uses c_idx_block
                
                var_block_cov = var_block_cov.at[row_start_slice:row_end_slice, col_start_slice:col_end_slice].set(current_block_scaled)

        # for r_block_idx in range(var_order):  # Index for the block row (0 to p-1)
        #     for c_block_idx in range(var_order):  # Index for the block column (0 to p-1)
                
                # lag_h = abs(r_block_idx - c_block_idx) # This is the lag h for Gamma_h

                # # gamma_list is [Gamma_0, Gamma_1, ..., Gamma_{p-1}]
                # # Access Gamma_h using gamma_list[lag_h]
                # if not (0 <= lag_h < len(gamma_list)): # This check should ideally always pass given len(gamma_list) == var_order
                #      raise RuntimeError(f"Build P0 Cov Error: Calculated lag difference {lag_h} is out of bounds for gamma_list (len {len(gamma_list)}). This indicates an internal logic error with var_order.")

                # gamma_h_matrix = gamma_list[lag_h]
                
                # # Validate the fetched Gamma_h matrix
                # if gamma_h_matrix is None or \
                #    gamma_h_matrix.shape != (n_stationary, n_stationary) or \
                #    not jnp.all(jnp.isfinite(gamma_h_matrix)):
                #     raise RuntimeError(
                #         f"Build P0 Cov Error: gamma_list[{lag_h}] (Gamma_{lag_h}) is invalid. "
                #         f"Shape: {gamma_h_matrix.shape if hasattr(gamma_h_matrix, 'shape') else 'None'}, "
                #         f"Expected: ({n_stationary},{n_stationary}), or contains non-finite values."
                #     )

                # current_block_scaled = gamma_h_matrix * gamma_scaling
                
        #         # For off-diagonal blocks, use transpose if it's the lower part of the block Toeplitz
        #         if r_block_idx > c_block_idx: 
        #             current_block_scaled = current_block_scaled.T
                
        #         # Define slice indices for inserting the current_block_scaled
        #         row_start_slice = r_block_idx * n_stationary
        #         row_end_slice = (r_block_idx + 1) * n_stationary
        #         col_start_slice = c_block_idx * n_stationary
        #         col_end_slice = (c_idx_block + 1) * n_stationary
                
        #         var_block_cov = var_block_cov.at[row_start_slice:row_end_slice, col_start_slice:col_end_slice].set(current_block_scaled)
        
        # Insert the populated VAR block covariance into the full P0 matrix
        if not (var_start_idx + var_state_total_dim <= state_dim): # Should be equal
             raise RuntimeError(f"Build P0 Cov Error: VAR block index range [{var_start_idx}:{var_start_idx + var_state_total_dim}] inconsistency with state_dim {state_dim}.")
        
        init_cov = init_cov.at[var_start_idx : var_start_idx + var_state_total_dim, 
                               var_start_idx : var_start_idx + var_state_total_dim].set(var_block_cov)

    # Finalize: ensure symmetry and add jitter for positive definiteness
    init_cov = (init_cov + init_cov.T) / 2.0 
    init_cov = init_cov + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    # Final checks on the constructed P0
    if not jnp.all(jnp.isfinite(init_cov)):
         raise RuntimeError("Build P0 Cov Error: Final P0 matrix contains non-finite values after construction and jitter.")
    try:
        # Test for Positive Semi-Definiteness
        jnp.linalg.cholesky(init_cov) 
    except Exception as e:
        diag_str = str(jnp.diag(init_cov)[:min(10, state_dim)]) + ("..." if state_dim > 10 else "")
        cond_num_approx = np.linalg.cond(np.array(init_cov)) if state_dim > 0 else "N/A"
        raise RuntimeError(
            f"Build P0 Cov Error: Final P0 matrix (dim {state_dim}x{state_dim}) "
            f"is not positive semi-definite after jitter. Cholesky failed: {e}. "
            f"Diagonal starts: {diag_str}. Approx Condition #: {cond_num_approx}"
        ) from e
        
    return init_cov

if __name__ == "__main__":
    test_gpm_prior_evaluator_module()