# clean_gpm_bvar_trends/gpm_prior_evaluator.py - FIXED

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import pandas as pd
import os
import matplotlib.pyplot as plt

from .integration_orchestrator import create_integration_orchestrator
from .common_types import EnhancedBVARParams, SmootherResults
from .gpm_model_parser import ReducedModel, VarPriorSetup, PriorSpec, VariableSpec, ReducedExpression
from .state_space_builder import StateSpaceBuilder
from .constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER, _SP_JITTER

# Import the ROBUST reconstruction function
from .variable_reconstruction import _reconstruct_original_variables

try:
    from .P0_utils import (
        _build_gamma_based_p0,
        _create_standard_p0,
        _extract_gamma_matrices_from_params
    )
except ImportError:
     _build_gamma_based_p0 = None
     _create_standard_p0 = None
     print("Warning: p0_utils not available. P0 building will likely fail.")

try:
    from .Kalman_filter_jax import KalmanFilter
except ImportError:
    KalmanFilter = None
    print("Warning: KalmanFilter not available in gpm_prior_evaluator.py")

try:
    from .simulation_smoothing import jarocinski_corrected_simulation_smoother
except ImportError:
    jarocinski_corrected_simulation_smoother = None
    print("Warning: jarocinski_corrected_simulation_smoother not available in gpm_prior_evaluator.py")

try:
    from .stationary_prior_jax_simplified import make_stationary_var_transformation_jax
except ImportError:
    make_stationary_var_transformation_jax = None
    print("Warning: make_stationary_var_transformation_jax not available in gpm_prior_evaluator.py")

try:
    from .reporting_plots import (
        plot_smoother_results,
        plot_custom_series_comparison,
        plot_observed_vs_single_trend_component,
        compute_hdi_robust,
        compute_summary_statistics
    )
    PLOTTING_AVAILABLE_EVAL = True
except ImportError:
    PLOTTING_AVAILABLE_EVAL = False
    # Define dummy functions
    # def plot_smoother_results(*args, **kwargs): return None, None
    # def plot_custom_series_comparison(*args, **kwargs): return None
    # def plot_observed_vs_single_trend_component(*args, **kwargs): return None
    # def compute_hdi_robust(*args, **kwargs): return (np.nan, np.nan)
    # def compute_summary_statistics(*args, **kwargs): return {}

# (All helper functions like _resolve_parameter_value, _build_trend_covariance, etc. remain the same)
# ...
# [Keep all existing helper functions _resolve_parameter_value through _build_initial_mean_for_test here]
# ...

def _resolve_parameter_value(param_key_base: str, param_values_input: Dict[str, Any], estimated_params_gpm: Dict[str, Any], is_shock_std_dev: bool = False) -> float:
    val_resolved = None; checked_keys = []; checked_keys.append(param_key_base)
    if param_key_base in param_values_input: val_resolved = param_values_input[param_key_base]
    elif is_shock_std_dev:
        sigma_key = f"sigma_{param_key_base}"; checked_keys.append(sigma_key)
        if sigma_key in param_values_input: val_resolved = param_values_input[sigma_key]
    if val_resolved is None:
        prior_spec_to_use = None; key_in_gpm_prior = None
        checked_keys.append(f"GPM prior for '{param_key_base}'")
        if param_key_base in estimated_params_gpm: prior_spec_to_use = estimated_params_gpm[param_key_base]; key_in_gpm_prior = param_key_base
        elif is_shock_std_dev:
            sigma_key_gpm = f"sigma_{param_key_base}"; checked_keys.append(f"GPM prior for '{sigma_key_gpm}'")
            if sigma_key_gpm in estimated_params_gpm: prior_spec_to_use = estimated_params_gpm[sigma_key_gpm]; key_in_gpm_prior = sigma_key_gpm
        if prior_spec_to_use:
            if prior_spec_to_use.distribution == 'inv_gamma_pdf' and len(prior_spec_to_use.params) >= 2:
                alpha, beta = prior_spec_to_use.params
                mode = beta / (alpha + 1.0) if alpha > -1.0 else None
                if mode is not None and mode > 0: val_resolved = mode
                elif is_shock_std_dev: raise ValueError(f"Parameter '{key_in_gpm_prior}': Prior '{prior_spec_to_use.distribution}' implies non-positive std dev (mode: {mode}).")
            elif prior_spec_to_use.distribution == 'normal_pdf' and len(prior_spec_to_use.params) >= 1:
                mean_val = prior_spec_to_use.params[0]
                if is_shock_std_dev:
                    val_resolved = abs(mean_val)
                    if val_resolved == 0: raise ValueError(f"Parameter '{key_in_gpm_prior}': Normal prior for std dev has zero mean, leading to zero std dev.")
                else: val_resolved = mean_val
            else: raise ValueError(f"Parameter '{key_in_gpm_prior}': Unsupported prior distribution '{prior_spec_to_use.distribution}' or insufficient params in GPM estimated_params.")
    if val_resolved is None: raise ValueError(f"Parameter '{param_key_base}' could not be resolved. Attempted lookups: {', '.join(checked_keys)}.")
    try: val_float = float(val_resolved)
    except (TypeError, ValueError) as e: raise ValueError(f"Parameter '{param_key_base}': Resolved value '{val_resolved}' ('{type(val_resolved).__name__}') cannot be converted to float.") from e
    if is_shock_std_dev and val_float <= 0: raise RuntimeError(f"Parameter '{param_key_base}' (shock std dev): Final resolved value must be positive, got {val_float}.")
    return val_float

def _build_trend_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
    if not dynamic_trend_names: return jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
    num_dynamic_trends = len(dynamic_trend_names); trend_sigmas_sq = jnp.zeros(num_dynamic_trends, dtype=_DEFAULT_DTYPE)
    for idx_dynamic_trend, trend_name in enumerate(dynamic_trend_names):
        associated_shock_name = next((eq.shock for eq in gpm_model.core_equations if eq.lhs == trend_name), None)
        if associated_shock_name:
            sigma_val = _resolve_parameter_value(param_key_base=associated_shock_name, param_values_input=param_values_input, estimated_params_gpm=gpm_model.estimated_params, is_shock_std_dev=True)
            trend_sigmas_sq = trend_sigmas_sq.at[idx_dynamic_trend].set(sigma_val ** 2)
    Sigma_eta = jnp.diag(trend_sigmas_sq)
    if not jnp.all(jnp.isfinite(Sigma_eta)): raise RuntimeError("Built Sigma_eta for trends contains non-finite values.")
    return Sigma_eta

def _build_var_parameters(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    n_stationary = len(gpm_model.stationary_variables)
    if n_stationary == 0: return jnp.empty((0,0), dtype=_DEFAULT_DTYPE), jnp.empty((0,0,0), dtype=_DEFAULT_DTYPE), []
    if not gpm_model.var_prior_setup or not hasattr(gpm_model.var_prior_setup, 'var_order') or gpm_model.var_prior_setup.var_order <= 0: raise ValueError("VAR parameters require valid 'var_prior_setup' with var_order > 0 attribute.")
    setup = gpm_model.var_prior_setup; n_vars, n_lags = n_stationary, setup.var_order
    mean_diag = setup.es[0] if setup.es and len(setup.es) > 0 else 0.0; mean_offdiag = setup.es[1] if setup.es and len(setup.es) > 1 else 0.0
    raw_A_list = []
    for _ in range(n_lags):
        A_lag = jnp.full((n_vars, n_vars), mean_offdiag, dtype=_DEFAULT_DTYPE)
        A_lag = A_lag.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(mean_diag); raw_A_list.append(jnp.asarray(A_lag, dtype=_DEFAULT_DTYPE))
    if not gpm_model.stationary_shocks or len(gpm_model.stationary_shocks) != n_vars: raise ValueError(f"Mismatch between #stationary_variables ({n_vars}) and #stationary_shocks ({len(gpm_model.stationary_shocks if gpm_model.stationary_shocks else 'None')}).")
    stat_sigmas_std_dev = [_resolve_parameter_value(shock, param_values_input, gpm_model.estimated_params, True) for shock in gpm_model.stationary_shocks]
    sigma_u_diag_vec = jnp.array(stat_sigmas_std_dev, dtype=_DEFAULT_DTYPE)
    Omega_u_chol_val = param_values_input.get("_var_innovation_corr_chol")
    if Omega_u_chol_val is not None:
        if not isinstance(Omega_u_chol_val, (jnp.ndarray, np.ndarray)) or Omega_u_chol_val.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(Omega_u_chol_val)): raise ValueError(f"_var_innovation_corr_chol is invalid. Shape: {Omega_u_chol_val.shape if hasattr(Omega_u_chol_val, 'shape') else 'N/A' }, Expected: ({n_vars},{n_vars}).")
        Omega_u_chol_val = jnp.asarray(Omega_u_chol_val, dtype=_DEFAULT_DTYPE)
    else: Omega_u_chol_val = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    Sigma_u = jnp.diag(sigma_u_diag_vec) @ Omega_u_chol_val @ Omega_u_chol_val.T @ jnp.diag(sigma_u_diag_vec)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _SP_JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    if not jnp.all(jnp.isfinite(Sigma_u)): raise RuntimeError("Built Sigma_u (VAR innovation cov) contains non-finite values.")
    try: jnp.linalg.cholesky(Sigma_u)
    except Exception as e_chol: raise RuntimeError(f"Built Sigma_u not PSD: {e_chol}") from e_chol
    if make_stationary_var_transformation_jax is None:
        A_transformed = jnp.stack(raw_A_list); gamma_list_for_P0 = []
        return Sigma_u, A_transformed, gamma_list_for_P0
    try:
        phi_list, gamma_list_from_transform = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
        if not (phi_list and len(phi_list) == n_lags and all(p_mat is not None and hasattr(p_mat, 'shape') and p_mat.shape == (n_vars,n_vars) and jnp.all(jnp.isfinite(p_mat)) for p_mat in phi_list)):
             A_transformed = jnp.stack(raw_A_list); print("Warning: Stationary transformation produced invalid phi list. Using raw A for dynamics.")
        else: A_transformed = jnp.stack(phi_list)
        if not (gamma_list_from_transform and len(gamma_list_from_transform) == n_lags and all(g is not None and hasattr(g, 'shape') and g.shape == (n_vars, n_vars) and jnp.all(jnp.isfinite(g)) for g in gamma_list_from_transform)):
             print("Warning: Stationary transformation produced invalid gamma list. P0 will use fallback."); final_gamma_list_for_P0 = []
        else: final_gamma_list_for_P0 = gamma_list_from_transform
    except Exception as e: A_transformed = jnp.stack(raw_A_list); final_gamma_list_for_P0 = []; print(f"Warning: Stationary VAR transformation failed: {e}. Using raw A for dynamics and fallback P0.")
    return Sigma_u, A_transformed, final_gamma_list_for_P0

def _build_measurement_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    n_observed = len(gpm_model.gpm_observed_variables_original);
    if n_observed == 0: return jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    H_val = param_values_input.get("_measurement_error_cov_full")
    if H_val is not None:
        if not isinstance(H_val, (jnp.ndarray, np.ndarray)) or H_val.shape != (n_observed, n_observed) or not jnp.all(jnp.isfinite(H_val)): raise ValueError(f"Provided _measurement_error_cov_full is invalid. Shape: {H_val.shape if hasattr(H_val,'shape') else 'N/A'}, Expected: ({n_observed},{n_observed})")
        H = jnp.asarray(H_val, dtype=_DEFAULT_DTYPE); H = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_observed, dtype=_DEFAULT_DTYPE)
    else: obs_err_sigmas_sq = jnp.full(n_observed, 1e-10, dtype=_DEFAULT_DTYPE); H = jnp.diag(obs_err_sigmas_sq) + _KF_JITTER * jnp.eye(n_observed, dtype=_DEFAULT_DTYPE)
    if not jnp.all(jnp.isfinite(H)): raise RuntimeError("Built measurement covariance H contains non-finite values.")
    try: jnp.linalg.cholesky(H)
    except Exception as e_chol_h: raise RuntimeError(f"Built measurement covariance H not PSD: {e_chol_h}") from e_chol_h
    return H

def _build_initial_mean_for_test(gpm_model: ReducedModel, ss_builder: StateSpaceBuilder, initial_state_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> jnp.ndarray:
    state_dim = ss_builder.state_dim; init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    if initial_state_prior_overrides is None: initial_state_prior_overrides = {}
    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
    for trend_name in dynamic_trend_names:
        override_spec = initial_state_prior_overrides.get(trend_name); mean_val_to_set = None
        if override_spec and "mean" in override_spec: mean_val_to_set = override_spec["mean"]
        elif trend_name in gpm_model.initial_values:
            var_spec = gpm_model.initial_values[trend_name]
            if not (isinstance(var_spec, VariableSpec) and var_spec.init_dist == 'normal_pdf' and var_spec.init_params and len(var_spec.init_params) >= 1): raise ValueError(f"Build P0 Mean Error: 'initval' for dynamic trend '{trend_name}' requires 'normal_pdf' with at least a mean parameter.")
            mean_val_to_set = var_spec.init_params[0]
        else: raise ValueError(f"Build P0 Mean Error: Dynamic trend '{trend_name}' must have an 'initval' entry or an override for its mean.")
        state_idx = ss_builder.core_var_map.get(trend_name)
        if state_idx is None or not (0 <= state_idx < ss_builder.n_dynamic_trends): raise RuntimeError(f"Build P0 Mean Error: Builder map index {state_idx} for dynamic trend '{trend_name}' is invalid.")
        init_mean = init_mean.at[state_idx].set(jnp.array(mean_val_to_set, dtype=_DEFAULT_DTYPE))
    var_block_start_idx = ss_builder.n_dynamic_trends
    for i_stat_in_block, stat_var_name in enumerate(gpm_model.stationary_variables):
        override_spec_stat = initial_state_prior_overrides.get(stat_var_name); mean_val_stat_to_set = 0.0
        if override_spec_stat and "mean" in override_spec_stat: mean_val_stat_to_set = override_spec_stat["mean"]
        elif stat_var_name in gpm_model.initial_values:
            var_spec_stat = gpm_model.initial_values[stat_var_name]
            if isinstance(var_spec_stat, VariableSpec) and var_spec_stat.init_dist == 'normal_pdf' and var_spec_stat.init_params and len(var_spec_stat.init_params) >= 1: mean_val_stat_to_set = var_spec_stat.init_params[0]
        idx_for_this_stat_lag0 = var_block_start_idx + i_stat_in_block
        if not (ss_builder.n_dynamic_trends <= idx_for_this_stat_lag0 < ss_builder.n_dynamic_trends + ss_builder.n_stationary): raise RuntimeError(f"Build P0 Mean Error: Calculated index {idx_for_this_stat_lag0} for stationary var '{stat_var_name}' (lag 0) is out of range.")
        init_mean = init_mean.at[idx_for_this_stat_lag0].set(jnp.array(mean_val_stat_to_set, dtype=_DEFAULT_DTYPE))
    if not jnp.all(jnp.isfinite(init_mean)): raise RuntimeError("Build P0 Mean Error: Resulting initial mean vector contains non-finite values.")
    return init_mean

def evaluate_gpm_at_parameters(gpm_file_path: str,
                                y: Union[jnp.ndarray, pd.DataFrame],
                                param_values: Dict[str, Any],
                                initial_state_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                                num_sim_draws: int = 50,
                                rng_key: jax.Array = random.PRNGKey(42),
                                plot_results: bool = True,
                                plot_default_observed_vs_trend_components: bool = True,
                                custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
                                variable_names: Optional[List[str]] = None,
                                use_gamma_init_for_test: bool = True,
                                gamma_init_scaling: float = 1.0,
                                hdi_prob: float = 0.9,
                                trend_P0_var_scale: float = 1e4,
                                var_P0_var_scale: float = 1.0,
                                save_plots_path_prefix: Optional[str] = None,
                                show_plot_info_boxes: bool = False
                        ) -> SmootherResults:
    print(f"\n--- Evaluating GPM: {gpm_file_path} at fixed parameters (fixed eval context) ---")
    if initial_state_prior_overrides: print(f"  Using initial_state_prior_overrides: {initial_state_prior_overrides}")
    if not os.path.exists(gpm_file_path): raise FileNotFoundError(f"GPM file not found: {gpm_file_path}")
    if KalmanFilter is None: raise RuntimeError("KalmanFilter is not available for evaluation.")
    if jarocinski_corrected_simulation_smoother is None and num_sim_draws > 0:
         print("Warning: Simulation smoother function not available. Setting num_sim_draws to 0."); num_sim_draws = 0

    time_index_for_plots: Optional[pd.Index] = None
    if isinstance(y, pd.DataFrame):
        y_jax_data = jnp.asarray(y.values, dtype=_DEFAULT_DTYPE); y_np_data = np.asarray(y.values)
        time_index_for_plots = y.index
        if variable_names is None: variable_names = list(y.columns)
    elif isinstance(y, (jnp.ndarray, np.ndarray)):
        y_jax_data = jnp.asarray(y, dtype=_DEFAULT_DTYPE); y_np_data = np.asarray(y)
        time_index_for_plots = pd.RangeIndex(start=0, stop=y_jax_data.shape[0], step=1)
    else: raise TypeError(f"Unsupported data type for 'y': {type(y)}. Must be JAX/NumPy array or Pandas DataFrame.")
    
    orchestrator = create_integration_orchestrator(gpm_file_path, strict_validation=True)
    gpm_model = orchestrator.reduced_model; ss_builder = orchestrator.ss_builder
    T_data_len, n_obs_data = y_jax_data.shape

    if variable_names is not None and len(variable_names) != n_obs_data:
        print(f"Warning: Length of 'variable_names' ({len(variable_names)}) does not match number of observed series ({n_obs_data}). Using defaults.")
        variable_names = [f"Obs{i+1}" for i in range(n_obs_data)]
    elif variable_names is None:
        variable_names = gpm_model.gpm_observed_variables_original
        if len(variable_names) != n_obs_data: variable_names = [f"Obs{i+1}" for i in range(n_obs_data)]
    
    print(f"  State Dim: {ss_builder.state_dim}, Dynamic Trends: {ss_builder.n_dynamic_trends}, Stat Vars: {ss_builder.n_stationary}, VAR Order: {ss_builder.var_order if ss_builder.n_stationary > 0 else 'N/A'}")
    
    structural_params_resolved = {}
    for param_name in gpm_model.parameters:
        try: structural_params_resolved[param_name] = jnp.array(_resolve_parameter_value(param_name, param_values, gpm_model.estimated_params, False), dtype=_DEFAULT_DTYPE)
        except Exception as e: raise ValueError(f"Failed to resolve structural parameter '{param_name}': {e}") from e
    
    Sigma_eta = _build_trend_covariance(gpm_model, param_values)
    Sigma_u_innov, A_transformed_coeffs, gamma_list_for_P0 = _build_var_parameters(gpm_model, param_values)
    Sigma_eps = _build_measurement_covariance(gpm_model, param_values)
    
    init_mean = _build_initial_mean_for_test(gpm_model, ss_builder, initial_state_prior_overrides)
    if hasattr(ss_builder, 'core_var_map') and hasattr(ss_builder, 'n_dynamic_trends'):
        core_var_map_val = ss_builder.core_var_map
        dynamic_trend_names_list = [name for name, idx in sorted(core_var_map_val.items(), key=lambda item: item[1]) if idx < ss_builder.n_dynamic_trends]
    else: core_var_map_val = {}; dynamic_trend_names_list = []; print("Warning: ss_builder missing core_var_map or n_dynamic_trends in gpm_prior_evaluator.py. P0 construction might be incorrect.")
    
    gamma_list_is_valid_for_p0_building = (gamma_list_for_P0 and len(gamma_list_for_P0) == ss_builder.var_order and all(g is not None and hasattr(g, 'shape') and g.shape == (ss_builder.n_stationary, ss_builder.n_stationary) and jnp.all(jnp.isfinite(g)) for g in gamma_list_for_P0))
    if use_gamma_init_for_test and ss_builder.n_stationary > 0 and ss_builder.var_order > 0 and gamma_list_is_valid_for_p0_building:
        if _build_gamma_based_p0: init_cov = _build_gamma_based_p0(state_dim=ss_builder.state_dim, n_dynamic_trends=ss_builder.n_dynamic_trends, dynamic_trend_names=dynamic_trend_names_list, core_var_map=core_var_map_val, gamma_list=gamma_list_for_P0, n_stationary=ss_builder.n_stationary, var_order=ss_builder.var_order, gamma_scaling=gamma_init_scaling, context="fixed_eval", trend_P0_scales_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale)
        else: print("Warning: _build_gamma_based_p0 utility function is not available. Using standard P0 fallback."); init_cov = _create_standard_p0(state_dim=ss_builder.state_dim, n_dynamic_trends=ss_builder.n_dynamic_trends, dynamic_trend_names=dynamic_trend_names_list, core_var_map=core_var_map_val, context="fixed_eval", trend_P0_scales_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale)
    else:
        if use_gamma_init_for_test and (ss_builder.n_stationary == 0 or not gamma_list_is_valid_for_p0_building): print("  Info: Gamma P0 requested but conditions not met. Using standard P0.")
        else: print("  Using standard P0 for fixed parameter evaluation.")
        if _create_standard_p0: init_cov = _create_standard_p0(state_dim=ss_builder.state_dim, n_dynamic_trends=ss_builder.n_dynamic_trends, dynamic_trend_names=dynamic_trend_names_list, core_var_map=core_var_map_val, context="fixed_eval", trend_P0_scales_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale)
        else: raise RuntimeError("Standard P0 utility function is not available.")

    params_for_ss_builder = EnhancedBVARParams(A=A_transformed_coeffs, Sigma_u=Sigma_u_innov, Sigma_eta=Sigma_eta, structural_params=structural_params_resolved, Sigma_eps=Sigma_eps)
    F, Q, C, H_obs_matrix = ss_builder.build_state_space_from_enhanced_bvar(params_for_ss_builder)
    for mat, name in zip([F, Q, C, H_obs_matrix], ["F", "Q", "C", "H_obs (from builder)"]):
        if not jnp.all(jnp.isfinite(mat)): raise RuntimeError(f"Builder Matrix {name} contains non-finite values.")
    try: jnp.linalg.cholesky(Q + _KF_JITTER * jnp.eye(Q.shape[0]))
    except Exception as e: raise RuntimeError(f"Builder State Cov Q not PSD: {e}")
    if H_obs_matrix.shape[0] > 0:
        try: jnp.linalg.cholesky(H_obs_matrix + _KF_JITTER * jnp.eye(H_obs_matrix.shape[0]))
        except Exception as e: raise RuntimeError(f"Builder Measurement Cov H_obs not PSD: {e}")
    R_sim_chol_Q = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(Q.shape[0]))

    if KalmanFilter is None:
        print("Warning: KalmanFilter not available. Skipping filtering and smoothing."); loglik_val = jnp.array(np.nan, dtype=_DEFAULT_DTYPE)
        sim_draws_core_state = jnp.full((max(1,num_sim_draws), T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE) if num_sim_draws > 0 else jnp.empty((0, T_data_len, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)
    else:
        kf = KalmanFilter(T=F, R=R_sim_chol_Q, C=C, H=H_obs_matrix, init_x=init_mean, init_P=init_cov)
        valid_obs_idx = jnp.arange(n_obs_data); I_obs = jnp.eye(n_obs_data) if n_obs_data > 0 else jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        loglik_val = kf.log_likelihood(y_jax_data, valid_obs_idx, n_obs_data, C, H_obs_matrix, I_obs)
        if not jnp.isfinite(loglik_val): print(f"Warning: Log-likelihood is non-finite ({loglik_val}). Check parameters or model stability.")
        print(f"  Log-likelihood: {loglik_val:.3f}")
        sim_draws_core_state = jnp.empty((0, T_data_len, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)
        if num_sim_draws > 0:
            sim_draws_list = []; print(f"  Running simulation smoother for {num_sim_draws} draws...")
            rng_key, sim_key_for_smoother = random.split(rng_key)
            for i in range(num_sim_draws):
                sim_key_for_smoother, current_draw_key = random.split(sim_key_for_smoother)
                try:
                    s_states = jarocinski_corrected_simulation_smoother(y_jax_data, F, R_sim_chol_Q, C, H_obs_matrix, init_mean, init_cov, current_draw_key)
                    if jnp.all(jnp.isfinite(s_states)): sim_draws_list.append(s_states)
                    else: print(f"  Warning: Sim draw {i+1} had non-finite values, discarding.")
                except Exception as e_sim: print(f"  Warning: Sim draw {i+1} failed: {e_sim}, discarding.")
            if sim_draws_list: sim_draws_core_state = jnp.stack(sim_draws_list)
            else: sim_draws_core_state = jnp.full((0, T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE)
            print(f"  Completed {sim_draws_core_state.shape[0]} simulation draws.")
        else: print("  num_sim_draws is 0. Skipping simulation smoother.")

    n_actual_sim_draws = sim_draws_core_state.shape[0]
    reconstructed_all_trends_draws = jnp.full((n_actual_sim_draws, T_data_len, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
    reconstructed_all_stationary_draws = jnp.full((n_actual_sim_draws, T_data_len, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)

    if n_actual_sim_draws > 0:
        print("  Reconstructing original GPM variables from simulated states...")
        
        # --- FIX STARTS HERE ---
        # Initialize empty lists to hold the time series for each draw
        all_draws_trends_list = []
        all_draws_stationary_list = []

        for i_draw in range(n_actual_sim_draws):
            # Call the new reconstruction function which returns dictionaries
            reconstructed_trends_dict, reconstructed_stationary_dict = _reconstruct_original_variables(
                core_states_draw=sim_draws_core_state[i_draw],
                gpm_model=gpm_model,
                ss_builder=ss_builder,
                current_builder_params_draw=structural_params_resolved,
                T_data=T_data_len,
                state_dim=ss_builder.state_dim
            )
            
            # Assemble the trend array for this draw in the correct, canonical order
            trend_ts_for_this_draw = [
                reconstructed_trends_dict.get(name, jnp.full(T_data_len, np.nan))
                for name in gpm_model.gpm_trend_variables_original
            ]
            all_draws_trends_list.append(jnp.stack(trend_ts_for_this_draw, axis=-1))
            
            # Assemble the stationary array for this draw in the correct, canonical order
            stationary_ts_for_this_draw = [
                reconstructed_stationary_dict.get(name, jnp.full(T_data_len, np.nan))
                for name in gpm_model.gpm_stationary_variables_original
            ]
            all_draws_stationary_list.append(jnp.stack(stationary_ts_for_this_draw, axis=-1))

        # Stack the draws together
        reconstructed_all_trends_draws = jnp.stack(all_draws_trends_list, axis=0)
        reconstructed_all_stationary_draws = jnp.stack(all_draws_stationary_list, axis=0)
        print("  Reconstruction complete.")
    else:
        print("  Skipping reconstruction as no simulation draws were generated.")
        reconstructed_all_trends_draws = jnp.empty((0, T_data_len, len(gpm_model.gpm_trend_variables_original)), dtype=_DEFAULT_DTYPE)
        reconstructed_all_stationary_draws = jnp.empty((0, T_data_len, len(gpm_model.gpm_stationary_variables_original)), dtype=_DEFAULT_DTYPE)

    trend_stats = compute_summary_statistics(np.asarray(reconstructed_all_trends_draws))
    stationary_stats = compute_summary_statistics(np.asarray(reconstructed_all_stationary_draws))
    trend_hdi_lower, trend_hdi_upper = (None, None); stationary_hdi_lower, stationary_hdi_upper = (None, None)
    if n_actual_sim_draws > 1:
        trend_hdi_lower, trend_hdi_upper = compute_hdi_robust(np.asarray(reconstructed_all_trends_draws), hdi_prob)
        stationary_hdi_lower, stationary_hdi_upper = compute_hdi_robust(np.asarray(reconstructed_all_stationary_draws), hdi_prob)

    results = SmootherResults(
        observed_data=y_np_data, observed_variable_names=variable_names, time_index=time_index_for_plots,
        trend_draws=np.asarray(reconstructed_all_trends_draws), trend_names=list(gpm_model.gpm_trend_variables_original),
        trend_stats=trend_stats, trend_hdi_lower=trend_hdi_lower, trend_hdi_upper=trend_hdi_upper,
        stationary_draws=np.asarray(reconstructed_all_stationary_draws), stationary_names=list(gpm_model.gpm_stationary_variables_original),
        stationary_stats=stationary_stats, stationary_hdi_lower=stationary_hdi_lower, stationary_hdi_upper=stationary_hdi_upper,
        reduced_measurement_equations=gpm_model.reduced_measurement_equations, gpm_model=gpm_model,
        parameters_used=param_values, log_likelihood=float(loglik_val.item()) if hasattr(loglik_val, 'item') and jnp.isfinite(loglik_val) else (float(loglik_val) if isinstance(loglik_val, (float,int)) and np.isfinite(loglik_val) else np.nan),
        n_draws=n_actual_sim_draws, hdi_prob=hdi_prob
    )

    if plot_results and PLOTTING_AVAILABLE_EVAL and results.n_draws > 0:
        print("  Generating plots for fixed parameter evaluation...")
        plot_save_prefix_fixed = None
        if save_plots_path_prefix:
             plot_save_prefix_fixed = save_plots_path_prefix
             plot_dir = os.path.dirname(plot_save_prefix_fixed)
             if plot_dir and not os.path.exists(plot_dir): os.makedirs(plot_dir, exist_ok=True)
        if callable(plot_smoother_results):
            plot_smoother_results(results, save_path=plot_save_prefix_fixed, show_info_box=show_plot_info_boxes)
        # if plot_default_observed_vs_trend_components and callable(plot_observed_vs_single_trend_component):
        #      plot_observed_vs_single_trend_component(results, save_path=plot_save_prefix_fixed, show_info_box=show_plot_info_boxes, use_median_for_trend_line=True)
        if custom_plot_specs and callable(plot_custom_series_comparison):
            for spec_idx, spec_dict_item in enumerate(custom_plot_specs):
                plot_custom_series_comparison(plot_title=spec_dict_item.get("title", f"Custom Plot {spec_idx+1}") + " (Fixed Params)", series_specs=spec_dict_item.get("series_to_plot", []), results=results, show_info_box=show_plot_info_boxes, save_path=plot_save_prefix_fixed)
    elif plot_results: print("  Plotting was requested but is unavailable or no draws were generated. Skipping plots.")
    
    print(f"--- Evaluation for {gpm_file_path} complete ---")
    return results