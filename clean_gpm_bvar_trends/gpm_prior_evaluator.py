# clean_gpm_bvar_trends/gpm_prior_evaluator.py

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union # Added Union
import time
import pandas as pd # Added for time_index handling
import os
import matplotlib.pyplot as plt # Needed for plt.close() if plotting

# Import necessary components from other modules using relative imports
from .integration_orchestrator import create_integration_orchestrator
from .common_types import EnhancedBVARParams, SmootherResults # Import SmootherResults
from .gpm_model_parser import ReducedModel, VarPriorSetup, PriorSpec, VariableSpec, ReducedExpression # Ensure all needed parser classes are imported
from .state_space_builder import StateSpaceBuilder
from .constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER, _SP_JITTER

# Import P0 building utilities
try:
    from .P0_utils import (
        _build_gamma_based_p0,
        _create_standard_p0,
        _extract_gamma_matrices_from_params # Might be needed if recreating gamma list here
    )
except ImportError:
     _build_gamma_based_p0 = None
     _create_standard_p0 = None
     #_extract_gamma_matrices_from_params = None
     print("Warning: p0_utils not available. P0 building will likely fail.")


# Conditional imports with error handling
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
    # _SP_JITTER should be imported from constants now
    print("Warning: make_stationary_var_transformation_jax not available in gpm_prior_evaluator.py")


# Plotting imports (ensure PLOTTING_AVAILABLE_EVAL is defined and functions accept SmootherResults)
try:
    from .reporting_plots import (
        plot_smoother_results, # Assumed to accept SmootherResults
        plot_custom_series_comparison, # Assumed to accept SmootherResults
        plot_observed_vs_single_trend_component, # Assumed to accept SmootherResults
        compute_hdi_robust,
        compute_summary_statistics
    )
    PLOTTING_AVAILABLE_EVAL = True
    print("✓ Plotting functions imported in gpm_prior_evaluator.py (assumed SmootherResults compatible)")
except ImportError as e:
    print(f"⚠️  Warning (gpm_prior_evaluator.py): Could not import plotting functions or they might not be SmootherResults compatible: {e}")
    PLOTTING_AVAILABLE_EVAL = False

    def plot_smoother_results(*args, **kwargs):
        print("Plotting disabled (gpm_prior_evaluator) - plot_smoother_results skipped")
        return None, None # Original returns two figures
    def plot_custom_series_comparison(*args, **kwargs):
        print("Plotting disabled (gpm_prior_evaluator) - plot_custom_series_comparison skipped")
        return None
    def plot_observed_vs_single_trend_component(*args, **kwargs):
         print("Plotting disabled (gpm_prior_evaluator) - plot_observed_vs_single_trend_component skipped")
         return None
    def compute_hdi_robust(*args, **kwargs):
        print("Warning: compute_hdi_robust not available.")
        # Return NaNs based on input shape if possible
        if len(args) > 0 and hasattr(args[0], 'shape') and len(args[0].shape) > 1:
            return (np.full_like(np.asarray(args[0])[0], np.nan), np.full_like(np.asarray(args[0])[0], np.nan))
        return (np.nan, np.nan)
    def compute_summary_statistics(*args, **kwargs):
        print("Warning: compute_summary_statistics not available.")
        # Return NaNs based on input shape if possible
        if len(args) > 0 and hasattr(args[0], 'shape') and len(args[0].shape) > 1:
            nan_array = np.full_like(np.asarray(args[0])[0], np.nan)
            return {'mean': nan_array, 'median': nan_array, 'mode': nan_array, 'std': nan_array}
        return {'mean':np.nan, 'median':np.nan, 'mode':np.nan, 'std':np.nan}


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# --- Helper functions defined locally ---
# These helpers are needed by evaluate_gpm_at_parameters and depend on GPM/SSBuilder structure.

def _resolve_parameter_value(param_key_base: str,
                             param_values_input: Dict[str, Any],
                             estimated_params_gpm: Dict[str, Any],
                             is_shock_std_dev: bool = False
                             ) -> float:
    # (Implementation as provided in previous context, depends on PriorSpec)
    val_resolved = None
    checked_keys = []

    checked_keys.append(param_key_base)
    if param_key_base in param_values_input:
        val_resolved = param_values_input[param_key_base]
    elif is_shock_std_dev:
        sigma_key = f"sigma_{param_key_base}"
        checked_keys.append(sigma_key)
        if sigma_key in param_values_input:
            val_resolved = param_values_input[sigma_key]

    if val_resolved is None:
        prior_spec_to_use = None
        key_in_gpm_prior = None

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

        if prior_spec_to_use:
            if prior_spec_to_use.distribution == 'inv_gamma_pdf' and len(prior_spec_to_use.params) >= 2:
                alpha, beta = prior_spec_to_use.params
                mode = beta / (alpha + 1.0) if alpha > -1.0 else None
                if mode is not None and mode > 0:
                    val_resolved = mode
                elif is_shock_std_dev:
                     raise ValueError(f"Parameter '{key_in_gpm_prior}': Prior '{prior_spec_to_use.distribution}' implies non-positive std dev (mode: {mode}).")
            elif prior_spec_to_use.distribution == 'normal_pdf' and len(prior_spec_to_use.params) >= 1:
                mean_val = prior_spec_to_use.params[0]
                if is_shock_std_dev:
                    val_resolved = abs(mean_val)
                    if val_resolved == 0:
                         raise ValueError(f"Parameter '{key_in_gpm_prior}': Normal prior for std dev has zero mean, leading to zero std dev.")
                else:
                    val_resolved = mean_val
            else:
                raise ValueError(f"Parameter '{key_in_gpm_prior}': Unsupported prior distribution '{prior_spec_to_use.distribution}' or insufficient params in GPM estimated_params.")

    if val_resolved is None:
        raise ValueError(f"Parameter '{param_key_base}' could not be resolved. Attempted lookups: {', '.join(checked_keys)}.")

    try:
        val_float = float(val_resolved)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter '{param_key_base}': Resolved value '{val_resolved}' ('{type(val_resolved).__name__}') cannot be converted to float.") from e

    if is_shock_std_dev and val_float <= 0:
        raise RuntimeError(f"Parameter '{param_key_base}' (shock std dev): Final resolved value must be positive, got {val_float}.")
    return val_float


def _build_trend_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    # (Implementation as provided in previous context, uses _resolve_parameter_value)
    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
    if not dynamic_trend_names:
        return jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    num_dynamic_trends = len(dynamic_trend_names)
    trend_sigmas_sq = jnp.zeros(num_dynamic_trends, dtype=_DEFAULT_DTYPE)

    for idx_dynamic_trend, trend_name in enumerate(dynamic_trend_names):
        associated_shock_name = next((eq.shock for eq in gpm_model.core_equations if eq.lhs == trend_name), None)

        if associated_shock_name:
            sigma_val = _resolve_parameter_value(
                param_key_base=associated_shock_name,
                param_values_input=param_values_input,
                estimated_params_gpm=gpm_model.estimated_params,
                is_shock_std_dev=True
            )
            trend_sigmas_sq = trend_sigmas_sq.at[idx_dynamic_trend].set(sigma_val ** 2)

    Sigma_eta = jnp.diag(trend_sigmas_sq)
    if not jnp.all(jnp.isfinite(Sigma_eta)):
        raise RuntimeError("Built Sigma_eta for trends contains non-finite values.")
    return Sigma_eta


def _build_var_parameters(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    # (Implementation as provided in previous context, uses _resolve_parameter_value, make_stationary_var_transformation_jax)
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
        raw_A_list.append(jnp.asarray(A_lag, dtype=_DEFAULT_DTYPE))

    if not gpm_model.stationary_shocks or len(gpm_model.stationary_shocks) != n_vars:
        raise ValueError(f"Mismatch between #stationary_variables ({n_vars}) and #stationary_shocks ({len(gpm_model.stationary_shocks if gpm_model.stationary_shocks else 'None')}).")

    stat_sigmas_std_dev = [_resolve_parameter_value(shock, param_values_input, gpm_model.estimated_params, True) for shock in gpm_model.stationary_shocks]
    sigma_u_diag_vec = jnp.array(stat_sigmas_std_dev, dtype=_DEFAULT_DTYPE)

    Omega_u_chol_val = param_values_input.get("_var_innovation_corr_chol")
    if Omega_u_chol_val is not None:
        if not isinstance(Omega_u_chol_val, (jnp.ndarray, np.ndarray)) or Omega_u_chol_val.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(Omega_u_chol_val)):
            raise ValueError(f"_var_innovation_corr_chol is invalid. Shape: {Omega_u_chol_val.shape if hasattr(Omega_u_chol_val, 'shape') else 'N/A' }, Expected: ({n_vars},{n_vars}).")
        Omega_u_chol_val = jnp.asarray(Omega_u_chol_val, dtype=_DEFAULT_DTYPE)
    else:
        Omega_u_chol_val = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    Sigma_u = jnp.diag(sigma_u_diag_vec) @ Omega_u_chol_val @ Omega_u_chol_val.T @ jnp.diag(sigma_u_diag_vec)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _SP_JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    if not jnp.all(jnp.isfinite(Sigma_u)): raise RuntimeError("Built Sigma_u (VAR innovation cov) contains non-finite values.")
    try: jnp.linalg.cholesky(Sigma_u)
    except Exception as e_chol: raise RuntimeError(f"Built Sigma_u not PSD: {e_chol}") from e_chol

    if make_stationary_var_transformation_jax is None:
        # No transformation function available. Use raw A for dynamics and empty gamma list for P0.
        A_transformed = jnp.stack(raw_A_list)
        gamma_list_for_P0 = []
        return Sigma_u, A_transformed, gamma_list_for_P0

    try:
        phi_list, gamma_list_from_transform = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)

        if not (phi_list and len(phi_list) == n_lags and all(p_mat is not None and hasattr(p_mat, 'shape') and p_mat.shape == (n_vars,n_vars) and jnp.all(jnp.isfinite(p_mat)) for p_mat in phi_list)):
             # Fallback to raw A if transformed phi list is invalid
             A_transformed = jnp.stack(raw_A_list)
             print("Warning: Stationary transformation produced invalid phi list. Using raw A for dynamics.")
        else:
             A_transformed = jnp.stack(phi_list)


        if not (gamma_list_from_transform and len(gamma_list_from_transform) == n_lags and
                all(g is not None and hasattr(g, 'shape') and g.shape == (n_vars, n_vars) and jnp.all(jnp.isfinite(g)) for g in gamma_list_from_transform)):
             # If gamma list structure is wrong or contains NaNs/Infs, provide empty list for P0 building to fall back
             print("Warning: Stationary transformation produced invalid gamma list. P0 will use fallback.")
             final_gamma_list_for_P0 = []
        else:
             final_gamma_list_for_P0 = gamma_list_from_transform # Use if valid list structure

    except Exception as e:
        # Transformation failed numerically for this draw.
        # Use raw A for dynamics and empty gamma list for P0 fallback.
        A_transformed = jnp.stack(raw_A_list)
        final_gamma_list_for_P0 = []
        print(f"Warning: Stationary VAR transformation failed: {e}. Using raw A for dynamics and fallback P0.")

    return Sigma_u, A_transformed, final_gamma_list_for_P0


def _build_measurement_covariance(gpm_model: ReducedModel, param_values_input: Dict[str, Any]) -> jnp.ndarray:
    # (Implementation as provided in previous context, uses _resolve_parameter_value)
    n_observed = len(gpm_model.gpm_observed_variables_original)
    if n_observed == 0: return jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    H_val = param_values_input.get("_measurement_error_cov_full")
    if H_val is not None:
        if not isinstance(H_val, (jnp.ndarray, np.ndarray)) or H_val.shape != (n_observed, n_observed) or not jnp.all(jnp.isfinite(H_val)):
             raise ValueError(f"Provided _measurement_error_cov_full is invalid. Shape: {H_val.shape if hasattr(H_val,'shape') else 'N/A'}, Expected: ({n_observed},{n_observed})")
        H = jnp.asarray(H_val, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_observed, dtype=_DEFAULT_DTYPE)
    else:
        # Default to a small diagonal matrix if no specific measurement error parameters are handled.
        obs_err_sigmas_sq = jnp.full(n_observed, 1e-10, dtype=_DEFAULT_DTYPE)
        H = jnp.diag(obs_err_sigmas_sq) + _KF_JITTER * jnp.eye(n_observed, dtype=_DEFAULT_DTYPE)

    if not jnp.all(jnp.isfinite(H)): raise RuntimeError("Built measurement covariance H contains non-finite values.")
    try: jnp.linalg.cholesky(H)
    except Exception as e_chol_h: raise RuntimeError(f"Built measurement covariance H not PSD: {e_chol_h}") from e_chol_h
    return H


def _build_initial_mean_for_test(
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    initial_state_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None
) -> jnp.ndarray:
    # (Implementation as provided in previous context, depends on VariableSpec)
    state_dim = ss_builder.state_dim
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    if initial_state_prior_overrides is None:
        initial_state_prior_overrides = {}

    dynamic_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]

    for trend_name in dynamic_trend_names:
        override_spec = initial_state_prior_overrides.get(trend_name)
        mean_val_to_set = None

        if override_spec and "mean" in override_spec:
            mean_val_to_set = override_spec["mean"]
        elif trend_name in gpm_model.initial_values:
            var_spec = gpm_model.initial_values[trend_name]
            if not (isinstance(var_spec, VariableSpec) and var_spec.init_dist == 'normal_pdf' and var_spec.init_params and len(var_spec.init_params) >= 1):
                raise ValueError(f"Build P0 Mean Error: 'initval' for dynamic trend '{trend_name}' requires 'normal_pdf' with at least a mean parameter.")
            mean_val_to_set = var_spec.init_params[0]
        else:
            raise ValueError(f"Build P0 Mean Error: Dynamic trend '{trend_name}' must have an 'initval' entry or an override for its mean.")

        state_idx = ss_builder.core_var_map.get(trend_name)
        if state_idx is None or not (0 <= state_idx < ss_builder.n_dynamic_trends):
            raise RuntimeError(f"Build P0 Mean Error: Builder map index {state_idx} for dynamic trend '{trend_name}' is invalid.")
        init_mean = init_mean.at[state_idx].set(jnp.array(mean_val_to_set, dtype=_DEFAULT_DTYPE))

    var_block_start_idx = ss_builder.n_dynamic_trends
    for i_stat_in_block, stat_var_name in enumerate(gpm_model.stationary_variables):
        override_spec_stat = initial_state_prior_overrides.get(stat_var_name)
        mean_val_stat_to_set = 0.0

        if override_spec_stat and "mean" in override_spec_stat:
            mean_val_stat_to_set = override_spec_stat["mean"]
        elif stat_var_name in gpm_model.initial_values:
            var_spec_stat = gpm_model.initial_values[stat_var_name]
            if isinstance(var_spec_stat, VariableSpec) and var_spec_stat.init_dist == 'normal_pdf' and var_spec_stat.init_params and len(var_spec_stat.init_params) >= 1:
                mean_val_stat_to_set = var_spec_stat.init_params[0]

        idx_for_this_stat_lag0 = var_block_start_idx + i_stat_in_block
        if not (ss_builder.n_dynamic_trends <= idx_for_this_stat_lag0 < ss_builder.n_dynamic_trends + ss_builder.n_stationary):
            raise RuntimeError(f"Build P0 Mean Error: Calculated index {idx_for_this_stat_lag0} for stationary var '{stat_var_name}' (lag 0) is out of range.")
        init_mean = init_mean.at[idx_for_this_stat_lag0].set(jnp.array(mean_val_stat_to_set, dtype=_DEFAULT_DTYPE))

    if not jnp.all(jnp.isfinite(init_mean)):
        raise RuntimeError("Build P0 Mean Error: Resulting initial mean vector contains non-finite values.")
    return init_mean


# _build_initial_covariance_for_test is removed, logic moved into evaluate_gpm_at_parameters


# This is the main evaluate_gpm_at_parameters function
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
                        ) -> SmootherResults: # !!! Return SmootherResults !!!
    """
    Evaluates the GPM model's state space representation at fixed parameter values.
    Allows overriding initial state prior parameters (mean) for trends and lag 0 stationary variables.
    P0 covariance initialization uses "fixed parameter evaluation" context.
    Generates and returns a SmootherResults object. Optionally generates plots.
    """
    print(f"\n--- Evaluating GPM: {gpm_file_path} at fixed parameters (fixed eval context) ---")
    if initial_state_prior_overrides:
        print(f"  Using initial_state_prior_overrides: {initial_state_prior_overrides}")

    if not os.path.exists(gpm_file_path): raise FileNotFoundError(f"GPM file not found: {gpm_file_path}")
    if KalmanFilter is None: raise RuntimeError("KalmanFilter is not available for evaluation.")
    if jarocinski_corrected_simulation_smoother is None and num_sim_draws > 0:
         print("Warning: Simulation smoother function not available. Setting num_sim_draws to 0.")
         num_sim_draws = 0


    # --- Handle y_data and time_index ---
    time_index_for_plots: Optional[pd.Index] = None
    y_jax_data: jnp.ndarray
    y_np_data: np.ndarray # Define y_np_data early

    if isinstance(y, pd.DataFrame):
        y_jax_data = jnp.asarray(y.values, dtype=_DEFAULT_DTYPE)
        y_np_data = np.asarray(y.values) # Get numpy version too
        time_index_for_plots = y.index
        if variable_names is None: # If not overridden, use columns from DataFrame
            variable_names = list(y.columns)
    elif isinstance(y, (jnp.ndarray, np.ndarray)):
        y_jax_data = jnp.asarray(y, dtype=_DEFAULT_DTYPE)
        y_np_data = np.asarray(y) # Get numpy version
        time_index_for_plots = pd.RangeIndex(start=0, stop=y_jax_data.shape[0], step=1)
    else:
        raise TypeError(f"Unsupported data type for 'y': {type(y)}. Must be JAX/NumPy array or Pandas DataFrame.")
    # --- End Handle y_data ---

    orchestrator = create_integration_orchestrator(gpm_file_path, strict_validation=True)
    gpm_model = orchestrator.reduced_model
    ss_builder = orchestrator.ss_builder
    T_data_len, n_obs_data = y_jax_data.shape

    # Ensure variable_names matches n_obs_data if provided
    if variable_names is not None and len(variable_names) != n_obs_data:
        print(f"Warning: Length of 'variable_names' ({len(variable_names)}) does not match number of observed series ({n_obs_data}). Using defaults.")
        variable_names = [f"Obs{i+1}" for i in range(n_obs_data)]
    elif variable_names is None:
        variable_names = gpm_model.gpm_observed_variables_original
        if len(variable_names) != n_obs_data: # Fallback if GPM varobs also mismatch
            variable_names = [f"Obs{i+1}" for i in range(n_obs_data)]

    print(f"  State Dim: {ss_builder.state_dim}, Dynamic Trends: {ss_builder.n_dynamic_trends}, Stat Vars: {ss_builder.n_stationary}, VAR Order: {ss_builder.var_order if ss_builder.n_stationary > 0 else 'N/A'}")

    # Resolve structural parameters (used for evaluating non-core definitions)
    structural_params_resolved = {}
    for param_name in gpm_model.parameters:
        try:
            structural_params_resolved[param_name] = jnp.array(
                _resolve_parameter_value(param_name, param_values, gpm_model.estimated_params, False),
                dtype=_DEFAULT_DTYPE
            )
        except Exception as e:
            raise ValueError(f"Failed to resolve structural parameter '{param_name}': {e}") from e


    # Build state space matrices inputs
    # Sigma_eta from trends (using resolved shock std devs)
    Sigma_eta = _build_trend_covariance(gpm_model, param_values)

    # Sigma_u and A_transformed for VAR (and gamma_list for P0)
    # _build_var_parameters resolves stationary shock stds and builds Sigma_u
    # It also resolves A_transformed and computes gamma_list_for_P0
    Sigma_u_innov, A_transformed_coeffs, gamma_list_for_P0 = _build_var_parameters(gpm_model, param_values) # param_values contains _var_innovation_corr_chol

    # Sigma_eps (Measurement Error)
    Sigma_eps = _build_measurement_covariance(gpm_model, param_values)

    # --- Build Initial Conditions (x0 and P0) ---
    init_mean = _build_initial_mean_for_test(gpm_model, ss_builder, initial_state_prior_overrides)

    # P0 Covariance Initialization: Use gamma if conditions met and requested, otherwise standard
    if use_gamma_init_for_test and ss_builder.n_stationary > 0 and ss_builder.var_order > 0 and gamma_list_for_P0:
        # Check validity of gamma_list_for_P0 before passing to _build_gamma_based_p0
        gamma_list_is_valid_for_p0_building = (
            gamma_list_for_P0 is not None and
            len(gamma_list_for_P0) == ss_builder.var_order and
            all(g is not None and hasattr(g, 'shape') and g.shape == (ss_builder.n_stationary, ss_builder.n_stationary) and jnp.all(jnp.isfinite(g)) for g in gamma_list_for_P0)
        )

        if _build_gamma_based_p0 is None:
             print("Warning: _build_gamma_based_p0 utility function is not available. Using standard P0 fallback.")
             init_cov = _create_standard_p0( # <<< Call imported function
                ss_builder.state_dim, ss_builder.n_dynamic_trends,
                context="fixed_eval", trend_P0_var_scale_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale
             )
        elif gamma_list_is_valid_for_p0_building:
             init_cov = _build_gamma_based_p0( # <<< Call imported function
                ss_builder.state_dim,
                ss_builder.n_dynamic_trends,
                gamma_list_for_P0, # Pass the gamma list
                ss_builder.n_stationary,
                ss_builder.var_order,
                gamma_init_scaling,
                context="fixed_eval", # Specify context
                trend_P0_var_scale_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale
             )
        else:
             print("  Warning: Gamma P0 requested, but gamma_list_for_P0 is invalid. Using standard P0.")
             # Fallback to standard P0 if gamma_list is bad
             init_cov = _create_standard_p0( # <<< Call imported function
                ss_builder.state_dim, ss_builder.n_dynamic_trends,
                context="fixed_eval", trend_P0_var_scale_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale
            )
    else:
        if use_gamma_init_for_test:
             print("  Info: Gamma P0 requested but conditions (stat vars, var order, gamma_list) not met. Using standard P0.")
        else:
             print("  Using standard P0 for fixed parameter evaluation.")

        if _create_standard_p0 is None:
             raise RuntimeError("Standard P0 utility function is not available.")
        init_cov = _create_standard_p0( # <<< Call imported function
           ss_builder.state_dim, ss_builder.n_dynamic_trends,
           context="fixed_eval", trend_P0_var_scale_override=trend_P0_var_scale, var_P0_var_scale_override=var_P0_var_scale
       )


    # --- Build State Space Matrices (F, Q, C, H) ---
    params_for_ss_builder = EnhancedBVARParams(
        A=A_transformed_coeffs, Sigma_u=Sigma_u_innov, Sigma_eta=Sigma_eta,
        structural_params=structural_params_resolved, Sigma_eps=Sigma_eps
    )
    F, Q, C, H_obs_matrix = ss_builder.build_state_space_from_enhanced_bvar(params_for_ss_builder)


    # --- Validate Built Matrices ---
    for mat, name in zip([F, Q, C, H_obs_matrix], ["F", "Q", "C", "H_obs (from builder)"]):
        if not jnp.all(jnp.isfinite(mat)): raise RuntimeError(f"Builder Matrix {name} contains non-finite values.")
    try: jnp.linalg.cholesky(Q + _KF_JITTER * jnp.eye(Q.shape[0]))
    except Exception as e: raise RuntimeError(f"Builder State Cov Q not PSD: {e}")
    if H_obs_matrix.shape[0] > 0:
        try: jnp.linalg.cholesky(H_obs_matrix + _KF_JITTER * jnp.eye(H_obs_matrix.shape[0]))
        except Exception as e: raise RuntimeError(f"Builder Measurement Cov H_obs not PSD: {e}")

    R_sim_chol_Q = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(Q.shape[0]))


    # --- Kalman Filter and Smoother ---
    if KalmanFilter is None:
        print("Warning: KalmanFilter not available. Skipping filtering and smoothing.")
        loglik_val = jnp.array(np.nan, dtype=_DEFAULT_DTYPE)
        # Return empty/NaN arrays for smoothed/simulated states with correct shapes for reconstruction
        smoothed_means_filter = jnp.full((T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE)
        smoothed_covs_filter = jnp.full((T_data_len, ss_builder.state_dim, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE)
        # Ensure sim_draws_core_state is created with expected shape even if NaNs
        sim_draws_core_state = jnp.full((max(1,num_sim_draws), T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE) if num_sim_draws > 0 else jnp.empty((0, T_data_len, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)

    else:
        kf = KalmanFilter(T=F, R=R_sim_chol_Q, C=C, H=H_obs_matrix, init_x=init_mean, init_P=init_cov)
        valid_obs_idx = jnp.arange(n_obs_data)
        I_obs = jnp.eye(n_obs_data) if n_obs_data > 0 else jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

        loglik_val = kf.log_likelihood(y_jax_data, valid_obs_idx, n_obs_data, C, H_obs_matrix, I_obs)
        if not jnp.isfinite(loglik_val):
            print(f"Warning: Log-likelihood is non-finite ({loglik_val}). Check parameters or model stability.")

        print(f"  Log-likelihood: {loglik_val:.3f}")

        filter_results = kf.filter(y_jax_data, valid_obs_idx, n_obs_data, C, H_obs_matrix, I_obs)
        smoothed_means_filter, smoothed_covs_filter = kf.smooth(y_jax_data, filter_results=filter_results) # These are filter/smoother means, not sim smoother draws

        # --- Run Simulation Smoother to get Draws ---
        sim_draws_core_state = jnp.empty((0, T_data_len, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)
        if num_sim_draws > 0:
            if jarocinski_corrected_simulation_smoother is None:
                print("  Warning: Simulation smoother function not available. Skipping simulation draws.")
                sim_draws_core_state = jnp.full((num_sim_draws, T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE) # Still create array with NaNs
            else:
                sim_draws_list = []
                print(f"  Running simulation smoother for {num_sim_draws} draws...")
                rng_key, sim_key_for_smoother = random.split(rng_key) # Split key for smoother
                for i in range(num_sim_draws):
                    sim_key_for_smoother, current_draw_key = random.split(sim_key_for_smoother) # Split for each draw
                    try:
                        s_states = jarocinski_corrected_simulation_smoother(
                            y_jax_data, F, R_sim_chol_Q, C, H_obs_matrix, init_mean, init_cov, current_draw_key)
                        if jnp.all(jnp.isfinite(s_states)): sim_draws_list.append(s_states)
                        else: print(f"  Warning: Sim draw {i+1} had non-finite values, discarding.")
                    except Exception as e_sim: print(f"  Warning: Sim draw {i+1} failed: {e_sim}, discarding.")
                if sim_draws_list: sim_draws_core_state = jnp.stack(sim_draws_list)
                else: sim_draws_core_state = jnp.full((0, T_data_len, ss_builder.state_dim), np.nan, dtype=_DEFAULT_DTYPE) # Empty if no successful draws
                print(f"  Completed {sim_draws_core_state.shape[0]} simulation draws.")
        else:
             print("  num_sim_draws is 0. Skipping simulation smoother.")
             # sim_draws_core_state is already initialized as empty


    n_actual_sim_draws = sim_draws_core_state.shape[0] # Update actual number of draws


    # --- Reconstruct original GPM variables from simulated core states ---
    reconstructed_all_trends_draws = jnp.full((n_actual_sim_draws, T_data_len, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
    reconstructed_all_stationary_draws = jnp.full((n_actual_sim_draws, T_data_len, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)

    # Only attempt reconstruction if simulation draws were generated
    if n_actual_sim_draws > 0:
        print("  Reconstructing original GPM variables from simulated states...")
        # Need SymbolicReducerUtils instance here to evaluate non-core definitions
        # Assume SymbolicReducerUtils is available or imported. Let's import it.
        try:
             from .gpm_model_parser import SymbolicReducerUtils
             utils = SymbolicReducerUtils()
        except ImportError:
             utils = None
             print("Warning: SymbolicReducerUtils not available. Non-core trend reconstruction might fail.")

        if utils is not None:
            core_var_map = ss_builder.core_var_map
            non_core_trend_defs = gpm_model.non_core_trend_definitions
            params_for_reconstruction = structural_params_resolved # Use the resolved structural params

            for i_draw in range(n_actual_sim_draws):
                core_states_draw_t = sim_draws_core_state[i_draw] # Shape (T_data_len, state_dim)

                # Map core state time series by name using core_var_map
                current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
                dynamic_core_trend_names = [cv for cv in gpm_model.core_variables if cv not in gpm_model.stationary_variables]
                for trend_name_dt in dynamic_core_trend_names:
                    state_idx_dt = core_var_map.get(trend_name_dt)
                    if state_idx_dt is not None and state_idx_dt < ss_builder.state_dim: # Check against full state_dim
                         current_draw_core_state_values_ts[trend_name_dt] = core_states_draw_t[:, state_idx_dt]

                for stat_name_map in gpm_model.stationary_variables:
                     state_idx_stat_lag0 = core_var_map.get(stat_name_map) # This should resolve to n_dynamic_trends + index in stationary_variables list
                     if state_idx_stat_lag0 is not None and state_idx_stat_lag0 < ss_builder.state_dim:
                          current_draw_core_state_values_ts[stat_name_map] = core_states_draw_t[:, state_idx_stat_lag0]


                # Reconstruct original GPM trend variables (can be core or non-core)
                for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
                    if orig_trend_name in current_draw_core_state_values_ts: # It's a core trend
                        reconstructed_all_trends_draws = reconstructed_all_trends_draws.at[i_draw, :, i_orig_trend].set(
                            current_draw_core_state_values_ts[orig_trend_name])
                    elif orig_trend_name in non_core_trend_defs: # It's a non-core trend defined by an expression
                        expr_def = non_core_trend_defs[orig_trend_name]
                        reconstructed_value_ts = jnp.full(T_data_len, 0.0, dtype=_DEFAULT_DTYPE)

                        const_val_eval = utils._evaluate_coefficient_expression(expr_def.constant_str, params_for_reconstruction)
                        if hasattr(const_val_eval, 'ndim') and const_val_eval.ndim == 0:
                            reconstructed_value_ts += float(const_val_eval)
                        elif isinstance(const_val_eval, (float, int, np.number)):
                             reconstructed_value_ts += float(const_val_eval)


                        for var_key, coeff_str in expr_def.terms.items():
                            term_var_name, term_lag = utils._parse_var_key_for_rules(var_key)
                            coeff_val_eval = utils._evaluate_coefficient_expression(coeff_str, params_for_reconstruction)
                            coeff_num = None
                            if hasattr(coeff_val_eval, 'ndim') and coeff_val_eval.ndim == 0:
                                coeff_num = float(coeff_val_eval)
                            elif isinstance(coeff_val_eval, (float, int, np.number)):
                                coeff_num = float(coeff_val_eval)

                            if coeff_num is not None:
                                if term_lag == 0:
                                    if term_var_name in current_draw_core_state_values_ts:
                                        reconstructed_value_ts += coeff_num * current_draw_core_state_values_ts[term_var_name]
                                    elif term_var_name in params_for_reconstruction:
                                         param_val_eval = utils._evaluate_coefficient_expression(term_var_name, params_for_reconstruction)
                                         if hasattr(param_val_eval, 'ndim') and param_val_eval.ndim == 0:
                                              reconstructed_value_ts += coeff_num * float(param_val_eval)
                                         elif isinstance(param_val_eval, (float, int, np.number)):
                                              reconstructed_value_ts += coeff_num * float(param_val_eval)

                        reconstructed_all_trends_draws = reconstructed_all_trends_draws.at[i_draw, :, i_orig_trend].set(reconstructed_value_ts)

                # Reconstruct original GPM stationary variables (typically just the current VAR states)
                for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
                    if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
                         reconstructed_all_stationary_draws = reconstructed_all_stationary_draws.at[i_draw, :, i_orig_stat].set(
                              current_draw_core_state_values_ts[orig_stat_name]
                          )
        else:
             print("Skipping reconstruction as SymbolicReducerUtils is not available.")

        print("  Reconstruction complete.")
    else:
        print("  Skipping reconstruction as no simulation draws were generated.")


    # --- Compute Summary Statistics and HDIs for Reconstructed Components ---
    # Ensure summary stats and hdi functions are available before calling
    if compute_summary_statistics is None or compute_hdi_robust is None:
        print("Warning: Summary statistics or HDI computation functions not available. Skipping stats/hdi calculation.")
        trend_stats = {'mean': jnp.full_like(reconstructed_all_trends_draws[0] if reconstructed_all_trends_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_trend_variables_original))), np.nan),
                       'median': jnp.full_like(reconstructed_all_trends_draws[0] if reconstructed_all_trends_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_trend_variables_original))), np.nan),
                       'mode': jnp.full_like(reconstructed_all_trends_draws[0] if reconstructed_all_trends_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_trend_variables_original))), np.nan),
                       'std': jnp.full_like(reconstructed_all_trends_draws[0] if reconstructed_all_trends_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_trend_variables_original))), np.nan)}
        stationary_stats = {'mean': jnp.full_like(reconstructed_all_stationary_draws[0] if reconstructed_all_stationary_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_stationary_variables_original))), np.nan),
                            'median': jnp.full_like(reconstructed_all_stationary_draws[0] if reconstructed_all_stationary_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_stationary_variables_original))), np.nan),
                            'mode': jnp.full_like(reconstructed_all_stationary_draws[0] if reconstructed_all_stationary_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_stationary_variables_original))), np.nan),
                            'std': jnp.full_like(reconstructed_all_stationary_draws[0] if reconstructed_all_stationary_draws.shape[0]>0 else jnp.empty((T_data_len, len(gpm_model.gpm_stationary_variables_original))), np.nan)}
        trend_hdi_lower, trend_hdi_upper = None, None
        stationary_hdi_lower, stationary_hdi_upper = None, None
    else:
        # Convert JAX arrays to NumPy arrays before computing stats/hdi
        trend_stats = compute_summary_statistics(np.asarray(reconstructed_all_trends_draws))
        stationary_stats = compute_summary_statistics(np.asarray(reconstructed_all_stationary_draws))

        trend_hdi_lower, trend_hdi_upper = None, None
        stationary_hdi_lower, stationary_hdi_upper = None, None

        if n_actual_sim_draws > 1: # Need at least 2 draws for HDI
            trend_hdi_lower, trend_hdi_upper = compute_hdi_robust(np.asarray(reconstructed_all_trends_draws), hdi_prob)
            stationary_hdi_lower, stationary_hdi_upper = compute_hdi_robust(np.asarray(reconstructed_all_stationary_draws), hdi_prob)


    # --- Package Results into SmootherResults Object ---
    # Ensure all arrays are converted to NumPy before storing in the dataclass if it expects NumPy
    results = SmootherResults(
        observed_data=y_np_data, # Store as NumPy array
        observed_variable_names=variable_names,
        time_index=time_index_for_plots,

        trend_draws=np.asarray(reconstructed_all_trends_draws), # Store as NumPy array
        trend_names=list(gpm_model.gpm_trend_variables_original),
        trend_stats=trend_stats,
        trend_hdi_lower=trend_hdi_lower,
        trend_hdi_upper=trend_hdi_upper,

        stationary_draws=np.asarray(reconstructed_all_stationary_draws), # Store as NumPy array
        stationary_names=list(gpm_model.gpm_stationary_variables_original),
        stationary_stats=stationary_stats,
        stationary_hdi_lower=stationary_hdi_lower,
        stationary_hdi_upper=stationary_hdi_upper,

        reduced_measurement_equations=gpm_model.reduced_measurement_equations,
        gpm_model=gpm_model, # Store the full parsed model

        parameters_used=param_values, # Store the fixed parameter values used
        log_likelihood=float(loglik_val.item()) if hasattr(loglik_val, 'item') and jnp.isfinite(loglik_val) else (float(loglik_val) if isinstance(loglik_val, (float,int)) and np.isfinite(loglik_val) else np.nan), # Store loglik as float

        n_draws=n_actual_sim_draws,
        hdi_prob=hdi_prob
    )

    # --- Plotting Section (Call plotting functions with SmootherResults) ---
    if plot_results and PLOTTING_AVAILABLE_EVAL:
        print("  Generating plots for fixed parameter evaluation...")

        # Determine save path prefix
        plot_save_prefix_fixed = None
        if save_plots_path_prefix: # This prefix should be provided by the caller (main_custom_plots.py)
             plot_save_prefix_fixed = save_plots_path_prefix
             # Ensure directory exists (caller should handle this, but double-check)
             plot_dir = os.path.dirname(plot_save_prefix_fixed)
             if plot_dir and not os.path.exists(plot_dir):
                  os.makedirs(plot_dir, exist_ok=True)

        # Check if there are any draws to plot before calling plotting functions that expect draws
        if results.n_draws > 0:

            # 2. Smoother Components Plot (Individual Trends & Stationary)
            # This plots the estimated components themselves with their uncertainty bands.
            if callable(plot_smoother_results): # Assumed plot_smoother_results accepts SmootherResults now
                print("    Generating smoother component plots (fixed params)...")
                plot_smoother_results(
                    results, # Pass the SmootherResults object
                    save_path=plot_save_prefix_fixed,
                    show_info_box=show_plot_info_boxes
                )
                # plot_smoother_results should handle plt.close() internally

            # 3. Default Observed vs. Single Trend Component (if enabled)
            # This plots observed data against individual estimated trend components.
            if plot_default_observed_vs_trend_components and callable(plot_observed_vs_single_trend_component):
                 print("    Generating default observed vs. trend component plots (fixed params)...")
                 plot_observed_vs_single_trend_component(
                     results, # Pass the SmootherResults object
                     save_path=plot_save_prefix_fixed, # Pass save path prefix
                     show_info_box=show_plot_info_boxes,
                     use_median_for_trend_line=True # Example flag
                 )
                 # plot_observed_vs_single_trend_component should handle plt.close() internally


            # 4. Custom Plots (based on custom_plot_specs)
            if custom_plot_specs and callable(plot_custom_series_comparison):
                print("    Generating custom plots (fixed params)...")
                for spec_idx, spec_dict_item in enumerate(custom_plot_specs):
                    # plot_custom_series_comparison handles saving and closing internally
                    plot_custom_series_comparison(
                        plot_title=spec_dict_item.get("title", f"Custom Plot {spec_idx+1}") + " (Fixed Params)",
                        series_specs=spec_dict_item.get("series_to_plot", []),
                        results=results, # Pass the SmootherResults object
                        show_info_box=show_plot_info_boxes,
                        save_path=plot_save_prefix_fixed # Pass save path prefix
                    )
        else:
             print("  Skipping plotting of results relying on simulation draws as no draws were generated.")

    elif plot_results and not PLOTTING_AVAILABLE_EVAL:
        print("  Plotting was requested for fixed param eval but is unavailable. Skipping plots.")
    # --- End of Plotting Section ---


    print(f"--- Evaluation for {gpm_file_path} complete ---")
    # The function signature says it returns SmootherResults, so return the created object.
    return results

# --- Keep the example __main__ block for local testing if desired ---
# It would need to import the necessary components from .p0_utils
# if __name__ == "__main__":
#     # Example of how to test this file directly
#     print("Running example test for gpm_prior_evaluator.py")

#     # Need to create a dummy GPM file and dummy data here for the test
#     # Need to define or import _build_gamma_based_p0, _create_standard_p0, etc. for the test block to run
#     # Example: from .p0_utils import _build_gamma_based_p0, _create_standard_p0, _extract_gamma_matrices_from_params

#     # Create a dummy GPM file
#     dummy_gpm_content = """
# parameters rho;
# estimated_params;
#     rho, normal_pdf, 0.5, 0.1;
#     stderr SHK_TREND1, inv_gamma_pdf, 2.0, 0.01;
#     stderr shk_stat1, inv_gamma_pdf, 2.0, 0.01;
# end;
# trends_vars TREND1;
# stationary_variables stat1;
# trend_shocks; var SHK_TREND1; end;
# shocks; var shk_stat1; end;
# trend_model; TREND1 = TREND1(-1) + rho*TREND1(-1) + SHK_TREND1; end;
# varobs OBS1;
# measurement_equations; OBS1 = TREND1 + stat1; end;
# var_prior_setup; var_order = 1; es = 0.8, 0.1; fs = 0.1, 0.1; gs = 3, 3; hs = 1, 1; eta = 2; end;
# initval; TREND1, normal_pdf, 0, 1; stat1, normal_pdf, 0, 1; end;
#     """
#     dummy_gpm_file = "dummy_eval_test_model.gpm"
#     with open(dummy_gpm_file, "w") as f:
#         f.write(dummy_gpm_content)

#     # Create dummy data
#     T_dummy = 100
#     np.random.seed(123)
#     dummy_data_np = np.random.randn(T_dummy, 1) * 0.5 + np.linspace(0, 5, T_dummy)[:, None]
#     dummy_data_df = pd.DataFrame(dummy_data_np, columns=['OBS1'], index=pd.date_range(start='2000-01-01', periods=T_dummy, freq='QE'))

#     # Define test parameters
#     test_params = {
#         'rho': 0.7, # Example fixed value
#         'SHK_TREND1': 0.1, # Example fixed std dev
#         'shk_stat1': 0.05, # Example fixed std dev
#         '_var_coefficients': jnp.array([[[0.6]]], dtype=_DEFAULT_DTYPE), # Example fixed VAR coeff
#         '_var_innovation_corr_chol': jnp.array([[1.0]], dtype=_DEFAULT_DTYPE), # Example fixed VAR corr (identity for 1 var)
#     }

#     # Define custom plot specs
#     custom_plots_example = [
#         {
#             "title": "OBS1 vs TREND1",
#             "series_to_plot": [
#                 {"type": "observed", "name": "OBS1", "label": "Observed", "style": "k.-"},
#                 {"type": "trend", "name": "TREND1", "label": "Estimated Trend", "show_hdi": True, "color": "blue"}
#             ]
#         },
#         {
#             "title": "STAT1 Component",
#             "series_to_plot": [
#                 {"type": "stationary", "name": "stat1", "label": "Estimated Stationary", "show_hdi": True, "color": "green"}
#             ]
#         }
#     ]


#     try:
#         # You would need to mock or provide access to the P0 utility functions here if running this __main__ directly
#         # Example: from .p0_utils import _build_gamma_based_p0 as mock_build_gamma, _create_standard_p0 as mock_create_standard
#         # and update the calls in evaluate_gpm_at_parameters within this __main__ scope or globally.
#         # Or, simplest: ensure your environment correctly allows importing .p0_utils when running this file.

#         eval_results = evaluate_gpm_at_parameters(
#             gpm_file_path=dummy_gpm_file,
#             y=dummy_data_df, # Pass DataFrame
#             param_values=test_params,
#             num_sim_draws=100, # Generate draws for plotting
#             plot_results=True,
#             plot_default_observed_vs_trend_components=True, # Plot default OvT
#             custom_plot_specs=custom_plots_example, # Use custom plots
#             variable_names=['OBS1'], # Explicitly name observed var
#             use_gamma_init_for_test=True, # Test gamma P0
#             gamma_init_scaling=1.0,
#             hdi_prob=0.95,
#             trend_P0_var_scale=100,
#             var_P0_var_scale=0.1,
#             save_plots_path_prefix="eval_test_output/plots/eval_plot", # Example save path prefix
#             show_plot_info_boxes=False
#         )
#         print("\nEvaluation Results Summary (example):")
#         print(f"Log-likelihood: {eval_results.log_likelihood:.4f}")
#         # Check if plots were generated in eval_test_output/plots/
#         print(f"Check 'eval_test_output/plots/' for generated plot files.")

#     except Exception as e:
#         print(f"Error during example evaluation: {e}")
#         import traceback; traceback.print_exc()
#     finally:
#         # Clean up dummy file
#         if os.path.exists(dummy_gpm_file):
#             os.remove(dummy_gpm_file)
#         # Note: Example save path creates a directory, which is not removed here.