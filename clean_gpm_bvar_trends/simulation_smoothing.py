# clean_gpm_bvar_trends/simulation_smoothing.py

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpyro # Retaining if MCMC object passed needs it for type or methods
from typing import Tuple, Optional, Dict, Any, List, Union # Added Union
import numpy as np # Use np for numpy
import xarray as xr
import arviz as az

# Local imports from the new refactored structure
from .common_types import EnhancedBVARParams # Used by _extract_parameters_for_ss_builder
from .gpm_model_parser import ReducedModel, ReducedExpression, SymbolicReducerUtils # Import ReducedExpression and SymbolicReducerUtils for reconstruction
from .state_space_builder import StateSpaceBuilder           # For evaluating expressions
# ParameterContract not directly used here, but ss_builder uses it internally
from .constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER, _SP_JITTER # Ensure all jitters and dtype are here

# Import P0 building utilities

from .P0_utils import (
    _build_gamma_based_p0,
    _create_standard_p0,
    _extract_gamma_matrices_from_params # Needed here to get gamma list from draw params
)
# except ImportError:
#     _build_gamma_based_p0 = None
#     _create_standard_p0 = None
#     _extract_gamma_matrices_from_params = None
#     print("Warning: p0_utils not available. P0 building in simulation_smoothing will likely fail.")


# Conditional imports with error handling
try:
    from .Kalman_filter_jax import KalmanFilter # Note the leading dot for relative import
except ImportError:
    KalmanFilter = None
    print("Warning: KalmanFilter not available in simulation_smoothing.py")

try:
    # Assuming jarocinski_corrected_simulation_smoother is defined in this file or imported correctly
    # If defined locally, no import needed here. If elsewhere, import from there.
    # Let's assume it's defined later in this file for now.
    pass
except ImportError:
    # If it were imported from elsewhere, you'd do the import here.
    # Example: from .simulation_smoother_backend import jarocinski_corrected_simulation_smoother
    pass



def jarocinski_corrected_simulation_smoother(
    y: jnp.ndarray,
    F: jnp.ndarray,
    R_ss: jnp.ndarray, # Shock impact matrix (n_states x n_shocks_actual)
    C: jnp.ndarray,
    H: jnp.ndarray,    # Observation noise covariance
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    key: jax.Array
) -> jax.Array:
   """
   Jarocinski (2015) corrected Durbin & Koopman simulation smoother.
   (Implementation as provided in your file, assuming it's correct)
   """
   T_local, _ = y.shape
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
    R_ss: jnp.ndarray, # Shock impact matrix
    C: jnp.ndarray,
    H: jnp.ndarray,
    zero_init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    T_sim: int,
    key_sim: jax.Array
) -> Tuple[jax.Array, jax.Array]:
   """Forward simulation with zero initial mean."""
   # (Implementation as provided in your file, seems correct)
   state_dim_fwd = F.shape[0]
   n_obs_fwd = C.shape[0]
   n_actual_shocks = R_ss.shape[1] if R_ss.ndim == 2 and R_ss.shape[0] == state_dim_fwd else 0
   alpha_p = jnp.zeros((T_sim, state_dim_fwd), dtype=_DEFAULT_DTYPE)
   y_p = jnp.zeros((T_sim, n_obs_fwd), dtype=_DEFAULT_DTYPE)
   current_key, init_key_fwd = random.split(key_sim)
   init_cov_reg_fwd = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim_fwd, dtype=_DEFAULT_DTYPE)
   try: alpha_0_fwd = random.multivariate_normal(init_key_fwd, zero_init_mean, init_cov_reg_fwd, dtype=_DEFAULT_DTYPE)
   except Exception: alpha_0_fwd = zero_init_mean # Fallback
   current_s = alpha_0_fwd
   for t_idx in range(T_sim):
       alpha_p = alpha_p.at[t_idx].set(current_s)
       current_key, obs_key_fwd = random.split(current_key)
       obs_m = C @ current_s
       H_reg_fwd = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_obs_fwd, dtype=_DEFAULT_DTYPE)
       try: y_t_fwd = random.multivariate_normal(obs_key_fwd, obs_m, H_reg_fwd, dtype=_DEFAULT_DTYPE)
       except Exception: y_t_fwd = obs_m # Fallback
       y_p = y_p.at[t_idx].set(y_t_fwd)
       if t_idx < T_sim - 1:
           current_key, state_key_fwd = random.split(current_key)
           shock_term=R_ss@random.normal(state_key_fwd,(n_actual_shocks,),dtype=_DEFAULT_DTYPE) if n_actual_shocks>0 else jnp.zeros(state_dim_fwd,dtype=_DEFAULT_DTYPE)
           current_s = F @ current_s + shock_term
   return alpha_p, y_p


def compute_smoothed_expectation(
    y_star: jnp.ndarray,
    F: jnp.ndarray,
    R_ss: jnp.ndarray, # Shock impact matrix
    C: jnp.ndarray,
    H: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray
) -> jax.Array:
   """Compute E(α|y*) using Kalman filter/smoother."""
   # (Implementation as provided, seems correct)
   T_exp, n_obs_exp = y_star.shape
   state_dim_exp = F.shape[0]
   if KalmanFilter is None: return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE)
   kf_instance = KalmanFilter(T=F, R=R_ss, C=C, H=H, init_x=init_mean, init_P=init_cov)
   valid_obs_idx_static = jnp.arange(n_obs_exp, dtype=jnp.int32)
   I_obs_static = jnp.eye(n_obs_exp, dtype=_DEFAULT_DTYPE)
   try:
       filter_results = kf_instance.filter(y_star, static_valid_obs_idx=valid_obs_idx_static, static_n_obs_actual=n_obs_exp, static_C_obs=C, static_H_obs=H, static_I_obs=I_obs_static)
       smoothed_means, _ = kf_instance.smooth(y_star, filter_results=filter_results)
       if not jnp.all(jnp.isfinite(smoothed_means)): return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE)
       return smoothed_means
   except Exception as e_smooth_exp:
       print(f"Error during smoothed expectation computation: {e_smooth_exp}")
       return jnp.full((T_exp, state_dim_exp), jnp.nan, dtype=_DEFAULT_DTYPE)


# --- MCMC Helper/Interface functions ---
# These helpers are used by extract_reconstructed_components_fixed

def _identify_required_sites(mcmc_samples: Dict, gpm_model_struct: ReducedModel) -> list:
    """Identify required MCMC sites for state space construction."""
    # (Implementation as provided, seems correct)
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
    ) -> Dict[str, Any]: # Return Dict for builder, not EnhancedBVARParams
    """
    Extracts parameters for a specific MCMC draw and formats them for StateSpaceBuilder.
    Returns a dictionary where keys match StateSpaceBuilder's expected input names.
    """
    # This is a simplified version to get a dictionary; StateSpaceBuilder handles EnhancedBVARParams.
    # It's cleaner if this function just gets the raw values from the draw.
    # Let's adapt it to return a dictionary with *potential* builder keys.

    builder_params_draw: Dict[str, Any] = {}

    # Get structural parameters
    if hasattr(gpm_model_struct, 'parameters'):
        for p_n in gpm_model_struct.parameters:
            if p_n in mcmc_samples_dict and mcmc_samples_dict[p_n].shape[0] > mcmc_draw_idx:
                builder_params_draw[p_n] = mcmc_samples_dict[p_n][mcmc_draw_idx]

    # Get shock standard deviations (builder expects shock name, not "sigma_")
    if hasattr(gpm_model_struct, 'trend_shocks'):
        for s_bn in gpm_model_struct.trend_shocks:
            mcmc_sn = f"sigma_{s_bn}";
            if mcmc_sn in mcmc_samples_dict and mcmc_samples_dict[mcmc_sn].shape[0] > mcmc_draw_idx:
                builder_params_draw[s_bn] = mcmc_samples_dict[mcmc_sn][mcmc_draw_idx]
            # Default if not found in samples could be added here, but better to raise error if expected but missing?

    if hasattr(gpm_model_struct, 'stationary_shocks'):
        for s_bn in gpm_model_struct.stationary_shocks:
            mcmc_sn = f"sigma_{s_bn}";
            if mcmc_sn in mcmc_samples_dict and mcmc_samples_dict[mcmc_sn].shape[0] > mcmc_draw_idx:
                builder_params_draw[s_bn] = mcmc_samples_dict[mcmc_sn][mcmc_draw_idx]
            # Default if not found

    # Get VAR parameters (A_transformed, Omega_u_chol) if they were sampled/determined
    if gpm_model_struct.var_prior_setup:
        if "A_transformed" in mcmc_samples_dict and mcmc_samples_dict["A_transformed"].shape[0] > mcmc_draw_idx:
             builder_params_draw["_var_coefficients"] = mcmc_samples_dict["A_transformed"][mcmc_draw_idx]
        if "Omega_u_chol" in mcmc_samples_dict and mcmc_samples_dict["Omega_u_chol"].shape[0] > mcmc_draw_idx:
             builder_params_draw["_var_innovation_corr_chol"] = mcmc_samples_dict["Omega_u_chol"][mcmc_draw_idx]
        # Sigma_u_full is often derived, not directly sampled.

    # Get initial mean (if sampled/determined)
    if "init_mean_full" in mcmc_samples_dict and mcmc_samples_dict["init_mean_full"].shape[0] > mcmc_draw_idx:
         builder_params_draw["init_mean_full"] = mcmc_samples_dict["init_mean_full"][mcmc_draw_idx]


    # Note: This function doesn't return the full Sigma_u, Sigma_eta, Sigma_eps covariances.
    # StateSpaceBuilder is responsible for building those from the individual pieces.
    # If needed, we could add "_trend_innovation_cov_full", etc. here if they were sampled/determined.
    # Let's rely on StateSpaceBuilder's fallback logic to build them from individual stddevs if the full covs aren't provided.

    return builder_params_draw


def _extract_initial_mean(mcmc_samples_dict: Dict, mcmc_draw_idx: int, state_dim: int) -> jnp.ndarray:
    """Extracts initial mean for a specific MCMC draw."""
    # (Implementation as provided, seems correct)
    if "init_mean_full" in mcmc_samples_dict and mcmc_samples_dict["init_mean_full"].shape[0] > mcmc_draw_idx:
        init_mean = mcmc_samples_dict["init_mean_full"][mcmc_draw_idx]
        return jnp.where(jnp.isfinite(init_mean), init_mean, jnp.zeros_like(init_mean))
    # Fallback to zeros if init_mean_full site is missing or index is out of bounds
    return jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)


# _create_reasonable_initial_covariance is not used by extract_reconstructed_components_fixed
# as it uses P0 builders from p0_utils.py or the logic within gpm_numpyro_models.py.
# If it's needed elsewhere in simulation_smoothing.py, keep it. Otherwise, it can be removed.
# Let's assume it's not needed for the main extraction workflow.
# def _create_reasonable_initial_covariance(state_dim: int, n_dynamic_trends_in_state: int) -> jnp.ndarray:
#     pass


# --- Main Extraction Function ---

def extract_reconstructed_components_fixed( # Corrected name from extract_reconstructed_components
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray,
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None,
    use_gamma_init_for_smoother: bool = True,
    gamma_init_scaling_for_smoother: float = 1.0,
    hdi_prob: float = 0.9, # Added hdi_prob
    observed_variable_names: Optional[List[str]] = None, # Added observed_variable_names
    time_index: Optional[Any] = None, # Added time_index
    # Removed current_builder_params as an input to this function,
    # it's extracted *inside* the loop for each draw.
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str,List[str]]]: # Return raw draws and names
    """
    Extracts reconstructed components using the Jarocinski smoother for multiple MCMC draws.
    Returns raw draws and names for post-processing (e.g., plotting).
    """
    print(f"\n=== FIXED SIMULATION SMOOTHER ===")
    print(f"Use gamma-based P0: {use_gamma_init_for_smoother}")
    print(f"Gamma scaling: {gamma_init_scaling_for_smoother}")

    if rng_key_smooth is None: rng_key_smooth = random.PRNGKey(0)

    mcmc_samples = mcmc_output.get_samples()
    if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
        print("Warning: No MCMC samples available")
        num_orig_trends = len(gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
        # Return empty arrays with correct dimensions for the output structure
        return jnp.empty((0, 0, num_orig_trends)), jnp.empty((0, 0, num_orig_stat)), {'trends': list(gpm_model.gpm_trend_variables_original), 'stationary': list(gpm_model.gpm_stationary_variables_original)}


    T_data, _ = y_data.shape
    state_dim = ss_builder.state_dim # Get state dim once

    first_param_key = list(mcmc_samples.keys())[0]
    total_posterior_draws = mcmc_samples[first_param_key].shape[0]
    actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

    if actual_num_smooth_draws <= 0:
        num_orig_trends = len(gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
        return jnp.empty((0, T_data, num_orig_trends)), \
               jnp.empty((0, T_data, num_orig_stat)), \
               {'trends': list(gpm_model.gpm_trend_variables_original),
                'stationary': list(gpm_model.gpm_stationary_variables_original)}

    #draw_indices = onp.round(onp.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    # Check if smoother function is available
    if jarocinski_corrected_simulation_smoother is None:
        print("ERROR: Jarocinski simulation smoother function not available.")
        num_orig_trends = len(gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
        return jnp.empty((0, T_data, num_orig_trends)), jnp.empty((0, T_data, num_orig_stat)), {'trends': list(gpm_model.gpm_trend_variables_original), 'stationary': list(gpm_model.gpm_stationary_variables_original)}

    # Check if P0 builders are available if gamma init is requested
    # This check relies on the import block earlier setting the variables to None on failure
    if use_gamma_init_for_smoother and (_build_gamma_based_p0 is None or _create_standard_p0 is None or _extract_gamma_matrices_from_params is None):
         print("ERROR: P0 building helper functions not available for smoother (gamma init requested).")
         # Fallback to standard P0 logic if helpers are missing
         use_gamma_init_for_smoother = False
         print("Warning: Falling back to standard P0 for all draws due to missing P0 utilities.")

    # If P0 builders are still missing even for standard P0
    if _create_standard_p0 is None: # This should only be None if the import failed and gamma was false
         print("ERROR: Standard P0 utility function is not available.")
         num_orig_trends = len(gpm_model.gpm_trend_variables_original)
         num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
         return jnp.empty((0, T_data, num_orig_trends)), jnp.empty((0, T_data, num_orig_stat)), {'trends': list(gpm_model.gpm_trend_variables_original), 'stationary': list(gpm_model.gpm_stationary_variables_original)}


    output_trend_draws_list = []
    output_stationary_draws_list = []
    successful_draws = 0

    # Need SymbolicReducerUtils instance here for reconstruction if not imported globally
    try:
        from .gpm_model_parser import SymbolicReducerUtils
        utils = SymbolicReducerUtils()
    except ImportError:
        utils = None
        print("Warning: SymbolicReducerUtils not available for reconstruction.")


    for i_loop, mcmc_draw_idx in enumerate(draw_indices):
        rng_key_smooth, sim_key = random.split(rng_key_smooth)

        # 1. Extract parameters for the current draw (using the helper)
        # This dictionary contains builder-friendly parameter values for THIS draw.
        current_builder_params_draw = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx, gpm_model)

        # 2. Build state space matrices for the current draw (using ss_builder)
        # ss_builder needs EnhancedBVARParams, which it builds from the dictionary.
        # It internally handles building Sigma_u, Sigma_eta from components/full cov.
        F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_from_enhanced_bvar(
             ss_builder._build_enhanced_bvar_from_params_dict(current_builder_params_draw) # Use ss_builder's helper
         )

        # 3. Get initial mean (from the draw params)
        init_mean_for_smoother = _extract_initial_mean(mcmc_samples, mcmc_draw_idx, state_dim)

        # 4. Build initial covariance (P0) for the current draw
        init_cov_for_smoother = None # Initialize
        gamma_list_for_this_draw = None # Initialize

        # Attempt to build gamma list for this draw if gamma init is requested and possible
        if use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and ss_builder.var_order > 0 and _extract_gamma_matrices_from_params is not None:
             # _extract_gamma_matrices_from_params needs A_transformed and Sigma_u values from the draw
             A_trans_d_for_gamma = current_builder_params_draw.get("_var_coefficients")
             # ss_builder builds _var_innovation_cov_full in build_state_space_from_enhanced_bvar
             # so we need to get it from the resulting Q or rebuild it here.
             # Rebuilding from components is more reliable if Q might be problematic.
             # Let's rebuild Sigma_u from the components (shock stds and corr_chol) in current_builder_params_draw
             # Similar logic to _build_var_parameters's Sigma_u part is needed here.
             # This requires access to _resolve_parameter_value or a portable version.
             # For simplicity and to avoid recreating that logic, let's try to get the Sigma_u_innov built by ss_builder
             # or rely on the builders in p0_utils to handle reconstruction if they can.
             # The _extract_gamma_matrices_from_params *in p0_utils* takes A_transformed and Sigma_u (matrices), not components.
             # So we need to get Sigma_u_innov as built by ss_builder for this draw.
             # ss_builder._build_matrices_internal returns Q. Sigma_u is the top-left block of the VAR part of Q.
             n_dynamic_core_trends = ss_builder.n_core - ss_builder.n_stationary
             var_block_start_in_state = n_dynamic_core_trends
             n_stat_vars = ss_builder.n_stationary

             # Extract Sigma_u from the Q matrix built by ss_builder for this draw
             Sigma_u_d_for_gamma = Q_draw[var_block_start_in_state : var_block_start_in_state + n_stat_vars,
                                          var_block_start_in_state : var_block_start_in_state + n_stat_vars]

             gamma_list_for_this_draw = _extract_gamma_matrices_from_params( # <<< Call imported function
                 A_trans_d_for_gamma, Sigma_u_d_for_gamma,
                 n_stat_vars, ss_builder.var_order
             )

        # Decide whether to use gamma-based P0 or standard based on request and gamma availability
        use_gamma_for_this_draw = use_gamma_init_for_smoother and (gamma_list_for_this_draw is not None)

        if use_gamma_for_this_draw:
            if _build_gamma_based_p0 is None: # Should be checked earlier, but defensive
                 print(f"  Draw {i_loop}: _build_gamma_based_p0 not available. Using standard P0.")
                 init_cov_for_smoother = _create_standard_p0(state_dim, ss_builder.n_dynamic_trends, context="mcmc_smoother")
            else:
                 init_cov_for_smoother = _build_gamma_based_p0( # <<< Call imported function
                     state_dim, ss_builder.n_dynamic_trends,
                     gamma_list_for_this_draw, ss_builder.n_stationary, ss_builder.var_order,
                     gamma_init_scaling_for_smoother, context="mcmc_smoother"
                 )
        else:
            # Use standard P0 (gamma not requested, not possible, or builders missing)
            init_cov_for_smoother = _create_standard_p0( # <<< Call imported function
                state_dim, ss_builder.n_dynamic_trends, context="mcmc_smoother"
            )

        # Ensure P0 is finite before proceeding
        if not jnp.all(jnp.isfinite(init_cov_for_smoother)):
             print(f"  Draw {i_loop}: Generated P0 contains non-finite values. Skipping draw.")
             continue # Skip this draw if P0 is bad


        # 5. Ensure matrices are finite before running smoother
        matrices_finite = (
            jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(Q_draw)) and
            jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and
            jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))
        )

        if not matrices_finite:
            print(f"  Draw {i_loop}: Skipping due to non-finite matrices")
            continue

        # 6. Regularize Q for Cholesky needed by smoother
        Q_reg_sm = (Q_draw + Q_draw.T) / 2.0 + _JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
        try:
            R_sm_draw = jnp.linalg.cholesky(Q_reg_sm)
        except Exception:
             print(f"  Draw {i_loop}: Q not PSD, falling back to diagonal R.")
             R_sm_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg_sm), _JITTER)))


        # 7. Run the Jarocinski corrected simulation smoother
        try:
            # The smoother operates on the *core* state vector
            core_states_smoothed_draw = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_sm_draw, C_draw, H_draw,
                init_mean_for_smoother, init_cov_for_smoother, sim_key
            )

            if not jnp.all(jnp.isfinite(core_states_smoothed_draw)):
                print(f"  Draw {i_loop}: Skipping due to non-finite smoother output")
                continue

        except Exception as e:
            print(f"  Draw {i_loop}: Jarocinski smoother failed: {e}")
            continue

        # 8. Reconstruct original GPM variables from smoothed core states
        # Use the current_builder_params_draw (which contains structural_params) for reconstruction
        if utils is None:
             print("  Draw {i_loop}: Skipping reconstruction as SymbolicReducerUtils is not available.")
             continue # Skip this draw if reconstruction utils are missing

        current_draw_core_state_values_ts: Dict[str, jnp.ndarray] = {}
        core_var_map = ss_builder.core_var_map # Get map from ss_builder
        non_core_trend_defs = gpm_model.non_core_trend_definitions # Get non-core defs

        # Map core state time series by name using core_var_map
        for var_name, state_idx in core_var_map.items():
             if state_idx is not None and state_idx < state_dim:
                 current_draw_core_state_values_ts[var_name] = core_states_smoothed_draw[:, state_idx]


        reconstructed_trends_this_draw = jnp.full((T_data, len(gpm_model.gpm_trend_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        # Need structural params for evaluating non-core definitions
        params_for_reconstruction = {k:v for k,v in current_builder_params_draw.items() if k in gpm_model.parameters} # Filter for structural params

        # Reconstruct original GPM trend variables (can be core or non-core)
        for i_orig_trend, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
            if orig_trend_name in current_draw_core_state_values_ts: # It's a core trend
                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(
                    current_draw_core_state_values_ts[orig_trend_name]
                )
            elif orig_trend_name in non_core_trend_defs: # It's a non-core trend defined by an expression
                expr_def = non_core_trend_defs[orig_trend_name]
                reconstructed_value_ts = jnp.full(T_data, 0.0, dtype=_DEFAULT_DTYPE)

                const_val_eval = utils.evaluate_numeric_expression(expr_def.constant_str, params_for_reconstruction)
                if isinstance(const_val_eval, (float, int, np.number)): # Check it's a number
                    reconstructed_value_ts += float(const_val_eval)

                for var_key, coeff_str in expr_def.terms.items():
                    term_var_name, term_lag = utils._parse_variable_key(var_key) # Using existing _parse_variable_key

                    # --- Start New Logic for processing a term ---
                    term_var_name_val_ts_or_scalar = None
                    if term_lag == 0:
                        if term_var_name in current_draw_core_state_values_ts:
                            term_var_name_val_ts_or_scalar = current_draw_core_state_values_ts[term_var_name]
                        elif term_var_name in params_for_reconstruction:
                            term_var_name_val_ts_or_scalar = utils.evaluate_numeric_expression(term_var_name, params_for_reconstruction)
                        elif utils._is_numeric_string(term_var_name):
                            term_var_name_val_ts_or_scalar = float(term_var_name)
                        else:
                            print(f"Warning (sim_smoother): Term variable '{term_var_name}' for non-core trend '{orig_trend_name}' is not a known core state, parameter, or numeric literal. Skipping term '{var_key}'.")
                            continue
                    else:
                        print(f"Warning (sim_smoother): Term '{var_key}' for non-core trend '{orig_trend_name}' has unhandled lag {term_lag}. Skipping term.")
                        continue

                    actual_coeff_val_or_ts = None
                    if coeff_str is None:
                        actual_coeff_val_or_ts = 1.0
                    elif utils._is_numeric_string(coeff_str):
                        actual_coeff_val_or_ts = float(coeff_str)
                    elif coeff_str in params_for_reconstruction:
                        actual_coeff_val_or_ts = utils.evaluate_numeric_expression(coeff_str, params_for_reconstruction)
                    elif coeff_str in current_draw_core_state_values_ts:
                        actual_coeff_val_or_ts = current_draw_core_state_values_ts[coeff_str]
                    else:
                        print(f"Warning (sim_smoother): Coefficient string '{coeff_str}' for term '{var_key}' of non-core trend '{orig_trend_name}' is not numeric, not a parameter, and not a known state variable. Skipping term.")
                        continue

                    if actual_coeff_val_or_ts is not None and term_var_name_val_ts_or_scalar is not None:
                        term_contribution = actual_coeff_val_or_ts * term_var_name_val_ts_or_scalar
                        reconstructed_value_ts += term_contribution
                    else:
                        print(f"Warning (sim_smoother): Could not fully evaluate term '{coeff_str} * {term_var_name}' for non-core trend '{orig_trend_name}'. Skipping.")
                        continue
                    # --- End New Logic ---

                reconstructed_trends_this_draw = reconstructed_trends_this_draw.at[:, i_orig_trend].set(reconstructed_value_ts)

        reconstructed_stationary_this_draw = jnp.full((T_data, len(gpm_model.gpm_stationary_variables_original)), jnp.nan, dtype=_DEFAULT_DTYPE)
        for i_orig_stat, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
            if orig_stat_name in current_draw_core_state_values_ts and orig_stat_name in gpm_model.stationary_variables:
                 reconstructed_stationary_this_draw = reconstructed_stationary_this_draw.at[:, i_orig_stat].set(
                      current_draw_core_state_values_ts[orig_stat_name]
                  )

        output_trend_draws_list.append(reconstructed_trends_this_draw)
        output_stationary_draws_list.append(reconstructed_stationary_this_draw)
        successful_draws += 1

    print(f"Successfully processed {successful_draws}/{len(draw_indices)} draws")

    # Stack draws and return results
    if not output_trend_draws_list:
        print("ERROR: No valid simulation smoother draws!")
        num_orig_trends = len(gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
        return jnp.empty((0, T_data, num_orig_trends)), \
               jnp.empty((0, T_data, num_orig_stat)), \
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


