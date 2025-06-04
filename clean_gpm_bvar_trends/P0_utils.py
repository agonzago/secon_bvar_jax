# clean_gpm_bvar_trends/p0_utils.py
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, Any, Union
# We need specific imports for types used in function signatures (like ReducedModel, StateSpaceBuilder)
# If these types aren't defined in this file, we can use quoted type hints or import them.
# Let's use quoted type hints to avoid circular imports if those modules needed p0_utils.
# Alternatively, define a simple common_types_model_structure.py with these data classes.
# For now, quoted type hints are simpler.

# Import constants
from .constants import _DEFAULT_DTYPE, _KF_JITTER # Ensure relative imports

# Import necessary functions/helpers that P0 building depends on
# make_stationary_var_transformation_jax is needed for gamma list calculation
try:
    from .stationary_prior_jax_simplified import make_stationary_var_transformation_jax, _JITTER as _SP_JITTER
except ImportError:
    make_stationary_var_transformation_jax = None; _SP_JITTER = 1e-8
    print("Warning: stationary_prior_jax_simplified not available for P0 utils.")

# Need _resolve_parameter_value or a similar portable version if building Sigma_u manually
# Assuming a portable version or dependency injection is handled elsewhere.
# For now, assuming param_values comes in with resolved values or a helper is accessible.
# Let's add a simple portable resolver here if it's only for P0 building.

def _resolve_parameter_value_portable(param_key_base: str, param_values: Dict[str, Any], estimated_params: Dict[str, Any], is_shock_std_dev: bool = False) -> float:
    """
    A simple, portable version to resolve a parameter value for P0 utility,
    avoiding dependency on the full gpm_prior_evaluator._resolve_parameter_value.
    This assumes param_values dict has the resolved values or a simple fallback is possible.
    """
    if param_key_base in param_values:
        val = param_values[param_key_base]
        return float(val.item() if hasattr(val, 'item') else val) # Safely get float
    elif is_shock_std_dev and f"sigma_{param_key_base}" in param_values:
         val = param_values[f"sigma_{param_key_base}"]
         return float(val.item() if hasattr(val, 'item') else val) # Safely get float
    # Add fallback to prior mean/mode if needed and estimated_params structure is known
    # For this utility, let's rely on param_values having the necessary shock stddevs
    # as resolved by _build_var_parameters or equivalent elsewhere.
    # If not found, maybe raise an error or return a default? Raising is safer for debugging.
    raise ValueError(f"Parameter '{param_key_base}' not found in provided param_values for portable resolution.")


def _extract_gamma_matrices_from_params(A_transformed: Any, Sigma_u: Any,
                                         n_stationary: int, var_order: int) -> Optional[List[jnp.ndarray]]:
    """
    Extracts and computes gamma matrices from A_transformed and Sigma_u.
    This function does NOT resolve parameter names; it expects A_transformed and Sigma_u JAX arrays.
    """
    if A_transformed is None or Sigma_u is None:
        return None
    if not (isinstance(A_transformed, (jnp.ndarray, jnp.ndarray)) and A_transformed.shape == (var_order, n_stationary, n_stationary)):
        return None # Bad draw or parameters
    if not (isinstance(Sigma_u, (jnp.ndarray, jnp.ndarray)) and Sigma_u.shape == (n_stationary, n_stationary)):
        return None  # Bad draw or parameters

    if make_stationary_var_transformation_jax is None:
        # print("Warning: make_stationary_var_transformation_jax not available.")
        return None # Cannot compute gamma without the transform

    try:
        A_raw_list = [A_transformed[lag] for lag in range(var_order)]
        _, gamma_list = make_stationary_var_transformation_jax(
            Sigma_u, A_raw_list, n_stationary, var_order
        )
        # Validate gamma list properties
        if gamma_list and len(gamma_list) == var_order:
            valid_gammas = all(g is not None and hasattr(g, 'shape') and g.shape == (n_stationary, n_stationary) and jnp.all(jnp.isfinite(g))
                               for g in gamma_list)
            if valid_gammas:
                return gamma_list
            # else: print("Warning: Generated gamma list contains invalid matrices.")
    except Exception as e:
        # print(f"Warning: Error during stationary VAR transformation: {e}")
        pass # Return None if transform fails

    return None # Return None if gamma list is invalid or transform failed


def _build_gamma_based_p0(
    state_dim: int, n_dynamic_trends: int, gamma_list: List[jnp.ndarray],
    n_stationary: int, var_order: int, gamma_scaling: float,
    dynamic_trend_names: List[str],  # New argument
    core_var_map: Dict[str, Any],  # New argument
    context: str = "mcmc", # "mcmc", "fixed_eval"
    trend_P0_scales_override: Optional[Union[float, Dict[str, float]]] = None,  # Renamed and type updated
    var_P0_var_scale_override: Optional[float] = None
) -> jnp.ndarray:
    """Build gamma-based P0."""
    # Default trend scale, used if trend_P0_scales_override is None or a float
    default_trend_scale = 1e6 if context == "mcmc" else 1e4
    var_fallback_scale = var_P0_var_scale_override if var_P0_var_scale_override is not None else (4.0 if context == "mcmc" else 1.0)

    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    if n_dynamic_trends > 0:
        trend_scales_diag = jnp.ones(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * default_trend_scale
        if trend_P0_scales_override is not None:
            if isinstance(trend_P0_scales_override, float):
                trend_scales_diag = jnp.ones(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * trend_P0_scales_override
            elif isinstance(trend_P0_scales_override, dict):
                temp_scales = []
                for i, trend_name in enumerate(dynamic_trend_names): # dynamic_trend_names should correspond to the first n_dynamic_trends states
                    scale = trend_P0_scales_override.get(trend_name, default_trend_scale)
                    temp_scales.append(scale)
                trend_scales_diag = jnp.array(temp_scales, dtype=_DEFAULT_DTYPE)

        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
            jnp.diag(trend_scales_diag)
        )

    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order

    gamma_list_is_valid = (
        gamma_list is not None and
        len(gamma_list) == var_order and
        gamma_list[0] is not None and
        hasattr(gamma_list[0], 'shape') and
        gamma_list[0].shape == (n_stationary, n_stationary) and
        all(g is not None and hasattr(g, 'shape') and g.shape == (n_stationary, n_stationary) and jnp.all(jnp.isfinite(g))
            for g in gamma_list)
    )

    if n_stationary > 0 and var_order > 0 and gamma_list_is_valid:
        var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
        for r_block_idx in range(var_order):
            for c_block_idx in range(var_order):
                lag_d = abs(r_block_idx - c_block_idx)
                # Fallback to Gamma0 * decay if lag_d is out of bounds (shouldn't happen if valid)
                blk_unscaled = gamma_list[lag_d] if lag_d < len(gamma_list) else gamma_list[0] * (0.5**lag_d)
                curr_blk = blk_unscaled * gamma_scaling
                if r_block_idx > c_block_idx: curr_blk = curr_blk.T

                row_start_slice = r_block_idx * n_stationary
                row_end_slice = (r_block_idx + 1) * n_stationary
                col_start_slice = c_block_idx * n_stationary
                col_end_slice = (c_block_idx + 1) * n_stationary

                if row_end_slice <= var_state_total_dim and col_end_slice <= var_state_total_dim:
                     var_block_cov = var_block_cov.at[row_start_slice:row_end_slice, col_start_slice:col_end_slice].set(curr_blk)
                # else: print("Warning: P0 VAR block indexing out of bounds.") # Tracing

        if var_start_idx + var_state_total_dim <= state_dim:
            init_cov = init_cov.at[var_start_idx:var_start_idx + var_state_total_dim,
                                  var_start_idx:var_start_idx + var_state_total_dim].set(var_block_cov)
    elif var_state_total_dim > 0:
        # print("Info: Gamma list not valid for VAR block. Using fallback.") # Tracing
        # Fallback for VAR part uses var_P0_var_scale_override
        var_block_scale = var_P0_var_scale_override if var_P0_var_scale_override is not None else (4.0 if context == "mcmc" else 1.0)
        init_cov = init_cov.at[var_start_idx:var_start_idx + var_state_total_dim,
                              var_start_idx:var_start_idx + var_state_total_dim].set(
                                  jnp.eye(var_state_total_dim, dtype=_DEFAULT_DTYPE) * var_block_scale
                              )

    regularization = _KF_JITTER * (10 if context == "mcmc" else 1)
    init_cov = (init_cov + init_cov.T) / 2.0 + regularization * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    try: jnp.linalg.cholesky(init_cov)
    except Exception as e:
        # print(f"Warning: P0 not PSD after regularization ({context}): {e}. Adding more jitter.") # Tracing
        init_cov = init_cov + regularization * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    return init_cov

def _extract_gamma_matrices_from_params(A_transformed: Any, Sigma_u: Any,
                                         n_stationary: int, var_order: int) -> Optional[List[jnp.ndarray]]:
    """
    Extracts and computes gamma matrices [Gamma_0, ..., Gamma_{p-1}]
    from A_transformed ([Phi_1, ..., Phi_p]) and Sigma_u (VAR innovation covariance).

    This function does NOT resolve parameter names; it expects JAX/Numpy arrays.
    Returns a list of JAX arrays or None if computation fails or inputs are invalid.
    """
    # Check if either of these values are NaNs, or have bad shapes
    if A_transformed is None or Sigma_u is None:
        # print("Info: Skipping gamma extraction due to None inputs.") # Tracing
        return None
    if not (isinstance(A_transformed, (jnp.ndarray, jnp.ndarray)) and A_transformed.shape == (var_order, n_stationary, n_stationary)):
        # print(f"Warning: Skipping gamma extraction due to A_transformed shape mismatch: {A_transformed.shape if hasattr(A_transformed, 'shape') else 'N/A'}. Expected ({var_order}, {n_stationary}, {n_stationary}).") # Tracing
        return None # Invalid shape/type
    if not (isinstance(Sigma_u, (jnp.ndarray, jnp.ndarray)) and Sigma_u.shape == (n_stationary, n_stationary)):
        # print(f"Warning: Skipping gamma extraction due to Sigma_u shape mismatch: {Sigma_u.shape if hasattr(Sigma_u, 'shape') else 'N/A'}. Expected ({n_stationary}, {n_stationary}).") # Tracing
        return None  # Invalid shape/type
    if not jnp.all(jnp.isfinite(A_transformed)) or not jnp.all(jnp.isfinite(Sigma_u)):
         # print("Warning: Skipping gamma extraction due to non-finite A_transformed or Sigma_u.") # Tracing
         return None # Non-finite values

    if make_stationary_var_transformation_jax is None:
        # print("Warning: make_stationary_var_transformation_jax not available in p0_utils.") # Tracing
        return None # Cannot compute gamma without the transform

    try:
        A_list = [A_transformed[lag] for lag in range(var_order)] # Convert stacked A back to list
        # This function is expected to return phi_list (which we don't need here) and gamma_list.
        phi_list_dummy, gamma_list = make_stationary_var_transformation_jax(
            Sigma_u, A_list, n_stationary, var_order
        )
        # Validate the structure and finiteness of the returned gamma list
        if gamma_list is not None and len(gamma_list) == var_order:
            valid_gammas = all(g is not None and hasattr(g, 'shape') and g.shape == (n_stationary, n_stationary) and jnp.all(jnp.isfinite(g))
                               for g in gamma_list)
            if valid_gammas:
                # print(f"Info: Successfully extracted {len(gamma_list)} gamma matrices.") # Tracing
                return gamma_list
            # else: print("Warning: Generated gamma list contains invalid matrices after transformation.") # Tracing
        # else: print(f"Warning: Transformation returned gamma list of wrong length ({len(gamma_list) if gamma_list else 'None'}). Expected {var_order}.") # Tracing

    except Exception as e:
        # Catch any numerical errors or other issues during the transformation
        # print(f"Warning: Error during stationary VAR transformation: {e}") # Tracing
        pass # Return None if transform fails

    return None # Return None if gamma list is invalid or transform failed

def _create_standard_p0(state_dim: int, n_dynamic_trends: int,
                       dynamic_trend_names: List[str],  # New argument
                       core_var_map: Dict[str, Any],  # New argument
                       context: str = "mcmc",
                       trend_P0_scales_override: Optional[Union[float, Dict[str, float]]] = None,  # Renamed and type updated
                       var_P0_var_scale_override: Optional[float] = None) -> jnp.ndarray:
    """Standard initial covariance."""
    default_trend_scale = 1e6 if context == "mcmc" else 1e4
    var_scale = var_P0_var_scale_override if var_P0_var_scale_override is not None else (4.0 if context == "mcmc" else 1.0)
    # print(f"Standard P0 ({context}): default trend scale = {default_trend_scale}, var scale = {var_scale}") # Tracing

    init_cov = jnp.zeros((state_dim, state_dim), dtype=_DEFAULT_DTYPE) # Initialize with zeros

    # Handle trends part
    if n_dynamic_trends > 0:
        trend_scales_diag = jnp.ones(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * default_trend_scale
        if trend_P0_scales_override is not None:
            if isinstance(trend_P0_scales_override, float):
                trend_scales_diag = jnp.ones(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * trend_P0_scales_override
            elif isinstance(trend_P0_scales_override, dict):
                temp_scales = []
                # dynamic_trend_names should list the names of the trend components in order
                for i, trend_name in enumerate(dynamic_trend_names): # Assumes dynamic_trend_names are for the first n_dynamic_trends states
                    scale = trend_P0_scales_override.get(trend_name, default_trend_scale)
                    temp_scales.append(scale)
                if len(temp_scales) == n_dynamic_trends:
                     trend_scales_diag = jnp.array(temp_scales, dtype=_DEFAULT_DTYPE)
                # else: print warning or error if lengths don't match

        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(jnp.diag(trend_scales_diag))

    # Handle VAR part (states after dynamic trends)
    if state_dim > n_dynamic_trends:
        var_states_dim = state_dim - n_dynamic_trends
        init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
            jnp.eye(var_states_dim, dtype=_DEFAULT_DTYPE) * var_scale
        )

    regularization = _KF_JITTER * (10 if context == "mcmc" else 1)
    init_cov = (init_cov + init_cov.T) / 2.0 + regularization * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    try: jnp.linalg.cholesky(init_cov)
    except Exception as e:
        # print(f"Warning: Standard P0 not PSD after regularization ({context}): {e}. Adding more jitter.") # Tracing
        init_cov = init_cov + regularization * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    return init_cov

