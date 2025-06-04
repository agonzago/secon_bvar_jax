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

    # Refactored JAX-friendly gamma_list_is_valid check
    preliminary_check = (
        gamma_list is not None and
        isinstance(gamma_list, list) and
        len(gamma_list) == var_order and
        var_order > 0 and # Ensure gamma_list[0] is safe to access if var_order > 0
        n_stationary > 0 and # Ensure shape check (n_stationary, n_stationary) is meaningful
        gamma_list[0] is not None and
        hasattr(gamma_list[0], 'shape') and
        isinstance(gamma_list[0], jnp.ndarray) and
        gamma_list[0].ndim == 2 and
        gamma_list[0].shape == (n_stationary, n_stationary)
    )

    if preliminary_check:
        # Check structure for all elements
        structural_checks_pass = True
        # If var_order is 0, gamma_list should be empty, caught by preliminary_check (len(gamma_list) == var_order)
        # If var_order is > 0, preliminary_check ensures gamma_list[0] is valid.
        # Loop for g in gamma_list (or gamma_list[1:] if gamma_list[0] fully checked)
        for g_idx, g in enumerate(gamma_list): # Check all, including gamma_list[0] for consistency
            if not (g is not None and
                    hasattr(g, 'shape') and
                    isinstance(g, jnp.ndarray) and
                    g.ndim == 2 and
                    g.shape == (n_stationary, n_stationary)):
                structural_checks_pass = False
                break

        if structural_checks_pass:
            # Perform JAX-based finiteness check on all elements together
            try:
                # Attempt to stack. If shapes were inconsistent (missed by checks), this might fail.
                stacked_gammas = jnp.stack(gamma_list) # Shape (var_order, n_stationary, n_stationary)
                all_finite = jnp.all(jnp.isfinite(stacked_gammas))
                gamma_list_is_valid = all_finite # JAX bool
            except Exception:
                gamma_list_is_valid = jnp.array(False) # JAX bool false
        else:
            gamma_list_is_valid = jnp.array(False) # JAX bool false
    else:
        # If var_order is 0 (so gamma_list is empty and preliminary_check is False if n_stationary >0, or var_order > 0 was False),
        # or n_stationary is 0, then gamma_list_is_valid should be False for the purpose of building a gamma-based block.
        # However, if var_order == 0 or n_stationary == 0, the condition `n_stationary > 0 and var_order > 0 and gamma_list_is_valid` later will be false anyway.
        # For clarity, set it to False explicitly if preliminary checks fail for relevant cases.
        if var_order > 0 and n_stationary > 0: # Only if we expected a valid gamma list
             gamma_list_is_valid = jnp.array(False)
        else: # Otherwise, it's vacuously not a "valid gamma list for processing" but not an error state for this flag
             gamma_list_is_valid = jnp.array(False) # Treat as not valid for building VAR block

    # The condition for building the VAR block using gammas:
    # Original Python bool `gamma_list_is_valid` is now a JAX array.
    # Python `if` statements with JAX arrays are problematic if the condition is traced.
    # Assuming this part of the code is setting up structures and values *before* any JAX transformation (like jit or grad),
    # using the JAX array in a Python `if` might be okay if it evaluates to a concrete True/False at this stage.
    # However, to be safe, especially if this function could be jitted later,
    # one might need lax.cond here, or ensure this logic runs in Python before JAX tracing.
    # For now, let's assume it's okay, as error message pointed to the check itself.
    # If `gamma_list_is_valid` is a JAX array, `and gamma_list_is_valid` will be a JAX expression.

    # We need to ensure the `if` condition below can handle a JAX boolean.
    # A common pattern is to evaluate it to a Python bool if this code runs during model setup (not traced execution).
    # If it's traced, then lax.cond is needed for the whole block.
    # Given the error, it's likely this `if` is being traced.
    # However, the request is to make `gamma_list_is_valid` JAX-friendly, not necessarily to rewrite the whole `if/else` with `lax.cond` yet.
    # Let's assume the Python `if` will be used with a JAX bool that JAX can handle in `jit` by staging out.

    # Define helper functions for jax.lax.cond
    def _build_var_block_from_gamma_body(operands_tuple):
        # n_stationary, var_order, n_dynamic_trends are from outer scope
        (init_cov_in, gamma_list_in, gamma_scaling_in,
         _,  # var_fallback_scale ignored
         state_dim_in) = operands_tuple # Unpack 5 items

        var_start_idx_val = n_dynamic_trends # Calculate var_start_idx from outer scope n_dynamic_trends
        var_state_total_dim_calc = n_stationary * var_order # Use n_stationary and var_order from outer scope

        var_block_cov = jnp.zeros((var_state_total_dim_calc, var_state_total_dim_calc), dtype=_DEFAULT_DTYPE)
        for r_block_idx in range(var_order):
            for c_block_idx in range(var_order):
                lag_d = abs(r_block_idx - c_block_idx)
                blk_unscaled = gamma_list_in[lag_d]
                curr_blk = blk_unscaled * gamma_scaling_in
                if r_block_idx > c_block_idx:
                    curr_blk = curr_blk.T

                row_start_slice = r_block_idx * n_stationary
                row_end_slice = (r_block_idx + 1) * n_stationary
                col_start_slice = c_block_idx * n_stationary
                col_end_slice = (c_block_idx + 1) * n_stationary

                var_block_cov = var_block_cov.at[row_start_slice:row_end_slice, col_start_slice:col_end_slice].set(curr_blk)

        return init_cov_in.at[var_start_idx_val:var_start_idx_val + var_state_total_dim_calc,
                              var_start_idx_val:var_start_idx_val + var_state_total_dim_calc].set(var_block_cov)

    def _build_var_block_fallback_body(operands_tuple):
        # n_stationary, var_order, n_dynamic_trends are from outer scope
        (init_cov_in, _, _,  # gamma_list_in, gamma_scaling_in ignored
         var_fallback_scale_in, state_dim_in) = operands_tuple # Unpack 5 items

        var_start_idx_val = n_dynamic_trends # Calculate var_start_idx from outer scope n_dynamic_trends
        var_state_total_dim_calc = n_stationary * var_order # Use n_stationary and var_order from outer scope

        # Original: elif var_state_total_dim > 0:
        # This means if not (n_stationary > 0 and var_order > 0 and gamma_list_is_valid)
        # AND var_state_total_dim > 0.
        # The lax.cond structure implies this function is called if the main_condition is false.
        # So, if main_condition is false, we then check if var_state_total_dim_in > 0.

        # If var_state_total_dim_in is 0, this branch should effectively do nothing to the VAR part.
        # The P0 utils are designed such that if n_stationary or var_order is 0, var_state_total_dim_in will be 0.
        # So, the condition for applying fallback is simply var_state_total_dim_in > 0.

        # Note: var_fallback_scale_in is already resolved var_P0_var_scale_override or default.

        # This function is only called if gamma_list_is_valid is False.
        # We still need to apply fallback if var_state_total_dim > 0.
        def apply_fallback(init_c):
            return init_c.at[var_start_idx_val:var_start_idx_val + var_state_total_dim_calc,
                             var_start_idx_val:var_start_idx_val + var_state_total_dim_calc].set(
                                 jnp.eye(var_state_total_dim_calc, dtype=_DEFAULT_DTYPE) * var_fallback_scale_in)

        def do_nothing(init_c):
            return init_c

        # Only apply fallback if there are VAR states to initialize
        return jax.lax.cond(var_state_total_dim_calc > 0, # Use calculated dim
                            apply_fallback,
                            do_nothing,
                            init_cov_in)

    # gamma_list_is_valid is already a JAX boolean array from the previous refactoring.
    # It correctly evaluates to false if n_stationary or var_order is not positive.
    main_condition = gamma_list_is_valid

    # Operands for the true branch (_build_var_block_from_gamma_body)
    # (init_cov_in, gamma_list_in, gamma_scaling_in, _, state_dim_in)
    # Operands for the false branch (_build_var_block_fallback_body)
    # (init_cov_in, _, _, var_fallback_scale_in, state_dim_in)
    # The placeholder `_` means that element from the combined operands tuple won't be used by that branch.

    # Operands tuple now has 5 elements
    operands = (init_cov, gamma_list, gamma_scaling,
                var_fallback_scale, state_dim)

    init_cov = jax.lax.cond(
        main_condition,
        _build_var_block_from_gamma_body, # True branch
        _build_var_block_fallback_body,   # False branch
        operands
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

