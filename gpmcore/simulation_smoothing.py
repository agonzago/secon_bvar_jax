import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpyro
from functools import partial
from typing import Tuple, Optional, Dict, Any
import numpy as np
import time

import xarray as xr
import arviz as az
# Import required modules
try:
    from .stationary_prior_jax_simplified import _JITTER
except ImportError:
    print("Warning: Could not import _JITTER from stationary_prior_jax_simplified")
    _JITTER = 1e-8

try:
    from .Kalman_filter_jax import KalmanFilter, _KF_JITTER
except ImportError:
    print("Warning: Could not import KalmanFilter")
    _KF_JITTER = 1e-8
    class KalmanFilter:
        def __init__(self, *args, **kwargs): 
            raise NotImplementedError("KalmanFilter import failed")

# Configure constants
_DEFAULT_DTYPE = jnp.float64


def jarocinski_corrected_simulation_smoother(y: jnp.ndarray, F: jnp.ndarray, R: jnp.ndarray, 
                                           C: jnp.ndarray, H: jnp.ndarray,
                                           init_mean: jnp.ndarray, init_cov: jnp.ndarray,
                                           key: jnp.ndarray) -> jnp.ndarray:
    """
    Correctly implement Jarocinski's (2015) corrected Durbin & Koopman simulation smoother.
    
    The key insight from Jarocinski (2015) is that we need to reset the initial mean to zero
    in EITHER Step 1 OR Step 2, but use the same covariance structure throughout.
    
    We choose to do the correction in Step 1 (simulation step) as suggested by most implementations.
    
    Algorithm 2a from Jarocinski (2015):
    1. Draw α⁺ and y⁺ with ZERO initial mean but original covariance
    2. Construct y* = y - y⁺  
    3. Compute α̂* = E(α|y*) with original model (non-zero mean)
    4. Return α̃ = α̂* + α⁺
    """
    T, n_obs = y.shape
    state_dim = F.shape[0]
    
    if T == 0:
        return jnp.empty((0, state_dim), dtype=_DEFAULT_DTYPE)
    
    # Step 1: Generate α⁺ and y⁺ with ZERO initial mean (Jarocinski correction)
    key, step1_key = random.split(key)
    zero_init_mean = jnp.zeros_like(init_mean)
    
    alpha_plus, y_plus = simulate_forward_with_zero_mean(
        F, R, C, H, zero_init_mean, init_cov, T, step1_key
    )
    
    # Step 2: Construct artificial series y* = y - y⁺
    y_star = y - y_plus
    
    # Step 3: Compute α̂* = E(α|y*) using ORIGINAL model (with non-zero init_mean)
    # This is where we differ from our previous implementation
    alpha_hat_star = compute_smoothed_expectation(
        y_star, F, R, C, H, init_mean, init_cov  # Use ORIGINAL init_mean here
    )
    
    # Step 4: Return α̃ = α̂* + α⁺
    alpha_tilde = alpha_hat_star + alpha_plus
    
    return alpha_tilde


def simulate_forward_with_zero_mean(F: jnp.ndarray, R: jnp.ndarray, C: jnp.ndarray, H: jnp.ndarray,
                                   zero_init_mean: jnp.ndarray, init_cov: jnp.ndarray, 
                                   T: int, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward simulation with zero initial mean but preserving covariance structure.
    """
    state_dim = F.shape[0]
    n_obs = C.shape[0]
    n_shocks = R.shape[1]
    
    # Initialize storage
    alpha_plus = jnp.zeros((T, state_dim), dtype=_DEFAULT_DTYPE)
    y_plus = jnp.zeros((T, n_obs), dtype=_DEFAULT_DTYPE)
    
    # Sample initial state with zero mean but original covariance
    key, init_key = random.split(key)
    init_cov_reg = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    try:
        alpha_0 = random.multivariate_normal(init_key, zero_init_mean, init_cov_reg)
        if not jnp.all(jnp.isfinite(alpha_0)):
            alpha_0 = zero_init_mean
    except:
        alpha_0 = zero_init_mean
    
    current_state = alpha_0
    
    # Forward simulation
    for t in range(T):
        # Store current state
        alpha_plus = alpha_plus.at[t].set(current_state)
        
        # Generate observation
        key, obs_key = random.split(key)
        obs_mean = C @ current_state
        
        # Sample observation with noise
        H_reg = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        try:
            y_t = random.multivariate_normal(obs_key, obs_mean, H_reg)
            if not jnp.all(jnp.isfinite(y_t)):
                y_t = obs_mean
        except:
            y_t = obs_mean
        
        y_plus = y_plus.at[t].set(y_t)
        
        # Evolve state for next period
        if t < T - 1:
            key, state_key = random.split(key)
            
            # Generate state innovations
            try:
                eta_t = random.normal(state_key, shape=(n_shocks,), dtype=_DEFAULT_DTYPE)
                innovation = R @ eta_t
                if not jnp.all(jnp.isfinite(innovation)):
                    innovation = jnp.zeros_like(innovation)
            except:
                innovation = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
            
            current_state = F @ current_state + innovation
    
    return alpha_plus, y_plus


def compute_smoothed_expectation(y_star: jnp.ndarray, F: jnp.ndarray, R: jnp.ndarray,
                               C: jnp.ndarray, H: jnp.ndarray,
                               init_mean: jnp.ndarray, init_cov: jnp.ndarray) -> jnp.ndarray:
    """
    Compute E(α|y*) using Kalman filter and smoother with ORIGINAL initial conditions.
    This is the key difference from our previous implementation.
    """
    T, n_obs = y_star.shape
    state_dim = F.shape[0]
    
    # Use ORIGINAL initial mean (not zero) - this is the key correction
    kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
    
    # Set up observation info
    valid_obs_idx = jnp.arange(n_obs, dtype=int)
    I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
    
    try:
        # Run filter
        filter_results = kf.filter(
            y_star, 
            static_valid_obs_idx=valid_obs_idx,
            static_n_obs_actual=n_obs,
            static_C_obs=C, 
            static_H_obs=H, 
            static_I_obs=I_obs
        )
        
        # Run smoother
        smoothed_means, smoothed_covs = kf.smooth(
            y_star,
            filter_results=filter_results,
            static_valid_obs_idx=valid_obs_idx,
            static_n_obs_actual=n_obs,
            static_C_obs_for_filter=C,
            static_H_obs_for_filter=H,
            static_I_obs_for_filter=I_obs
        )
        
        # Validate results
        if not jnp.all(jnp.isfinite(smoothed_means)):
            print("Warning: Kalman smoother produced non-finite results, returning zeros")
            return jnp.zeros((T, state_dim), dtype=_DEFAULT_DTYPE)
        
        return smoothed_means
        
    except Exception as e:
        print(f"Error in smoothed expectation computation: {e}")
        return jnp.zeros((T, state_dim), dtype=_DEFAULT_DTYPE)


def extract_gpm_trends_and_components(mcmc, y: jnp.ndarray, gpm_model, ss_builder,
                                num_draws: int = 100, 
                                rng_key: jnp.ndarray = random.PRNGKey(42)):
    """
    Extract components using the FIXED Jarocinski-corrected simulation smoother.
    """
    samples = mcmc.get_samples()
    T, n_obs = y.shape
    n_trends = ss_builder.n_trends
    n_stationary = ss_builder.n_stationary
    state_dim = ss_builder.state_dim

    # Storage for draws
    trend_draws = []
    stationary_draws = []

    # Check required sites
    required_sites = _identify_required_sites(samples, gpm_model)
    
    missing_sites = [site for site in required_sites if site not in samples]
    if missing_sites:
        print(f"Error: Required sites {missing_sites} not found in MCMC samples. Cannot proceed with extraction.")
        return jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE), jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)

    n_posterior = len(samples[required_sites[0]])
    num_draws = min(num_draws, n_posterior)
    
    if num_draws > 0:
        draw_indices_float = np.linspace(0, n_posterior - 1, num_draws)
        draw_indices = np.round(draw_indices_float).astype(int)
    else:
        draw_indices = np.array([], dtype=int)

    # Create more reasonable initial covariance (less extreme)
    init_cov_fixed = _create_reasonable_initial_covariance(state_dim, n_trends)

    print(f"Processing {len(draw_indices)} posterior draws with FIXED Jarocinski-corrected simulation smoother...")

    for i, idx in enumerate(draw_indices):
        if (i + 1) % 10 == 0:
            print(f"Processing draw {i+1}/{len(draw_indices)}")
        
        try:
            # Extract parameters for this draw
            params_draw = _extract_gpm_parameters(samples, idx, gpm_model)
            
            # Build state space matrices
            F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_matrices(params_draw)
            
            # Check for NaNs
            if jnp.any(jnp.isnan(F_draw)) or jnp.any(jnp.isnan(Q_draw)) or jnp.any(jnp.isnan(C_draw)) or jnp.any(jnp.isnan(H_draw)):
                print(f"Warning: State space matrices contain NaNs for draw {idx}. Skipping.")
                continue

            # Create R matrix from Q
            try:
                R_draw = jnp.linalg.cholesky(Q_draw + _JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE))
            except:
                R_draw = jnp.diag(jnp.sqrt(jnp.diag(Q_draw) + _JITTER))

            # Extract initial state mean
            init_mean_draw = _extract_initial_mean(samples, idx, state_dim)

            # Run FIXED Jarocinski-corrected simulation smoother
            rng_key, sim_key = random.split(rng_key)

            try:
                states_draw = jarocinski_corrected_simulation_smoother(
                    y, F_draw, R_draw, C_draw, H_draw,
                    init_mean_draw, init_cov_fixed, sim_key
                )
            except Exception as sim_error:
                print(f"Fixed simulation smoother failed for draw {idx}: {sim_error}")
                continue

            # Validate results
            if jnp.any(jnp.isnan(states_draw)) or jnp.any(jnp.isinf(states_draw)):
                print(f"Warning: Fixed simulation smoother produced NaNs/Infs for draw {idx}. Skipping.")
                continue

            if states_draw.shape != (T, state_dim):
                print(f"Warning: Unexpected shape {states_draw.shape} for draw {idx}. Expected ({T}, {state_dim}). Skipping.")
                continue

            # Extract components
            trends = states_draw[:, :n_trends]
            stationary = states_draw[:, n_trends:n_trends + n_stationary]

            trend_draws.append(trends)
            stationary_draws.append(stationary)

        except Exception as e:
            print(f"Error processing draw {idx}: {e}")
            continue

    # Stack results
    if len(trend_draws) > 0:
        trend_draws = jnp.stack(trend_draws)
        stationary_draws = jnp.stack(stationary_draws)
        print(f"Successfully extracted {len(trend_draws)} draws using FIXED Jarocinski-corrected simulation smoother")
    else:
        print("No draws were successfully extracted")
        trend_draws = jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)

    return trend_draws, stationary_draws


def _create_reasonable_initial_covariance(state_dim: int, n_trends: int) -> jnp.ndarray:
    """
    Create more reasonable initial state covariance matrix (less extreme than before).
    """
    # Use more moderate values instead of extreme 1e6/1e-6
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 10.0  # Moderate diffuse prior for trends
    
    # Moderate prior for VAR states
    if state_dim > n_trends:
        init_cov = init_cov.at[n_trends:, n_trends:].set(
            jnp.eye(state_dim - n_trends, dtype=_DEFAULT_DTYPE) * 1.0  # Less tight
        )
    
    # Ensure positive definite
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


# Helper functions (reuse existing ones)
def _identify_required_sites(samples: Dict, gpm_model) -> list:
    """Identify required parameter sites based on GPM model structure"""
    required = []
    
    # Add trend shock parameters
    for shock in gpm_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            required.append(param_name)
    
    # Add stationary shock parameters  
    for shock in gpm_model.stationary_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            required.append(param_name)
    
    # Add VAR parameters if present
    if gpm_model.var_prior_setup:
        if "A_transformed" in samples:
            required.append("A_transformed")
        elif "A_raw" in samples:
            required.append("A_raw")
        
        if "Omega_u_chol" in samples:
            required.append("Omega_u_chol")
    
    # Add structural parameters
    for param in gpm_model.parameters:
        if param in samples:
            required.append(param)
    
    # Add initial conditions
    if "init_mean_full" in samples:
        required.append("init_mean_full")
    
    return required


def _extract_gpm_parameters(samples: Dict, idx: int, gpm_model):
    """Extract parameters for a specific draw from GPM model"""
    from .gpm_bvar_trends import EnhancedBVARParams
    
    # Extract trend covariance
    trend_sigmas = []
    for shock in gpm_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            trend_sigmas.append(samples[param_name][idx])
        else:
            trend_sigmas.append(0.1)  # Default fallback
    
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas) ** 2)
    
    # Extract VAR parameters
    if gpm_model.var_prior_setup and gpm_model.stationary_variables:
        # Get VAR coefficient matrices
        if "A_transformed" in samples:
            A_matrices = samples["A_transformed"][idx]
        elif "A_raw" in samples:
            A_matrices = samples["A_raw"][idx]
        else:
            n_vars = len(gpm_model.stationary_variables)
            n_lags = gpm_model.var_prior_setup.var_order
            A_matrices = jnp.zeros((n_lags, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        
        # Get stationary covariance
        if "Omega_u_chol" in samples:
            stat_sigmas = []
            for shock in gpm_model.stationary_shocks:
                param_name = f"sigma_{shock}"
                if param_name in samples:
                    stat_sigmas.append(samples[param_name][idx])
                else:
                    stat_sigmas.append(0.1)
            
            sigma_u = jnp.array(stat_sigmas)
            Omega_u_chol = samples["Omega_u_chol"][idx]
            Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
        else:
            n_vars = len(gpm_model.stationary_variables)
            Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) * 0.1
    else:
        A_matrices = jnp.zeros((1, 1, 1), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(1, dtype=_DEFAULT_DTYPE)
    
    # Extract structural parameters
    structural_params = {}
    for param in gpm_model.parameters:
        if param in samples:
            structural_params[param] = samples[param][idx]
    
    return EnhancedBVARParams(
        A=A_matrices,
        Sigma_u=Sigma_u,
        Sigma_eta=Sigma_eta,
        structural_params=structural_params,
        Sigma_eps=None
    )


def _extract_initial_mean(samples: Dict, idx: int, state_dim: int) -> jnp.ndarray:
    """Extract initial state mean for a specific draw"""
    if "init_mean_full" in samples:
        init_mean = samples["init_mean_full"][idx]
        # Ensure it's finite
        init_mean = jnp.where(jnp.isfinite(init_mean), init_mean, jnp.zeros_like(init_mean))
    else:
        init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_mean


def compute_hdi_with_percentiles(draws: jnp.ndarray, hdi_prob: float = 0.94):
    """Compute credible intervals using percentiles."""
    if draws.shape[0] < 2:
        print("Warning: Not enough draws to compute credible interval. Need at least 2.")
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}

    draws_np = np.asarray(draws)
    lower_percentile = (1 - hdi_prob) / 2 * 100
    upper_percentile = (1 + hdi_prob) / 2 * 100
    percentiles = np.array([lower_percentile, upper_percentile])

    try:
        hdi_bounds_np = np.percentile(draws_np, percentiles, axis=0)
        hdi_low_np = hdi_bounds_np[0, ...]  
        hdi_high_np = hdi_bounds_np[1, ...]
        return {'low': jnp.asarray(hdi_low_np), 'high': jnp.asarray(hdi_high_np)}
    except Exception as e:
        print(f"Warning: Percentile computation failed with error: {e}. Returning NaNs.")
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}
    

# Revised _extract_hdi_from_arviz_output

def _extract_hdi_from_arviz_output(hdi_output: Any) -> Dict[str, np.ndarray]:
    """
    Safely extracts lower and higher bounds from arviz.hdi output,
    handling both DataArray and raw ndarray returns.

    Args:
        hdi_output: The object returned by arviz.hdi.

    Returns:
        A dictionary {'low': np.ndarray, 'high': np.ndarray}. The arrays
        will contain NaNs if extraction fails.
    """
    try:
        if isinstance(hdi_output, xr.DataArray):
            # Expected case for multi-dimensional input
            if 'hdi' in hdi_output.dims and hdi_output.shape[0] == 2:
                 low = hdi_output.loc['lower', ...].values
                 high = hdi_output.loc['higher', ...].values
            elif hdi_output.shape[0] == 2: # Maybe hdi dim isn't named
                 low = hdi_output[0, ...].values
                 high = hdi_output[1, ...].values
            else:
                 raise ValueError(f"Unexpected DataArray shape from arviz.hdi: {hdi_output.shape}")

        elif isinstance(hdi_output, np.ndarray):
            # Fallback case if arviz.hdi returns raw ndarray
            if hdi_output.shape[0] == 2:
                 low = hdi_output[0, ...]
                 high = hdi_output[1, ...]
            else:
                 raise ValueError(f"Unexpected ndarray shape from arviz.hdi: {hdi_output.shape}")

        else:
            raise TypeError(f"Unexpected type returned by arviz.hdi: {type(hdi_output)}")

        # Final check for NaNs in results
        if np.any(np.isnan(low)) or np.any(np.isnan(high)):
             print(f"Warning: Computed HDI contains NaN values.")

        return {'low': low, 'high': high}

    except Exception as e:
        print(f"Error extracting HDI bounds from arviz output: {e}.")
        # Return NaNs with the shape of the *expected* output arrays, which is hdi_output.shape[1:]
        nan_shape = hdi_output.shape[1:] if hdi_output is not None and hasattr(hdi_output, 'shape') and hdi_output.shape[0] == 2 else (1, 1) # Fallback shape if shape is weird
        return {'low': np.full(nan_shape, np.nan, dtype=_DEFAULT_DTYPE),
                'high': np.full(nan_shape, np.nan, dtype=_DEFAULT_DTYPE)}    
    


def _compute_and_format_hdi_az(draws_np: np.ndarray, hdi_prob: float = 0.9) -> Dict[str, np.ndarray]:
    """
    Computes HDI using ArviZ on multi-dimensional draws by reshaping,
    handles potential shape inconsistencies from arviz.hdi output,
    and formats the output into a {'low': ..., 'high': ...} dictionary
    with the original time/variable dimensions.

    Args:
        draws_np: NumPy array of draws, expected shape (num_draws, T, n_vars).
        hdi_prob: The HDI probability.

    Returns:
        A dictionary {'low': np.ndarray, 'high': np.ndarray}. The arrays
        will contain NaNs if computation, reshaping, or extraction fails.
    """
    if draws_np.ndim < 2:
         print(f"Warning: Draws array has unexpected shape {draws_np.shape}. Need at least 2 dimensions (draws, ...).")
         # Return NaNs with a minimal shape
         return {'low': np.array(np.nan, dtype=_DEFAULT_DTYPE),
                 'high': np.array(np.nan, dtype=_DEFAULT_DTYPE)}

    num_draws = draws_np.shape[0]
    original_shape_after_draws = draws_np.shape[1:] # This will be (T, n_vars) or similar

    if num_draws < 2:
        print("Warning: Not enough draws to compute HDI (need at least 2).")
        # Return NaNs with the correct final shape (T, n_vars)
        return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE),
                'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}

    flat_size = int(np.prod(original_shape_after_draws))
    # Expected shape from az.hdi is (2, flat_size) or possibly (flat_size, 2)
    expected_shape_bounds_first = (2, flat_size)
    expected_shape_params_first = (flat_size, 2)

    final_hdi_shape = (2,) + original_shape_after_draws # Desired shape (2, T, n_vars)


    try:
        # Reshape input to 2D: (num_draws, T * n_vars)
        draws_reshaped = draws_np.reshape(num_draws, flat_size)

        # Compute HDI using arviz.hdi on the 2D reshaped array
        hdi_output_reshaped = az.hdi(draws_reshaped, hdi_prob=hdi_prob)

        hdi_for_reshape = None

        # Check the actual shape and handle dimension order
        if hdi_output_reshaped.shape == expected_shape_bounds_first:
             hdi_for_reshape = hdi_output_reshaped
        elif hdi_output_reshaped.shape == expected_shape_params_first:
             # Transpose if parameters are in the first dimension
             hdi_for_reshape = hdi_output_reshaped.T
        else:
             raise ValueError(f"ArviZ HDI returned unexpected shape: {hdi_output_reshaped.shape}. Expected {expected_shape_bounds_first} or {expected_shape_params_first}.")


        # Reshape the corrected 2D HDI results back to (2, T, n_vars)
        hdi_full_shape = hdi_for_reshape.reshape(final_hdi_shape)

        # Extract lower and higher bounds from the first dimension
        low = hdi_full_shape[0, ...]
        high = hdi_full_shape[1, ...]

        # Final check for NaNs in results
        if np.any(np.isnan(low)) or np.any(np.isnan(high)):
             print(f"Warning: Computed HDI contains NaN values for hdi_prob={hdi_prob}.")

        return {'low': low, 'high': high}

    except Exception as e:
        print(f"Error during ArviZ HDI computation and formatting: {e}. Returning NaNs.")
        # Return NaNs with the correct final shape (T, n_vars)
        return {'low': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE),
                'high': np.full(original_shape_after_draws, np.nan, dtype=_DEFAULT_DTYPE)}
    
# Ensure the module is importable