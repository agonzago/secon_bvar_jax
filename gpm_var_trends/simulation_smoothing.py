import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpyro
from functools import partial
from typing import Tuple, Optional, Dict, Any
import numpy as np
import time

# Import required modules
try:
    from stationary_prior_jax_simplified import _JITTER
except ImportError:
    print("Warning: Could not import _JITTER from stationary_prior_jax_simplified")
    _JITTER = 1e-8

try:
    from Kalman_filter_jax import KalmanFilter, _KF_JITTER
except ImportError:
    print("Warning: Could not import KalmanFilter")
    _KF_JITTER = 1e-8
    class KalmanFilter:
        def __init__(self, *args, **kwargs): 
            raise NotImplementedError("KalmanFilter import failed")

# Configure constants
_DEFAULT_DTYPE = jnp.float64


def durbin_koopman_smoother_step(carry, t, F, a, r, m, P):
    """
    Single backward step of the Durbin and Koopman simulation smoother.
    """
    sampled_state_t_plus_1, key = carry
    state_dim = F.shape[0]

    # Extract quantities for time t and t+1 from filter results
    m_t, P_t = m[t], P[t]  # Filtered at t
    a_t_plus_1, r_t_plus_1 = a[t + 1], r[t + 1]  # Predicted at t+1

    # Compute the smoother gain: A_t = P_t @ F.T @ jnp.linalg.inv(r_{t+1})
    r_t_plus_1_reg = (r_t_plus_1 + r_t_plus_1.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    try:
        # Compute A_t = P_t @ F.T @ inv(r_{t+1})
        P_t_F_T = P_t @ F.T
        A_t = jax.scipy.linalg.solve(r_t_plus_1_reg, P_t_F_T.T, assume_a='pos').T
        
        # Conditional mean: m_t^* = m_t + A_t @ (x_{t+1} - a_{t+1})
        m_t_star = m_t + A_t @ (sampled_state_t_plus_1 - a_t_plus_1)
        
        # Conditional covariance: P_t^* = P_t - A_t @ r_{t+1} @ A_t.T
        P_t_star = P_t - A_t @ r_t_plus_1 @ A_t.T
        
        # Ensure symmetry
        P_t_star = (P_t_star + P_t_star.T) / 2.0
        
        solve_ok = jnp.all(jnp.isfinite(m_t_star)) & jnp.all(jnp.isfinite(P_t_star))
        
        # Fallback if solve fails
        m_t_star = jnp.where(solve_ok, m_t_star, m_t)
        P_t_star = jnp.where(solve_ok, P_t_star, P_t)
        
    except Exception:
        m_t_star = m_t
        P_t_star = P_t

    # Ensure P_t_star is positive definite for sampling
    P_t_star_reg = (P_t_star + P_t_star.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    # Sample from the conditional distribution
    key, subkey = random.split(key)
    try:
        sampled_state_t = random.multivariate_normal(subkey, m_t_star, P_t_star_reg)
        sample_ok = jnp.all(jnp.isfinite(sampled_state_t))
        sampled_state_t = jnp.where(sample_ok, sampled_state_t, m_t_star)
    except Exception:
        sampled_state_t = m_t_star

    new_carry = (sampled_state_t, key)
    return new_carry, sampled_state_t


def durbin_koopman_simulation_smoother_from_results(
    filter_a: jnp.ndarray, filter_r: jnp.ndarray,
    filter_m: jnp.ndarray, filter_P: jnp.ndarray,
    smooth_mu: jnp.ndarray, smooth_V: jnp.ndarray,
    F: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Durbin and Koopman simulation smoother using results from KF and KFS.
    """
    T = filter_a.shape[0]
    state_dim = F.shape[0]

    # Handle empty time series
    if T == 0:
        return jnp.empty((0, state_dim), dtype=_DEFAULT_DTYPE)

    # Handle single time step
    if T == 1:
        key, subkey = random.split(key)
        smooth_V_0_reg = (smooth_V[0] + smooth_V[0].T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
        try:
            sampled_state = random.multivariate_normal(subkey, smooth_mu[0], smooth_V_0_reg)
            sample_ok = jnp.all(jnp.isfinite(sampled_state))
            sampled_state = jnp.where(sample_ok, sampled_state, smooth_mu[0])
        except Exception:
            sampled_state = smooth_mu[0]
        return sampled_state[None, :]

    # Sample the last state x_{T-1} from N(mu_{T-1}, V_{T-1})
    key, subkey = random.split(key)
    smooth_V_T_minus_1_reg = (smooth_V[T-1] + smooth_V[T-1].T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    try:
        sampled_state_T_minus_1 = random.multivariate_normal(
            subkey, smooth_mu[T-1], smooth_V_T_minus_1_reg
        )
        sample_ok_final = jnp.all(jnp.isfinite(sampled_state_T_minus_1))
        sampled_state_T_minus_1 = jnp.where(sample_ok_final, sampled_state_T_minus_1, smooth_mu[T-1])
    except Exception:
        sampled_state_T_minus_1 = smooth_mu[T-1]

    # Backward simulation from t = T-2 down to 0
    backward_times = jnp.arange(T - 2, -1, -1)

    # Partialize the step function
    step_fn = partial(
        durbin_koopman_smoother_step,
        F=F, a=filter_a, r=filter_r, m=filter_m, P=filter_P
    )

    # Initial carry: (sampled state at T-1, key)
    key, scan_key = random.split(key)
    init_carry_dk = (sampled_state_T_minus_1, scan_key)

    # Run the backward simulation scan
    final_carry_dk, sampled_states_rev_dk = lax.scan(
        step_fn,
        init_carry_dk,
        backward_times
    )

    # Concatenate: reverse the draws for 0..T-2 and append the draw for T-1
    sampled_states_dk = jnp.concatenate([
        sampled_states_rev_dk[::-1],  # Times 0 to T-2 (reversed)
        sampled_state_T_minus_1[None, :]  # Time T-1
    ])

    return sampled_states_dk


def extract_gpm_trends_and_components(mcmc, y: jnp.ndarray, gmp_model, ss_builder,
                                    num_draws: int = 100, rng_key: jnp.ndarray = random.PRNGKey(42)):
    """
    Extract trend and stationary components using Durbin & Koopman simulation smoother
    for GPM-based models.
    """
    samples = mcmc.get_samples()
    T, n_obs = y.shape
    n_trends = ss_builder.n_trends
    n_stationary = ss_builder.n_stationary
    state_dim = ss_builder.state_dim

    # Storage for draws
    trend_draws = []
    stationary_draws = []

    # Check required sites (this will depend on your GPM model structure)
    # The exact parameter names will depend on how your GPM model names them
    required_sites = _identify_required_sites(samples, gmp_model)
    
    missing_sites = [site for site in required_sites if site not in samples]
    if missing_sites:
        print(f"Error: Required sites {missing_sites} not found in MCMC samples. Cannot proceed with extraction.")
        return jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE), jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)

    n_posterior = len(samples[required_sites[0]])  # Use first required site to get length
    num_draws = min(num_draws, n_posterior)
    
    if num_draws > 0:
        draw_indices_float = np.linspace(0, n_posterior - 1, num_draws)
        draw_indices = np.round(draw_indices_float).astype(int)
    else:
        draw_indices = np.array([], dtype=int)

    # Static observation info
    valid_obs_idx = jnp.arange(n_obs, dtype=int)
    I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)

    # Fixed initial state covariance
    init_cov_fixed = _create_gmp_initial_covariance(state_dim, n_trends)

    print(f"Processing {len(draw_indices)} posterior draws...")

    for i, idx in enumerate(draw_indices):
        if (i + 1) % 10 == 0:
            print(f"Processing draw {i+1}/{len(draw_indices)}")
        
        try:
            # Extract parameters for this draw - this will be GPM-specific
            params_draw = _extract_gmp_parameters(samples, idx, gmp_model)
            
            # Build state space matrices using the GPM builder
            F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_matrices(params_draw)
            
            # Check for NaNs
            state_space_nan_draw = (jnp.any(jnp.isnan(F_draw)) | jnp.any(jnp.isnan(Q_draw)) | 
                                   jnp.any(jnp.isnan(C_draw)) | jnp.any(jnp.isnan(H_draw)))

            if state_space_nan_draw:
                print(f"Warning: State space matrices contain NaNs for draw {idx}. Skipping.")
                continue

            # Create R matrix from Q (assuming Q = R @ R.T)
            try:
                R_draw = jnp.linalg.cholesky(Q_draw + _JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE))
            except:
                R_draw = jnp.diag(jnp.sqrt(jnp.diag(Q_draw) + _JITTER))

            # Initial state mean
            init_mean_draw = _extract_initial_mean(samples, idx, state_dim)

            # Create Kalman Filter instance
            kf_draw = KalmanFilter(T=F_draw, R=R_draw, C=C_draw, H=H_draw, 
                                  init_x=init_mean_draw, init_P=init_cov_fixed)

            # Run Kalman Filter - returns a dictionary
            try:
                filter_results_dict = kf_draw.filter(
                    y, static_valid_obs_idx=valid_obs_idx,
                    static_n_obs_actual=n_obs,
                    static_C_obs=C_draw, static_H_obs=H_draw, static_I_obs=I_obs
                )
                
                # Extract the arrays from the dictionary
                filter_a_draw = filter_results_dict['x_pred']
                filter_r_draw = filter_results_dict['P_pred']
                filter_m_draw = filter_results_dict['x_filt']
                filter_P_draw = filter_results_dict['P_filt']
                
            except Exception as filter_error:
                print(f"Filter failed for draw {idx}: {filter_error}")
                continue

            # Run Kalman Smoother
            try:
                smoothed_means_draw, smoothed_covs_draw = kf_draw.smooth(
                    y, filter_results=filter_results_dict,
                    static_valid_obs_idx=valid_obs_idx,
                    static_n_obs_actual=n_obs,
                    static_C_obs_for_filter=C_draw, 
                    static_H_obs_for_filter=H_draw, 
                    static_I_obs_for_filter=I_obs
                )
                
            except Exception as smooth_error:
                print(f"Smoother failed for draw {idx}: {smooth_error}")
                continue

            # Check results for validity
            filter_results_ok = (jnp.all(jnp.isfinite(filter_a_draw)) & jnp.all(jnp.isfinite(filter_r_draw)) & 
                               jnp.all(jnp.isfinite(filter_m_draw)) & jnp.all(jnp.isfinite(filter_P_draw)))

            smooth_results_ok = (jnp.all(jnp.isfinite(smoothed_means_draw)) & 
                                jnp.all(jnp.isfinite(smoothed_covs_draw)))

            if not (filter_results_ok and smooth_results_ok):
                print(f"Warning: Kalman filter/smoother produced NaNs/Infs for draw {idx}. Skipping.")
                continue

            # Run Durbin & Koopman Simulation Smoother
            rng_key, sim_key = random.split(rng_key)

            try:
                states_draw = durbin_koopman_simulation_smoother_from_results(
                    filter_a=filter_a_draw, filter_r=filter_r_draw,
                    filter_m=filter_m_draw, filter_P=filter_P_draw,
                    smooth_mu=smoothed_means_draw,
                    smooth_V=smoothed_covs_draw,
                    F=F_draw, key=sim_key
                )
            except Exception as sim_error:
                print(f"Simulation smoother failed for draw {idx}: {sim_error}")
                continue

            # Check simulation smoother output
            if jnp.any(jnp.isnan(states_draw)) or jnp.any(jnp.isinf(states_draw)):
                print(f"Warning: Simulation smoother produced NaNs/Infs for draw {idx}. Skipping.")
                continue

            if states_draw.shape != (T, state_dim):
                print(f"Warning: Unexpected shape {states_draw.shape} for draw {idx}. Expected ({T}, {state_dim}). Skipping.")
                continue

            # Extract components based on GPM structure
            # Trends are typically the first n_trends components
            trends = states_draw[:, :n_trends]
            
            # Stationary components are the next n_stationary components (current period)
            # The state structure is [trends, current_stationary, lagged_stationary, ...]
            stationary = states_draw[:, n_trends:n_trends + n_stationary]

            trend_draws.append(trends)
            stationary_draws.append(stationary)

        except Exception as e:
            print(f"Error processing draw {idx}: {e}")
            continue

    # Stack draws
    if len(trend_draws) > 0:
        trend_draws = jnp.stack(trend_draws)
        stationary_draws = jnp.stack(stationary_draws)
        print(f"Successfully extracted {len(trend_draws)} draws")
    else:
        print("No draws were successfully extracted")
        trend_draws = jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)

    return trend_draws, stationary_draws


def _identify_required_sites(samples: Dict, gmp_model) -> list:
    """Identify required parameter sites based on GPM model structure"""
    required = []
    
    # Add trend shock parameters
    for shock in gmp_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            required.append(param_name)
    
    # Add stationary shock parameters  
    for shock in gmp_model.stationary_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            required.append(param_name)
    
    # Add VAR parameters if present
    if gmp_model.var_prior_setup:
        if "A_transformed" in samples:
            required.append("A_transformed")
        elif "A_raw" in samples:
            required.append("A_raw")
        
        if "Omega_u_chol" in samples:
            required.append("Omega_u_chol")
    
    # Add structural parameters
    for param in gmp_model.parameters:
        if param in samples:
            required.append(param)
    
    # Add initial conditions
    if "init_mean_full" in samples:
        required.append("init_mean_full")
    
    return required


def _extract_gmp_parameters(samples: Dict, idx: int, gmp_model):
    """Extract parameters for a specific draw from GPM model"""
    from gpm_bvar_trends import EnhancedBVARParams
    
    # Extract trend covariance
    trend_sigmas = []
    for shock in gmp_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in samples:
            trend_sigmas.append(samples[param_name][idx])
        else:
            trend_sigmas.append(0.1)  # Default fallback
    
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas) ** 2)
    
    # Extract VAR parameters
    if gmp_model.var_prior_setup and gmp_model.stationary_variables:
        # Get VAR coefficient matrices
        if "A_transformed" in samples:
            A_matrices = samples["A_transformed"][idx]
        elif "A_raw" in samples:
            A_matrices = samples["A_raw"][idx]
        else:
            n_vars = len(gmp_model.stationary_variables)
            n_lags = gmp_model.var_prior_setup.var_order
            A_matrices = jnp.zeros((n_lags, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        
        # Get stationary covariance
        if "Omega_u_chol" in samples:
            stat_sigmas = []
            for shock in gmp_model.stationary_shocks:
                param_name = f"sigma_{shock}"
                if param_name in samples:
                    stat_sigmas.append(samples[param_name][idx])
                else:
                    stat_sigmas.append(0.1)
            
            sigma_u = jnp.array(stat_sigmas)
            Omega_u_chol = samples["Omega_u_chol"][idx]
            Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
        else:
            n_vars = len(gmp_model.stationary_variables)
            Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) * 0.1
    else:
        A_matrices = jnp.zeros((1, 1, 1), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(1, dtype=_DEFAULT_DTYPE)
    
    # Extract structural parameters
    structural_params = {}
    for param in gmp_model.parameters:
        if param in samples:
            structural_params[param] = samples[param][idx]
    
    return EnhancedBVARParams(
        A=A_matrices,
        Sigma_u=Sigma_u,
        Sigma_eta=Sigma_eta,
        structural_params=structural_params,
        Sigma_eps=None
    )


def _create_gmp_initial_covariance(state_dim: int, n_trends: int) -> jnp.ndarray:
    """Create initial state covariance matrix for GPM models"""
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6  # Diffuse for trends
    
    # Tighter prior for VAR states
    if state_dim > n_trends:
        init_cov = init_cov.at[n_trends:, n_trends:].set(
            jnp.eye(state_dim - n_trends, dtype=_DEFAULT_DTYPE) * 1e-6
        )
    
    # Ensure positive definite
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


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
    """
    Compute equal-tailed credible interval (approximate HDI) using percentiles.
    This is a robust alternative if az.hdi has shape issues.
    """
    # Requires at least 2 draws
    if draws.shape[0] < 2:
        print("Warning: Not enough draws to compute credible interval. Need at least 2.")
        # Return NaNs with shape matching the data dimensions
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}

    # Convert JAX array to NumPy array for percentile computation
    draws_np = np.asarray(draws)

    # Calculate the percentiles
    lower_percentile = (1 - hdi_prob) / 2 * 100
    upper_percentile = (1 + hdi_prob) / 2 * 100
    percentiles = np.array([lower_percentile, upper_percentile])

    try:
        # Compute percentiles along the sample dimension (axis=0)
        hdi_bounds_np = np.percentile(draws_np, percentiles, axis=0)

        # Rearrange to get low and high arrays
        hdi_low_np = hdi_bounds_np[0, ...]  
        hdi_high_np = hdi_bounds_np[1, ...]

        # Convert back to JAX arrays
        return {'low': jnp.asarray(hdi_low_np), 'high': jnp.asarray(hdi_high_np)}

    except Exception as e:
        print(f"Warning: Percentile computation failed with error: {e}. Returning NaNs.")
        # Return NaNs with correct shape in case of a general error
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}