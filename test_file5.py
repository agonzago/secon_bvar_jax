import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
# linalg and multivariate_normal are used internally by imported modules, keep imports for clarity/potential direct use
from jax.scipy import linalg
from jax.scipy.stats import multivariate_normal
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from functools import partial
from typing import NamedTuple, Tuple, Optional, List, Dict, Any, Sequence
import numpy as np # Use standard numpy for simulation and manual summary
import jax.scipy.linalg as jsl
from jax.typing import ArrayLike # Import ArrayLike
import xarray # Import xarray for type checking in HDI (not directly used for HDI computation logic)
import time # Import time to measure MCMC duration


# Import functions and jitter from your stationary prior module
# Ensure stationary_prior_jax_simplified.py is in your Python path
try:
    from stationary_prior_jax_simplified import (
        AtoP_jax, rev_mapping_jax, make_stationary_var_transformation_jax,
        quad_form_sym_jax, # Import quad_form_sym_jax
        _JITTER # Use the jitter from your stationary prior module
    )
except ImportError:
    print("Error: Could not import from stationary_prior_jax_simplified.py")
    print("Please ensure the file exists and is in your Python path.")
    # Define dummy values to allow code parsing to continue, but it will fail at runtime
    _JITTER = 1e-8
    def AtoP_jax(*args, **kwargs): raise NotImplementedError("Import failed")
    def rev_mapping_jax(*args, **kwargs): raise NotImplementedError("Import failed")
    def make_stationary_var_transformation_jax(*args, **kwargs): raise NotImplementedError("Import failed")
    def quad_form_sym_jax(*args, **kwargs): raise NotImplementedError("Import failed")


# Import the KalmanFilter class from your Kalman filter module
# Ensure Kalman_filter_jax.py is in your Python path
try:
    # Assuming KalmanFilter.filter returns (a, r, m, P) and .smooth returns (mu, V)
    from Kalman_filter_jax import KalmanFilter, _KF_JITTER # Import KalmanFilter and its jitter
except ImportError:
    print("Error: Could not import from Kalman_filter_jax.py")
    print("Please ensure the file exists and is in your Python path.")
    # Define dummy values to allow code parsing to continue, but it will fail at runtime
    _KF_JITTER = 1e-8
    class KalmanFilter:
        def __init__(self, *args, **kwargs): raise NotImplementedError("Import failed")
        def filter(self, *args, **kwargs): raise NotImplementedError("Import failed")
        def smooth(self, *args, **kwargs): raise NotImplementedError("Import failed")


try:
    import multiprocessing
    # Attempt to get physical CPU count or logical if physical is not available
    num_cpu = multiprocessing.cpu_count()
    # Set host device count, ensuring it's at least 1 and not excessively large
    numpyro.set_host_device_count(min(num_cpu, 8)) # Cap at 8 for safety/common hardware
except Exception as e:
    print(f"Could not set host device count: {e}. Falling back to default (likely 1 or 4).")
    # If setting fails, numpyro will use its default, which is usually okay.
    pass
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

_DEFAULT_DTYPE = jnp.float64 # Use JAX default dtype

#Import plotting functions from reporting_plots.py
from reporting_plots import plot_decomposition_results, plot_observed_and_trend

# Note: We are now using _JITTER from stationary_prior_jax_simplified
# and _KF_JITTER from Kalman_filter_jax. They are currently both set to 1e-8 in those files.
# If you change them in the original files, the imported values will reflect that.


# --- Data Structures ---
# (These are data structures specific to the BVAR with trends model logic, not part of the imported modules)

class BVARState(NamedTuple):
    """State for BVAR with trends"""
    trends: jnp.ndarray      # Random walk trends
    stationary: jnp.ndarray  # Stationary VAR states

class BVARParams(NamedTuple):
    """Parameters for BVAR with trends"""
    # VAR parameters (stationary component)
    A: jnp.ndarray          # VAR coefficient matrices (stacked: A_1, ..., A_p)
    Sigma_u: jnp.ndarray    # VAR innovation covariance

    # Trend parameters
    Sigma_eta: jnp.ndarray  # Trend innovation covariance

    # Measurement equation (if any)
    Sigma_eps: Optional[jnp.ndarray] = None  # Measurement error covariance (assuming H is fixed as [I | I | 0 ...])


# --- Durbin & Koopman Simulation Smoother ---

# --- Corrected Extract Function for Dictionary-Based KalmanFilter ---

def extract_trends_and_components(mcmc, y: jnp.ndarray, n_lags: int = 2,
                                 num_draws: int = 100, rng_key: jnp.ndarray = random.PRNGKey(42)):
    """Extract trend and stationary components using Durbin & Koopman simulation smoother"""
    samples = mcmc.get_samples()
    T, n_vars = y.shape
    state_dim = n_vars + n_vars * n_lags

    # Storage for draws
    trend_draws = []
    stationary_draws = []

    # Check required sites
    required_sites = ['A_transformed', 'sigma_u', 'Omega_u_chol', 'sigma_eta', 'Omega_eta_chol', 'init_mean']
    for site in required_sites:
        if site not in samples:
            print(f"Error: '{site}' not found in MCMC samples. Cannot proceed with extraction.")
            return jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE), jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE)

    n_posterior = len(samples['A_transformed'])
    num_draws = min(num_draws, n_posterior)
    
    if num_draws > 0:
        draw_indices_float = np.linspace(0, n_posterior - 1, num_draws)
        draw_indices = np.round(draw_indices_float).astype(int)
    else:
        draw_indices = np.array([], dtype=int)

    # Static observation info
    n_obs_actual = n_vars
    valid_obs_idx = jnp.arange(n_vars, dtype=int)
    I_obs = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    # Fixed initial state covariance
    init_cov_fixed = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6
    init_cov_fixed = init_cov_fixed.at[n_vars:, n_vars:].set(jnp.eye(n_vars * n_lags, dtype=_DEFAULT_DTYPE) * 1e-6)
    init_cov_fixed = (init_cov_fixed + init_cov_fixed.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    print(f"Processing {len(draw_indices)} posterior draws...")

    for i, idx in enumerate(draw_indices):
        if (i + 1) % 10 == 0:
            print(f"Processing draw {i+1}/{len(draw_indices)}")
        
        try:
            # Extract parameters for this draw
            A_transformed_draw = samples['A_transformed'][idx]
            sigma_u_draw = samples['sigma_u'][idx]
            Omega_u_chol_draw = samples['Omega_u_chol'][idx]
            sigma_eta_draw = samples['sigma_eta'][idx]
            Omega_eta_chol_draw = samples['Omega_eta_chol'][idx]
            init_mean_draw = samples['init_mean'][idx]

            # Compute covariance matrices
            Sigma_u_draw = jnp.diag(sigma_u_draw) @ Omega_u_chol_draw @ Omega_u_chol_draw.T @ jnp.diag(sigma_u_draw)
            Sigma_u_draw = (Sigma_u_draw + Sigma_u_draw.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

            Sigma_eta_draw = jnp.diag(sigma_eta_draw) @ Omega_eta_chol_draw @ Omega_eta_chol_draw.T @ jnp.diag(sigma_eta_draw)
            Sigma_eta_draw = (Sigma_eta_draw + Sigma_eta_draw.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

            # Create parameter structure
            params_draw = BVARParams(A=A_transformed_draw, Sigma_u=Sigma_u_draw, Sigma_eta=Sigma_eta_draw, Sigma_eps=None)

            # Build state space matrices
            F_draw, R_aug_draw, C_mat_draw, H_obs_draw, Q_mat_draw = build_state_space_matrices(params_draw, n_vars, n_lags)

            # Check for NaNs
            state_space_nan_draw = (jnp.any(jnp.isnan(F_draw)) | jnp.any(jnp.isnan(R_aug_draw)) | 
                                   jnp.any(jnp.isnan(C_mat_draw)) | jnp.any(jnp.isnan(H_obs_draw)) | 
                                   jnp.any(jnp.isnan(Q_mat_draw)))

            if state_space_nan_draw:
                print(f"Warning: State space matrices contain NaNs for draw {idx}. Skipping.")
                continue

            # Initial state
            init_mean_kf_draw = jnp.where(jnp.isfinite(init_mean_draw), init_mean_draw, jnp.zeros_like(init_mean_draw))

            # Create Kalman Filter instance
            kf_draw = KalmanFilter(T=F_draw, R=R_aug_draw, C=C_mat_draw, H=H_obs_draw, 
                                  init_x=init_mean_kf_draw, init_P=init_cov_fixed)

            # Run Kalman Filter - returns a dictionary
            try:
                filter_results_dict = kf_draw.filter(
                    y, static_valid_obs_idx=valid_obs_idx,
                    static_n_obs_actual=n_obs_actual,
                    static_C_obs=C_mat_draw, static_H_obs=H_obs_draw, static_I_obs=I_obs
                )
                
                # Extract the arrays from the dictionary
                filter_a_draw = filter_results_dict['x_pred']  # Predicted states
                filter_r_draw = filter_results_dict['P_pred']  # Predicted covariances
                filter_m_draw = filter_results_dict['x_filt']  # Filtered states
                filter_P_draw = filter_results_dict['P_filt']  # Filtered covariances
                
            except Exception as filter_error:
                print(f"Filter failed for draw {idx}: {filter_error}")
                continue

            # Run Kalman Smoother - returns a tuple (smoothed_means, smoothed_covs)
            try:
                # Pass the filter results dictionary to smooth
                smoothed_means_draw, smoothed_covs_draw = kf_draw.smooth(
                    y, filter_results=filter_results_dict,
                    static_valid_obs_idx=valid_obs_idx,
                    static_n_obs_actual=n_obs_actual,
                    static_C_obs_for_filter=C_mat_draw, 
                    static_H_obs_for_filter=H_obs_draw, 
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

            # Extract components
            trends = states_draw[:, :n_vars]
            stationary = states_draw[:, n_vars:n_vars*2]  # Current VAR states

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
        trend_draws = jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE)

    return trend_draws, stationary_draws


# --- Updated Durbin & Koopman Simulation Smoother ---

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


    

# --- State Space Matrix Building ---
# This function is specific to the BVAR with trends model structure

def build_state_space_matrices(params: BVARParams, n_vars: int, n_lags: int):
    """Build state space representation matrices F, R_aug, C, H_obs, and Q_mat"""
    # State dimension: n_vars (trends) + n_vars * n_lags (VAR states)
    state_dim = n_vars + n_vars * n_lags

    # Transition matrix F
    F = jnp.zeros((state_dim, state_dim), dtype=_DEFAULT_DTYPE)

    # Trends evolve as random walks: tau_t = tau_{t-1} + eta_t
    F = F.at[:n_vars, :n_vars].set(jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))

    # VAR component: X_t = A_1 X_{t-1} + ... + A_p X_{t-p} + u_t
    # State is [tau_t; X_t; X_{t-1}; ...; X_{t-p+1}]
    var_start = n_vars

    # Set VAR coefficient matrices (params.A is (n_lags, n_vars, n_vars))
    # params.A[lag] is A_{lag+1} in X_t = A_1 X_{t-1} + ...
    # State indices for X_{t-1} are var_start + n_vars * 0, X_{t-2} are var_start + n_vars * 1, ...
    # So A_i should multiply state corresponding to X_{t-i}.
    # For A_1, multiply state at var_start + n_vars*0. For A_p, multiply state at var_start + n_vars*(p-1).
    # This means params.A[0] is A_1, params.A[1] is A_2, ..., params.A[n_lags-1] is A_p.
    # The indices in the state are for X_t, X_{t-1}, ..., X_{t-p+1}.
    # X_t depends on X_{t-1} (at index var_start+n_vars*0), X_{t-2} (at index var_start+n_vars*1), ...
    # F[var_start:var_start+n_vars, var_start + i*n_vars : var_start + (i+1)*n_vars] should be A_{i+1}.
    # So F[var_start:var_start+n_vars, var_start + lag*n_vars : var_start + (lag+1)*n_vars] should be params.A[lag]. This seems correct.
    for lag in range(n_lags):
        F = F.at[var_start:var_start + n_vars,
                 var_start + lag * n_vars:var_start + (lag + 1) * n_vars].set(params.A[lag])

    # Set identity matrices for lagged states (X_{t-i} state at time t becomes X_{t-i-1} state at time t+1)
    # X_{t-i} at time t corresponds to index var_start + i*n_vars
    # X_{t-i-1} at time t+1 corresponds to index var_start + (i+1)*n_vars
    if n_lags > 1:
        for i in range(n_lags - 1):
            start_row = var_start + (i + 1) * n_vars
            start_col = var_start + i * n_vars
            F = F.at[start_row:start_row + n_vars, start_col:start_col + n_vars].set(jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))


    # Shock impact matrix R_aug (maps state shocks to state)
    # State shocks: eta_t (trend), u_t (VAR)
    # State = F @ State[-1] + [eta_t; u_t; 0; ...; 0]
    # This implies the state shock vector is [eta_t; u_t; 0; ...; 0].
    # R_aug should map a standard normal shock vector (dim = n_vars + n_vars = 2*n_vars)
    # to the state shock vector.
    # State shock vector = R_aug @ standard_normal_shocks
    # [eta_t; u_t; 0; ...; 0] = R_aug @ [eps_eta; eps_u] where eta_t = Chol(Sigma_eta) @ eps_eta, u_t = Chol(Sigma_u) @ eps_u
    # So R_aug should be diag(Chol(Sigma_eta), Chol(Sigma_u), 0, ..._
    # The dimension of standard_normal_shocks vector is n_vars (for trend) + n_vars (for VAR) = 2*n_vars.
    # R_aug shape: state_dim x (2 * n_vars)

    n_state_shocks = 2 * n_vars
    R_aug = jnp.zeros((state_dim, n_state_shocks), dtype=_DEFAULT_DTYPE)

    # Trend shock block (Cholesky factor of Sigma_eta)
    try:
        # Add jitter from the imported module for stability
        chol_Sigma_eta = jnp.linalg.cholesky(params.Sigma_eta + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))
        R_aug = R_aug.at[:n_vars, :n_vars].set(chol_Sigma_eta)
    except Exception:
         # Fallback if Cholesky fails (should be rare with jitter and PSD enforcement)
         # Use sqrt of diagonal as a robust fallback, although not a true Cholesky
         R_aug = R_aug.at[:n_vars, :n_vars].set(jnp.diag(jnp.sqrt(jnp.diag(params.Sigma_eta) + _JITTER)))


    # VAR shock block (Cholesky factor of Sigma_u)
    try:
        # Add jitter from the imported module for stability
        chol_Sigma_u = jnp.linalg.cholesky(params.Sigma_u + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))
        R_aug = R_aug.at[var_start:var_start + n_vars, n_vars:n_vars*2].set(chol_Sigma_u) # Map to the second block of shocks
    except Exception:
        # Fallback if Cholesky fails
        R_aug = R_aug.at[var_start:var_start + n_vars, n_vars:n_vars*2].set(jnp.diag(jnp.sqrt(jnp.diag(params.Sigma_u) + _JITTER)))

    # The state innovation covariance matrix Q is R_aug @ R_aug.T
    Q_mat = R_aug @ R_aug.T


    # Observation matrix C (observing trends + current VAR states)
    # y_t = [I | I | 0 | ... | 0] @ state_t + measurement_error_t
    C_mat = jnp.zeros((n_vars, state_dim), dtype=_DEFAULT_DTYPE)
    C_mat = C_mat.at[:, :n_vars].set(jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))  # Trends
    C_mat = C_mat.at[:, var_start:var_start + n_vars].set(jnp.eye(n_vars, dtype=_DEFAULT_DTYPE))  # Current VAR states

    # Measurement error covariance H_obs (if applicable)
    # If Sigma_eps is None, H_obs is zero matrix.
    # The KalmanFilter class expects H to be provided, even if zero.
    # If Sigma_eps is None, create a zero matrix with appropriate jittering for the KF.
    if params.Sigma_eps is not None:
        H_obs = params.Sigma_eps
        # Ensure H_obs is symmetric and PSD for Kalman Filter
        H_obs = (H_obs + H_obs.T) / 2.0 + _KF_JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    else:
         # If no measurement error, H_obs is zero. The KF handles H=0 correctly.
         H_obs = jnp.zeros((n_vars, n_vars), dtype=_DEFAULT_DTYPE)


    # Return F, R_aug, C_mat, H_obs, and Q_mat
    return F, R_aug, C_mat, H_obs, Q_mat


# --- Numpyro Model ---

def bvar_model(y: jnp.ndarray, n_lags: int = 2):
    """Numpyro model for BVAR with trends using Stationary Prior and Kalman Filter"""
    T, n_vars = y.shape
    state_dim = n_vars + n_vars * n_lags

    # --- Priors ---

    # Priors for VAR coefficients (using stationary prior transformation)
    # Sample the raw parameters (these will be transformed into the P matrices via AtoP_jax)
    # The total number of raw parameters is n_lags * n_vars * n_vars.
    A_vec_raw = numpyro.sample("A_vec_raw",
                               dist.Normal(0, 1).expand([n_lags * n_vars * n_vars]))

    # Reshape A_vec_raw into a list of p m x m matrices (these are the *raw* A matrices for make_stationary_var_transformation_jax)
    raw_A_list = [A_vec_raw[i*n_vars*n_vars:(i+1)*n_vars*n_vars].reshape((n_vars, n_vars)) for i in range(n_lags)]


    # LKJ prior for VAR innovation correlation
    Omega_u_chol = numpyro.sample("Omega_u_chol",
                                  dist.LKJCholesky(n_vars, concentration=2.0))
    sigma_u = numpyro.sample("sigma_u", dist.InverseGamma(2.0, 1.0).expand([n_vars]))
    Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
    # Ensure Sigma_u is symmetric and positive definite for the stationary transformation
    # Use _JITTER from stationary_prior_jax_simplified
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)


    # Apply stationary transformation using make_stationary_var_transformation_jax
    # This transforms the raw_A_list into the stationary VAR coefficients (phi_list)
    # make_stationary_var_transformation_jax takes Sigma (VAR innovation cov) and A_list (raw matrices)
    phi_list, gamma_list = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)


    # Check if transformation resulted in NaNs (indicating numerical issues or non-stationarity)
    # If NaNs are present, assign a very low log-likelihood.
    transformation_failed = jnp.any(jnp.stack([jnp.any(jnp.isnan(phi)) for phi in phi_list]))

    # Stack the transformed VAR coefficients (phi_list) to get the A matrix for the state space
    # If transformation failed, A_transformed will contain NaNs. build_state_space_matrices might produce NaNs.
    A_transformed = jnp.stack(phi_list) # Shape (n_lags, n_vars, n_vars)

    # --- Store transformed VAR coefficients as a deterministic ---
    numpyro.deterministic("A_transformed", A_transformed)


    # LKJ prior for trend innovation correlation
    Omega_eta_chol = numpyro.sample("Omega_eta_chol",
                                    dist.LKJCholesky(n_vars, concentration=2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.InverseGamma(2.0, 1.0).expand([n_vars]))
    Sigma_eta = jnp.diag(sigma_eta) @ Omega_eta_chol @ Omega_eta_chol.T @ jnp.diag(sigma_eta)
    # Ensure Sigma_eta is symmetric and positive definite
    # Use _JITTER from stationary_prior_jax_simplified
    Sigma_eta = (Sigma_eta + Sigma_eta.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)


    # Measurement error covariance (optional - assuming zero for now as in simple_idea.py)
    Sigma_eps = None # Set to None for no measurement error


    # --- Priors for Initial State Mean ---
    # The initial state vector is [tau_0; X_0; X_{-1}; ...; X_{-p+1}]
    # Priors for tau_0 (trends) can be diffuse. Priors for X_0, X_{-1}, etc. (VAR) can be tight around 0.
    # Define the mean and diagonal standard deviation for the Normal prior on the initial state mean.
    init_mean_prior_loc = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)

    # Define the diagonal standard deviations (sqrt of variances)
    init_mean_prior_diag_stds = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    # Diffuse prior for trends (first n_vars elements)
    init_mean_prior_diag_stds = init_mean_prior_diag_stds.at[:n_vars].set(jnp.full(n_vars, jnp.sqrt(10), dtype=_DEFAULT_DTYPE))
    # Tight prior for VAR states (next n_vars * n_lags elements)
    init_mean_prior_diag_stds = init_mean_prior_diag_stds.at[n_vars:].set(jnp.full(n_vars * n_lags, jnp.sqrt(1), dtype=_DEFAULT_DTYPE))

    # Sample the initial state mean vector
    init_mean_sampled = numpyro.sample("init_mean", dist.Normal(init_mean_prior_loc, init_mean_prior_diag_stds))


    # --- Fixed Initial State Covariance for Kalman Filter ---
    # This is the P_0 matrix in the Kalman Filter, typically treated as fixed or hyperparameter
    init_cov_fixed = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6  # Diffuse prior for all states initially
    # Tight prior for VAR states (indices n_vars onwards)
    init_cov_fixed = init_cov_fixed.at[n_vars:, n_vars:].set(jnp.eye(n_vars * n_lags, dtype=_DEFAULT_DTYPE) * 1e-6)
    # Ensure init_cov is symmetric and positive definite for the KF
    init_cov_fixed = (init_cov_fixed + init_cov_fixed.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


    # Create parameter structure (only includes sampled structural parameters)
    # Note: This params structure is used ONLY for build_state_space_matrices here.
    # The sampled initial state mean is handled separately for the KF init_x.
    params_for_ss = BVARParams(A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta, Sigma_eps=Sigma_eps)

    # --- State Space Representation ---
    # build_state_space_matrices returns F, R_aug, C_mat, H_obs, Q_mat
    F, R_aug, C_mat, H_obs, Q_mat = build_state_space_matrices(params_for_ss, n_vars, n_lags)


    # --- Kalman Filter for Likelihood ---
    # Instantiate KalmanFilter class with the SAMPLED initial mean and FIXED initial covariance
    kf = KalmanFilter(T=F, R=R_aug, C=C_mat, H=H_obs, init_x=init_mean_sampled, init_P=init_cov_fixed)

    # Determine static observation info (assuming no NaNs or static NaNs in y)
    n_obs_actual = n_vars # All variables are observed in this model structure
    valid_obs_idx = jnp.arange(n_vars, dtype=int)
    # C_obs = C_mat # If all observed, C_obs is the full C_mat (used for filter likelihood)
    # H_obs_actual = H_obs # If all observed, H_obs_actual is the full H_obs (used for filter likelihood)
    I_obs = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) # Identity for observed variables dimension


    # Compute log-likelihood using the Kalman Filter
    # If transformation failed or state space matrices built incorrectly (NaNs), loglik should be very low.
    state_space_nan = jnp.any(jnp.isnan(F)) | jnp.any(jnp.isnan(R_aug)) | jnp.any(jnp.isnan(C_mat)) | jnp.any(jnp.isnan(H_obs)) | jnp.any(jnp.isnan(Q_mat)) | jnp.any(jnp.isnan(init_mean_sampled)) | jnp.any(jnp.isnan(init_cov_fixed))


    # The kf.log_likelihood method takes static observation info
    # Need to handle the case where T=0 explicitly, although the KF should handle it.
    loglik = jax.lax.cond(
        transformation_failed | state_space_nan, # Check if anything failed
        lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE), # Assign -inf loglik if failed
        lambda: kf.log_likelihood(y, valid_obs_idx, n_obs_actual, C_mat, H_obs, I_obs) # Pass C_mat, H_obs for filter
    )

    # Add likelihood to the model
    numpyro.factor("loglik", loglik)


def bvar_model_hierarchical(y: jnp.ndarray, n_lags: int = 2, hyperparams: dict = None):
    """Numpyro model for BVAR with trends using Hierarchical Stationary Prior and Kalman Filter"""
    T, n_vars = y.shape
    state_dim = n_vars + n_vars * n_lags

    # Default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            'es': [0.5, 0.3],  # Prior means for diagonal and off-diagonal
            'fs': [0.5, 0.5],  # Prior std devs for means
            'gs': [2.0, 2.0],  # Gamma shape parameters for precisions
            'hs': [1.0, 1.0]   # Gamma rate parameters for precisions
        }

    # Extract hyperparameters
    es = jnp.array(hyperparams['es'])
    fs = jnp.array(hyperparams['fs'])
    gs = jnp.array(hyperparams['gs'])
    hs = jnp.array(hyperparams['hs'])

    # --- Hierarchical Priors for VAR Coefficients ---
    
    # Sample hyperparameters for diagonal and off-diagonal elements
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(es[i], fs[i])) for i in range(2)]
    Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(gs[i], hs[i])) for i in range(2)]

    # Sample all elements as matrices, then apply hierarchical structure
    raw_A_list = []
    
    for lag in range(n_lags):
        # Sample a full n_vars x n_vars matrix with hierarchical priors
        A_full = numpyro.sample(f"A_full_{lag}", 
                               dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
        
        # Sample diagonal elements separately
        A_diag_vals = numpyro.sample(f"A_diag_{lag}", 
                                    dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
        
        # Replace diagonal of A_full with the hierarchical diagonal values
        A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag_vals)
        
        raw_A_list.append(A_lag)

    # Store the raw A matrices as deterministic for inspection
    A_raw_stacked = jnp.stack(raw_A_list)  # Shape: (n_lags, n_vars, n_vars)
    numpyro.deterministic("A_raw", A_raw_stacked)

    # --- Continue with existing priors for covariance matrices ---
    
    # LKJ prior for VAR innovation correlation
    Omega_u_chol = numpyro.sample("Omega_u_chol",
                                  dist.LKJCholesky(n_vars, concentration=2.0))
    sigma_u = numpyro.sample("sigma_u", dist.InverseGamma(2.0, 1.0).expand([n_vars]))
    Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    # Apply stationary transformation using make_stationary_var_transformation_jax
    phi_list, gamma_list = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)

    # Check if transformation resulted in NaNs
    transformation_failed = jnp.any(jnp.stack([jnp.any(jnp.isnan(phi)) for phi in phi_list]))

    # Stack the transformed VAR coefficients
    A_transformed = jnp.stack(phi_list)  # Shape (n_lags, n_vars, n_vars)
    numpyro.deterministic("A_transformed", A_transformed)

    # LKJ prior for trend innovation correlation
    Omega_eta_chol = numpyro.sample("Omega_eta_chol",
                                    dist.LKJCholesky(n_vars, concentration=2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.InverseGamma(2.0, 1.0).expand([n_vars]))
    Sigma_eta = jnp.diag(sigma_eta) @ Omega_eta_chol @ Omega_eta_chol.T @ jnp.diag(sigma_eta)
    Sigma_eta = (Sigma_eta + Sigma_eta.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    # Measurement error covariance (optional)
    Sigma_eps = None

    # --- Priors for Initial State Mean ---
    init_mean_prior_loc = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_mean_prior_diag_stds = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_mean_prior_diag_stds = init_mean_prior_diag_stds.at[:n_vars].set(jnp.full(n_vars, jnp.sqrt(10), dtype=_DEFAULT_DTYPE))
    init_mean_prior_diag_stds = init_mean_prior_diag_stds.at[n_vars:].set(jnp.full(n_vars * n_lags, jnp.sqrt(1), dtype=_DEFAULT_DTYPE))
    init_mean_sampled = numpyro.sample("init_mean", dist.Normal(init_mean_prior_loc, init_mean_prior_diag_stds))

    # --- Fixed Initial State Covariance for Kalman Filter ---
    init_cov_fixed = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6
    init_cov_fixed = init_cov_fixed.at[n_vars:, n_vars:].set(jnp.eye(n_vars * n_lags, dtype=_DEFAULT_DTYPE) * 1e-6)
    init_cov_fixed = (init_cov_fixed + init_cov_fixed.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    # Create parameter structure
    params_for_ss = BVARParams(A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta, Sigma_eps=Sigma_eps)

    # --- State Space Representation ---
    F, R_aug, C_mat, H_obs, Q_mat = build_state_space_matrices(params_for_ss, n_vars, n_lags)

    # --- Kalman Filter for Likelihood ---
    kf = KalmanFilter(T=F, R=R_aug, C=C_mat, H=H_obs, init_x=init_mean_sampled, init_P=init_cov_fixed)

    # Determine static observation info
    n_obs_actual = n_vars
    valid_obs_idx = jnp.arange(n_vars, dtype=int)
    I_obs = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)

    # Compute log-likelihood
    state_space_nan = (jnp.any(jnp.isnan(F)) | jnp.any(jnp.isnan(R_aug)) | 
                      jnp.any(jnp.isnan(C_mat)) | jnp.any(jnp.isnan(H_obs)) | 
                      jnp.any(jnp.isnan(Q_mat)) | jnp.any(jnp.isnan(init_mean_sampled)) | 
                      jnp.any(jnp.isnan(init_cov_fixed)))

    loglik = jax.lax.cond(
        transformation_failed | state_space_nan,
        lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
        lambda: kf.log_likelihood(y, valid_obs_idx, n_obs_actual, C_mat, H_obs, I_obs)
    )

    numpyro.factor("loglik", loglik)


# --- Fitting Function ---

def fit_bvar_trends_hierarchical(y: jnp.ndarray, n_lags: int = 2, hyperparams: dict = None,
                                num_warmup: int = 1000, num_samples: int = 2000, 
                                num_chains: int = 4, rng_key: jnp.ndarray = random.PRNGKey(0)):
    """Fit BVAR with trends using hierarchical prior and MCMC"""
    
    kernel = NUTS(bvar_model_hierarchical)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains)
    
    mcmc.run(rng_key, y=y, n_lags=n_lags, hyperparams=hyperparams)
    
    return mcmc

def fit_bvar_trends(y: jnp.ndarray, n_lags: int = 2, num_warmup: int = 1000,
                   num_samples: int = 2000, num_chains: int = 4, rng_key: jnp.ndarray = random.PRNGKey(0)):
    """Fit BVAR with trends using MCMC"""

    # Run MCMC
    kernel = NUTS(bvar_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains)

    # Use the provided rng_key or generate a new one
    mcmc.run(rng_key, y=y, n_lags=n_lags)

    return mcmc


# --- HDI Computation (using Percentiles) ---
# Reverted to the percentile based approach which is more robust to ArviZ shape issues

def compute_hdi_with_percentiles(draws: jnp.ndarray, hdi_prob: float = 0.94):
    """
    Compute equal-tailed credible interval (approximate HDI) using percentiles.
    This is a robust alternative if az.hdi has shape issues.
    """
    # Requires at least 2 draws
    if draws.shape[0] < 2:
        print("Warning: Not enough draws to compute credible interval. Need at least 2.")
        # Return NaNs with shape (T, n_vars)
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}

    # Convert JAX array to NumPy array for percentile computation
    draws_np = np.asarray(draws) # Shape (num_draws, T, n_vars)

    # Calculate the percentiles
    lower_percentile = (1 - hdi_prob) / 2 * 100
    upper_percentile = (1 + hdi_prob) / 2 * 100
    percentiles = np.array([lower_percentile, upper_percentile])

    try:
        # Compute percentiles along the sample dimension (axis=0)
        # Resulting shape will be (2, T, n_vars)
        hdi_bounds_np = np.percentile(draws_np, percentiles, axis=0)

        # Rearrange to get low and high arrays of shape (T, n_vars)
        hdi_low_np = hdi_bounds_np[0, :, :] # Shape (T, n_vars)
        hdi_high_np = hdi_bounds_np[1, :, :] # Shape (T, n_vars)

        # Convert back to JAX arrays
        return {'low': jnp.asarray(hdi_low_np), 'high': jnp.asarray(hdi_high_np)}

    except Exception as e:
        print(f"Warning: Percentile computation failed with error: {e}. Returning NaNs.")
        # Return NaNs with correct shape (T, n_vars) in case of a general error
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}





# --- Example Usage Function ---

def example_usage():
    """Example of how to use the BVAR with trends model"""

    # Define parameters for the example
    T = 100
    n_vars = 3
    n_lags = 2
    num_warmup = 500
    num_samples = 1000
    num_chains = 2 # Set the desired number of chains here
    num_extract_draws_desired = 50 # Desired number of simulation smoother draws

    # Generate synthetic data using standard NumPy
    # This avoids potential JAX/GPU issues during data generation
    np_rng = np.random.default_rng(123)


    print(f"Generating synthetic data with T={T}, n_vars={n_vars}, n_lags={n_lags} using NumPy...")

    # Trends: Random Walk
    # Use np for simulation
    trends_np = np.cumsum(np_rng.normal(loc=0, scale=0.1, size=(T, n_vars)), axis=0).astype(_DEFAULT_DTYPE)


    # Stationary Component: VAR(n_lags)
    # Generate stable VAR coefficients using np
    # For a real simulation, might sample A from a stable distribution or check eigenvalues.
    # Using small random coefficients to increase likelihood of stability for simulation
    A_sim_list_np = [np_rng.normal(loc=0, scale=0.05, size=(n_vars, n_vars)).astype(_DEFAULT_DTYPE) for _ in range(n_lags)]

    # Generate VAR shocks using np
    Sigma_u_sim_np = np.eye(n_vars, dtype=_DEFAULT_DTYPE) * 0.5 # Simple diagonal covariance
    stationary_shocks_np = np_rng.multivariate_normal(mean=np.zeros(n_vars), cov=Sigma_u_sim_np, size=T).astype(_DEFAULT_DTYPE)

    # Simulate stationary VAR using np
    # We need initial values for lags (X_{0}, X_{-1}, ..., X_{-(p-1)}) to start the recursion at t=0
    # Use random noise for simplicity for initial lags
    initial_var_lags_np = np_rng.normal(loc=0, scale=0.1, size=(n_lags, n_vars)).astype(_DEFAULT_DTYPE)

    # Create a placeholder array for the full sequence of VAR states [X_0, X_1, ..., X_{T-1}]
    stationary_np = np.zeros((T, n_vars), dtype=_DEFAULT_DTYPE)

    # Keep track of the last p VAR states for the recursion
    # Use a fixed-size np array for the lags
    current_lags_np = initial_var_lags_np # current_lags_np[0] is X_{t-1}, current_lags_np[p-1] is X_{t-p}

    # Simulate forward the VAR process X_t = A_1 X_{t-1} + ... + A_p X_{t-p} + u_t
    # The simulation loop runs from t=0 to T-1
    for t in range(T):
        # Compute X_t using the current lags
        # sum_{i=0}^{n_lags-1} A_sim_list_np[i] @ current_lags_np[i]
        # A_1 uses X_{t-1}, which is current_lags_np[0]
        # A_p uses X_{t-p}, which is current_lags_np[p-1]
        var_contribution_np = np.sum(np.stack([A_sim_list_np[i] @ current_lags_np[i] for i in range(n_lags)]), axis=0)

        X_t_np = var_contribution_np + stationary_shocks_np[t]

        # Store the computed X_t
        stationary_np[t] = X_t_np

        # Update the lags for the next time step: X_t becomes the new X_{t-1}
        # Shift the lags down: [X_t, X_{t-1}, ..., X_{t-p+2}]
        # current_lags_np[0] becomes X_t, current_lags_np[1] becomes old current_lags_np[0], ...
        # Need to prepend X_t to the front of the lags and drop the last one.
        current_lags_np = np.concatenate([X_t_np[None, :], current_lags_np[:-1]], axis=0)


    # Combine components using np
    y_np = trends_np + stationary_np # No measurement error in this simple version
    
    # Convert NumPy array to JAX array for the model fitting step
    y_jax = jnp.asarray(y_np)


    import pandas as pd

    output_filename_pandas = 'sim_data.csv'

    # Convert NumPy array to a Pandas DataFrame
    df = pd.DataFrame(y_np, columns=['OBS1', 'OBS2', 'OBS3']) # You can add column names here

    # Save the DataFrame to CSV
    df.to_csv(output_filename_pandas, index=False) #
    print(f"Synthetic data saved to {output_filename_pandas}")
    print("Synthetic data generated.")

    print("Fitting BVAR with trends...")
    start_time_mcmc = time.time() # Start timing MCMC

    # Fit model using JAX/Numpyro
    # Use fewer samples for a quicker example run
    # Reduce num_chains if you only have 1 CPU core available (as suggested by warning)
    # Define num_chains_to_use here
    num_chains_to_use = min(num_chains, jax.local_device_count())
    if num_chains > num_chains_to_use:
         print(f"Reducing number of chains to {num_chains_to_use} due to available devices ({jax.local_device_count()}).")
    # Set number of host devices for CPU chains if needed
    if jax.local_device_count() == 1 and num_chains_to_use > 1:
         print(f"Setting number of host devices to {num_chains_to_use} for parallel CPU chains.")
         numpyro.set_host_device_count(num_chains_to_use)


       # Define custom hyperparameters
    custom_hyperparams = {
        'es': [0.8, 0.0],   # Diagonal elements centered at 0.8, off-diagonal at 0.0
        'fs': [0.3, 0.2],   # Tighter prior for diagonal, looser for off-diagonal means
        'gs': [3.0, 2.0],   # Shape parameters for precision priors
        'hs': [2.0, 1.0]    # Rate parameters for precision priors
    }
    
    # Generate or load your data
    T, n_vars, n_lags = 100, 3, 2
    
    # For demonstration, create simple synthetic data
    np_rng = np.random.default_rng(123)
    y_synthetic = np_rng.normal(0, 1, (T, n_vars)).astype(_DEFAULT_DTYPE)
    y_jax = jnp.asarray(y_synthetic)
    
    print("Fitting BVAR with hierarchical priors...")
    
    # Fit the model
    mcmc = fit_bvar_trends_hierarchical(
        y_jax, 
        n_lags=n_lags, 
        hyperparams=custom_hyperparams,
        num_warmup=500, 
        num_samples=1000, 
        num_chains=2,
        rng_key=random.PRNGKey(42)
    )
    
    # Print summary
    mcmc.print_summary(exclude_deterministic=False)

    #mcmc = fit_bvar_trends(y_jax, n_lags=n_lags, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains_to_use, rng_key=random.PRNGKey(1))

    end_time_mcmc = time.time() # End timing MCMC
    print(f"MCMC completed in {end_time_mcmc - start_time_mcmc:.2f} seconds.")

    mcmc.print_summary(exclude_deterministic=False)
    print("Extracting trend and stationary components using Durbin & Koopman Simulation Smoother...")

    # Extract components
    start_time_extraction = time.time() # Start timing extraction
    # Use a new key for simulation smoother draws
    # Ensure num_draws is reasonable compared to total posterior draws
    # Check A_transformed length as it was made deterministic
    if 'A_transformed' not in mcmc.get_samples() or 'init_mean' not in mcmc.get_samples():
         print("Error: Required sites ('A_transformed' or 'init_mean') not found in MCMC samples. Cannot proceed with extraction.")
         # Set extraction results to empty/None and skip plotting/HDI
         trend_draws, stationary_draws = jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE), jnp.empty((0, T, n_vars), dtype=_DEFAULT_DTYPE)
         trend_hdi, stationary_hdi = None, None

    else:
        total_posterior_draws = mcmc.get_samples()['A_transformed'].shape[0]
        num_extract_draws = min(num_extract_draws_desired, total_posterior_draws)
        if num_extract_draws < num_extract_draws_desired:
             print(f"Warning: Reducing requested simulation smoother draws from {num_extract_draws_desired} to {num_extract_draws} as only {total_posterior_draws} posterior samples are available.")

        # Call the extraction function which now uses the DK smoother
        trend_draws, stationary_draws = extract_trends_and_components(mcmc, y_jax, n_lags=n_lags, num_draws=num_extract_draws, rng_key=random.PRNGKey(2))

        end_time_extraction = time.time() # End timing extraction
        print(f"Component extraction completed in {end_time_extraction - start_time_extraction:.2f} seconds.")

        print("Computing HDI intervals using percentiles...") # Updated print statement

        # Compute HDI using the percentile-based function
        if trend_draws.shape[0] > 1:
            trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=0.9)
            stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=0.9)
            print("HDI computed successfully using percentiles!") # Updated print statement
        else:
            trend_hdi = None
            stationary_hdi = None
            print("Not enough simulation smoother draws to compute HDI (need at least 2).")


    print(f"Synthetic data (NumPy) shape: {y_np.shape}")
    print(f"Synthetic data (JAX) shape: {y_jax.shape}")
    print(f"Trend component draws shape: {trend_draws.shape}")
    print(f"Stationary component draws shape: {stationary_draws.shape}")

    # --- Add calls to plotting functions here ---
    # Only attempt plotting if extraction was successful and produced draws
    if trend_draws.shape[0] > 0 and stationary_draws.shape[0] > 0:
        print("Generating plots...")

        # You can provide variable names if you have them, otherwise defaults are used.
        # For the synthetic data example, let's just use the defaults.
        variable_names = [f'Var {i+1}' for i in range(n_vars)] # Example variable names

        # Plot estimated vs true components
        plot_decomposition_results(
            y_np=y_np,
            trends_true_np=trends_np,
            stationary_true_np=stationary_np,
            trend_draws=trend_draws,
            stationary_draws=stationary_draws,
            trend_hdi=trend_hdi,
            stationary_hdi=stationary_hdi,
            variable_names=variable_names
        )

        # Plot observed data and estimated trend
        plot_observed_and_trend(
            y_np=y_np,
            trend_draws=trend_draws,
            trend_hdi=trend_hdi,
            variable_names=variable_names
        )

        print("Plotting complete.")
    else:
        print("Skipping plotting due to insufficient extracted draws.")
    # --- End plotting calls ---


    return {
        'mcmc': mcmc,
        'y_synthetic_np': y_np,
        'y_synthetic_jax': y_jax,
        'trends_true_np': trends_np,
        'stationary_true_np': stationary_np,
        'trend_draws': trend_draws,
        'stationary_draws': stationary_draws,
        'trend_hdi': trend_hdi,
        'stationary_hdi': stationary_hdi
    }

if __name__ == "__main__":
    results = example_usage()