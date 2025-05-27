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
import numpy as np # Use standard numpy for simulation
import jax.scipy.linalg as jsl
from jax.typing import ArrayLike # Import ArrayLike

# *** Refined device count setting ***
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

# Import functions and jitter from your stationary prior module
# Ensure stationary_prior_jax_simplified.py is in your Python path
try:
    from gpmcore.stationary_prior_jax_simplified import (
        AtoP_jax, rev_mapping_jax, make_stationary_var_transformation_jax,
        quad_form_sym_jax, # Import quad_form_sym_jax
        _JITTER # Use the jitter from your stationary prior module
    )
except ImportError:
    print("Error: Could not import from gpmcore.stationary_prior_jax_simplified.py")
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
    from gpmcore.Kalman_filter_jax import KalmanFilter, _KF_JITTER # Import KalmanFilter and its jitter
except ImportError:
    print("Error: Could not import from gpmcore.Kalman_filter_jax.py")
    print("Please ensure the file exists and is in your Python path.")
    # Define dummy values to allow code parsing to continue, but it will fail at runtime
    _KF_JITTER = 1e-8
    class KalmanFilter:
        def __init__(self, *args, **kwargs): raise NotImplementedError("Import failed")


# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64 # Use JAX default dtype

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


# --- Simulation Smoother (based on simple_idea.py logic) ---
# Note: This simulation smoother uses smoothed states/covs directly, which is
# different from the standard algorithm but matches the logic in your simple_idea.py.
# It also requires the state innovation covariance Q.

def simulation_smoother_step(carry, x):
    """Single step of simulation smoother (based on simple_idea.py logic)"""
    # carry: (next_state, F, Q, key) -> Q is the state innovation covariance matrix (R_aug @ R_aug.T)
    # x: (smooth_mean_t, smooth_cov_t) for time t

    next_state_t_plus_1, F, Q, key = carry
    smooth_mean_t, smooth_cov_t = x # Smoothed mean and cov at time t

    # Predict state at t+1 based on smoothed state at t
    pred_mean_t_plus_1_from_t = F @ smooth_mean_t
    pred_cov_t_plus_1_from_t = F @ smooth_cov_t @ F.T + Q # Predicted cov using smoothed cov at t
    # Ensure symmetry for numerical stability
    pred_cov_t_plus_1_from_t = (pred_cov_t_plus_1_from_t + pred_cov_t_plus_1_from_t.T) / 2.0


    # Smoother gain (as defined in simple_idea.py, using smoothed values)
    # A = smooth_cov_t @ F.T @ linalg.solve(pred_cov_t_plus_1_from_t, jnp.eye(pred_cov_t_plus_1_from_t.shape[0]))
    # Use safe solve with jitter - using _KF_JITTER as it's related to Kalman Filter operations
    pred_cov_t_plus_1_reg = pred_cov_t_plus_1_from_t + _KF_JITTER * jnp.eye(pred_cov_t_plus_1_from_t.shape[0], dtype=_DEFAULT_DTYPE)

    try:
        # Attempt standard solve with PSD assumption
        A = smooth_cov_t @ F.T @ jax.scipy.linalg.solve(pred_cov_t_plus_1_reg, jnp.eye(pred_cov_t_plus_1_reg.shape[0]), assume_a='pos')
        solve_ok = jnp.all(jnp.isfinite(A))
        # If standard solve fails, try pinv as a fallback
        def pinv_fallback(ops):
             smooth_cov_loc, F_loc, pred_cov_reg_loc = ops
             return smooth_cov_loc @ F_loc.T @ jnp.linalg.pinv(pred_cov_reg_loc, rcond=1e-6)
        A = jax.lax.cond(solve_ok, lambda ops: ops[0], pinv_fallback, operand=(A, smooth_cov_t, F, pred_cov_t_plus_1_reg))
        # Final check for NaNs/Infs after potential fallback
        A = jnp.where(jnp.all(jnp.isfinite(A)), A, jnp.zeros_like(A)) # Use zero gain if all attempts fail

    except Exception:
        # If even the cond+pinv structure fails unexpectedly, return zero gain
        A = jnp.zeros_like(smooth_cov_t @ F.T)
        # print("Warning: Kalman gain computation failed in simulation smoother step. Using zero gain.")


    # Conditional distribution: p(x_t | x_{t+1}, Y_T) ~ N(m_t, V_t) as implemented in simple_idea.py
    # m_t = smooth_mean_t + A @ (x_{t+1} - F @ smooth_mean_t)
    # V_t = smooth_cov_t - A @ pred_cov_t_plus_1_from_t @ A.T

    # Calculate conditional mean and covariance
    cond_mean_t = smooth_mean_t + A @ (next_state_t_plus_1 - pred_mean_t_plus_1_from_t)
    # Ensure pred_cov is symmetric before quad_form_sym
    pred_cov_t_plus_1_from_t_sym = (pred_cov_t_plus_1_from_t + pred_cov_t_plus_1_from_t.T)/2.0
    cond_cov_t = smooth_cov_t - quad_form_sym_jax(pred_cov_t_plus_1_from_t_sym, A.T) # J @ P @ J.T = quad_form_sym(P, J.T)
    cond_cov_t = (cond_cov_t + cond_cov_t.T) / 2.0 # Ensure symmetry

    # Handle potential NaNs/Infs in mean/cov
    cond_mean_t = jnp.where(jnp.isfinite(cond_mean_t), cond_mean_t, smooth_mean_t) # Fallback to smoothed mean
    cond_cov_t = jnp.where(jnp.all(jnp.isfinite(cond_cov_t)), cond_cov_t, smooth_cov_t) # Fallback to smoothed cov

    # Ensure positive definiteness of conditional covariance for sampling
    cond_cov_t_reg = cond_cov_t + _KF_JITTER * jnp.eye(cond_cov_t.shape[0], dtype=_DEFAULT_DTYPE)
    cond_cov_t_reg = (cond_cov_t_reg + cond_cov_t_reg.T) / 2.0 # Ensure symmetry


    # Sample from conditional distribution
    key, subkey = random.split(key)
    try:
        sampled_state_t = random.multivariate_normal(subkey, cond_mean_t, cond_cov_t_reg)
        sample_ok = jnp.all(jnp.isfinite(sampled_state_t))
        sampled_state_t = jnp.where(sample_ok, sampled_state_t, cond_mean_t) # Fallback to mean if sample fails
    except Exception:
        # If sampling fails even with regularized cov, fallback to mean
        sampled_state_t = cond_mean_t
        # print("Warning: Sampling failed in simulation smoother step. Using conditional mean.")


    # The carry for the next step (t-1) is the sampled state at time t
    new_carry = (sampled_state_t, F, Q, key)
    return new_carry, sampled_state_t


def simulation_smoother(smoothed_means: jnp.ndarray, smoothed_covs: jnp.ndarray,
                       F: jnp.ndarray, Q: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """Simulation smoother for drawing states (implementation based on simple_idea.py)"""
    T = smoothed_means.shape[0]

    # Handle empty time series
    if T == 0:
        return jnp.empty((0, F.shape[0]), dtype=_DEFAULT_DTYPE)

    # Sample final state x_{T-1} ~ N(smoothed_means[T-1], smoothed_covs[T-1])
    key, subkey = random.split(key)
    # Ensure final smoothed_cov is symmetric and regularized for sampling
    final_smooth_cov_reg = smoothed_covs[T-1] + _KF_JITTER * jnp.eye(smoothed_covs[T-1].shape[0], dtype=_DEFAULT_DTYPE)
    final_smooth_cov_reg = (final_smooth_cov_reg + final_smooth_cov_reg.T) / 2.0 # Ensure symmetry

    try:
        final_state_simple = random.multivariate_normal(
            subkey, smoothed_means[T-1], final_smooth_cov_reg
        )
        sample_ok_final_simple = jnp.all(jnp.isfinite(final_state_simple))
        final_state_simple = jnp.where(sample_ok_final_simple, final_state_simple, smoothed_means[T-1]) # Fallback to mean if sample fails
    except Exception:
         # Fallback if sampling fails even with regularized cov
         final_state_simple = smoothed_means[T-1]
         # print("Warning: Sampling final state failed in simulation smoother. Using smoothed mean.")


    # Backward simulation from t = T-2 down to 0
    # Scan inputs: smoothed_means[T-2]...smoothed_means[0] and smoothed_covs[T-2]...smoothed_covs[0]
    rev_smooth_means = smoothed_means[:-1][::-1] # smoothed_means[T-2]...smoothed_means[0]
    rev_smooth_covs = smoothed_covs[:-1][::-1]   # smoothed_covs[T-2]...smoothed_covs[0]

    # Handle case where T <= 1 (no steps for backward scan)
    if T <= 1:
        # Just return the final state sampled (which is the only state if T=1)
        return final_state_simple[None, :] # Add time dimension


    # Initial carry: (sampled state at T-1, F, Q, key)
    init_carry_sim_simple = (final_state_simple, F, Q, key)

    # Run the backward simulation scan (using the simple_idea.py step function)
    # Scan over the reversed smoothed means/covs from time T-2 down to 0
    # The scan receives (smooth_mean_t, smooth_cov_t) in reversed time order.
    final_carry_sim_simple, sampled_states_rev_simple = lax.scan(
        simulation_smoother_step, # This uses the simplified logic
        init_carry_sim_simple,
        (rev_smooth_means, rev_smooth_covs)
    )

    # Concatenate: reverse the draws for 0..T-2 and prepend the draw for T-1.
    sampled_states_simple = jnp.concatenate([
        sampled_states_rev_simple[::-1], # Times 0 to T-2
        final_state_simple[None, :]      # Time T-1 (using 0-based indexing for T steps)
    ])

    return sampled_states_simple


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


    # Return F, R_aug, C_mat, H_obs for KalmanFilter constructor
    # Also return Q_mat for the simulation smoother (as per simple_idea.py's simulation_smoother signature)
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


    # Create parameter structure using transformed A
    params = BVARParams(A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta, Sigma_eps=Sigma_eps)

    # --- State Space Representation ---
    # build_state_space_matrices returns F, R_aug, C_mat, H_obs, Q_mat
    F, R_aug, C_mat, H_obs, Q_mat = build_state_space_matrices(params, n_vars, n_lags)


    # Initial state (diffuse prior for trends, zero for VAR states)
    # Matches simple_idea.py's approach
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6  # Diffuse prior for all states initially
    # Tight prior for VAR states (indices n_vars onwards)
    init_cov = init_cov.at[n_vars:, n_vars:].set(jnp.eye(n_vars * n_lags, dtype=_DEFAULT_DTYPE) * 1e-6)
    # Ensure init_cov is symmetric and positive definite for the KF
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


    # --- Kalman Filter for Likelihood ---
    # Instantiate KalmanFilter class
    # Note: Your KalmanFilter class expects R to be the shock impact matrix,
    # H to be the observation noise covariance, init_x/init_P, and T/C.
    # build_state_space_matrices provides F (T), R_aug (R), C_mat (C), H_obs (H).
    kf = KalmanFilter(T=F, R=R_aug, C=C_mat, H=H_obs, init_x=init_mean, init_P=init_cov)

    # Determine static observation info (assuming no NaNs or static NaNs in y)
    # For this simple model y = trend + stationary, we observe all variables at all times.
    # If y has NaNs, we need to identify which elements are *never* NaN across the time series.
    # For simplicity of the example, assume no NaNs in the input `y`.
    # If NaNs were present, we would need to preprocess y to find observed indices.

    # Assuming no NaNs in y for this example:
    n_obs_actual = n_vars # All variables are observed
    valid_obs_idx = jnp.arange(n_vars, dtype=int)
    C_obs = C_mat # If all observed, C_obs is the full C_mat
    H_obs_actual = H_obs # If all observed, H_obs_actual is the full H_obs
    I_obs = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)


    # Compute log-likelihood using the Kalman Filter
    # If transformation failed, loglik should be very low.
    # Also check if state space matrices built correctly (no NaNs)
    state_space_nan = jnp.any(jnp.isnan(F)) | jnp.any(jnp.isnan(R_aug)) | jnp.any(jnp.isnan(C_mat)) | jnp.any(jnp.isnan(H_obs)) | jnp.any(jnp.isnan(init_mean)) | jnp.any(jnp.isnan(init_cov))

    # The kf.log_likelihood method takes static observation info
    loglik = jax.lax.cond(
        transformation_failed | state_space_nan, # Check if transformation or matrix building failed
        lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE), # Assign -inf loglik if either failed
        lambda: kf.log_likelihood(y, valid_obs_idx, n_obs_actual, C_obs, H_obs_actual, I_obs)
    )

    # Add likelihood to the model
    numpyro.factor("loglik", loglik)


# --- Fitting Function ---

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


# --- Extraction Function (using Simulation Smoother) ---

def extract_trends_and_components(mcmc, y: jnp.ndarray, n_lags: int = 2,
                                 num_draws: int = 100, rng_key: jnp.ndarray = random.PRNGKey(42)):
    """Extract trend and stationary components using simulation smoother"""
    samples = mcmc.get_samples()
    T, n_vars = y.shape
    state_dim = n_vars + n_vars * n_lags

    # Storage for draws
    trend_draws = []
    stationary_draws = []

    # Select subset of posterior draws
    # Ensure num_draws doesn't exceed available samples
    n_posterior = len(samples['A_vec_raw'])
    num_draws = min(num_draws, n_posterior)
    # Select indices, handling the case where num_draws is 0 or negative
    if num_draws > 0:
        # Use np.linspace with retstep=False to avoid precision issues with floating point stops
        # and ensure integer indices
        draw_indices_float = np.linspace(0, n_posterior - 1, num_draws)
        draw_indices = np.round(draw_indices_float).astype(int)
    else:
        draw_indices = np.array([], dtype=int) # Empty array if no draws requested


    # Determine static observation info (assuming no NaNs or static NaNs in y)
    # This needs to be consistent with how the model was fitted.
    # Assuming no NaNs in y for this example:
    n_obs_actual = n_vars # All variables are observed
    valid_obs_idx = jnp.arange(n_vars, dtype=int)
    I_obs = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)


    # We need smoothed states/covs, F, and Q for the simulation smoother (based on simple_idea.py logic).
    # Need to run Kalman Filter *and* Smoother for each posterior draw to get smoothed_means/covs.
    # F and Q (state innovation cov) are constructed from parameters for each draw.

    for i, idx in enumerate(draw_indices):
        # Extract parameters for this draw
        A_vec_raw_draw = samples['A_vec_raw'][idx]
        sigma_u_draw = samples['sigma_u'][idx]
        Omega_u_chol_draw = samples['Omega_u_chol'][idx]
        sigma_eta_draw = samples['sigma_eta'][idx]
        Omega_eta_chol_draw = samples['Omega_eta_chol'][idx]
        # Sigma_eps is assumed None or zero based on the model definition

        # Compute Sigma_u and Sigma_eta
        Sigma_u_draw = jnp.diag(sigma_u_draw) @ Omega_u_chol_draw @ Omega_u_chol_draw.T @ jnp.diag(sigma_u_draw)
        # Use _JITTER from stationary_prior_jax_simplified
        Sigma_u_draw = (Sigma_u_draw + Sigma_u_draw.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) # Ensure PSD

        Sigma_eta_draw = jnp.diag(sigma_eta_draw) @ Omega_eta_chol_draw @ Omega_eta_chol_draw.T @ jnp.diag(sigma_eta_draw)
        # Use _JITTER from stationary_prior_jax_simplified
        Sigma_eta_draw = (Sigma_eta_draw + Sigma_eta_draw.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) # Ensure PSD


        # Reshape A_vec_raw into raw_A_list for the stationary transformation
        raw_A_list_draw = [A_vec_raw_draw[i*n_vars*n_vars:(i+1)*n_vars*n_vars].reshape((n_vars, n_vars)) for i in range(n_lags)]

        # Apply stationary transformation to get A (phi_list)
        phi_list_draw, _ = make_stationary_var_transformation_jax(Sigma_u_draw, raw_A_list_draw, n_vars, n_lags)

        # Check if transformation resulted in NaNs for this draw
        transformation_failed_draw = jnp.any(jnp.stack([jnp.any(jnp.isnan(phi)) for phi in phi_list_draw]))

        # If transformation failed, skip this draw's extraction
        if transformation_failed_draw:
             print(f"Warning: Stationary transformation failed for draw {idx}. Skipping extraction for this draw.")
             continue # Skip to next draw

        A_transformed_draw = jnp.stack(phi_list_draw)

        params_draw = BVARParams(A=A_transformed_draw, Sigma_u=Sigma_u_draw, Sigma_eta=Sigma_eta_draw, Sigma_eps=None) # Sigma_eps is None

        # --- State Space Representation for this draw ---
        # build_state_space_matrices returns F, R_aug, C_mat, H_obs, Q_mat
        F_draw, R_aug_draw, C_mat_draw, H_obs_draw, Q_mat_draw = build_state_space_matrices(params_draw, n_vars, n_lags)

        # Check if state space matrices built correctly (no NaNs) for this draw
        # Note: Checking R_aug might not be needed if Q_mat is checked, as Q = R_aug @ R_aug.T
        state_space_nan_draw = jnp.any(jnp.isnan(F_draw)) | jnp.any(jnp.isnan(C_mat_draw)) | jnp.any(jnp.isnan(H_obs_draw)) | jnp.any(jnp.isnan(Q_mat_draw))

        if state_space_nan_draw:
             print(f"Warning: State space matrices contain NaNs for draw {idx}. Skipping extraction for this draw.")
             continue # Skip to next draw


        # Initial state (consistent with model)
        init_mean_draw = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
        init_cov_draw = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6
        init_cov_draw = init_cov_draw.at[n_vars:, n_vars:].set(jnp.eye(n_vars * n_lags, dtype=_DEFAULT_DTYPE) * 1e-6)
        # Ensure init_cov is symmetric and positive definite for the KF
        init_cov_draw = (init_cov_draw + init_cov_draw.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


        # --- Run Kalman Filter and Smoother ---
        # Instantiate KalmanFilter for this draw using the imported class
        kf_draw = KalmanFilter(T=F_draw, R=R_aug_draw, C=C_mat_draw, H=H_obs_draw, init_x=init_mean_draw, init_P=init_cov_draw)

        # Run smoother to get smoothed states/covs
        # Provide static observation info required by the smooth method
        # The smooth method internally calls filter, so we don't need to call filter separately first.
        try:
             # Use the smooth method from the imported KalmanFilter class
             smoothed_means_draw, smoothed_covs_draw = kf_draw.smooth(y, filter_results=None, # Let smooth compute filter internally
                                                                     static_valid_obs_idx=valid_obs_idx,
                                                                     static_n_obs_actual=n_obs_actual,
                                                                     static_C_obs_for_filter=C_mat_draw, # Pass C_mat
                                                                     static_H_obs_for_filter=H_obs_draw, # Pass H_obs
                                                                     static_I_obs_for_filter=I_obs)

             # Check smoothed outputs for NaNs/Infs before simulation smoothing
             if jnp.any(jnp.isnan(smoothed_means_draw)) or jnp.any(jnp.isinf(smoothed_means_draw)) or \
                jnp.any(jnp.isnan(smoothed_covs_draw)) or jnp.any(jnp.isinf(smoothed_covs_draw)):
                  print(f"Warning: Kalman smoother produced NaNs/Infs for draw {idx}. Skipping extraction for this draw.")
                  continue # Skip to next draw

        except Exception as e:
             print(f"Warning: Kalman smoother failed for draw {idx} with error {e}. Skipping extraction for this draw.")
             continue # Skip to next draw


        # --- Run Simulation Smoother (using simple_idea.py logic) ---
        # This requires F, Q, smoothed_means, smoothed_covs, and a rng_key
        rng_key, sim_key = random.split(rng_key) # Use a new key for each simulation draw
        states_draw = simulation_smoother(
            smoothed_means_draw, smoothed_covs_draw, F_draw, Q_mat_draw, sim_key
        )

        # Check simulation smoother output for NaNs/Infs
        if jnp.any(jnp.isnan(states_draw)) or jnp.any(jnp.isinf(states_draw)):
             print(f"Warning: Simulation smoother produced NaNs/Infs for draw {idx}. Skipping extraction for this draw.")
             continue # Skip to next draw


        # Extract trends and stationary components from the sampled state vector
        # State is [tau_t; X_t; X_{t-1}; ...; X_{t-p+1}]
        trends = states_draw[:, :n_vars]
        stationary = states_draw[:, n_vars:n_vars*2]  # Current VAR states (X_t)

        # Append the extracted components for this successful draw
        trend_draws.append(trends)
        stationary_draws.append(stationary)

    # Stack draws if any were collected
    if len(trend_draws) > 0:
        trend_draws = jnp.stack(trend_draws)  # (effective_num_draws, T, n_vars)
        stationary_draws = jnp.stack(stationary_draws)  # (effective_num_draws, T, n_vars)
    else:
        # Return empty arrays with correct shape if no draws were possible
        # Determine T and n_vars from the input y
        T_in, n_vars_in = y.shape
        trend_draws = jnp.empty((0, T_in, n_vars_in), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T_in, n_vars_in), dtype=_DEFAULT_DTYPE)


    return trend_draws, stationary_draws


# --- HDI Computation ---

def compute_hdi_with_arviz(draws: jnp.ndarray, hdi_prob: float = 0.94):
    """Compute HDI using ArviZ directly on draws."""
    # Requires at least 2 draws
    if draws.shape[0] < 2:
        print("Warning: Not enough draws to compute HDI. Need at least 2.")
        # Return NaNs with shape (T, n_vars)
        # Shape of draws is (num_draws, T, n_vars). We need shape (T, n_vars).
        hdi_nan_shape = draws.shape[1:]
        return {'low': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE),
                'high': jnp.full(hdi_nan_shape, jnp.nan, dtype=_DEFAULT_DTYPE)}


    # ArviZ expects array-like input for hdi. Our draws are (num_draws, T, n_vars).
    # By default, ArviZ's hdi often treats the first dimension as the sample dimension.
    # The result should be an xarray DataArray with dimensions corresponding to
    # the non-sample dimensions (time, variable) and a new 'interval' dimension.

    try:
        # Call az.hdi directly on the JAX array.
        # Pass dimensions explicitly for clarity and robustness if needed,
        # but let's try without first as it often works.
        # hdi_result = az.hdi(draws, hdi_prob=hdi_prob, dims=['draw', 'time', 'variable']) # Alternative with explicit dims

        hdi_result = az.hdi(draws, hdi_prob=hdi_prob) # Simple call

        # The resulting DataArray `hdi_result` is expected to have dimensions like ('time', 'variable', 'interval')
        # Access the values as a NumPy array (ArviZ returns NumPy arrays from .values)
        hdi_values = hdi_result.values # Shape (T, n_vars, 2)

        # Extract low and high bounds
        hdi_low = hdi_values[:, :, 0] # Shape (T, n_vars)
        hdi_high = hdi_values[:, :, 1] # Shape (T, n_vars)

        # Convert back to JAX arrays if desired, although NumPy arrays from ArviZ are usually fine for plotting etc.
        return {'low': jnp.asarray(hdi_low), 'high': jnp.asarray(hdi_high)}

    except Exception as e:
        print(f"Warning: ArviZ HDI computation failed with error: {e}. Returning NaNs.")
        # Return NaNs with correct shape (T, n_vars) in case of ArviZ error
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
    num_samples = 500
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

    print("Synthetic data generated.")

    print("Fitting BVAR with trends...")

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


    mcmc = fit_bvar_trends(y_jax, n_lags=n_lags, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains_to_use, rng_key=random.PRNGKey(1))

    print("MCMC fitting complete.")

    print("Extracting trend and stationary components using Simulation Smoother...")

    # Extract components
    # Use a new key for simulation smoother draws
    # Ensure num_draws is reasonable compared to total posterior draws
    total_posterior_draws = mcmc.get_samples()['A_vec_raw'].shape[0]
    num_extract_draws = min(num_extract_draws_desired, total_posterior_draws)
    if num_extract_draws < num_extract_draws_desired:
         print(f"Warning: Reducing requested simulation smoother draws from {num_extract_draws_desired} to {num_extract_draws} as only {total_posterior_draws} posterior samples are available.")

    trend_draws, stationary_draws = extract_trends_and_components(mcmc, y_jax, n_lags=n_lags, num_draws=num_extract_draws, rng_key=random.PRNGKey(2))

    print("Component extraction complete.")

    print("Computing HDI intervals...")

    # Compute HDI - only if enough draws were extracted
    if trend_draws.shape[0] > 1:
        trend_hdi = compute_hdi_with_arviz(trend_draws, hdi_prob=0.9)
        stationary_hdi = compute_hdi_with_arviz(stationary_draws, hdi_prob=0.9)
        print("HDI computed successfully!")
    else:
        trend_hdi = None
        stationary_hdi = None
        print("Not enough simulation smoother draws to compute HDI (need at least 2).")


    print(f"Synthetic data (NumPy) shape: {y_np.shape}")
    print(f"Synthetic data (JAX) shape: {y_jax.shape}")
    print(f"Trend component draws shape: {trend_draws.shape}")
    print(f"Stationary component draws shape: {stationary_draws.shape}")


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