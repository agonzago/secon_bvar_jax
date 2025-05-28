# --- START OF FILE gpm_test_utils.py ---

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
# Removed matplotlib.pyplot as plt - plotting is now handled by reporting_plots
from typing import Dict, List, Optional, Tuple
import time

# Import necessary components from other modules
from gpm_parser import GPMParser, GPMModel
from gpm_bvar_trends import (
    GPMStateSpaceBuilder, EnhancedBVARParams,
    _DEFAULT_DTYPE, _JITTER, _KF_JITTER,
    _has_measurement_error, _sample_measurement_covariance,
    _create_initial_covariance # Need this for the fallback option
)
from simulation_smoothing import jarocinski_corrected_simulation_smoother
from Kalman_filter_jax import KalmanFilter # Assuming KalmanFilter is importable

# Import stationary transformation
try:
    from stationary_prior_jax_simplified import make_stationary_var_transformation_jax
except ImportError:
    print("Warning: Could not import stationary transformation (make_stationary_var_transformation_jax)")
    make_stationary_var_transformation_jax = None

# Import plotting functions from reporting_plots.py
try:
    from reporting_plots import (
        plot_observed_vs_true_decomposition, # Useful if you have true components later
        plot_observed_and_trend,
        plot_observed_and_fitted,
        plot_estimated_components
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Could not import plotting functions from reporting_plots.py.")
    PLOTTING_AVAILABLE = False


# Helper function: Build initial mean for testing (no sampling)
def _build_initial_mean_for_test(gpm_model: GPMModel, state_dim: int) -> jnp.ndarray:
    """
    Builds the initial state mean vector for testing with fixed parameters.
    Uses the mean of specified initial value priors where available,
    otherwise defaults to zero.
    """
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)

    # Assuming trends are the first n_trends states in the state vector
    n_trends = len(gpm_model.trend_variables)
    n_stationary = len(gpm_model.stationary_variables)
    var_order = gpm_model.var_prior_setup.var_order if gpm_model.var_prior_setup else 1
    n_stat_states = n_stationary * var_order # This is the size of the VAR state block

    # Iterate through specified initial values in the GPM file
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]

            # Find the index of this variable in the state space vector
            state_idx = -1 # Default to not found

            if var_name in gpm_model.trend_variables:
                idx = gpm_model.trend_variables.index(var_name)
                if idx < n_trends:
                    state_idx = idx
            elif var_name in gpm_model.stationary_variables and var_order > 0:
                # Assuming stationary variables map to the first block of VAR states (lag 0)
                # This needs careful mapping depending on how your state vector is structured
                # GPMStateSpaceBuilder seems to structure it as [Trends, Stat_t, Stat_t-1, ...]
                # So Stat_t should be at indices n_trends to n_trends + n_stationary - 1
                 idx = gpm_model.stationary_variables.index(var_name)
                 if idx < n_stationary: # Ensure index is within the first lag block
                      state_idx = n_trends + idx


            # If found and within bounds, set the mean
            if state_idx != -1 and state_idx < state_dim:
                init_mean = init_mean.at[state_idx].set(jnp.array(mean, dtype=_DEFAULT_DTYPE)) # Ensure dtype
            elif state_idx == -1:
                 print(f"Warning: Initial value specified for variable '{var_name}' not found in state vector or mapping is ambiguous. Skipping setting mean.")
            else: # state_idx >= state_dim
                 print(f"Warning: Computed state index {state_idx} for variable '{var_name}' is out of bounds ({state_dim}). Skipping setting mean.")


    # For states not covered by GPM initval, the mean remains the initialized zero.
    return init_mean

# Helper function: Build initial covariance for testing (deterministic)
def _build_initial_covariance_for_test(
    state_dim: int, n_trends: int,
    gamma_list: List[jnp.ndarray],
    n_stationary: int, var_order: int,
    gamma_scaling: float = 1.0 # Added scaling parameter
) -> jnp.ndarray:
    """
    Builds the initial state covariance matrix for testing with fixed parameters.
    Uses diffuse prior for trends and theoretical VAR unconditional covariance
    for the stationary part based on the provided gamma_list, with optional scaling.
    """
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    # Diffuse prior for trends (unchanged)
    if n_trends > 0:
        init_cov = init_cov.at[:n_trends, :n_trends].set(
            jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * 1e6
        )

    # Use theoretical VAR covariances for stationary components if available
    if len(gamma_list) > 0 and n_stationary > 0 and var_order > 0:
        var_start = n_trends
        var_state_dim = n_stationary * var_order

        # Build VAR state covariance matrix using gamma matrices
        var_state_cov = jnp.zeros((var_state_dim, var_state_dim), dtype=_DEFAULT_DTYPE)

        try:
            # Check if gamma_list has enough elements and gamma_list[0] is valid
            gamma_0 = gamma_list[0] if len(gamma_list) > 0 else None
            is_gamma_0_valid = (
                gamma_0 is not None and
                gamma_0.shape == (n_stationary, n_stationary) and
                jnp.all(jnp.isfinite(gamma_0)) and
                jnp.all(jnp.diag(gamma_0) > _JITTER) # Check for positive diagonal
            )

            if is_gamma_0_valid:
                for i in range(var_order):
                    for j in range(var_order):
                        lag_diff = abs(i - j)

                        # Use gamma matrix if available, otherwise use contemporaneous covariance
                        if lag_diff < len(gamma_list):
                            # Ensure gamma matrix for this lag is valid
                            gamma_lag = gamma_list[lag_diff]
                            is_gamma_lag_valid = (
                                gamma_lag is not None and
                                gamma_lag.shape == (n_stationary, n_stationary) and
                                jnp.all(jnp.isfinite(gamma_lag))
                            )

                            if is_gamma_lag_valid:
                                block_cov = gamma_lag * gamma_scaling # Apply scaling
                                if i > j:
                                    block_cov = block_cov.T # Transpose scaled block
                            else:
                                # Fallback if specific gamma(lag) is bad but gamma(0) is OK
                                decay_factor = 0.5 ** lag_diff
                                block_cov = gamma_0 * gamma_scaling * decay_factor # Apply scaling here too
                                if i > j:
                                     block_cov = block_cov.T

                        else:
                            # Fallback for lags beyond available gamma matrices
                            decay_factor = (0.5 ** lag_diff)
                            block_cov = gamma_0 * gamma_scaling * decay_factor # Apply scaling here too
                            if i > j:
                                 block_cov = block_cov.T


                        # Insert block into VAR state covariance
                        row_start, row_end = i * n_stationary, (i + 1) * n_stationary
                        col_start, col_end = j * n_stationary, (j + 1) * n_stationary

                        # Ensure calculated block_cov is the right shape before inserting
                        if block_cov.shape == (n_stationary, n_stationary):
                            var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)
                        else:
                            print(f"Warning: Calculated block_cov shape mismatch ({block_cov.shape} vs ({n_stationary}, {n_stationary})). Skipping block ({i},{j}).")


                # Insert VAR covariance into full state covariance matrix
                var_end = var_start + var_state_dim
                if var_end <= state_dim:
                     init_cov = init_cov.at[var_start:var_end, var_start:var_end].set(var_state_cov)
            else:
                 print("Warning: gamma_list[0] is invalid or missing. Using a moderate default for VAR initial covariance.")
                 # Fallback if gamma[0] is bad
                 if var_state_dim > 0:
                     init_cov = init_cov.at[var_start:var_start + var_state_dim,
                                           var_start:var_start + var_state_dim].set(
                         jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE) * 0.1 # Use a moderate value
                     )

        except Exception as e:
            print(f"Error building gamma-based covariance for test: {e}. Using a moderate default.")
            # Emergency fallback
            if var_state_dim > 0:
                init_cov = init_cov.at[var_start:var_start + var_state_dim,
                                      var_start:var_start + var_state_dim].set(
                    jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE) * 0.1
                )

    # Ensure positive definite with sufficient regularization
    # Increased jitter slightly for robustness in tests
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

    return init_cov


# Helper function: Build trend covariance from fixed params (or prior mode/mean)
def _build_trend_covariance(gpm_model: GPMModel, param_values: Dict[str, float]) -> jnp.ndarray:
    """Build trend innovation covariance matrix using fixed parameter values or prior modes/means"""
    n_trends = len(gpm_model.trend_variables)

    # Handle case with no trends
    if n_trends == 0:
        return jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    # Get individual shock standard deviations
    trend_sigmas = []
    for shock in gpm_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in param_values:
            sigma = param_values[param_name]
            trend_sigmas.append(sigma)
            #print(f"  Using fixed {param_name} = {sigma}") # Commented out for cleaner output
        elif shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            if prior_spec.distribution == 'inv_gamma_pdf' and len(prior_spec.params) >= 2:
                alpha, beta = prior_spec.params
                mode = beta / (alpha + 1.0) if alpha > -1.0 else beta
                sigma = mode if mode > 0 else 0.1 # Ensure mode is positive, fallback to 0.1
            elif prior_spec.distribution == 'normal_pdf' and len(prior_spec.params) >= 1:
                sigma = abs(prior_spec.params[0]) # Use abs of mean for a stddev
                sigma = sigma if sigma > 0 else 0.1 # Ensure positive, fallback to 0.1
            else:
                sigma = 0.1 # Fallback default
            trend_sigmas.append(sigma)
            #print(f"  Using prior mode/mean for {param_name} = {sigma}") # Commented out
        else:
            # Default fallback if not in estimated_params
            sigma = 0.1
            trend_sigmas.append(sigma)
            #print(f"  Using default for {param_name} = {sigma}") # Commented out

    # For simplicity, assume diagonal covariance for trends
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas, dtype=_DEFAULT_DTYPE) ** 2)

    return Sigma_eta

# Helper function: Build VAR parameters from fixed values (or prior mode/mean)
def _build_var_parameters(gpm_model: GPMModel, param_values: Dict[str, float]) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """Build VAR parameters using fixed values or prior modes/means and stationary transformation"""

    n_stationary = len(gpm_model.stationary_variables)
    # Handle case with no stationary variables
    if n_stationary == 0:
        # Return placeholder matrices with correct dtypes but minimal/zero dimensions
        A = jnp.empty((0, 0, 0), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
        gamma_list = []
        return Sigma_u, A, gamma_list

    if not gpm_model.var_prior_setup:
        # Fallback: simple VAR(0) or VAR(1) structure if setup is missing but stat vars exist
        print("Warning: VAR prior setup missing. Assuming simple VAR(0) structure for building.")
        A = jnp.zeros((1, n_stationary, n_stationary), dtype=_DEFAULT_DTYPE) # Treat as VAR(1) with A=0
        Sigma_u = jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE) * 0.1 # Default shock cov
        gamma_list = [Sigma_u] # Contemporaneous cov as gamma0
        return Sigma_u, A, gamma_list


    setup = gpm_model.var_prior_setup
    n_vars = n_stationary # n_vars here refers to stationary variables
    n_lags = setup.var_order

    # Use the hierarchical prior means from GPM specification (same as MCMC version)
    # Ensure es list is long enough or use fallback
    mean_diag = setup.es[0] if len(setup.es) > 0 else 0.5
    mean_offdiag = setup.es[1] if len(setup.es) > 1 else 0.3

    #print(f"Using hierarchical VAR prior structure for building:") # Commented out
    #print(f"  Diagonal mean: {mean_diag}, Off-diagonal mean: {mean_offdiag}") # Commented out
    #print(f"  Number of lags: {n_lags}") # Commented out

    # Create raw VAR coefficient matrices using prior means (SAME FOR ALL LAGS)
    raw_A_list = []
    for lag in range(n_lags):
        # Initialize with off-diagonal mean
        A_lag = jnp.full((n_vars, n_vars), mean_offdiag, dtype=_DEFAULT_DTYPE)
        # Set diagonal to diagonal mean
        A_lag = A_lag.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(mean_diag)
        raw_A_list.append(A_lag)

        #print(f"  Lag {lag+1}: diagonal = {jnp.diag(A_lag)}, off-diagonal = {mean_offdiag}") # Commented out

    # Build stationary innovation covariance
    stat_sigmas = []
    for shock in gpm_model.stationary_shocks:
        param_name = f"sigma_{shock}"
        if param_name in param_values:
            sigma = param_values[param_name]
            stat_sigmas.append(sigma)
            #print(f"  Using fixed {param_name} = {sigma}") # Commented out
        elif shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            if prior_spec.distribution == 'inv_gamma_pdf' and len(prior_spec.params) >= 2:
                alpha, beta = prior_spec.params
                mode = beta / (alpha + 1.0) if alpha > -1.0 else beta
                sigma = mode if mode > 0 else 0.2
            elif prior_spec.distribution == 'normal_pdf' and len(prior_spec.params) >= 1:
                 sigma = abs(prior_spec.params[0]) # Use abs of mean for a stddev
                 sigma = sigma if sigma > 0 else 0.2
            else:
                sigma = 0.2 # Fallback default
            stat_sigmas.append(sigma)
            #print(f"  Using prior mode/mean for {param_name} = {sigma}") # Commented out
        else:
            sigma = 0.2 # Default if not in estimated_params
            stat_sigmas.append(sigma)
            #print(f"  Using default for {param_name} = {sigma}") # Commented out


    # Handle case where there are stationary vars but no specified shocks
    if n_stationary > 0 and not stat_sigmas:
         print(f"Warning: {n_stationary} stationary variables defined but no stationary shocks found. Using default Sigma_u.")
         Sigma_u = jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE) * 0.2 # Default shock cov
    else:
         sigma_u = jnp.array(stat_sigmas, dtype=_DEFAULT_DTYPE)

         # Handle Omega_u (correlation part)
         # Assuming identity correlation for fixed parameter testing if not explicitly handled
         Omega_u_chol = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) # Default to identity correlation
         # if "Omega_u_chol" in param_values: # This would require fixing the cholesky value
         #     print("Warning: Fixed value for Omega_u_chol ignored. Using identity correlation.")

         Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
         # Ensure Sigma_u is symmetric and PSD
         Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)


    # Apply stationary transformation
    if make_stationary_var_transformation_jax is not None:
        try:
            #print(f"Applying stationary transformation...") # Commented out
            #print(f"  Sigma_u shape: {Sigma_u.shape}") # Commented out
            #print(f"  Raw A shapes: {[A.shape for A in raw_A_list]}") # Commented out

            # Check Sigma_u before passing
            if Sigma_u.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(Sigma_u)) or jnp.any(jnp.diag(Sigma_u) <= 0):
                 print("Warning: Sigma_u is problematic for stationary transformation. Using raw coefficients and fallback gamma.")
                 A_transformed = jnp.stack(raw_A_list)
                 gamma_list = [_safe_default_gamma0(n_vars) * 0.2] # Use a very safe default for gamma0
                 for lag in range(1, n_lags + 1):
                     gamma_list.append(gamma_list[0] * 0.5 ** lag) # Decay from safe gamma0
            else:
                # Ensure raw_A_list matrices are valid shape and finite
                if all(A_mat.shape == (n_vars, n_vars) and jnp.all(jnp.isfinite(A_mat)) for A_mat in raw_A_list):
                     phi_list, gamma_list = make_stationary_var_transformation_jax(
                        Sigma_u, raw_A_list, n_vars, n_lags
                    )
                     A_transformed = jnp.stack(phi_list)
                     #print(f"  Transformation successful!") # Commented out
                     #print(f"  Transformed A shape: {A_transformed.shape}") # Commented out
                     #print(f"  Transformed A[0] diagonal: {jnp.diag(A_transformed[0])}") # Commented out
                else:
                     print("Warning: Raw A matrices are problematic. Using raw coefficients and fallback gamma.")
                     A_transformed = jnp.stack(raw_A_list)
                     gamma_list = [_safe_default_gamma0(n_vars) * 0.2] # Use a very safe default for gamma0
                     for lag in range(1, n_lags + 1):
                         gamma_list.append(gamma_list[0] * 0.5 ** lag) # Decay from safe gamma0


        except Exception as e:
            print(f"  Stationary transformation failed: {e}. Using raw coefficients instead")
            A_transformed = jnp.stack(raw_A_list)

            # Create fallback gamma list
            gamma_list = [Sigma_u] # Use potentially problematic Sigma_u as gamma0
            # Check if Sigma_u is completely useless, if so use a safe default
            if Sigma_u.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(Sigma_u)):
                 print("  Sigma_u is problematic. Using a safer default for gamma list.")
                 gamma_list = [_safe_default_gamma0(n_vars) * 0.2] # Use a very safe default for gamma0

            # Add decaying terms based on the (possibly problematic) gamma0
            gamma0_for_decay = gamma_list[0]
            if gamma0_for_decay.shape != (n_vars, n_vars) or not jnp.all(jnp.isfinite(gamma0_for_decay)):
                 gamma0_for_decay = _safe_default_gamma0(n_vars) * 0.2 # Even safer decay base if gamma_list[0] is bad


            for lag in range(1, n_lags + 1):
                decay_factor = 0.7 ** lag
                gamma_list.append(gamma0_for_decay * decay_factor)


        return Sigma_u, A_transformed, gamma_list
    else:
        print("Stationary transformation not available, using raw coefficients")
        A_transformed = jnp.stack(raw_A_list)
        Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) * 0.1 # Fallback Sigma_u
        gamma_list = [Sigma_u] # Fallback gamma0
        return Sigma_u, A_transformed, gamma_list

def _safe_default_gamma0(n_vars: int) -> jnp.ndarray:
    """Provides a safe default for gamma_list[0] (unconditional covariance)"""
    return jnp.eye(n_vars, dtype=_DEFAULT_DTYPE) * 1.0 # Identity with variance 1.0


# Removed _plot_test_results - plotting is now done by reporting_plots

# Main function to run the test workflow
def test_gpm_model_with_parameters(gpm_file_path: str,
                                 y: jnp.ndarray,
                                 param_values: Dict[str, float],
                                 num_sim_draws: int = 50,
                                 rng_key: jnp.ndarray = random.PRNGKey(42),
                                 plot_results: bool = True,
                                 variable_names: Optional[List[str]] = None,
                                 use_gamma_init_for_test: bool = True, # <-- New toggle flag
                                 gamma_init_scaling: float = 1.0, # <-- New scaling for gamma init
                                 hdi_prob: float = 0.9 # Probability for HDI in plots
                                 ) -> Dict:
    """
    Test GPM model with fixed parameter values without running MCMC.
    Allows switching between gamma-based and standard 1e6/1e-6 initial covariance.
    Uses plotting functions from reporting_plots.py.
    """

    print(f"Testing GPM model: {gpm_file_path}")

    # Parse GPM file
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)

    T, n_obs = y.shape

    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.trend_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {gpm_model.observed_variables}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")

    # Build fixed parameters
    print("Building parameter structure with fixed values...")

    # Structural parameters (use fixed values or prior mean/mode)
    structural_params = {}
    for param_name in gpm_model.parameters:
        if param_name in gpm_model.estimated_params:
            if param_name in param_values:
                structural_params[param_name] = jnp.array(param_values[param_name], dtype=_DEFAULT_DTYPE)
                #print(f"  Using fixed {param_name} = {param_values[param_name]}") # Commented out
            else:
                # Use prior mean as default (if normal) or mode (if inv_gamma, etc.)
                prior_spec = gpm_model.estimated_params[param_name]
                if prior_spec.distribution == 'normal_pdf' and len(prior_spec.params) >= 1:
                     structural_params[param_name] = jnp.array(prior_spec.params[0], dtype=_DEFAULT_DTYPE)
                     #print(f"  Using prior mean for {param_name} = {prior_spec.params[0]}") # Commented out
                elif prior_spec.distribution == 'inv_gamma_pdf' and len(prior_spec.params) >= 2:
                    alpha, beta = prior_spec.params
                    mode = beta / (alpha + 1.0) if alpha > -1.0 else beta
                    structural_params[param_name] = jnp.array(mode, dtype=_DEFAULT_DTYPE)
                    #print(f"  Using prior mode for {param_name} = {mode}") # Commented out
                else:
                    structural_params[param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE) # Fallback default
                    #print(f"  Using default 0.0 for {param_name}") # Commented out

    # Build covariance matrices using fixed values (and get gamma_list)
    #print("\nBuilding trend and VAR parameters...") # Commented out
    Sigma_eta = _build_trend_covariance(gpm_model, param_values)
    Sigma_u, A_transformed, gamma_list = _build_var_parameters(gpm_model, param_values) # <-- gamma_list is returned here
    #print("Finished building trend and VAR parameters.\n") # Commented out

    # Measurement covariance
    Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None # This function is fine as it doesn't sample from a distribution


    # Initial conditions - Use dedicated functions for testing
    #print("Building initial conditions for testing...") # Commented out
    init_mean = _build_initial_mean_for_test(gpm_model, ss_builder.state_dim) # Use the new mean function

    # --- Initial Covariance Setup ---
    if use_gamma_init_for_test:
        print(f"Using gamma-based initial covariance (scaling={gamma_init_scaling})...")
        if ss_builder.n_stationary > 0:
             if len(gamma_list) > 0 and gamma_list[0] is not None and gamma_list[0].shape == (ss_builder.n_stationary, ss_builder.n_stationary):
                 print(f"  gamma_list[0] diagonal: {jnp.diag(gamma_list[0])}")
             else:
                 print("  gamma_list[0] is missing or malformed. Building gamma-based init_cov may fallback.")

        init_cov = _build_initial_covariance_for_test( # Use the new gamma-based cov function
            ss_builder.state_dim,
            ss_builder.n_trends,
            gamma_list, # Pass gamma_list
            ss_builder.n_stationary,
            ss_builder.var_order,
            gamma_scaling = gamma_init_scaling # Pass scaling
        )
        print(f"  Built init_cov diagonal: {jnp.diag(init_cov)}")
        try:
             cond_num = jnp.linalg.cond(init_cov)
             print(f"  Built init_cov condition number: {cond_num:.2e}")
        except Exception as e:
             print(f"  Could not compute init_cov condition number: {e}")


    else:
        print("Using standard 1e6/1e-6 initial covariance...")
        init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends) # Use the original 1e6/1e-6 function
        print(f"  Built init_cov diagonal: {jnp.diag(init_cov)}")
        try:
             cond_num = jnp.linalg.cond(init_cov)
             print(f"  Built init_cov condition number: {cond_num:.2e}")
        except Exception as e:
             print(f"  Could not compute init_cov condition number: {e}")
    # --- End Initial Covariance Setup ---


    print(f"Initial conditions shape: mean={init_mean.shape}, cov={init_cov.shape}")

    # Create parameter structure
    params = EnhancedBVARParams(
        A=A_transformed,
        Sigma_u=Sigma_u,
        Sigma_eta=Sigma_eta,
        structural_params=structural_params,
        Sigma_eps=Sigma_eps
    )

    # Build state space matrices
    print("Building state space matrices...")
    F, Q, C, H = ss_builder.build_state_space_matrices(params)

    # Check for numerical issues
    matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) &
                  jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) &
                  jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))

    if not matrices_ok:
        print("Warning: Some matrices contain non-finite values! Returning empty results.")
        # Return empty results with error message
        T_steps = y.shape[0]
        state_dim = ss_builder.state_dim
        n_trends = ss_builder.n_trends
        n_stationary = ss_builder.n_stationary
        return {
                'loglik': -jnp.inf,
                'error': 'Non-finite matrices',
                'filtered_means': jnp.empty((T_steps, state_dim), dtype=_DEFAULT_DTYPE),
                'filtered_covs': jnp.empty((T_steps, state_dim, state_dim), dtype=_DEFAULT_DTYPE),
                'smoothed_means': jnp.empty((T_steps, state_dim), dtype=_DEFAULT_DTYPE),
                'smoothed_covs': jnp.empty((T_steps, state_dim, state_dim), dtype=_DEFAULT_DTYPE),
                'sim_draws': jnp.empty((0, T_steps, state_dim), dtype=_DEFAULT_DTYPE),
                'trend_draws': jnp.empty((0, T_steps, n_trends), dtype=_DEFAULT_DTYPE),
                'stationary_draws': jnp.empty((0, T_steps, n_stationary), dtype=_DEFAULT_DTYPE),
                'trend_mean_smoothed': jnp.full((T_steps, n_trends), jnp.nan, dtype=_DEFAULT_DTYPE),
                'stationary_mean_smoothed': jnp.full((T_steps, n_stationary), jnp.nan, dtype=_DEFAULT_DTYPE),
                'F': F, 'Q': Q, 'C': C, 'H': H,
                'gpm_model': gpm_model,
                'ss_builder': ss_builder,
                'params': params
            }


    # Build R matrix from Q
    try:
        # Add jitter before cholesky
        Q_reg = Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        # Ensure symmetry
        Q_reg = (Q_reg + Q_reg.T) / 2.0
        R = jnp.linalg.cholesky(Q_reg)
    except Exception as e:
        print(f"Warning: Cholesky of Q failed ({e}). Using diagonal approximation for R.")
        R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER)) # Add jitter to diag as well


    # Create Kalman Filter
    kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)

    # Compute likelihood
    valid_obs_idx = jnp.arange(n_obs, dtype=int)
    I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)

    loglik = kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
    print(f"Log-likelihood: {loglik:.2f}")

    # Run filter and smoother for analysis
    print("Running Kalman filter and smoother...")

    # Filter
    filter_results = kf.filter(
        y,
        static_valid_obs_idx=valid_obs_idx,
        static_n_obs_actual=n_obs,
        static_C_obs=C,
        static_H_obs=H,
        static_I_obs=I_obs
    )

    # Smoother
    # Need to ensure filter_results is a dict for smooth method if it returns FilterResults object
    filter_results_dict = filter_results # Assuming filter returns dict


    smoothed_means, smoothed_covs = kf.smooth(
        y,
        filter_results=filter_results_dict, # Pass dictionary
        static_valid_obs_idx=valid_obs_idx,
        static_n_obs_actual=n_obs,
        static_C_obs_for_filter=C,
        static_H_obs_for_filter=H,
        static_I_obs_for_filter=I_obs
    )

    # Run simulation smoother
    print(f"Running simulation smoother with {num_sim_draws} draws...")
    sim_draws = []

    # Ensure init_cov is positive definite and symmetric before passing to smoother
    # This adds a final layer of safety after _build_initial_covariance_for_test or _create_initial_covariance
    init_cov_safe = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
    # Add extra safety check for init_cov_safe validity
    if not jnp.all(jnp.isfinite(init_cov_safe)):
         print("Warning: Initial covariance is not finite after regularization. Skipping simulation smoother.")
         sim_draws_stacked = jnp.empty((0, T, ss_builder.state_dim), dtype=_DEFAULT_DTYPE) # Return empty
    else:
        for i in range(num_sim_draws):
            if (i + 1) % 10 == 0:
                print(f"  Draw {i+1}/{num_sim_draws}")

            rng_key, sim_key = random.split(rng_key)

            try:
                sim_states = jarocinski_corrected_simulation_smoother(
                    y, F, R, C, H, init_mean, init_cov_safe, sim_key # Use the safe init_cov
                )
                # Add safety check for simulation results
                if not jnp.any(jnp.isnan(sim_states)) and not jnp.any(jnp.isinf(sim_states)):
                    sim_draws.append(sim_states)
                else:
                    print(f"Warning: Simulation draw {i+1} produced NaNs/Infs. Skipping.")

            except Exception as e:
                print(f"Warning: Simulation draw {i+1} failed ({e}). Skipping.")
                continue

        if len(sim_draws) > 0:
            sim_draws_stacked = jnp.stack(sim_draws)
            print(f"Successfully completed {len(sim_draws)} simulation draws")
        else:
            print("No simulation draws completed successfully")
            sim_draws_stacked = jnp.empty((0, T, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)


    # Extract components from simulation draws (only if draws are available)
    trend_draws = jnp.empty((0, T, ss_builder.n_trends), dtype=_DEFAULT_DTYPE)
    stationary_draws = jnp.empty((0, T, ss_builder.n_stationary), dtype=_DEFAULT_DTYPE)

    if sim_draws_stacked.shape[0] > 0:
        # Assuming state vector is [Trends, Stationary_t, Stationary_t-1, ..., Stationary_t-p+1]
        n_trends = ss_builder.n_trends
        n_stationary = ss_builder.n_stationary
        # Check if dimensions match expected
        expected_state_dim = n_trends + n_stationary * ss_builder.var_order
        if sim_draws_stacked.shape[-1] == expected_state_dim:
            trend_draws = sim_draws_stacked[:, :, :n_trends]
            stationary_draws = sim_draws_stacked[:, :, n_trends:n_trends + n_stationary]
        else:
             print(f"Warning: Simulation draws last dimension ({sim_draws_stacked.shape[-1]}) does not match expected state dimension ({expected_state_dim}). Cannot extract components.")


    # Also extract smoothed means for comparison (always available if filter/smoother ran)
    # Ensure smoothed means dimensions match expected
    n_trends = ss_builder.n_trends
    n_stationary = ss_builder.n_stationary
    expected_state_dim = n_trends + n_stationary * ss_builder.var_order
    if smoothed_means.shape[-1] == expected_state_dim:
         trend_mean_smoothed = smoothed_means[:, :n_trends]
         stationary_mean_smoothed = smoothed_means[:, n_trends:n_trends + n_stationary]
    else:
         print(f"Warning: Smoothed means last dimension ({smoothed_means.shape[-1]}) does not match expected state dimension ({expected_state_dim}). Cannot extract smoothed components.")
         trend_mean_smoothed = jnp.full((T, n_trends), jnp.nan, dtype=_DEFAULT_DTYPE)
         stationary_mean_smoothed = jnp.full((T, n_stationary), jnp.nan, dtype=_DEFAULT_DTYPE)


    # Generate plots using reporting_plots.py functions
    if plot_results and PLOTTING_AVAILABLE and sim_draws_stacked.shape[0] > 0: # Only plot if plotting available and draws were successful
        print("Generating plots using reporting_plots.py...")

        # Ensure the number of variable names matches the number of observed variables
        if variable_names is not None and len(variable_names) != n_obs:
            print(f"Warning: Number of variable names ({len(variable_names)}) does not match number of observed variables ({n_obs}). Using observed variable names from GPM.")
            variable_names = gpm_model.observed_variables
        elif variable_names is None:
             variable_names = gpm_model.observed_variables

        # Get trend and stationary variable names from GPM for plot titles
        trend_var_names_gpm = gpm_model.trend_variables
        stat_var_names_gpm = gpm_model.stationary_variables


        # Convert observed data to NumPy for plotting functions
        y_np = np.asarray(y)

        # Call plotting functions
        # Plot 1: Observed vs Fitted
        plot_observed_and_fitted(
            y_np=y_np,
            trend_draws=trend_draws, # Pass trend draws
            stationary_draws=stationary_draws, # Pass stationary draws
            hdi_prob=hdi_prob,
            variable_names=variable_names # Observed variable names
        )

        # Plot 2: Estimated Components (Trends and Stationary separately)
        plot_estimated_components(
            trend_draws=trend_draws,
            stationary_draws=stationary_draws,
            hdi_prob=hdi_prob,
            trend_variable_names=trend_var_names_gpm, # GPM trend names
            stationary_variable_names=stat_var_names_gpm # GPM stationary names
        )

        # Optional: Plot 3: Observed vs Trend (just the trend)
        plot_observed_and_trend(
            y_np=y_np,
            trend_draws=trend_draws,
            hdi_prob=hdi_prob,
            variable_names=variable_names # Observed variable names
        )

        print("Plotting complete.")
    elif plot_results and PLOTTING_AVAILABLE:
        print("Skipping plots: No simulation draws completed successfully or plotting not available.")
    elif plot_results and not PLOTTING_AVAILABLE:
         print("Skipping plots: reporting_plots.py could not be imported.")


    # Return results dictionary including extracted draws and smoothed means
    results = {
        'loglik': loglik,
        'filtered_means': smoothed_means, # Using smoothed means as the 'best' estimate
        'filtered_covs': smoothed_covs,
        'smoothed_means': smoothed_means,
        'smoothed_covs': smoothed_covs,
        'sim_draws': sim_draws_stacked,
        'trend_draws': trend_draws, # Extracted draws
        'stationary_draws': stationary_draws, # Extracted draws
        'trend_mean_smoothed': trend_mean_smoothed, # Smoothed means for components
        'stationary_mean_smoothed': stationary_mean_smoothed, # Smoothed means for components
        'F': F, 'Q': Q, 'C': C, 'H': H,
        'gpm_model': gpm_model,
        'ss_builder': ss_builder,
        'params': params
    }

    return results

# --- END OF FILE gpm_test_utils.py ---