import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time

# Import your existing modules
from gpm_parser import GPMParser, GPMModel
from gpm_bvar_trends import GPMStateSpaceBuilder, EnhancedBVARParams, _DEFAULT_DTYPE, _JITTER
from gpm_bvar_trends import  _create_initial_covariance
from gpm_bvar_trends import _has_measurement_error, _sample_measurement_covariance
from simulation_smoothing import jarocinski_corrected_simulation_smoother
from Kalman_filter_jax import KalmanFilter, _KF_JITTER

# Import stationary transformation
try:
    from stationary_prior_jax_simplified import make_stationary_var_transformation_jax
except ImportError:
    print("Warning: Could not import stationary transformation")
    make_stationary_var_transformation_jax = None



def test_gpm_model_with_parameters(gpm_file_path: str,
                                 y: jnp.ndarray,
                                 param_values: Dict[str, float],
                                 num_sim_draws: int = 50,
                                 rng_key: jnp.ndarray = random.PRNGKey(42),
                                 plot_results: bool = True,
                                 variable_names: Optional[List[str]] = None,
                                 use_gamma_init_for_test: bool = True, # <-- New toggle flag
                                 gamma_init_scaling: float = 1.0 # <-- New scaling for gamma init
                                 ) -> Dict:
    """
    Test GPM model with fixed parameter values without running MCMC.
    Allows switching between gamma-based and standard 1e6/1e-6 initial covariance.
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
                print(f"  Using fixed {param_name} = {param_values[param_name]}")
            else:
                # Use prior mean as default (if normal) or mode (if inv_gamma, etc.)
                prior_spec = gpm_model.estimated_params[param_name]
                if prior_spec.distribution == 'normal_pdf' and len(prior_spec.params) >= 1:
                     structural_params[param_name] = jnp.array(prior_spec.params[0], dtype=_DEFAULT_DTYPE)
                     print(f"  Using prior mean for {param_name} = {prior_spec.params[0]}")
                elif prior_spec.distribution == 'inv_gamma_pdf' and len(prior_spec.params) >= 2:
                    alpha, beta = prior_spec.params
                    mode = beta / (alpha + 1.0) if alpha > -1.0 else beta
                    structural_params[param_name] = jnp.array(mode, dtype=_DEFAULT_DTYPE)
                    print(f"  Using prior mode for {param_name} = {mode}")
                else:
                    structural_params[param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE) # Fallback default
                    print(f"  Using default 0.0 for {param_name}")


    # Build covariance matrices using fixed values (and get gamma_list)
    Sigma_eta = _build_trend_covariance(gpm_model, param_values)
    Sigma_u, A_transformed, gamma_list = _build_var_parameters(gpm_model, param_values) # <-- gamma_list is returned here

    # Measurement covariance
    Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None


    # Initial conditions - Use dedicated functions for testing
    print("Building initial conditions for testing...")
    init_mean = _build_initial_mean_for_test(gpm_model, ss_builder.state_dim) # Use the new mean function

    # --- Initial Covariance Setup ---
    if use_gamma_init_for_test:
        print(f"Using gamma-based initial covariance (scaling={gamma_init_scaling})...")
        if len(gamma_list) > 0 and gamma_list[0] is not None: # Check if gamma_list has content
             print(f"  gamma_list[0] diagonal: {jnp.diag(gamma_list[0])}")
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
        print("Warning: Some matrices contain non-finite values!")
        return {'loglik': -jnp.inf, 'error': 'Non-finite matrices'}

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
    # (KalmanFilter.filter returns a dict, so this conversion might not be strictly needed but is safe)
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


    for i in range(num_sim_draws):
        if (i + 1) % 10 == 0:
            print(f"  Draw {i+1}/{num_sim_draws}")

        rng_key, sim_key = random.split(rng_key)

        try:
            sim_states = jarocinski_corrected_simulation_smoother(
                y, F, R, C, H, init_mean, init_cov_safe, sim_key # Use the safe init_cov
            )
            sim_draws.append(sim_states)
        except Exception as e:
            print(f"Warning: Simulation draw {i+1} failed: {e}")
            continue

    if len(sim_draws) > 0:
        sim_draws = jnp.stack(sim_draws)
        print(f"Successfully completed {len(sim_draws)} simulation draws")
    else:
        print("No simulation draws completed successfully")
        sim_draws = jnp.empty((0, T, ss_builder.state_dim), dtype=_DEFAULT_DTYPE)

    # Extract components
    results = {
        'loglik': loglik,
        'filtered_means': filter_results_dict['x_filt'],
        'filtered_covs': filter_results_dict['P_filt'],
        'smoothed_means': smoothed_means,
        'smoothed_covs': smoothed_covs,
        'sim_draws': sim_draws,
        'F': F, 'Q': Q, 'C': C, 'H': H,
        'gpm_model': gpm_model,
        'ss_builder': ss_builder,
        'params': params
    }

    # Extract trend and stationary components from simulation draws
    if len(sim_draws) > 0:
        # Assuming state vector is [Trends, Stationary_t, Stationary_t-1, ..., Stationary_t-p+1]
        trend_draws = sim_draws[:, :, :ss_builder.n_trends]
        stationary_draws = sim_draws[:, :, ss_builder.n_trends:ss_builder.n_trends + ss_builder.n_stationary]

        results['trend_draws'] = trend_draws
        results['stationary_draws'] = stationary_draws

        # Also extract from smoothed means for comparison
        results['trend_mean_smoothed'] = smoothed_means[:, :ss_builder.n_trends]
        results['stationary_mean_smoothed'] = smoothed_means[:, ss_builder.n_trends:ss_builder.n_trends + ss_builder.n_stationary]

    # Generate plots
    if plot_results and len(sim_draws) > 0:
        _plot_test_results(y, results, variable_names)

    return results


# Add this function definition somewhere accessible, e.g., in gpm_bvar_trends.py
# or directly in single_parameter_run.py before test_gpm_model_with_parameters

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
            # Check if gamma_list[0] is valid
            gamma_0 = gamma_list[0]
            is_gamma_0_valid = (
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

                        var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)

                # Insert VAR covariance into full state covariance matrix
                var_end = var_start + var_state_dim
                if var_end <= state_dim:
                     init_cov = init_cov.at[var_start:var_end, var_start:var_end].set(var_state_cov)
            else:
                 print("Warning: gamma_list[0] is invalid. Using a moderate default for VAR initial covariance.")
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


def _build_trend_covariance(gpm_model: GPMModel, param_values: Dict[str, float]) -> jnp.ndarray:
    """Build trend innovation covariance matrix using fixed parameter values"""
    n_trends = len(gpm_model.trend_variables)
    
    # Get individual shock standard deviations
    trend_sigmas = []
    for shock in gpm_model.trend_shocks:
        param_name = f"sigma_{shock}"
        if param_name in param_values:
            sigma = param_values[param_name]
            trend_sigmas.append(sigma)
            print(f"  Using fixed {param_name} = {sigma}")
        elif shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            if prior_spec.distribution == 'inv_gamma_pdf':
                alpha, beta = prior_spec.params
                if alpha > 1:
                    sigma = (alpha - 1) * beta  # Mode of inverse gamma
                else:
                    sigma = 0.1
            else:
                sigma = 0.1
            trend_sigmas.append(sigma)
            print(f"  Using prior mode for {param_name} = {sigma}")
        else:
            # Default fallback
            sigma = 0.1
            trend_sigmas.append(sigma)
            print(f"  Using default for {param_name} = {sigma}")
    
    # For simplicity, assume diagonal covariance for trends
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas, dtype=_DEFAULT_DTYPE) ** 2)
    
    return Sigma_eta


def _build_var_parameters(gpm_model: GPMModel, param_values: Dict[str, float]) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """Build VAR parameters using fixed values and stationary transformation"""
    
    if not gpm_model.var_prior_setup or not gpm_model.stationary_variables:
        # Fallback: simple VAR with minimal structure
        n_vars = len(gpm_model.stationary_variables) if gpm_model.stationary_variables else 1
        A = jnp.zeros((1, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
        gamma_list = [Sigma_u]
        return Sigma_u, A, gamma_list
    
    setup = gpm_model.var_prior_setup
    n_vars = len(gpm_model.stationary_variables)
    n_lags = setup.var_order
    
    # Use the hierarchical prior means from GPM specification (same as MCMC version)
    mean_diag = setup.es[0]      # e.g., 0.5 for diagonal elements
    mean_offdiag = setup.es[1]   # e.g., 0.3 for off-diagonal elements
    
    print(f"Using hierarchical VAR prior structure:")
    print(f"  Diagonal mean: {mean_diag}, Off-diagonal mean: {mean_offdiag}")
    print(f"  Number of lags: {n_lags}")
    
    # Create raw VAR coefficient matrices using prior means (SAME FOR ALL LAGS)
    raw_A_list = []
    for lag in range(n_lags):
        # Initialize with off-diagonal mean
        A_lag = jnp.full((n_vars, n_vars), mean_offdiag, dtype=_DEFAULT_DTYPE)
        # Set diagonal to diagonal mean
        A_lag = A_lag.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(mean_diag)
        raw_A_list.append(A_lag)
        
        print(f"  Lag {lag+1}: diagonal = {mean_diag}, off-diagonal = {mean_offdiag}")
    
    # Build stationary innovation covariance
    stat_sigmas = []
    for shock in gpm_model.stationary_shocks:
        param_name = f"sigma_{shock}"
        if param_name in param_values:
            sigma = param_values[param_name]
            stat_sigmas.append(sigma)
            print(f"  Using fixed {param_name} = {sigma}")
        elif shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            if prior_spec.distribution == 'inv_gamma_pdf':
                alpha, beta = prior_spec.params
                if alpha > 1:
                    sigma = (alpha - 1) * beta  # Mode of inverse gamma
                else:
                    sigma = 0.2
            else:
                sigma = 0.2
            stat_sigmas.append(sigma)
            print(f"  Using prior mode for {param_name} = {sigma}")
        else:
            sigma = 0.2
            stat_sigmas.append(sigma)
            print(f"  Using default for {param_name} = {sigma}")
    
    Sigma_u = jnp.diag(jnp.array(stat_sigmas, dtype=_DEFAULT_DTYPE) ** 2)
    
    # Apply stationary transformation (SAME AS MCMC VERSION)
    if make_stationary_var_transformation_jax is not None:
        try:
            print(f"Applying stationary transformation...")
            print(f"  Sigma_u shape: {Sigma_u.shape}")
            print(f"  Raw A shapes: {[A.shape for A in raw_A_list]}")
            
            phi_list, gamma_list = make_stationary_var_transformation_jax(
                Sigma_u, raw_A_list, n_vars, n_lags
            )
            A_transformed = jnp.stack(phi_list)
            
            print(f"  Transformation successful!")
            print(f"  Transformed A shape: {A_transformed.shape}")
            print(f"  Transformed A[0] diagonal: {jnp.diag(A_transformed[0])}")
            
            return Sigma_u, A_transformed, gamma_list
            
        except Exception as e:
            print(f"  Stationary transformation failed: {e}")
            print(f"  Using raw coefficients instead")
            A_transformed = jnp.stack(raw_A_list)
            
            # Create fallback gamma list
            gamma_list = [Sigma_u]
            for lag in range(1, n_lags + 1):
                decay_factor = 0.7 ** lag
                gamma_list.append(Sigma_u * decay_factor)
            
            return Sigma_u, A_transformed, gamma_list
    else:
        print("Stationary transformation not available, using raw coefficients")
        A_transformed = jnp.stack(raw_A_list)
        gamma_list = [Sigma_u]
        return Sigma_u, A_transformed, gamma_list


def _plot_test_results(y: jnp.ndarray, results: Dict, variable_names: Optional[List[str]] = None):
    """Generate plots of test results"""
    
    T, n_obs = y.shape
    
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_obs)]
    
    # Plot observed data with trend and stationary components
    if 'trend_draws' in results and 'stationary_draws' in results:
        trend_draws = results['trend_draws']
        stationary_draws = results['stationary_draws']
        
        # Compute credible intervals
        trend_lower = jnp.percentile(trend_draws, 5, axis=0)
        trend_upper = jnp.percentile(trend_draws, 95, axis=0)
        trend_median = jnp.percentile(trend_draws, 50, axis=0)
        
        stat_lower = jnp.percentile(stationary_draws, 5, axis=0)
        stat_upper = jnp.percentile(stationary_draws, 95, axis=0)
        stat_median = jnp.percentile(stationary_draws, 50, axis=0)
        
        # Create subplots
        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        if n_obs == 1:
            axes = [axes]
        
        for i in range(n_obs):
            ax = axes[i]
            
            # Plot observed data
            ax.plot(y[:, i], 'k-', linewidth=2, label='Observed', alpha=0.8)
            
            # Plot trend component
            ax.plot(trend_median[:, i], 'b-', linewidth=1.5, label='Trend (median)')
            ax.fill_between(range(T), trend_lower[:, i], trend_upper[:, i], 
                          color='blue', alpha=0.2, label='Trend (90% CI)')
            
            # Plot stationary component
            ax.plot(stat_median[:, i], 'r-', linewidth=1.5, label='Stationary (median)')
            ax.fill_between(range(T), stat_lower[:, i], stat_upper[:, i], 
                          color='red', alpha=0.2, label='Stationary (90% CI)')
            
            ax.set_title(f'{variable_names[i]} - Decomposition')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: observed vs fitted
        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        if n_obs == 1:
            axes = [axes]
        
        for i in range(n_obs):
            ax = axes[i]
            
            # Fitted = trend + stationary
            fitted_median = trend_median[:, i] + stat_median[:, i]
            fitted_lower = trend_lower[:, i] + stat_lower[:, i]
            fitted_upper = trend_upper[:, i] + stat_upper[:, i]
            
            ax.plot(y[:, i], 'k-', linewidth=2, label='Observed', alpha=0.8)
            ax.plot(fitted_median, 'g-', linewidth=1.5, label='Fitted (median)')
            ax.fill_between(range(T), fitted_lower, fitted_upper, 
                          color='green', alpha=0.2, label='Fitted (90% CI)')
            
            ax.set_title(f'{variable_names[i]} - Observed vs Fitted')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def quick_test_example():
    """Quick example of testing a GPM model"""
    
    # Generate some test data
    T, n_vars = 100, 3
    np_rng = np.random.default_rng(123)
    
    # Generate trends (random walks)
    trends = np.cumsum(np_rng.normal(0, 0.1, (T, n_vars)), axis=0)
    
    # Generate stationary component
    stationary = np.zeros((T, n_vars))
    for i in range(n_vars):
        for t in range(1, T):
            stationary[t, i] = 0.7 * stationary[t-1, i] + np_rng.normal(0, 0.2)
    
    y = jnp.array(trends + stationary, dtype=_DEFAULT_DTYPE)
    
    # Create simple GPM file
    gpm_content = '''
    parameters ;
    
    estimated_params;
        stderr SHK_TREND1, inv_gamma_pdf, 2.1, 0.1;
        stderr SHK_TREND2, inv_gamma_pdf, 2.1, 0.1;
        stderr SHK_TREND3, inv_gamma_pdf, 2.1, 0.1;
        stderr SHK_STAT1, inv_gamma_pdf, 2.1, 0.2;
        stderr SHK_STAT2, inv_gamma_pdf, 2.1, 0.2;
        stderr SHK_STAT3, inv_gamma_pdf, 2.1, 0.2;
    end;
    
    trends_vars
        TREND1,
        TREND2,
        TREND3
    ;
    
    stationary_variables
        STAT1,
        STAT2,
        STAT3
    ;
    
    trend_shocks;
        var SHK_TREND1
        var SHK_TREND2
        var SHK_TREND3
    end;
    
    shocks;
        var SHK_STAT1
        var SHK_STAT2
        var SHK_STAT3
    end;
    
    trend_model;
        TREND1 = TREND1(-1) + SHK_TREND1;
        TREND2 = TREND2(-1) + SHK_TREND2;
        TREND3 = TREND3(-1) + SHK_TREND3;
    end;
    
    varobs 
        OBS1,
        OBS2,
        OBS3
    ;
    
    measurement_equations;
        OBS1 = TREND1 + STAT1;
        OBS2 = TREND2 + STAT2;
        OBS3 = TREND3 + STAT3;
    end;
    
    initval;
        TREND1, normal_pdf, 0, 10;
        TREND2, normal_pdf, 0, 10;
        TREND3, normal_pdf, 0, 10;
        STAT1, normal_pdf, 0, 0.1;
        STAT2, normal_pdf, 0, 0.1;
        STAT3, normal_pdf, 0, 0.1;
    end;
    
    var_prior_setup;
        var_order = 2;
        es = 0.5, 0.3;
        fs = 0.5, 0.5;
        gs = 3.0, 3.0;
        hs = 1.0, 1.0;
        eta = 2.0;
    end;
    '''
    
    with open('test_model.gpm', 'w') as f:
        f.write(gpm_content)
    
    # Test parameter values (using reasonable values)
    test_params = {
        'sigma_SHK_TREND1': 0.1,  # Larger trend shocks
        'sigma_SHK_TREND2': 0.1,
        'sigma_SHK_TREND3': 0.1,
        'sigma_SHK_STAT1': 1.5,   # Larger stationary shocks  
        'sigma_SHK_STAT2': 1.5,
        'sigma_SHK_STAT3': 1.5,
    }
    
    # Run test
    results = test_gpm_model_with_parameters(
        'test_model.gpm',
        y,
        test_params,
        num_sim_draws=30,
        plot_results=True,
        variable_names=['OBS1', 'OBS2', 'OBS3']
    )
    
    print(f"\nTest Results:")
    print(f"Log-likelihood: {results['loglik']:.2f}")
    print(f"Number of simulation draws: {results['sim_draws'].shape[0]}")
    
    return results


if __name__ == "__main__":
    print("Running GPM model test example...")
    results = quick_test_example()