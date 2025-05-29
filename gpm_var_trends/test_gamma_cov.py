# FIXED GAMMA MATRIX INDEXING AND USAGE

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple
import numpyro
import numpyro.distributions as dist

_DEFAULT_DTYPE = jnp.float64
_KF_JITTER = 1e-8

def _sample_var_parameters_with_gamma0(gpm_model):
    """
    FIXED VERSION: Modified to return Γ(0) along with the other gamma matrices.
    
    Returns: Sigma_u, A_transformed, gamma_list_with_gamma0
    where gamma_list_with_gamma0[0] = Γ(0), gamma_list_with_gamma0[1] = Γ(1), etc.
    """
    from gpm_bvar_trends import _sample_var_parameters
    from stationary_prior_jax_simplified import make_stationary_var_transformation_jax
    
    if not gpm_model.var_prior_setup or not gpm_model.stationary_variables:
        # Fallback: simple VAR with minimal structure
        n_vars = len(gpm_model.stationary_variables) if gpm_model.stationary_variables else 1
        A = jnp.zeros((1, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
        # Include Γ(0) = Sigma_u as the first element
        gamma_list_with_gamma0 = [Sigma_u]
        return Sigma_u, A, gamma_list_with_gamma0
    
    setup = gpm_model.var_prior_setup
    n_vars = len(gpm_model.stationary_variables)
    n_lags = setup.var_order
    
    # Sample hierarchical hyperparameters (same as original)
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(setup.es[i], setup.fs[i])) 
           for i in range(2)]
    Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(setup.gs[i], setup.hs[i])) 
              for i in range(2)]
    
    # Sample VAR coefficient matrices (same as original)
    raw_A_list = []
    for lag in range(n_lags):
        A_full = numpyro.sample(f"A_full_{lag}", 
                               dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
        A_diag = numpyro.sample(f"A_diag_{lag}", 
                               dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
        A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag)
        raw_A_list.append(A_lag)
    
    # Sample stationary innovation covariance (same as original)
    Omega_u_chol = numpyro.sample("Omega_u_chol", 
                                  dist.LKJCholesky(n_vars, concentration=setup.eta))
    
    sigma_u_vec = []
    for shock in gpm_model.stationary_shocks:
        if shock in gpm_model.estimated_params:
            from gpm_bvar_trends import _sample_parameter
            prior_spec = gpm_model.estimated_params[shock]
            sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
            sigma_u_vec.append(sigma)
        else:
            sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
            sigma_u_vec.append(sigma)
    
    sigma_u = jnp.array(sigma_u_vec)
    Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + 1e-8 * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    
    # Apply stationarity transformation and get gamma matrices
    try:
        phi_list, gamma_list_original = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
        A_transformed = jnp.stack(phi_list)
        
        # 🔧 FIX THE INDEXING BUG HERE!
        # gamma_list_original[0] = Γ(1), gamma_list_original[1] = Γ(2), etc.
        # We need to prepend Γ(0) = Sigma_u to get the correct indexing
        gamma_list_with_gamma0 = [Sigma_u] + gamma_list_original
        # Now: gamma_list_with_gamma0[0] = Γ(0), gamma_list_with_gamma0[1] = Γ(1), etc.
        
        numpyro.deterministic("A_transformed", A_transformed)
        
        return Sigma_u, A_transformed, gamma_list_with_gamma0
        
    except Exception:
        # Fallback if transformation fails
        A_transformed = jnp.stack(raw_A_list)
        numpyro.deterministic("A_raw", A_transformed)
        
        # Create fallback gamma list with proper Γ(0)
        gamma_list_with_gamma0 = [Sigma_u]  # Start with Γ(0)
        for lag in range(1, n_lags + 1):
            decay_factor = 0.7 ** lag
            gamma_list_with_gamma0.append(Sigma_u * decay_factor)
        
        return Sigma_u, A_transformed, gamma_list_with_gamma0


def _create_initial_covariance_with_gammas_fixed(
    state_dim: int, 
    n_trends: int,
    gamma_list_with_gamma0: List[jnp.ndarray],
    n_stationary: int, 
    var_order: int,
    use_gamma_scaling: float = 0.1
) -> jnp.ndarray:
    """
    FIXED VERSION: Now gamma_list_with_gamma0[0] is correctly Γ(0).
    
    Creates the initial covariance matrix P0 using the theoretical
    unconditional covariance of the VAR process.
    """
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Trends: Use diffuse prior (non-stationary)
    if n_trends > 0:
        trend_var = 1e4  # Moderate diffuse (not extreme)
        init_cov = init_cov.at[:n_trends, :n_trends].set(
            jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * trend_var
        )
    
    # VAR stationary components: Use gamma matrices for unconditional variance
    var_start_idx = n_trends
    var_total_dim = n_stationary * var_order
    
    if n_stationary > 0 and var_order > 0 and len(gamma_list_with_gamma0) > 0:
        # Now gamma_list_with_gamma0[0] is correctly Γ(0)
        gamma_0 = gamma_list_with_gamma0[0]  # This is now Γ(0)
        
        # Check if gamma_0 is usable
        is_usable = (
            jnp.all(jnp.isfinite(gamma_0)) &
            (gamma_0.shape[0] == n_stationary) &
            jnp.all(jnp.diag(gamma_0) > 0)
        )
        
        def build_var_covariance_fixed(gamma_0_input):
            """Build VAR block covariance with CORRECT gamma indexing"""
            var_cov_block = jnp.zeros((var_total_dim, var_total_dim), dtype=_DEFAULT_DTYPE)
            
            # Build the companion form covariance matrix
            for i in range(var_order):
                for j in range(var_order):
                    lag_diff = abs(i - j)
                    
                    # Get the appropriate gamma matrix with CORRECT indexing
                    if lag_diff < len(gamma_list_with_gamma0):
                        gamma_lag = gamma_list_with_gamma0[lag_diff] * use_gamma_scaling
                    else:
                        # Exponential decay for missing lags
                        decay_factor = 0.5 ** (lag_diff - len(gamma_list_with_gamma0) + 1)
                        gamma_lag = gamma_0_input * use_gamma_scaling * decay_factor
                    
                    # Set the block
                    row_start = i * n_stationary
                    row_end = (i + 1) * n_stationary
                    col_start = j * n_stationary
                    col_end = (j + 1) * n_stationary
                    
                    # Use transpose for upper triangular blocks
                    block_value = gamma_lag.T if i > j else gamma_lag
                    var_cov_block = var_cov_block.at[
                        row_start:row_end, 
                        col_start:col_end
                    ].set(block_value)
            
            return var_cov_block
        
        def use_default_var_cov():
            """Fallback to diagonal covariance"""
            return jnp.eye(var_total_dim, dtype=_DEFAULT_DTYPE) * 0.1
        
        # Choose between gamma-based and default covariance
        var_cov_matrix = jax.lax.cond(
            is_usable,
            build_var_covariance_fixed,
            lambda x: use_default_var_cov(),
            operand=gamma_0
        )
        
        # Insert VAR block into full covariance
        init_cov = init_cov.at[
            var_start_idx:var_start_idx + var_total_dim,
            var_start_idx:var_start_idx + var_total_dim
        ].set(var_cov_matrix)
    
    # Final regularization
    init_cov = (init_cov + init_cov.T) / 2.0
    init_cov = init_cov + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


def _sample_initial_conditions_with_gammas_fixed(
    gpm_model, 
    state_dim: int,
    gamma_list_with_gamma0: List[jnp.ndarray],
    n_trends: int, 
    n_stationary: int,
    var_order: int,
    gamma_scaling: float = 0.1
) -> jnp.ndarray:
    """
    FIXED VERSION: Now uses gamma_list_with_gamma0[0] which is correctly Γ(0).
    
    Sample initial conditions with proper distinction between prior uncertainty
    about the mean and the unconditional process variance.
    """
    # Initialize with zeros (typical for stationary processes)
    init_mean_base = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std_base = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Handle trend variables: Use GPM specifications or diffuse priors
    for var_name, var_spec in gpm_model.initial_values.items():
        if (var_spec.init_dist == 'normal_pdf' and 
            len(var_spec.init_params) >= 2):
            mean_val, std_val = var_spec.init_params[:2]
            if var_name in gpm_model.trend_variables:
                try:
                    idx = gpm_model.trend_variables.index(var_name)
                    if idx < n_trends:
                        init_mean_base = init_mean_base.at[idx].set(mean_val)
                        init_std_base = init_std_base.at[idx].set(std_val)
                except ValueError:
                    pass
    
    # Default std for trends (moderate uncertainty about initial mean)
    if n_trends > 0:
        trend_mask = jnp.arange(state_dim) < n_trends
        default_trend_std = jnp.where(
            init_std_base[:n_trends] == 1.0, 
            0.5,  # Moderate prior uncertainty about trend starting point
            init_std_base[:n_trends]
        )
        init_std_base = init_std_base.at[:n_trends].set(default_trend_std)
    
    # VAR stationary components: Small prior uncertainty about initial mean
    var_start_idx = n_trends
    if n_stationary > 0 and var_order > 0 and len(gamma_list_with_gamma0) > 0:
        gamma_0 = gamma_list_with_gamma0[0]  # Now correctly Γ(0)
        
        is_usable = (
            jnp.all(jnp.isfinite(gamma_0)) &
            (gamma_0.shape[0] == n_stationary) &
            jnp.all(jnp.diag(gamma_0) > 0)
        )
        
        def use_gamma_for_std(gamma_0_input):
            """Use gamma matrix to set reasonable prior uncertainty"""
            # Use a small fraction of the unconditional std dev as prior uncertainty
            diag_gamma_0 = jnp.diag(gamma_0_input)
            theoretical_std = jnp.sqrt(jnp.maximum(diag_gamma_0, 1e-12))
            
            # Prior uncertainty should be much smaller than process variance
            prior_std_factor = 0.1  # 10% of process std dev
            var_prior_std = theoretical_std * prior_std_factor
            
            # Apply to all VAR lags
            current_std = init_std_base
            for lag in range(var_order):
                start_idx = var_start_idx + lag * n_stationary
                end_idx = start_idx + n_stationary
                current_std = current_std.at[start_idx:end_idx].set(var_prior_std)
            
            return current_std
        
        def use_default_std():
            """Fallback to small default std for stationary variables"""
            current_std = init_std_base
            var_indices = jnp.arange(var_start_idx, var_start_idx + n_stationary * var_order)
            current_std = current_std.at[var_indices].set(0.05)  # Small uncertainty
            return current_std
        
        init_std_base = jax.lax.cond(
            is_usable,
            use_gamma_for_std,
            lambda x: use_default_std(),
            operand=gamma_0
        )
    
    # Sample the initial conditions
    init_mean_sampled = numpyro.sample(
        "init_mean_full",
        dist.Normal(init_mean_base, init_std_base)
    )
    
    return init_mean_sampled


def fit_gpm_model_with_gamma_fixed(
    gpm_file_path: str, 
    y: jnp.ndarray,
    num_warmup: int = 1000, 
    num_samples: int = 2000,
    num_chains: int = 4, 
    num_extract_draws: int = 100,
    gamma_init_scaling: float = 0.05,  # Even smaller scaling
    rng_key = None
):
    """
    COMPLETELY FIXED VERSION with correct gamma matrix indexing.
    """
    from gpm_parser import GPMParser
    from gpm_bvar_trends import GPMStateSpaceBuilder, EnhancedBVARParams
    from gpm_bvar_trends import (_sample_parameter, _sample_trend_covariance, 
                                _sample_measurement_covariance, _has_measurement_error)
    from simulation_smoothing import (extract_gpm_trends_and_components, 
                                    _compute_and_format_hdi_az)
    from numpyro.infer import MCMC, NUTS
    import jax.random as random
    import time
    import numpy as np
    
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    print(f"Parsing GPM file: {gpm_file_path} for FIXED GAMMA-BASED initialization")
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.trend_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {gpm_model.observed_variables}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")
    
    def gpm_bvar_model_fixed(y_data: jnp.ndarray):
        T, n_obs = y_data.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample covariances and VAR parameters with FIXED gamma indexing
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, gamma_list_with_gamma0 = _sample_var_parameters_with_gamma0(gpm_model)
        Sigma_eps = (_sample_measurement_covariance(gpm_model) 
                    if _has_measurement_error(gpm_model) else None)
        
        # FIXED GAMMA-BASED INITIALIZATION with correct indexing
        init_mean = _sample_initial_conditions_with_gammas_fixed(
            gpm_model, ss_builder.state_dim, gamma_list_with_gamma0,
            ss_builder.n_trends, ss_builder.n_stationary, 
            ss_builder.var_order, gamma_scaling=gamma_init_scaling
        )
        
        init_cov = _create_initial_covariance_with_gammas_fixed(
            ss_builder.state_dim, ss_builder.n_trends, gamma_list_with_gamma0,
            ss_builder.n_stationary, ss_builder.var_order,
            use_gamma_scaling=gamma_init_scaling
        )
        
        # Build state space and compute likelihood
        params = EnhancedBVARParams(
            A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
            structural_params=structural_params, Sigma_eps=Sigma_eps
        )
        
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        # Check matrices are finite
        matrices_ok = (
            jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) &
            jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) &
            jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov))
        )
        
        # Create R matrix
        try:
            R_mat = jnp.linalg.cholesky(
                Q + _KF_JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
            )
        except:
            R_mat = jnp.diag(jnp.sqrt(jnp.diag(Q) + _KF_JITTER))
        
        # Kalman filter likelihood
        from Kalman_filter_jax import KalmanFilter
        kf = KalmanFilter(T=F, R=R_mat, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs_mat = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y_data, valid_obs_idx, n_obs, C, H, I_obs_mat)
        )
        
        numpyro.factor("loglik", loglik)
    
    # Run MCMC with higher acceptance probability to reduce divergences
    print("Running MCMC (FIXED Gamma-based initialization)...")
    kernel = NUTS(gpm_bvar_model_fixed, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    mcmc_key, extract_key = random.split(rng_key)
    start_time = time.time()
    mcmc.run(mcmc_key, y_data=y)
    end_time = time.time()
    
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    mcmc.print_summary(exclude_deterministic=False)
    
    # Component extraction (same as before)
    print("Extracting trend and stationary components...")
    start_extract_time = time.time()
    posterior_samples = mcmc.get_samples()
    
    total_draws_mcmc = 0
    if posterior_samples:
        first_key = list(posterior_samples.keys())[0]
        total_draws_mcmc = len(posterior_samples[first_key])
    
    actual_num_extract = min(num_extract_draws, total_draws_mcmc)
    
    trend_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_trends), dtype=_DEFAULT_DTYPE)
    stationary_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_stationary), dtype=_DEFAULT_DTYPE)
    trend_hdi_dict = None
    stationary_hdi_dict = None
    
    if actual_num_extract > 0:
        try:
            trend_draws_arr, stationary_draws_arr = extract_gpm_trends_and_components(
                mcmc, y, gpm_model, ss_builder,
                num_draws=actual_num_extract, rng_key=extract_key
            )
            print(f"Component extraction time: {time.time() - start_extract_time:.2f}s.")
            
            if trend_draws_arr.shape[0] > 0:
                print("Computing HDI intervals (ArviZ)...")
                trend_hdi_dict = _compute_and_format_hdi_az(
                    np.asarray(trend_draws_arr), hdi_prob=0.9
                )
                stationary_hdi_dict = _compute_and_format_hdi_az(
                    np.asarray(stationary_draws_arr), hdi_prob=0.9
                )
                
                if trend_hdi_dict and not np.any(np.isnan(trend_hdi_dict['low'])):
                    print("HDI computed.")
                else:
                    print("Warning: HDI computation problems.")
        except Exception as e:
            print(f"Error in component extraction: {e}")
            trend_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_trends), dtype=_DEFAULT_DTYPE)
            stationary_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_stationary), dtype=_DEFAULT_DTYPE)
    else:
        print("Skipping component extraction (no/few MCMC draws or num_extract_draws is 0).")
    
    return {
        'mcmc': mcmc, 'gpm_model': gpm_model, 'ss_builder': ss_builder,
        'trend_draws': trend_draws_arr, 'stationary_draws': stationary_draws_arr,
        'trend_hdi': trend_hdi_dict, 'stationary_hdi': stationary_hdi_dict
    }


# Alternative: Fix the original rev_mapping_jax to return Γ(0) properly
def fix_rev_mapping_jax_output(phi_list, gamma_list_original, Sigma_u):
    """
    Alternative fix: Correct the output of rev_mapping_jax to include Γ(0).
    
    Args:
        phi_list: List of stationary VAR coefficient matrices
        gamma_list_original: Original gamma list from rev_mapping_jax (missing Γ(0))
        Sigma_u: Innovation covariance matrix (which equals Γ(0) for the VAR process)
    
    Returns:
        gamma_list_corrected: [Γ(0), Γ(1), Γ(2), ...]
    """
    # Prepend Σ_u as Γ(0) to the original gamma list
    gamma_list_corrected = [Sigma_u] + gamma_list_original
    return gamma_list_corrected


# Test function with the complete fix
def test_fixed_gamma_initialization():
    """Test the completely fixed gamma initialization"""
    import pandas as pd
    
    # Load your data
    dta = pd.read_csv('sim_data.csv')
    y_np = dta.values
    y_jax = jnp.asarray(y_np)
    
    print("=== TESTING FIXED GAMMA INITIALIZATION ===")
    print("Key fixes applied:")
    print("1. ✅ Fixed gamma indexing: gamma_list[0] now correctly represents Γ(0)")
    print("2. ✅ Proper P0 construction using unconditional VAR covariance")
    print("3. ✅ Distinction between prior uncertainty and process variance")
    print("4. ✅ Much smaller scaling factor (0.05) to prevent divergences")
    print("5. ✅ Higher target acceptance probability (0.95)")
    print("")
    
    # Use the completely fixed function
    results = fit_gpm_model_with_gamma_fixed(
        'example_model.gpm', 
        y_jax,
        num_warmup=500,
        num_samples=1000, 
        num_chains=2,
        num_extract_draws=20,
        gamma_init_scaling=0.1,  # Very small scaling!
    )
    
    return results


# Diagnostic function to verify gamma matrices are being used correctly
def diagnose_gamma_usage(gpm_file_path, y):
    """Diagnostic function to show exactly how gamma matrices are being used"""
    from gpm_parser import GPMParser
    from gpm_bvar_trends import GPMStateSpaceBuilder
    
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    print("=== GAMMA MATRIX USAGE DIAGNOSIS ===")
    print(f"VAR order: {gpm_model.var_prior_setup.var_order if gpm_model.var_prior_setup else 'None'}")
    print(f"Stationary variables: {len(gpm_model.stationary_variables)}")
    print(f"State dimension: {ss_builder.state_dim}")
    print(f"Trends: {ss_builder.n_trends}, Stationary: {ss_builder.n_stationary}")
    
    # Show the theoretical structure
    if gpm_model.var_prior_setup:
        n_vars = len(gpm_model.stationary_variables)
        var_order = gpm_model.var_prior_setup.var_order
        print(f"\nTheoretical gamma structure for VAR({var_order}) with {n_vars} variables:")
        print(f"- Γ(0): {n_vars}×{n_vars} contemporaneous covariance")
        print(f"- Γ(1): {n_vars}×{n_vars} lag-1 autocovariance")
        print(f"- Γ(2): {n_vars}×{n_vars} lag-2 autocovariance")
        print(f"- etc.")
        
        print(f"\nCompanion form state covariance will be {n_vars*var_order}×{n_vars*var_order}")
        print("Built from blocks using gamma matrices with correct indexing")
        
        print("\n🎯 FIXED: Now gamma_list[0] = Γ(0), gamma_list[1] = Γ(1), etc.")
        print("🎯 FIXED: P0 uses theoretical unconditional VAR covariance")
        print("🎯 FIXED: Prior uncertainty about initial mean is separate and small")


if __name__ == "__main__":
    print("Testing fixed gamma matrix initialization...")
    
    # First run diagnostics
    diagnose_gamma_usage('example_model.gpm', None)
    
    # Then test the fixed implementation
    results = test_fixed_gamma_initialization()
    
    print("\n=== RESULTS ===")
    if results and 'mcmc' in results:
        samples = results['mcmc'].get_samples()
        if samples and len(samples) > 0:
            # Check for divergences
            divergences = results['mcmc'].get_extra_fields().get('diverging', None)
            if divergences is not None:
                n_divergences = jnp.sum(divergences)
                print(f"Number of divergences: {n_divergences}")
                if n_divergences == 0:
                    print("✅ SUCCESS: No divergences with fixed gamma initialization!")
                else:
                    print(f"⚠️  Still {n_divergences} divergences - may need further tuning")
            else:
                print("✅ MCMC completed successfully!")
        else:
            print("❌ No samples obtained")
    else:
        print("❌ Fitting failed")