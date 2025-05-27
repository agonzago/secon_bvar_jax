import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import pandas as pd
import time
import os

# Import your existing modules
from .gpm_parser import GPMParser, GPMModel
from .gpm_bvar_trends import (
    GPMStateSpaceBuilder, EnhancedBVARParams, fit_gpm_model,
    _sample_parameter, _sample_trend_covariance, _sample_var_parameters,
    _sample_measurement_covariance, _has_measurement_error, 
    _sample_initial_conditions, _create_initial_covariance,
    _sample_initial_conditions_with_gammas, _create_initial_covariance_with_gammas,
    _DEFAULT_DTYPE, _JITTER, _KF_JITTER
)

# Import simulation smoother
from .simulation_smoothing import (
    extract_gpm_trends_and_components, 
    compute_hdi_with_percentiles,
    _compute_and_format_hdi_az
)

import arviz as az
import xarray as xr
# Import Kalman Filter
try:
    from .Kalman_filter_jax import KalmanFilter
except ImportError:
    print("Warning: Could not import KalmanFilter")

# Import plotting functions if available
try:
    from .reporting_plots import plot_decomposition_results, plot_observed_and_trend
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Plotting functions not available. Skipping plot generation.")
    PLOTTING_AVAILABLE = False


def fit_gpm_model_with_smoother(gpm_file_path: str, y: jnp.ndarray, 
                               num_warmup: int = 1000, num_samples: int = 2000, 
                               num_chains: int = 4, num_extract_draws: int = 100,
                               rng_key: jnp.ndarray = random.PRNGKey(0)):
    """
    Fit a GPM-based BVAR model and extract trend/stationary components.
    FIXED: Back to original initialization (no gamma matrices)
    """
    
    print(f"Parsing GPM file: {gpm_file_path}")
    
    # Parse GPM file
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    
    # Create state space builder
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.trend_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {gpm_model.observed_variables}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")
    
    # Create the Numpyro model function
    def gpm_bvar_model(y: jnp.ndarray):
        """Numpyro model - BACK TO ORIGINAL INITIALIZATION"""
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample shock standard deviations and build covariance matrices
        Sigma_eta = _sample_trend_covariance(gpm_model)
        
        # Get gamma matrices but DON'T use them for initialization
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # USE ORIGINAL INITIALIZATION - NO GAMMA MATRICES
        init_mean = _sample_initial_conditions(gpm_model, ss_builder.state_dim)
        init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Create parameter structure
        params = EnhancedBVARParams(
            A=A_transformed,
            Sigma_u=Sigma_u,
            Sigma_eta=Sigma_eta,
            structural_params=structural_params,
            Sigma_eps=Sigma_eps
        )
        
        # Build state space matrices
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        # Check for numerical issues
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        # Build R matrix from Q (assuming Q = R @ R.T)
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        # Create Kalman Filter
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        # Compute likelihood
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    print("Running MCMC...")
    kernel = NUTS(gpm_bvar_model, 
                  target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, 
                num_samples=num_samples, 
                num_chains=num_chains
                )
    
    # Split random key for MCMC and component extraction
    mcmc_key, extract_key = random.split(rng_key)
    
    start_time = time.time()
    mcmc.run(mcmc_key, y=y)
    end_time = time.time()
    
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    mcmc.print_summary(exclude_deterministic=False)
    
    # Extract trend and stationary components
    print("Extracting trend and stationary components using Durbin & Koopman Simulation Smoother...")
    
    start_extract_time = time.time()
    
    # Check if we have enough posterior samples
    samples = mcmc.get_samples()
    total_posterior_draws = len(list(samples.values())[0])
    num_extract_draws = min(num_extract_draws, total_posterior_draws)
    
    try:
        trend_draws, stationary_draws = extract_gpm_trends_and_components(
            mcmc, y, gpm_model, ss_builder, 
            num_draws=num_extract_draws, 
            rng_key=extract_key
        )
        
        end_extract_time = time.time()
        print(f"Component extraction completed in {end_extract_time - start_extract_time:.2f} seconds.")
        
        # Compute HDI intervals
        # print("Computing HDI intervals using percentiles...")
        # if trend_draws.shape[0] > 1:
        #     trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=0.9)
        #     stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=0.9)
        #     print("HDI computed successfully using percentiles!")
        # else:
        #     trend_hdi = None
        #     stationary_hdi = None
        
        # print(f"Trend component draws shape: {trend_draws.shape}")
        # print(f"Stationary component draws shape: {stationary_draws.shape}")
        # Compute HDI intervals using ArviZ
        print("Computing HDI intervals using ArviZ...") # Consistent print statement

        # Convert JAX draws to NumPy for arviz
        # trend_draws_np shape: (num_draws, T, n_trends)
        # stationary_draws_np shape: (num_draws, T, n_stationary)
        trend_draws_np = np.asarray(trend_draws)
        stationary_draws_np = np.asarray(stationary_draws)

        # Use the dedicated helper function to compute and format HDI
        trend_hdi = _compute_and_format_hdi_az(trend_draws_np, hdi_prob=0.9)
        stationary_hdi = _compute_and_format_hdi_az(stationary_draws_np, hdi_prob=0.9)

        # The helper handles the case of insufficient draws and returns NaNs
        # We can add a check if you need to know if HDI was successfully computed (i.e., not all NaNs)
        if np.any(np.isnan(trend_hdi['low'])) or np.any(np.isnan(stationary_hdi['low'])):
            print("Warning: HDI computation resulted in NaNs.")
            # If you want to treat all-NaN HDI as equivalent to None HDI for plotting, you could do:
            # trend_hdi = None if np.all(np.isnan(trend_hdi['low'])) else trend_hdi
            # stationary_hdi = None if np.all(np.isnan(stationary_hdi['low'])) else stationary_hdi
        else:
            print("HDI computed successfully using ArviZ!") # Success print

        print(f"Trend component draws shape: {trend_draws.shape}")
        print(f"Stationary component draws shape: {stationary_draws.shape}")       
        # print(f"Trend component draws shape: {trend_draws.shape}")
        # print(f"Stationary component draws shape: {stationary_draws.shape}")

    except Exception as e:
        print(f"Error during component extraction: {e}")
        # Return empty arrays if extraction fails
        T, n_obs = y.shape
        n_trends = ss_builder.n_trends
        n_stationary = ss_builder.n_stationary
        trend_draws = jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)
        trend_hdi = None
        stationary_hdi = None
    
    return {
        'mcmc': mcmc,
        'gpm_model': gpm_model,
        'ss_builder': ss_builder,
        'trend_draws': trend_draws,
        'stationary_draws': stationary_draws,
        'trend_hdi': trend_hdi,
        'stationary_hdi': stationary_hdi
    }


## This uses gamma matrices for initial conditions and covariance (still working on this)
def fit_gpm_model_with_smoother_with_gamma0(gpm_file_path: str, y: jnp.ndarray, 
                               num_warmup: int = 1000, num_samples: int = 2000, 
                               num_chains: int = 4, num_extract_draws: int = 100,
                               rng_key: jnp.ndarray = random.PRNGKey(0)):
    """
    Fit a GPM-based BVAR model and extract trend/stationary components using simulation smoother.
    UPDATED to handle gamma matrices.
    """
    
    print(f"Parsing GPM file: {gpm_file_path}")
    
    # Parse GPM file
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    
    # Create state space builder
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.trend_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {gpm_model.observed_variables}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")
    
    # Create the Numpyro model function
    def gpm_bvar_model(y: jnp.ndarray):
        """Numpyro model based on GPM specification with gamma matrix initialization"""
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample shock standard deviations and build covariance matrices
        Sigma_eta = _sample_trend_covariance(gpm_model)
        
        # CORRECTED: Now properly unpacks 3 return values including gamma_list
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # Sample initial conditions using gamma matrices
        init_mean = _sample_initial_conditions_with_gammas(
            gpm_model, ss_builder.state_dim, gamma_list, 
            ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order
        )
        init_cov = _create_initial_covariance_with_gammas(
            ss_builder.state_dim, ss_builder.n_trends, gamma_list,
            ss_builder.n_stationary, ss_builder.var_order
        )
        
        # Create parameter structure
        params = EnhancedBVARParams(
            A=A_transformed,
            Sigma_u=Sigma_u,
            Sigma_eta=Sigma_eta,
            structural_params=structural_params,
            Sigma_eps=Sigma_eps
        )
        
        # Build state space matrices
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        # Check for numerical issues
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        # Build R matrix from Q (assuming Q = R @ R.T)
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        # Create Kalman Filter
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        # Compute likelihood
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    print("Running MCMC...")
    kernel = NUTS(gpm_bvar_model)
    mcmc = MCMC(kernel, 
                num_warmup=num_warmup,
                num_samples=num_samples, 
                num_chains=num_chains)
    
    # Split random key for MCMC and component extraction
    mcmc_key, extract_key = random.split(rng_key)
    
    start_time = time.time()
    mcmc.run(mcmc_key, y=y)
    end_time = time.time()
    
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    mcmc.print_summary(exclude_deterministic=False)
    
    # Extract trend and stationary components
    print("Extracting trend and stationary components using Durbin & Koopman Simulation Smoother...")
    
    start_extract_time = time.time()
    
    # Check if we have enough posterior samples
    samples = mcmc.get_samples()
    total_posterior_draws = len(list(samples.values())[0])  # Get length from first parameter
    num_extract_draws = min(num_extract_draws, total_posterior_draws)
    
    if num_extract_draws < num_extract_draws:
        print(f"Warning: Reducing requested simulation smoother draws to {num_extract_draws} "
              f"as only {total_posterior_draws} posterior samples are available.")
    
    try:
        trend_draws, stationary_draws = extract_gpm_trends_and_components(
            mcmc, y, gpm_model, ss_builder, 
            num_draws=num_extract_draws, 
            rng_key=extract_key
        )
        
        end_extract_time = time.time()
        print(f"Component extraction completed in {end_extract_time - start_extract_time:.2f} seconds.")
        
        # # Compute HDI intervals
        # print("Computing HDI intervals using percentiles...")
        # if trend_draws.shape[0] > 1:
        #     trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=0.9)
        #     stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=0.9)
        #     print("HDI computed successfully using percentiles!")
        # else:
        #     trend_hdi = None
        #     stationary_hdi = None
        #     print("Not enough simulation smoother draws to compute HDI (need at least 2).")
        
        # Compute HDI intervals using ArviZ
        print("Computing HDI intervals using ArviZ...") # Consistent print statement

        # Convert JAX draws to NumPy for arviz
        # trend_draws_np shape: (num_draws, T, n_trends)
        # stationary_draws_np shape: (num_draws, T, n_stationary)
        trend_draws_np = np.asarray(trend_draws)
        stationary_draws_np = np.asarray(stationary_draws)

        # Use the dedicated helper function to compute and format HDI
        trend_hdi = _compute_and_format_hdi_az(trend_draws_np, hdi_prob=0.9)
        stationary_hdi = _compute_and_format_hdi_az(stationary_draws_np, hdi_prob=0.9)

        # The helper handles the case of insufficient draws and returns NaNs
        # We can add a check if you need to know if HDI was successfully computed (i.e., not all NaNs)
        if np.any(np.isnan(trend_hdi['low'])) or np.any(np.isnan(stationary_hdi['low'])):
            print("Warning: HDI computation resulted in NaNs.")
            # If you want to treat all-NaN HDI as equivalent to None HDI for plotting, you could do:
            # trend_hdi = None if np.all(np.isnan(trend_hdi['low'])) else trend_hdi
            # stationary_hdi = None if np.all(np.isnan(stationary_hdi['low'])) else stationary_hdi
        else:
            print("HDI computed successfully using ArviZ!") # Success print

        print(f"Trend component draws shape: {trend_draws.shape}")
        print(f"Stationary component draws shape: {stationary_draws.shape}")     
 
        
    except Exception as e:
        print(f"Error during component extraction: {e}")
        # Return empty arrays if extraction fails
        T, n_obs = y.shape
        n_trends = ss_builder.n_trends
        n_stationary = ss_builder.n_stationary
        trend_draws = jnp.empty((0, T, n_trends), dtype=_DEFAULT_DTYPE)
        stationary_draws = jnp.empty((0, T, n_stationary), dtype=_DEFAULT_DTYPE)
        trend_hdi = None
        stationary_hdi = None
    
    return {
        'mcmc': mcmc,
        'gpm_model': gpm_model,
        'ss_builder': ss_builder,
        'trend_draws': trend_draws,
        'stationary_draws': stationary_draws,
        'trend_hdi': trend_hdi,
        'stationary_hdi': stationary_hdi
    }



def complete_gpm_workflow(data_file: str = 'sim_data.csv', 
                         gpm_file: str = 'auto_model.gpm',
                         num_warmup: int = 500,
                         num_samples: int = 1000,
                         num_chains: int = 2,
                         num_extract_draws: int = 50,
                         generate_plots: bool = True):
    """
    Complete GPM workflow that includes data loading, model fitting, 
    component extraction, and optional plotting.
    """
    
    # Read data
    print(f"Reading data from {data_file}...")
    try:
        dta = pd.read_csv(data_file)
        y_np = dta.values
        y_jax = jnp.asarray(y_np)
        T, n_vars = y_np.shape
        print(f"Data shape: {y_np.shape}")
    except Exception as e:
        print(f"Error reading data: {e}")
        return None
    
    # Create GPM file if it doesn't exist
    if not os.path.exists(gpm_file):
        print(f"Creating GPM file: {gpm_file}")
        create_default_gpm_file(gpm_file, n_vars)
    
    # Fit model and extract components
    try:
        results = fit_gpm_model_with_smoother(
            gpm_file, y_jax,
            num_warmup=num_warmup,
            num_samples=num_samples, 
            num_chains=num_chains,
            num_extract_draws=num_extract_draws,
            rng_key=random.PRNGKey(42)
        )
        
        if results is None:
            print("Model fitting failed.")
            return None
            
        # Generate plots if requested and available
        if generate_plots and PLOTTING_AVAILABLE and results['trend_draws'].shape[0] > 0:
            print("Generating plots...")
            
            # Create variable names
            variable_names = [f'Var_{i+1}' for i in range(n_vars)]
            
            try:
                plot_observed_and_trend(
                    y_np=y_np,
                    trend_draws=results['trend_draws'],
                    trend_hdi=results['trend_hdi'],
                    variable_names=variable_names
                )
                print("Trend plots generated successfully.")
            except Exception as plot_error:
                print(f"Error generating plots: {plot_error}")
        
        

        print("\n=== RESULTS SUMMARY ===")
        print(f"Successfully fitted GPM model with {results['trend_draws'].shape[0]} component draws")
        print(f"Trend variables: {results['gpm_model'].trend_variables}")
        print(f"Stationary variables: {results['gpm_model'].stationary_variables}")
        
        return results
        
    except Exception as e:
        print(f"Error in complete workflow: {e}")
        return None


def create_default_gpm_file(filename: str, n_vars: int):
    """Create a default GPM file for n_vars variables"""
    
    gpm_content = f'''
parameters ;

estimated_params;
'''
    
    # Add shock standard deviations
    for i in range(1, n_vars + 1):
        gpm_content += f"    stderr SHK_TREND{i}, inv_gamma_pdf, 2.1, 0.81;\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    stderr SHK_STAT{i}, inv_gamma_pdf, 2.1, 0.38;\n"
    
    gpm_content += "end;\n\n"
    
    # Add trend variables
    gpm_content += "trends_vars\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    TREND{i}"
        if i < n_vars:
            gpm_content += ",\n"
        else:
            gpm_content += "\n"
    gpm_content += ";\n\n"
    
    # Add stationary variables
    gpm_content += "stationary_variables\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    STAT{i}"
        if i < n_vars:
            gpm_content += ",\n"
        else:
            gpm_content += "\n"
    gpm_content += ";\n\n"
    
    # Add trend shocks
    gpm_content += "trend_shocks;\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    var SHK_TREND{i}\n"
    gpm_content += "end;\n\n"
    
    # Add stationary shocks  
    gpm_content += "shocks;\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    var SHK_STAT{i}\n"
    gpm_content += "end;\n\n"
    
    # Add trend model
    gpm_content += "trend_model;\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    TREND{i} = TREND{i}(-1) + SHK_TREND{i};\n"
    gpm_content += "end;\n\n"
    
    # Add observed variables
    gpm_content += "varobs\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    OBS{i}"
        if i < n_vars:
            gpm_content += "\n"
        else:
            gpm_content += "\n"
    gpm_content += ";\n\n"
    
    # Add measurement equations
    gpm_content += "measurement_equations;\n"
    for i in range(1, n_vars + 1):
        gpm_content += f"    OBS{i} = TREND{i} + STAT{i};\n"
    gpm_content += "end;\n\n"
    
    # Add VAR prior setup
    gpm_content += """var_prior_setup;
    var_order = 2;
    es = 0.5, 0.3;
    fs = 0.5, 0.5;
    gs = 3.0, 3.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;
"""
    
    with open(filename, 'w') as f:
        f.write(gpm_content)
    
    print(f"Created default GPM file: {filename}")


def generate_synthetic_data(T: int = 100, n_vars: int = 3, filename: str = 'sim_data.csv'):
    """Generate synthetic data for testing"""
    
    np_rng = np.random.default_rng(123)
    
    # Generate trends (random walks)
    trends_np = np.cumsum(np_rng.normal(loc=0, scale=0.1, size=(T, n_vars)), axis=0)
    
    # Generate stationary component (simple AR(1) for each variable)
    stationary_np = np.zeros((T, n_vars))
    for i in range(n_vars):
        phi = 0.7  # AR coefficient
        for t in range(1, T):
            stationary_np[t, i] = phi * stationary_np[t-1, i] + np_rng.normal(0, 0.2)
    
    # Combine components
    y_np = trends_np + stationary_np
    
    # Save to CSV
    df = pd.DataFrame(y_np, columns=[f'OBS{i+1}' for i in range(n_vars)])
    df.to_csv(filename, index=False)
    
    print(f"Generated synthetic data: {filename} with shape {y_np.shape}")
    return y_np


def quick_example():
    """Quick example using the complete workflow (with data generation if needed)"""
    
    # Generate data if it doesn't exist
    if not os.path.exists('sim_data.csv'):
        print("Generating synthetic data...")
        generate_synthetic_data()
    
    # Run the complete workflow
    results = complete_gpm_workflow(
        data_file='sim_data.csv',
        gpm_file='example_model.gpm',
        num_warmup=200,
        num_samples=400, 
        num_chains=2,
        num_extract_draws=20,
        generate_plots=True
    )
    
    if results:
        print("\n=== EXTRACTION SUMMARY ===")
        print(f"Extracted {results['trend_draws'].shape[0]} trend component draws")
        print(f"Extracted {results['stationary_draws'].shape[0]} stationary component draws")
        
        if results['trend_hdi'] is not None:
            print("HDI intervals computed successfully")
        else:
            print("HDI intervals not available")
    
    return results


if __name__ == "__main__":
    print("Running complete GPM workflow with simulation smoother...")
    results = quick_example()
    
    if results:
        print("\n=== SUCCESS ===")
        print("Complete GPM workflow with simulation smoother completed successfully!")
        print("Results include MCMC samples and extracted trend/stationary components.")
    else:
        print("\n=== FAILED ===")
        print("Complete GPM workflow encountered errors.")