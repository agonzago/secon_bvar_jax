import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax 
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import pandas as pd
import time
import os

# Import your existing modules
# from gpm_parser import GPMParser, GPMModel
from new_parser.integration_helper import create_reduced_gpm_model, ReducedGPMIntegration
from gpm_bvar_trends import (
    GPMStateSpaceBuilder, EnhancedBVARParams, # fit_gpm_model (if still needed elsewhere)
    _sample_parameter, _sample_trend_covariance, _sample_var_parameters,
    _sample_measurement_covariance, _has_measurement_error,
    # Standard non-gamma initializers
    _sample_initial_conditions, 
    _create_initial_covariance,
    # Definitive gamma-based initializers (assuming you renamed them in gpm_bvar_trends.py)
    _sample_initial_conditions_with_gammas, 
    _create_initial_covariance_with_gammas,
    _DEFAULT_DTYPE, _JITTER, _KF_JITTER
)

# Import simulation smoother
from simulation_smoothing import (
    extract_gpm_trends_and_components, 
    compute_hdi_with_percentiles,
    _compute_and_format_hdi_az
)

import arviz as az
import xarray as xr
# Import Kalman Filter
try:
    from Kalman_filter_jax import KalmanFilter
except ImportError:
    print("Warning: Could not import KalmanFilter")

# Import plotting functions if available
try:
    from reporting_plots import plot_decomposition_results, plot_observed_and_trend
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Plotting functions not available. Skipping plot generation.")
    PLOTTING_AVAILABLE = False


def fit_gpm_model_with_smoother(gpm_file_path: str, y: jnp.ndarray,
                               num_warmup: int = 1000, num_samples: int = 2000,
                               num_chains: int = 4, num_extract_draws: int = 100,
                               rng_key: jnp.ndarray = random.PRNGKey(0)):
    """
    Fit a GPM-based BVAR model using STANDARD (non-gamma) P0 initialization.
    """
    print(f"Parsing GPM file: {gpm_file_path} for STANDARD P0 initialization")
    # parser = GPMParser() # Assuming old parser for now
    # gpm_model = parser.parse_file(gpm_file_path)
    # ss_builder = GPMStateSpaceBuilder(gpm_model)
    integration, gpm_model, ss_builder = create_reduced_gpm_model(gpm_file_path)
    # ... (GPM Model Summary print) ...

    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.core_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {list(gpm_model.reduced_measurement_equations.keys())}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")


    def gpm_bvar_model(y_data: jnp.ndarray): # Renamed y to y_data to avoid conflict
        T, n_obs = y_data.shape
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, _ = _sample_var_parameters(gpm_model) # gamma_list not used for P0 here
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # STANDARD P0 INITIALIZATION
        init_mean = _sample_initial_conditions(gpm_model, ss_builder.state_dim)
        init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        params = EnhancedBVARParams(A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
                                   structural_params=structural_params, Sigma_eps=Sigma_eps)
        #F, Q, C, H = ss_builder.build_state_space_matrices(params)

        F, Q, C, H = integration.build_state_space_matrices(params)

        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) &
                       jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) &
                       jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        try:
            R_mat = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R_mat = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        kf = KalmanFilter(T=F, R=R_mat, C=C, H=H, init_x=init_mean, init_P=init_cov)
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs_mat = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        loglik = lax.cond(~matrices_ok,
                          lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
                          lambda: kf.log_likelihood(y_data, valid_obs_idx, n_obs, C, H, I_obs_mat))
        numpyro.factor("loglik", loglik)

    print("Running MCMC (Standard P0 Initialization)...")
    kernel = NUTS(gpm_bvar_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    mcmc_key, extract_key = random.split(rng_key)
    start_time = time.time()
    mcmc.run(mcmc_key, y_data=y) # Pass y as y_data to the model
    end_time = time.time()
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    mcmc.print_summary(exclude_deterministic=False)

    # ... (Component extraction part - should be identical for both fitting functions) ...
    # This part can be refactored into a helper if desired, or duplicated.
    # For now, I'll include it, ensuring variable names are consistent.
    print("Extracting trend and stationary components...")
    start_extract_time = time.time()
    posterior_samples = mcmc.get_samples()
    total_draws_mcmc = 0
    if posterior_samples:
        first_key = list(posterior_samples.keys())[0]
        total_draws_mcmc = len(posterior_samples[first_key])
    
    actual_num_extract = min(num_extract_draws, total_draws_mcmc)

    if num_extract_draws > total_draws_mcmc and total_draws_mcmc > 0:
        print(f"Warning: Reducing smoother draws to {actual_num_extract} (available: {total_draws_mcmc}).")

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
                 trend_hdi_dict = _compute_and_format_hdi_az(np.asarray(trend_draws_arr), hdi_prob=0.9)
                 stationary_hdi_dict = _compute_and_format_hdi_az(np.asarray(stationary_draws_arr), hdi_prob=0.9)
                 if trend_hdi_dict and not np.any(np.isnan(trend_hdi_dict['low'])): print("HDI computed.")
                 else: print("Warning: HDI computation problems.")
        except Exception as e:
            print(f"Error in component extraction: {e}")
            # Ensure arrays are empty on error
            trend_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_trends), dtype=_DEFAULT_DTYPE)
            stationary_draws_arr = jnp.empty((0, y.shape[0], ss_builder.n_stationary), dtype=_DEFAULT_DTYPE)
    else:
        print("Skipping component extraction (no/few MCMC draws or num_extract_draws is 0).")
        
    return {
        'mcmc': mcmc, 'gpm_model': gpm_model, 'ss_builder': ss_builder,
        'trend_draws': trend_draws_arr, 'stationary_draws': stationary_draws_arr,
        'trend_hdi': trend_hdi_dict, 'stationary_hdi': stationary_hdi_dict
    }


def fit_gpm_model_with_smoother_with_gamma0(gpm_file_path: str, y: jnp.ndarray,
                               num_warmup: int = 1000, num_samples: int = 2000,
                               num_chains: int = 4, num_extract_draws: int = 100,
                               gamma_init_scaling: float = 0.1, 
                               rng_key: jnp.ndarray = random.PRNGKey(0)):
    """
    Fit a GPM-based BVAR model using GAMMA-BASED P0 initialization.
    """
    print(f"Parsing GPM file: {gpm_file_path} for GAMMA-BASED P0 initialization")
    # parser = GPMParser() # Assuming old parser
    # gpm_model = parser.parse_file(gpm_file_path)
    # ss_builder = GPMStateSpaceBuilder(gpm_model)
    integration, gpm_model, ss_builder = create_reduced_gpm_model(gpm_file_path)

    # ... (GPM Model Summary print - same as above) ...
    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.core_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {list(gpm_model.reduced_measurement_equations.keys())}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")

    def gpm_bvar_model(y_data: jnp.ndarray): # Renamed y to y_data
        T, n_obs = y_data.shape
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model) # gamma_list IS used for P0
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # GAMMA-BASED P0 INITIALIZATION
        init_mean = _sample_initial_conditions_with_gammas( # Using the definitive gamma version
            gpm_model, ss_builder.state_dim, gamma_list,
            ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order,
            gamma_scaling=gamma_init_scaling
        )
        init_cov = _create_initial_covariance_with_gammas( # Using the definitive gamma version
            ss_builder.state_dim, ss_builder.n_trends, gamma_list,
            ss_builder.n_stationary, ss_builder.var_order,
            use_gamma_scaling=gamma_init_scaling
        )
        
        params = EnhancedBVARParams(A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
                                   structural_params=structural_params, Sigma_eps=Sigma_eps)
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) &
                       jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) &
                       jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        try:
            R_mat = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R_mat = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        kf = KalmanFilter(T=F, R=R_mat, C=C, H=H, init_x=init_mean, init_P=init_cov)
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs_mat = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        loglik = lax.cond(~matrices_ok,
                          lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
                          lambda: kf.log_likelihood(y_data, valid_obs_idx, n_obs, C, H, I_obs_mat))
        numpyro.factor("loglik", loglik)

    print("Running MCMC (Gamma-based P0 Initialization)...")
    kernel = NUTS(gpm_bvar_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    mcmc_key, extract_key = random.split(rng_key)
    start_time = time.time()
    mcmc.run(mcmc_key, y_data=y) # Pass y as y_data to the model
    end_time = time.time()
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    mcmc.print_summary(exclude_deterministic=False)

    # ... (Component extraction part - should be identical to the one in fit_gpm_model_with_smoother) ...
    # For brevity, this is the same block as in fit_gpm_model_with_smoother
    print("Extracting trend and stationary components...")
    start_extract_time = time.time()
    posterior_samples = mcmc.get_samples()
    total_draws_mcmc = 0
    if posterior_samples:
        first_key = list(posterior_samples.keys())[0]
        total_draws_mcmc = len(posterior_samples[first_key])
    
    actual_num_extract = min(num_extract_draws, total_draws_mcmc)

    if num_extract_draws > total_draws_mcmc and total_draws_mcmc > 0:
        print(f"Warning: Reducing smoother draws to {actual_num_extract} (available: {total_draws_mcmc}).")

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
                 trend_hdi_dict = _compute_and_format_hdi_az(np.asarray(trend_draws_arr), hdi_prob=0.9)
                 stationary_hdi_dict = _compute_and_format_hdi_az(np.asarray(stationary_draws_arr), hdi_prob=0.9)
                 if trend_hdi_dict and not np.any(np.isnan(trend_hdi_dict['low'])): print("HDI computed.")
                 else: print("Warning: HDI computation problems.")
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


def fit_gpm_model_selectable_init(
    gpm_file_path: str, y: jnp.ndarray,
    num_warmup: int = 1000, num_samples: int = 2000,
    num_chains: int = 4, num_extract_draws: int = 100,
    use_gamma_initialization: bool = False, 
    gamma_init_scaling: float = 0.1,     
    rng_key: jnp.ndarray = random.PRNGKey(0)
):
    """
    Fits the GPM-BVAR model, allowing selection between standard P0 initialization
    and gamma-based P0 initialization.
    """
    if use_gamma_initialization:
        print("Calling fit_gpm_model_with_smoother_with_gamma0...")
        return fit_gpm_model_with_smoother_with_gamma0(
            gpm_file_path=gpm_file_path, y=y,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, num_extract_draws=num_extract_draws,
            gamma_init_scaling=gamma_init_scaling,
            rng_key=rng_key
        )
    else:
        print("Calling fit_gpm_model_with_smoother (Standard P0)...")
        return fit_gpm_model_with_smoother(
            gpm_file_path=gpm_file_path, y=y,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, num_extract_draws=num_extract_draws,
            rng_key=rng_key
        )
    
def complete_gpm_workflow(data_file: str = 'sim_data.csv',
                         gpm_file: str = 'auto_model.gpm',
                         num_warmup: int = 500,
                         num_samples: int = 1000,
                         num_chains: int = 2,
                         num_extract_draws: int = 50,
                         generate_plots: bool = True,
                         use_gamma_init: bool = False, # New parameter
                         gamma_scale_factor: float = 0.1 # New parameter
                         ):
    """
    Complete GPM workflow that includes data loading, model fitting,
    component extraction, and optional plotting.
    Allows selection of P0 initialization method.
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
        results = fit_gpm_model_selectable_init( # Call the new selectable function
            gpm_file, y_jax,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            num_extract_draws=num_extract_draws,
            use_gamma_initialization=use_gamma_init, # Pass the flag
            gamma_init_scaling=gamma_scale_factor,  # Pass the scaling
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
        use_gamma_init=False, # Pass the flag
        gamma_scale_factor=0.01,  # Pass the scaling ( use small values to approximate )
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