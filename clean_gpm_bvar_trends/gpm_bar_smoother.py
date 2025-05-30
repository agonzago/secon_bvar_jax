# clean_gpm_bvar_trends/gpm_bar_smoother.py

import jax
import jax.numpy as jnp # For data conversion if needed, and PRNGKey
import jax.random as random # For PRNGKey
import numpy as np # For general numpy operations, e.g., in plotting data prep
import pandas as pd # For data loading
import time
import os # For file operations like exists, remove

# Imports from your refactored library
from gpm_numpyro_models import fit_gpm_numpyro_model # Main fitting function
# ReducedModel and StateSpaceBuilder are returned by fit_gpm_numpyro_model

from simulation_smoothing import extract_reconstructed_components, _compute_and_format_hdi_az
# KalmanFilter might not be directly needed here if all smoothing is via extract_reconstructed_components

# Plotting (assuming it's in the same package or adjust import)
try:
    from reporting_plots import plot_observed_and_trend, plot_estimated_components # Using the newer functions
    # If plot_decomposition_results was the old name for plot_observed_vs_true_decomposition
    # and you have true data, you might import that too.
    # from .reporting_plots import plot_observed_vs_true_decomposition
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: gpm_bar_smoother: Plotting functions not available from .reporting_plots.")
    PLOTTING_AVAILABLE = False

from constants import _DEFAULT_DTYPE

# --- Workflow Functions ---

def create_default_gpm_file_if_needed(filename: str, num_obs_vars: int, num_stat_vars: int = 0):
    """Creates a simplified default GPM file if it doesn't exist."""
    if os.path.exists(filename):
        return

    # Basic parameter
    gpm_content = "parameters rho;\n" # Example parameter
    gpm_content += "estimated_params;\n"
    gpm_content += "    rho, normal_pdf, 0.5, 0.2;\n" # Prior for rho

    # Trend variables and shocks
    gpm_content += "trends_vars " + ", ".join([f"TREND{i+1}" for i in range(num_obs_vars)]) + ";\n"
    gpm_content += "trend_shocks;\n"
    for i in range(num_obs_vars):
        gpm_content += f"    var SHK_TREND{i+1};\n"
        gpm_content += f"    stderr SHK_TREND{i+1}, inv_gamma_pdf, 2.0, 0.02;\n" # Add prior for shock
    gpm_content += "end;\n"

    # Stationary variables and shocks (match num_stat_vars)
    if num_stat_vars > 0:
        gpm_content += "stationary_variables " + ", ".join([f"STAT{i+1}" for i in range(num_stat_vars)]) + ";\n"
        gpm_content += "shocks;\n"
        for i in range(num_stat_vars):
            gpm_content += f"    var SHK_STAT{i+1};\n"
            gpm_content += f"    stderr SHK_STAT{i+1}, inv_gamma_pdf, 2.0, 0.1;\n" # Add prior
        gpm_content += "end;\n"
    else: # No stationary variables, provide empty blocks if parser expects them
        gpm_content += "stationary_variables ;\n" # Empty but valid
        gpm_content += "shocks; end;\n"


    # Trend model (simple random walks or AR1s)
    gpm_content += "trend_model;\n"
    for i in range(num_obs_vars):
        gpm_content += f"    TREND{i+1} = TREND{i+1}(-1) + SHK_TREND{i+1};\n"
    gpm_content += "end;\n"

    # Observed variables
    gpm_content += "varobs " + ", ".join([f"OBS{i+1}" for i in range(num_obs_vars)]) + ";\n"

    # Measurement equations
    gpm_content += "measurement_equations;\n"
    for i in range(num_obs_vars):
        stat_term = f" + STAT{i+1}" if i < num_stat_vars else "" # Add stat term only if enough stat_vars
        gpm_content += f"    OBS{i+1} = TREND{i+1}{stat_term};\n"
    gpm_content += "end;\n"

    # VAR prior setup (only if stationary variables exist)
    if num_stat_vars > 0:
        gpm_content += """var_prior_setup;
    var_order = 1; 
    es = 0.5, 0.1; fs = 0.5, 0.5; 
    gs = 3.0, 3.0; hs = 1.0, 1.0; 
    eta = 2.0; 
end;
"""
    else: # Provide empty block if parser requires it
        gpm_content += "var_prior_setup; end;\n"


    gpm_content += "initial_values; TREND1, normal_pdf, 0, 1; end;\n" # Example initval

    with open(filename, 'w') as f:
        f.write(gpm_content)
    print(f"Created default GPM file: {filename} for {num_obs_vars} obs and {num_stat_vars} stat vars.")


def generate_synthetic_data_for_gpm(
    gpm_file_path: str, # To get structure via orchestrator
    true_params: Dict[str, TypingAny], # Builder-friendly keys
    num_steps: int = 150,
    rng_key_seed: int = 42
    ) -> Optional[jnp.ndarray]:
    """Generates synthetic data from a GPM specification and true parameters."""
    if simulate_state_space is None: # simulate_state_space needs to be imported
        print("ERROR: simulate_state_space (from Kalman_filter_jax) not available. Cannot generate data.")
        return None
    try:
        orchestrator = IntegrationOrchestrator(gpm_file_path)
        F_true, Q_true, C_true, H_true = orchestrator.build_ss_from_direct_dict(true_params)
        
        Q_true_reg = (Q_true + Q_true.T) / 2.0 + _JITTER * jnp.eye(Q_true.shape[0])
        R_true = jnp.linalg.cholesky(Q_true_reg)

        init_x_true = jnp.zeros(orchestrator.state_dim, dtype=_DEFAULT_DTYPE) # Simple init for sim
        init_P_true = jnp.eye(orchestrator.state_dim, dtype=_DEFAULT_DTYPE) * 0.01

        sim_key = random.PRNGKey(rng_key_seed)
        _, y_sim = simulate_state_space(
            P_aug=F_true, R_aug=R_true, Omega=C_true, H_obs=H_true,
            init_x=init_x_true, init_P=init_P_true,
            key=sim_key, num_steps=num_steps
        )
        print(f"Successfully generated synthetic data with shape: {y_sim.shape}")
        return y_sim
    except Exception as e:
        import traceback
        print(f"Error generating synthetic data: {e}")
        traceback.print_exc()
        return None


def complete_gpm_workflow_with_smoother(
    data_source: TypingAny, # Can be path to CSV string, or a preloaded DataFrame/ndarray
    gpm_file: str = 'model_for_smoother.gpm',
    num_obs_for_default_gpm: int = 2, # Used if creating default GPM
    num_stat_for_default_gpm: int = 2, # Used if creating default GPM
    # MCMC settings
    num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2,
    rng_seed_mcmc: int = 0, target_accept_prob:float = 0.85,
    # P0 initialization settings
    use_gamma_init: bool = False, gamma_scale_factor: float = 0.1,
    # Smoother settings
    num_extract_draws: int = 100, rng_seed_smoother: int = 42,
    # Plotting
    generate_plots: bool = True, hdi_prob_plot: float = 0.9
):
    """
    Complete workflow: Load/Generate Data -> Fit GPM -> Extract Components -> Plot.
    """
    # 1. Load or Generate Data
    if isinstance(data_source, str) and os.path.exists(data_source):
        print(f"Reading data from CSV: {data_source}...")
        try:
            dta = pd.read_csv(data_source)
            y_numpy = dta.values.astype(_DEFAULT_DTYPE) # Ensure correct dtype
        except Exception as e:
            print(f"Error reading data CSV {data_source}: {e}"); return None
    elif isinstance(data_source, pd.DataFrame):
        y_numpy = data_source.values.astype(_DEFAULT_DTYPE)
    elif isinstance(data_source, (np.ndarray, jnp.ndarray)):
        y_numpy = np.asarray(data_source, dtype=_DEFAULT_DTYPE)
    else:
        print(f"Invalid data_source: {data_source}. Must be CSV path, DataFrame, or ndarray.")
        return None
        
    y_jax = jnp.asarray(y_numpy)
    T_actual, N_actual_obs = y_jax.shape
    print(f"Data loaded/used, shape: ({T_actual}, {N_actual_obs})")

    # 2. Ensure GPM file exists (create a default if not)
    create_default_gpm_file_if_needed(gpm_file, N_actual_obs, num_stat_for_default_gpm)

    # 3. Fit GPM model using NumPyro
    print(f"\n--- Fitting GPM Model: {gpm_file} ---")
    try:
        mcmc_results, parsed_gpm_model, state_space_builder = fit_gpm_numpyro_model(
            gpm_file_path=gpm_file,
            y_data=y_jax,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            rng_key_seed=rng_seed_mcmc,
            use_gamma_init_for_P0=use_gamma_init,
            gamma_init_scaling_for_P0=gamma_scale_factor,
            target_accept_prob=target_accept_prob
        )
        if mcmc_results is None: raise RuntimeError("MCMC fitting failed to return results.")
        # mcmc_results.print_summary() # Optional summary
    except Exception as e_fit:
        import traceback
        print(f"Error during GPM model fitting: {e_fit}")
        traceback.print_exc(); return None
    
    # 4. Extract and Reconstruct Components using Simulation Smoother
    print(f"\n--- Extracting and Reconstructing Components ---")
    # Ensure the number of draws for smoother doesn't exceed available MCMC samples
    total_mcmc_samples = mcmc_results.get_samples(group_by_chain=False)[list(mcmc_results.get_samples().keys())[0]].shape[0]
    actual_extract_draws = min(num_extract_draws, total_mcmc_samples)
    
    if actual_extract_draws > 0:
        try:
            rng_key_for_smoother = random.PRNGKey(rng_seed_smoother)
            all_trends_draws, all_stationary_draws, component_names = extract_reconstructed_components(
                mcmc_output=mcmc_results,
                y_data=y_jax,
                gpm_model=parsed_gpm_model,
                ss_builder=state_space_builder,
                num_smooth_draws=actual_extract_draws,
                rng_key_smooth=rng_key_for_smoother
            )
            # print(f"Smoothed components extracted. Trends shape: {all_trends_draws.shape}, Stat shape: {all_stationary_draws.shape}")
        except Exception as e_smooth:
            import traceback
            print(f"Error during component extraction/reconstruction: {e_smooth}")
            traceback.print_exc()
            all_trends_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_trend_variables_original)))
            all_stationary_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_stationary_variables_original)))
            component_names = {'trends': [], 'stationary': []}
    else:
        print("Skipping component extraction due to insufficient MCMC samples or num_extract_draws=0.")
        all_trends_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_trend_variables_original)))
        all_stationary_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_stationary_variables_original)))
        component_names = {'trends': list(parsed_gpm_model.gpm_trend_variables_original), 
                           'stationary': list(parsed_gpm_model.gpm_stationary_variables_original)}


    # 5. Generate Plots (using reporting_plots.py)
    if generate_plots and PLOTTING_AVAILABLE:
        print("\n--- Generating Plots ---")
        if all_trends_draws.shape[0] > 0 or all_stationary_draws.shape[0] > 0:
            # Observed data vs. Fitted (Trend + Stationary)
            # For this, we need to combine trend and stationary draws if they are separate state components
            # that sum up to the observed.
            # If your measurement eq is y_obs = trend_i + stat_i, then sum the corresponding trend/stat draws.
            # This example assumes a simple 1-to-1 mapping for plotting against observed.
            
            # Plot 1: Estimated Components (All original trends and all original stationary vars)
            try:
                plot_estimated_components(
                    trend_draws=all_trends_draws, # These are now ALL original trends
                    stationary_draws=all_stationary_draws, # ALL original stationary
                    hdi_prob=hdi_prob_plot,
                    trend_variable_names=component_names['trends'],
                    stationary_variable_names=component_names['stationary']
                )
            except Exception as e_plot1: print(f"Error plotting estimated components: {e_plot1}")

            # Plot 2: Observed vs. Trend (for each observed series, plot against its main trend component)
            # This requires identifying which of the `all_trends_draws` corresponds to each `y_numpy[:, i]`
            # For a simple GPM like y_obs_i = TREND_i + STAT_i, it's straightforward.
            # For more complex MEs, this mapping needs care.
            # Assuming here a direct correspondence for example purposes.
            obs_var_names_for_plot = parsed_gpm_model.gpm_observed_variables_original
            if len(obs_var_names_for_plot) == all_trends_draws.shape[2] and all_trends_draws.shape[0]>0: # If num obs == num trends plotted
                try:
                    plot_observed_and_trend(
                        y_np=y_numpy,
                        trend_draws=all_trends_draws, # Pass all reconstructed trends
                        hdi_prob=hdi_prob_plot,
                        variable_names=obs_var_names_for_plot # Names of the *observed* variables
                    )
                except Exception as e_plot2: print(f"Error plotting observed and trend: {e_plot2}")
            # else: print("Skipping observed vs trend plot due to mismatch in trend/obs count or no trend draws.")

        else:
            print("No smoothed component draws available for plotting.")
    elif generate_plots:
        print("Plotting skipped as PLOTTING_AVAILABLE is False.")

    results_dict = {
        'mcmc_object': mcmc_results,
        'parsed_gpm_model': parsed_gpm_model,
        'state_space_builder': state_space_builder,
        'reconstructed_trend_draws': all_trends_draws,
        'reconstructed_stationary_draws': all_stationary_draws,
        'component_names': component_names,
        'observed_data': y_numpy
    }
    print("\n--- Workflow Complete ---")
    return results_dict


def quick_test_smoother_workflow():
    """A quick example to test the complete workflow."""
    print("--- Running Quick Test Smoother Workflow ---")
    
    # Define parameters for a default GPM and data simulation
    num_obs_vars_example = 2
    num_stat_vars_example = 2
    gpm_file_example = "smoother_test_model.gpm"
    
    # Create a default GPM file for this test
    create_default_gpm_file_if_needed(gpm_file_example, num_obs_vars_example, num_stat_vars_example)

    # Define some "true" parameters for data simulation
    # These keys should be what StateSpaceBuilder._standardize_direct_params can handle
    # (i.e., MCMC names like "sigma_SHK_..." or builder names like "SHK_...")
    true_params_for_sim = {
        "rho": 0.5, # Example structural param from default GPM
        # Standard deviations for shocks
        "SHK_TREND1": 0.1, "SHK_TREND2": 0.15,
        "SHK_STAT1": 0.2, "SHK_STAT2": 0.25,
        # VAR parameters for a 2-var VAR(1)
        "_var_coefficients": jnp.array([[[0.7, 0.1], [0.0, 0.6]]]), # (order, n_stat, n_stat)
        "_var_innovation_corr_chol": jnp.array([[1.0, 0.0],[0.3, jnp.sqrt(1-0.3**2)]]) # Cholesky of a corr matrix
    }

    # Generate synthetic data
    simulated_y = generate_synthetic_data_for_gpm(
        gpm_file_path=gpm_file_example,
        true_params=true_params_for_sim,
        num_steps=200,
        rng_key_seed=789
    )

    if simulated_y is None:
        print("Failed to generate synthetic data for quick test.")
        return

    # Run the complete workflow
    results = complete_gpm_workflow_with_smoother(
        data_source=simulated_y,
        gpm_file=gpm_file_example,
        num_warmup=100,       # Keep low for quick test
        num_samples=150,      # Keep low
        num_chains=1,         # Keep low
        num_extract_draws=50, # Extract a few draws
        generate_plots=True   # Enable plotting if available
    )

    if results:
        print("\nQuick test workflow successful.")
        # print(f"Trend draws shape: {results['reconstructed_trend_draws'].shape}")
        # print(f"Stationary draws shape: {results['reconstructed_stationary_draws'].shape}")
    else:
        print("\nQuick test workflow failed.")
    
    # Clean up dummy GPM
    if os.path.exists(gpm_file_example):
        try: os.remove(gpm_file_example)
        except: pass

if __name__ == "__main__":
    # To run this, you'd need data (e.g., 'my_data.csv') and a GPM file.
    # Example:
    # create_default_gpm_file_if_needed("default_model.gpm", num_obs_vars=2, num_stat_vars=2)
    # complete_gpm_workflow_with_smoother(
    # data_source="my_data.csv", # or preloaded numpy array
    # gpm_file="default_model.gpm",
    # num_warmup=50, num_samples=100, num_chains=1, num_extract_draws=20
    # )
    quick_test_smoother_workflow()