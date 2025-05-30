# clean_gpm_bvar_trends/gpm_bar_smoother.py

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Optional, Any, Tuple

# Imports from the current refactored library structure
from gpm_numpyro_models import fit_gpm_numpyro_model, define_gpm_numpyro_model
from integration_orchestrator import IntegrationOrchestrator, create_integration_orchestrator
from simulation_smoothing import extract_reconstructed_components, _compute_and_format_hdi_az
from constants import _DEFAULT_DTYPE

# Plotting imports
try:
    from reporting_plots import (
        plot_observed_and_trend, 
        plot_estimated_components,
        plot_observed_and_fitted        
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: gpm_bar_smoother: Plotting functions not available from reporting_plots.")
    PLOTTING_AVAILABLE = False

# Optional simulation import
try:
    from Kalman_filter_jax import simulate_state_space
except ImportError:
    simulate_state_space = None
    print("Warning: simulate_state_space not available from Kalman_filter_jax")

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# --- Workflow Functions ---

def create_default_gpm_file_if_needed(filename: str, num_obs_vars: int, num_stat_vars: int = 0):
    """Creates a simplified default gpm file if it doesn't exist."""
    if os.path.exists(filename):
        return

    print(f"Creating default gpm file: {filename} with {num_obs_vars} obs vars and {num_stat_vars} stat vars")

    # Basic parameter and ALL estimated_params in one block
    gpm_content = "parameters rho;\n"
    gpm_content += "\nestimated_params;\n"
    gpm_content += "    rho, normal_pdf, 0.5, 0.2;\n"
    
    # Add ALL stderr priors inside estimated_params block
    for i in range(num_obs_vars):
        gpm_content += f"    stderr SHK_TREND{i+1}, inv_gamma_pdf, 2.0, 0.02;\n"
    
    for i in range(num_stat_vars):
        gpm_content += f"    stderr SHK_STAT{i+1}, inv_gamma_pdf, 2.0, 0.1;\n"
    
    gpm_content += "end;\n"

    # Trend variables and shocks
    trend_names = [f"TREND{i+1}" for i in range(num_obs_vars)]
    gpm_content += f"\ntrends_vars {', '.join(trend_names)};\n"
    
    gpm_content += "\ntrend_shocks;\n"
    for i in range(num_obs_vars):
        gpm_content += f"    var SHK_TREND{i+1};\n"
    gpm_content += "end;\n"

    # Stationary variables and shocks
    if num_stat_vars > 0:
        stat_names = [f"STAT{i+1}" for i in range(num_stat_vars)]
        gpm_content += f"\nstationary_variables {', '.join(stat_names)};\n"
        
        gpm_content += "\nshocks;\n"
        for i in range(num_stat_vars):
            gpm_content += f"    var SHK_STAT{i+1};\n"
        gpm_content += "end;\n"
    else:
        gpm_content += "\nstationary_variables ;\n"
        gpm_content += "\nshocks;\nend;\n"

    # Trend model (simple random walks)
    gpm_content += "\ntrend_model;\n"
    for i in range(num_obs_vars):
        gpm_content += f"    TREND{i+1} = TREND{i+1}(-1) + SHK_TREND{i+1};\n"
    gpm_content += "end;\n"

    # Observed variables
    obs_names = [f"OBS{i+1}" for i in range(num_obs_vars)]
    gpm_content += f"\nvarobs {', '.join(obs_names)};\n"

    # Measurement equations
    gpm_content += "\nmeasurement_equations;\n"
    for i in range(num_obs_vars):
        stat_term = f" + STAT{i+1}" if i < num_stat_vars else ""
        gpm_content += f"    OBS{i+1} = TREND{i+1}{stat_term};\n"
    gpm_content += "end;\n"

    # VAR prior setup (only if stationary variables exist)
    if num_stat_vars > 0:
        gpm_content += """\nvar_prior_setup;
    var_order = 1; 
    es = 0.5, 0.1; 
    fs = 0.5, 0.5; 
    gs = 3.0, 3.0; 
    hs = 1.0, 1.0; 
    eta = 2.0; 
end;
"""

    # Initial values
    gpm_content += "\ninitval;\n"
    for i in range(num_obs_vars):
        gpm_content += f"    TREND{i+1}, normal_pdf, 0, 1;\n"
    gpm_content += "end;\n"

    with open(filename, 'w') as f:
        f.write(gpm_content)
    print(f"✓ Created default gpm file: {filename}")


def generate_synthetic_data_for_gpm(
    gpm_file_path: str,
    true_params: Dict[str, Any],
    num_steps: int = 150,
    rng_key_seed: int = 42
) -> Optional[jnp.ndarray]:
    """Generates synthetic data from a gpm specification and true parameters."""
    if simulate_state_space is None:
        print("ERROR: simulate_state_space not available. Cannot generate synthetic data.")
        return None
    
    try:
        orchestrator = create_integration_orchestrator(gpm_file_path)
        F_true, Q_true, C_true, H_true = orchestrator.build_ss_from_direct_dict(true_params)
        
        # Regularize Q for Cholesky
        Q_true_reg = (Q_true + Q_true.T) / 2.0 + 1e-8 * jnp.eye(Q_true.shape[0])
        R_true = jnp.linalg.cholesky(Q_true_reg)

        init_x_true = jnp.zeros(orchestrator.state_dim, dtype=_DEFAULT_DTYPE)
        init_P_true = jnp.eye(orchestrator.state_dim, dtype=_DEFAULT_DTYPE) * 0.01

        sim_key = random.PRNGKey(rng_key_seed)
        _, y_sim = simulate_state_space(
            P_aug=F_true, R_aug=R_true, Omega=C_true, H_obs=H_true,
            init_x=init_x_true, init_P=init_P_true,
            key=sim_key, num_steps=num_steps
        )
        print(f"✓ Generated synthetic data with shape: {y_sim.shape}")
        return y_sim
    except Exception as e:
        import traceback
        print(f"Error generating synthetic data: {e}")
        traceback.print_exc()
        return None


def complete_gpm_workflow_with_smoother(
    data_source: Any,
    gpm_file: str = 'model_for_smoother.gpm',
    num_obs_for_default_gpm: int = 2,
    num_stat_for_default_gpm: int = 2,
    # MCMC settings
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    rng_seed_mcmc: int = 0,
    target_accept_prob: float = 0.85,
    # P0 initialization settings
    use_gamma_init: bool = False,
    gamma_scale_factor: float = 0.1,
    # Smoother settings
    num_extract_draws: int = 100,
    rng_seed_smoother: int = 42,
    # Plotting
    generate_plots: bool = True,
    hdi_prob_plot: float = 0.9
) -> Optional[Dict[str, Any]]:
    """
    Complete workflow: Load/Generate Data -> Fit gpm -> Extract Components -> Plot.
    """
    print("=== Starting Complete gpm Workflow with Smoother ===")
    
    # 1. Load or Generate Data
    print("\n1. Loading/Processing Data...")
    if isinstance(data_source, str) and os.path.exists(data_source):
        print(f"   Reading data from CSV: {data_source}")
        try:
            dta = pd.read_csv(data_source)
            y_numpy = dta.values.astype(_DEFAULT_DTYPE)
        except Exception as e:
            print(f"   ERROR: Could not read CSV {data_source}: {e}")
            return None
    elif isinstance(data_source, pd.DataFrame):
        print("   Using provided DataFrame")
        y_numpy = data_source.values.astype(_DEFAULT_DTYPE)
    elif isinstance(data_source, (np.ndarray, jnp.ndarray)):
        print("   Using provided array")
        y_numpy = np.asarray(data_source, dtype=_DEFAULT_DTYPE)
    else:
        print(f"   ERROR: Invalid data_source type: {type(data_source)}")
        return None
        
    y_jax = jnp.asarray(y_numpy)
    T_actual, N_actual_obs = y_jax.shape
    print(f"   ✓ Data shape: ({T_actual}, {N_actual_obs})")

    # 2. Ensure gpm file exists
    print(f"\n2. Checking gpm file: {gpm_file}")
    create_default_gpm_file_if_needed(gpm_file, N_actual_obs, num_stat_for_default_gpm)

    # 3. Fit gpm model using NumPyro
    print(f"\n3. Fitting gpm Model...")
    print(f"   MCMC settings: {num_warmup} warmup, {num_samples} samples, {num_chains} chains")
    print(f"   Gamma init P0: {use_gamma_init} (scaling: {gamma_scale_factor})")
    
    try:
        start_time = time.time()
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
        fit_time = time.time() - start_time
        print(f"   ✓ MCMC completed in {fit_time:.1f}s")
        
        mcmc_results.print_summary()
        if mcmc_results is None:
            raise RuntimeError("MCMC fitting returned None")
            
    except Exception as e:
        import traceback
        print(f"   ERROR during gpm model fitting: {e}")
        traceback.print_exc()
        return None
    
    # 4. Extract and Reconstruct Components
    print(f"\n4. Extracting Components via Simulation Smoother...")
    
    # Check available MCMC samples
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    if not mcmc_samples:
        print("   ERROR: No MCMC samples available")
        return None
        
    total_mcmc_samples = list(mcmc_samples.values())[0].shape[0]
    actual_extract_draws = min(num_extract_draws, total_mcmc_samples)
    print(f"   Using {actual_extract_draws}/{total_mcmc_samples} available MCMC draws")
    
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
            print(f"   ✓ Extracted components:")
            print(f"     Trends: {all_trends_draws.shape}")
            print(f"     Stationary: {all_stationary_draws.shape}")
            
        except Exception as e:
            import traceback
            print(f"   ERROR during component extraction: {e}")
            traceback.print_exc()
            # Create empty arrays
            all_trends_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_trend_variables_original)))
            all_stationary_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_stationary_variables_original)))
            component_names = {'trends': [], 'stationary': []}
    else:
        print("   Skipping component extraction (no MCMC draws available)")
        all_trends_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_trend_variables_original)))
        all_stationary_draws = jnp.empty((0, T_actual, len(parsed_gpm_model.gpm_stationary_variables_original)))
        component_names = {
            'trends': list(parsed_gpm_model.gpm_trend_variables_original), 
            'stationary': list(parsed_gpm_model.gpm_stationary_variables_original)
        }

    # 5. Generate Plots
    if generate_plots and PLOTTING_AVAILABLE:
        print(f"\n5. Generating Plots...")
        
        if all_trends_draws.shape[0] > 0 or all_stationary_draws.shape[0] > 0:
            obs_var_names = parsed_gpm_model.gpm_observed_variables_original
            
            # Plot 1: Estimated Components
            if all_trends_draws.shape[0] > 0 and all_stationary_draws.shape[0] > 0:
                try:
                    print("   Creating estimated components plot...")
                    plot_estimated_components(
                        trend_draws=all_trends_draws,
                        stationary_draws=all_stationary_draws,
                        hdi_prob=hdi_prob_plot,
                        trend_variable_names=component_names['trends'],
                        stationary_variable_names=component_names['stationary']
                    )
                except Exception as e:
                    print(f"   Warning: Could not create estimated components plot: {e}")

            # Plot 2: Observed vs. Trend  
            if all_trends_draws.shape[0] > 0:
                try:
                    print("   Creating observed vs trend plot...")
                    plot_observed_and_trend(
                        y_np=y_numpy,
                        trend_draws=all_trends_draws,
                        hdi_prob=hdi_prob_plot,
                        variable_names=obs_var_names
                    )
                except Exception as e:
                    print(f"   Warning: Could not create observed vs trend plot: {e}")

            # Plot 3: Observed vs. Fitted
            if all_trends_draws.shape[0] > 0 and all_stationary_draws.shape[0] > 0:
                try:
                    print("   Creating observed vs fitted plot...")
                    plot_observed_and_fitted(
                        y_np=y_numpy,
                        trend_draws=all_trends_draws,
                        stationary_draws=all_stationary_draws,
                        hdi_prob=hdi_prob_plot,
                        variable_names=obs_var_names
                    )
                except Exception as e:
                    print(f"   Warning: Could not create observed vs fitted plot: {e}")
                    
            print("   ✓ Plotting completed")
        else:
            print("   Skipping plots (no component draws available)")
            
    elif generate_plots:
        print(f"\n5. Skipping plots (PLOTTING_AVAILABLE={PLOTTING_AVAILABLE})")

    # 6. Return Results
    results_dict = {
        'mcmc_object': mcmc_results,
        'parsed_gpm_model': parsed_gpm_model,
        'state_space_builder': state_space_builder,
        'reconstructed_trend_draws': all_trends_draws,
        'reconstructed_stationary_draws': all_stationary_draws,
        'component_names': component_names,
        'observed_data': y_numpy,
        'fitting_time_seconds': fit_time if 'fit_time' in locals() else None
    }
    
    print("\n=== Workflow Complete ===")
    return results_dict


def quick_test_smoother_workflow():
    """A quick example to test the complete workflow."""
    print("=== Running Quick Test Smoother Workflow ===")
    
    # Parameters
    num_obs_vars_example = 2
    num_stat_vars_example = 2
    gpm_file_example = "smoother_test_model.gpm"
    
    # Create default gpm file
    create_default_gpm_file_if_needed(gpm_file_example, num_obs_vars_example, num_stat_vars_example)

    # Define "true" parameters for data simulation
    true_params_for_sim = {
        "rho": 0.5,
        # Shock standard deviations (builder names)
        "SHK_TREND1": 0.1, "SHK_TREND2": 0.15,
        "SHK_STAT1": 0.2, "SHK_STAT2": 0.25,
        # VAR parameters
        "_var_coefficients": jnp.array([[[0.7, 0.1], [0.0, 0.6]]]),  # (order, n_stat, n_stat)
        "_var_innovation_corr_chol": jnp.array([[1.0, 0.0], [0.3, jnp.sqrt(1-0.3**2)]])
    }

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    simulated_y = generate_synthetic_data_for_gpm(
        gpm_file_path=gpm_file_example,
        true_params=true_params_for_sim,
        num_steps=100,  # Smaller for quick test
        rng_key_seed=789
    )

    if simulated_y is None:
        print("Failed to generate synthetic data for quick test.")
        return None

    # Run the complete workflow
    print("\nRunning complete workflow...")
    results = complete_gpm_workflow_with_smoother(
        data_source=simulated_y,
        gpm_file=gpm_file_example,
        num_warmup=50,        # Small for quick test
        num_samples=100,      # Small for quick test
        num_chains=1,         # Single chain for speed
        num_extract_draws=25, # Few draws for speed
        generate_plots=True,
        use_gamma_init = True,
        gamma_scale_factor  = 1.0
    )

    if results:
        print("\n✓ Quick test workflow successful!")
        print(f"   Trend draws shape: {results['reconstructed_trend_draws'].shape}")
        print(f"   Stationary draws shape: {results['reconstructed_stationary_draws'].shape}")
    else:
        print("\n✗ Quick test workflow failed.")
    
    # Clean up
    if os.path.exists(gpm_file_example):
        try:
            os.remove(gpm_file_example)
            print(f"   Cleaned up test file: {gpm_file_example}")
        except:
            pass
            
    return results


# --- Main Entry Points ---

def run_workflow_from_csv(
    csv_path: str,
    gpm_file: str = None,
    **workflow_kwargs
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to run workflow from a CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return None
        
    # Auto-generate gpm filename if not provided
    if gpm_file is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        gpm_file = f"{base_name}_model.gpm"
        
    return complete_gpm_workflow_with_smoother(
        data_source=csv_path,
        gpm_file=gpm_file,
        **workflow_kwargs
    )


if __name__ == "__main__":
    # Run quick test by default
    quick_test_smoother_workflow()