# clean_gpm_bvar_trends/gpm_bar_smoother.py

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Union

from pandas import DataFrame
import matplotlib.pyplot as plt
# Imports from the current refactored library structure
from gpm_numpyro_models import fit_gpm_numpyro_model, define_gpm_numpyro_model
from integration_orchestrator import IntegrationOrchestrator, create_integration_orchestrator
from simulation_smoothing import extract_reconstructed_components_fixed 
from constants import _DEFAULT_DTYPE

from reporting_plots import (
    plot_smoother_results,
    plot_observed_vs_fitted,
    # plot_time_series_with_uncertainty, # Not directly used by workflow, but by sub-functions
    compute_hdi_robust,
    compute_summary_statistics,
    plot_custom_series_comparison # New flexible plotter
)

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


def debug_mcmc_parameter_variation(mcmc_results, num_draws_to_check=5):
    """Debug function to check if MCMC parameters are actually varying"""
    print(f"\n=== DEBUGGING MCMC PARAMETER VARIATION ===")
    
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    total_draws = list(mcmc_samples.values())[0].shape[0]
    
    print(f"Total MCMC draws available: {total_draws}")
    
    # Check key parameters
    key_params = ['sigma_shk_cycle_y_us', 'sigma_shk_trend_y_us', 'init_mean_full']
    
    for param_name in key_params:
        if param_name in mcmc_samples:
            param_array = mcmc_samples[param_name]
            print(f"\nParameter: {param_name}")
            print(f"  Shape: {param_array.shape}")
            print(f"  Mean across draws: {jnp.mean(param_array):.6f}")
            print(f"  Std across draws: {jnp.std(param_array):.6f}")
            
            # Show first few draws
            if param_array.ndim == 1:
                first_draws = param_array[:num_draws_to_check]
                print(f"  First {num_draws_to_check} draws: {first_draws}")
            else:
                print(f"  First draw mean: {jnp.mean(param_array[0]):.6f}")
                print(f"  Last draw mean: {jnp.mean(param_array[-1]):.6f}")
    
    print("=== END MCMC DEBUGGING ===\n")

def complete_gpm_workflow_with_smoother_fixed(
    data: Union[DataFrame, np.ndarray, jnp.ndarray],
    gpm_file: str = 'model_for_smoother.gpm',
    # MCMC settings
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    rng_seed_mcmc: int = 0,
    target_accept_prob: float = 0.85,
    # P0 initialization settings
    use_gamma_init: bool = True,
    gamma_scale_factor: float = 1.0,
    # Smoother settings
    num_extract_draws: int = 100,
    rng_seed_smoother: int = 42,
    # Plotting settings
    generate_plots: bool = True,
    hdi_prob_plot: float = 0.9,
    save_plots: bool = False,
    plot_save_path: Optional[str] = None,
    variable_names_override: Optional[List[str]] = None,
    show_plot_info_boxes: bool = False,
    custom_plot_specs: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    FIXED VERSION: Complete workflow with proper P0 handling in simulation smoother
    and explicit ME usage in plotting.
    """
    print("=== Starting FIXED GPM Workflow ===")
    # ... (data processing code remains the same) ...
    y_numpy: np.ndarray
    time_index_actual: Optional[Any] = None
    obs_var_names_actual: Optional[List[str]] = variable_names_override

    if isinstance(data, pd.DataFrame):
        print("Input data is a Pandas DataFrame.")
        y_numpy = data.values.astype(_DEFAULT_DTYPE)
        if obs_var_names_actual is None:
            obs_var_names_actual = list(data.columns)
        time_index_actual = data.index
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        print("Input data is a NumPy/JAX array.")
        y_numpy = np.asarray(data, dtype=_DEFAULT_DTYPE)
        # obs_var_names_actual uses override or remains None
    else:
        print(f"ERROR: Unsupported data type {type(data)}")
        return None

    y_jax = jnp.asarray(y_numpy)
    T_actual, N_actual_obs = y_jax.shape
    
    if obs_var_names_actual is None: # Fallback if still None (e.g. array input with no override)
        obs_var_names_actual = [f"OBS{i+1}" for i in range(N_actual_obs)]
    elif len(obs_var_names_actual) != N_actual_obs:
        print(f"Warning: Mismatch between variable_names_override ({len(obs_var_names_actual)}) and data columns ({N_actual_obs}). Using default OBS names.")
        obs_var_names_actual = [f"OBS{i+1}" for i in range(N_actual_obs)]


    print(f"Data shape: ({T_actual}, {N_actual_obs})")
    print(f"Observed variable names for plotting: {obs_var_names_actual}")


    # Fit GPM model (unchanged)
    print(f"\nFitting GPM Model: {gpm_file}")
    fit_time = None
    try:
        start_time = time.time()
        mcmc_results, parsed_gpm_model, state_space_builder = fit_gpm_numpyro_model(
            gpm_file_path=gpm_file, y_data=y_jax,
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            rng_key_seed=rng_seed_mcmc, 
            use_gamma_init_for_P0=use_gamma_init,
            gamma_init_scaling_for_P0=gamma_scale_factor,
            target_accept_prob=target_accept_prob
        )
        fit_time = time.time() - start_time
        print(f"MCMC completed in {fit_time:.1f}s")
        
        mcmc_results.print_summary()

        if mcmc_results is None or parsed_gpm_model is None or state_space_builder is None:
            raise RuntimeError("MCMC fitting failed to return all necessary objects.")
    except Exception as e:
        import traceback
        print(f"ERROR during GPM model fitting: {e}")
        traceback.print_exc()
        return None
    
    # Extract components using fixed smoother (unchanged)
    print(f"\nExtracting Components via FIXED Simulation Smoother...")
    # ... (component extraction logic remains the same, calls extract_reconstructed_components_fixed) ...
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    if not mcmc_samples: # Should not happen if MCMC was successful
        print("ERROR: No MCMC samples available after successful MCMC run.")
        return None
    
    total_mcmc_samples = list(mcmc_samples.values())[0].shape[0]
    actual_extract_draws = min(num_extract_draws, total_mcmc_samples)
    print(f"Using {actual_extract_draws}/{total_mcmc_samples} available MCMC draws for smoother")
    
    all_trends_draws = jnp.empty((0, T_actual, 0))
    all_stationary_draws = jnp.empty((0, T_actual, 0))
    component_names = {'trends': [], 'stationary': []}

    if parsed_gpm_model:
        num_orig_trends = len(parsed_gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(parsed_gpm_model.gpm_stationary_variables_original)
        # Initialize with correct number of variables, even if 0 draws
        all_trends_draws = jnp.empty((0, T_actual, num_orig_trends if num_orig_trends > 0 else 0))
        all_stationary_draws = jnp.empty((0, T_actual, num_orig_stat if num_orig_stat > 0 else 0))
        component_names = {
            'trends': list(parsed_gpm_model.gpm_trend_variables_original),
            'stationary': list(parsed_gpm_model.gpm_stationary_variables_original)
        }

    if actual_extract_draws > 0:
        try:
            rng_key_for_smoother = random.PRNGKey(rng_seed_smoother)
            
            all_trends_draws, all_stationary_draws, component_names_from_extract = extract_reconstructed_components_fixed(
                mcmc_output=mcmc_results, 
                y_data=y_jax,
                gpm_model=parsed_gpm_model, 
                ss_builder=state_space_builder,
                num_smooth_draws=actual_extract_draws, 
                rng_key_smooth=rng_key_for_smoother,
                use_gamma_init_for_smoother=use_gamma_init,
                gamma_init_scaling_for_smoother=gamma_scale_factor
            )
            component_names = component_names_from_extract # Update with names from extraction
            print(f"Successfully extracted components with FIXED smoother:")
            print(f"  Trends: {all_trends_draws.shape}")
            print(f"  Stationary: {all_stationary_draws.shape}")
        except Exception as e:
            import traceback
            print(f"ERROR during FIXED component extraction: {e}")
            traceback.print_exc()
            # Keep initialized empty arrays if extraction fails
    else:
        print("Skipping component extraction (no MCMC draws selected or available)")


    # Rest of the workflow (plotting, etc.)
    if generate_plots:
        print(f"\nGenerating Plots...")
        plot_path_full = os.path.join(plot_save_path, "plot") if save_plots and plot_save_path else None
        if plot_path_full and not os.path.exists(plot_save_path): 
            os.makedirs(plot_save_path, exist_ok=True)
            
        # --- MODIFIED PART ---
        # Ensure current_trend_names and current_stationary_names are from component_names
        # which should be updated by extract_reconstructed_components_fixed
        current_trend_names = component_names.get('trends', [])
        current_stationary_names = component_names.get('stationary', [])
        
        # obs_var_names_actual should be set from input DataFrame columns or override
        # If it was an array input and no override, it's default e.g. ['OBS1', 'OBS2']
        # This list is used to iterate over the columns of y_numpy for plotting.

        # Condition for plotting: at least one type of component has draws
        can_plot_components = (hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0) or \
                              (hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0)

        if can_plot_components:
            try:
                print("Creating smoother results plots (trends and stationary)...")
                plot_smoother_results(
                    trend_draws=all_trends_draws, stationary_draws=all_stationary_draws,
                    trend_names=current_trend_names, stationary_names=current_stationary_names,
                    hdi_prob=hdi_prob_plot, save_path=plot_path_full,
                    time_index=time_index_actual, show_info_box=show_plot_info_boxes
                )
                
                print("Creating observed vs fitted plot...")
                plot_observed_vs_fitted(
                    observed_data=y_numpy, 
                    trend_draws=all_trends_draws, 
                    stationary_draws=all_stationary_draws,
                    variable_names=obs_var_names_actual, # Names for y_numpy columns
                    trend_names=current_trend_names,    # Names for 3rd dim of all_trends_draws
                    stationary_names=current_stationary_names, # Names for 3rd dim of all_stationary_draws
                    reduced_measurement_equations=parsed_gpm_model.reduced_measurement_equations, # MODIFIED: Pass ME
                    hdi_prob=hdi_prob_plot, 
                    save_path=plot_path_full,
                    time_index=time_index_actual, 
                    show_info_box=show_plot_info_boxes
                )

                # Custom plots if specified (ensure it also gets reduced_measurement_equations if needed)
                if custom_plot_specs:
                    for i, spec_dict in enumerate(custom_plot_specs):
                        # ... (custom plot logic, may need ME if 'combined' types rely on GPM definitions)
                        # For now, assuming plot_custom_series_comparison handles its data needs
                        # based on simple 'trend', 'stationary', 'observed' types.
                        # If it needs to interpret GPM MEs, it would need 'reduced_measurement_equations'.
                        plot_title = spec_dict.get("title", f"Custom Plot {i+1}")
                        series_to_draw = spec_dict.get("series_to_plot", [])
                        if series_to_draw:
                            print(f"Creating custom plot: {plot_title}")
                            plot_custom_series_comparison( # This function's internal logic is not being changed here
                                plot_title=plot_title,
                                series_specs=series_to_draw,
                                observed_data=y_numpy,
                                trend_draws=all_trends_draws,
                                stationary_draws=all_stationary_draws,
                                observed_names=obs_var_names_actual,
                                trend_names=current_trend_names,
                                stationary_names=current_stationary_names,
                                time_index=time_index_actual,
                                hdi_prob=hdi_prob_plot
                            )
                            if plot_path_full:
                                plt.savefig(f"{plot_path_full}_custom_{plot_title.lower().replace(' ','_')}.png", 
                                          dpi=300, bbox_inches='tight')

                # ... (summary statistics printing - unchanged) ...
                print("\n=== SUMMARY STATISTICS (FIXED SMOOTHER) ===")
                # (Trend stats)
                if hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0:
                    trend_stats = compute_summary_statistics(all_trends_draws)
                    print(f"\nTrend Components (mean of time series means):")
                    for i_ts, name_ts in enumerate(current_trend_names):
                         if trend_stats['mean'].ndim > 1 and i_ts < trend_stats['mean'].shape[1]:
                            mean_val_ts = np.mean(trend_stats['mean'][:, i_ts])
                            std_val_ts = np.mean(trend_stats['std'][:, i_ts])
                            print(f"  {name_ts}: Mean={mean_val_ts:.4f}, Std={std_val_ts:.4f}")
                # (Stationary stats)
                if hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0:
                    stat_stats = compute_summary_statistics(all_stationary_draws)
                    print(f"\nStationary Components (mean of time series means):")
                    for i_ts, name_ts in enumerate(current_stationary_names):
                         if stat_stats['mean'].ndim > 1 and i_ts < stat_stats['mean'].shape[1]:
                            mean_val_ts = np.mean(stat_stats['mean'][:, i_ts])
                            std_val_ts = np.mean(stat_stats['std'][:, i_ts])
                            print(f"  {name_ts}: Mean={mean_val_ts:.4f}, Std={std_val_ts:.4f}")
                print("✓ Plotting completed successfully with FIXED smoother")
            except Exception as e:
                import traceback
                print(f"Warning: Plotting failed: {e}")
                traceback.print_exc()
        else:
            print("Skipping plots (no valid component draws available or generate_plots=False)")
    
    # Return results (unchanged)
    # ... (results_dict assembly) ...
    results_dict = {
        'mcmc_object': mcmc_results,
        'parsed_gpm_model': parsed_gpm_model,
        'state_space_builder': state_space_builder,
        'reconstructed_trend_draws': all_trends_draws,
        'reconstructed_stationary_draws': all_stationary_draws,
        'component_names': component_names,
        'observed_data_numpy': y_numpy,
        'time_index': time_index_actual,
        'observed_variable_names': obs_var_names_actual, # The names used for plotting y_numpy
        'fitting_time_seconds': fit_time,
        'trend_summary_stats': compute_summary_statistics(all_trends_draws) if hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0 else None,
        'stationary_summary_stats': compute_summary_statistics(all_stationary_draws) if hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0 else None,
        'hdi_prob': hdi_prob_plot,
        'used_gamma_init': use_gamma_init,
        'gamma_scale_factor': gamma_scale_factor
    }
    
    print("\n=== FIXED Workflow Complete ===")
    return results_dict

def plot_existing_smoother_results( # Unchanged logic, but added time_index and show_info_box
    all_trends_draws: jnp.ndarray, all_stationary_draws: jnp.ndarray,
    observed_data: np.ndarray, component_names: Dict[str, List[str]],
    variable_names: Optional[List[str]] = None, hdi_prob: float = 0.9,
    save_plots: bool = False, save_path: Optional[str] = None,
    time_index: Optional[Any] = None, show_info_box: bool = False
) -> Dict[str, Any]:
    print("\n=== PLOTTING EXISTING SMOOTHER RESULTS ===")
    trends_np, stationary_np, obs_np = np.asarray(all_trends_draws), np.asarray(all_stationary_draws), np.asarray(observed_data)
    results = {}
    plot_path_full = os.path.join(save_path, "plot") if save_plots and save_path else None
    if plot_path_full and not os.path.exists(save_path): os.makedirs(save_path)

    try:
        trend_fig, stat_fig = plot_smoother_results(
            trends_np, stationary_np,
            trend_names=component_names.get('trends'), stationary_names=component_names.get('stationary'),
            hdi_prob=hdi_prob, save_path=plot_path_full, time_index=time_index, show_info_box=show_info_box
        )
        results['trend_figure'], results['stationary_figure'] = trend_fig, stat_fig
        
        fitted_fig = plot_observed_vs_fitted(
            obs_np, 
            trends_np, 
            stationary_np, 
            variable_names=variable_names,
            #reduced_measurement_equations=parsed_gpm_model.reduced_measurement_equations            
            trend_names=component_names.get('trends'), 
            stationary_names=component_names.get('stationary'),
            hdi_prob=hdi_prob, 
            save_path=plot_path_full, 
            time_index=time_index, 
            show_info_box=show_info_box
        )
        results['fitted_figure'] = fitted_fig
        
        if trends_np.shape[0] > 0 and trends_np.shape[2]>0: results['trend_stats'], results['trend_hdi'] = compute_summary_statistics(trends_np), compute_hdi_robust(trends_np, hdi_prob)
        if stationary_np.shape[0] > 0 and stationary_np.shape[2]>0: results['stationary_stats'], results['stationary_hdi'] = compute_summary_statistics(stationary_np), compute_hdi_robust(stationary_np, hdi_prob)
        if trends_np.shape[0] > 0 and stationary_np.shape[0] > 0 and trends_np.shape[2] == stationary_np.shape[2] and trends_np.shape[2]>0:
            fitted_draws = trends_np + stationary_np
            results['fitted_stats'], results['fitted_hdi'] = compute_summary_statistics(fitted_draws), compute_hdi_robust(fitted_draws, hdi_prob)
        print("✓ Plotting and analysis complete")
    except Exception as e: import traceback; traceback.print_exc(); results['error'] = str(e)
    return results


def debug_smoother_draws(
    all_trends_draws: jnp.ndarray,
    all_stationary_draws: jnp.ndarray,
    component_names: Dict[str, List[str]]
) -> None:
    """
    Debug function to examine the structure and content of smoother draws.
    """
    print("\n=== DEBUGGING SMOOTHER DRAWS ===")
    
    # Convert to numpy for easier inspection
    trends_np = np.asarray(all_trends_draws)
    stationary_np = np.asarray(all_stationary_draws)
    
    print(f"Trend draws:")
    print(f"  Shape: {trends_np.shape}")
    print(f"  Dtype: {trends_np.dtype}")
    print(f"  Has NaN: {np.any(np.isnan(trends_np))}")
    print(f"  Has Inf: {np.any(np.isinf(trends_np))}")
    print(f"  Min value: {np.nanmin(trends_np):.6f}")
    print(f"  Max value: {np.nanmax(trends_np):.6f}")
    print(f"  Mean value: {np.nanmean(trends_np):.6f}")
    
    print(f"\nStationary draws:")
    print(f"  Shape: {stationary_np.shape}")
    print(f"  Dtype: {stationary_np.dtype}")
    print(f"  Has NaN: {np.any(np.isnan(stationary_np))}")
    print(f"  Has Inf: {np.any(np.isinf(stationary_np))}")
    print(f"  Min value: {np.nanmin(stationary_np):.6f}")
    print(f"  Max value: {np.nanmax(stationary_np):.6f}")
    print(f"  Mean value: {np.nanmean(stationary_np):.6f}")
    
    print(f"\nComponent names:")
    print(f"  Trends: {component_names.get('trends', 'None')}")
    print(f"  Stationary: {component_names.get('stationary', 'None')}")
    
    # Check for variation across draws
    if trends_np.shape[0] > 1:
        trend_std_across_draws = np.nanstd(trends_np, axis=0)
        print(f"\nTrend variation across draws:")
        print(f"  Mean std across time/vars: {np.nanmean(trend_std_across_draws):.6f}")
        print(f"  Max std across time/vars: {np.nanmax(trend_std_across_draws):.6f}")
    
    if stationary_np.shape[0] > 1:
        stat_std_across_draws = np.nanstd(stationary_np, axis=0)
        print(f"\nStationary variation across draws:")
        print(f"  Mean std across time/vars: {np.nanmean(stat_std_across_draws):.6f}")
        print(f"  Max std across time/vars: {np.nanmax(stat_std_across_draws):.6f}")
    
    print("=== END DEBUG ===\n")


# Example of how to use with your existing workflow
def example_enhanced_workflow():
    """
    Example showing how to integrate the enhanced plotting with your workflow.
    """
    import pandas as pd
    
    # Create some example data
    np.random.seed(42)
    T = 100
    n_obs = 2
    
    # Generate synthetic observed data
    time_trend = np.linspace(0, 5, T)
    observed_data = np.column_stack([
        time_trend + np.random.randn(T) * 0.5,  # GDP with trend
        np.sin(time_trend) + np.random.randn(T) * 0.3  # Cyclical inflation
    ])
    
    # Convert to DataFrame with proper column names
    data_df = pd.DataFrame(observed_data, columns=['GDP', 'Inflation'])
    
    # Run enhanced workflow
    results = enhanced_complete_gpm_workflow_with_smoother(
        data=data_df,
        gpm_file='your_model.gpm',  # Replace with your actual GPM file
        num_warmup=100,  # Reduced for example
        num_samples=200,  # Reduced for example
        num_chains=1,    # Reduced for example
        num_extract_draws=50,  # Reduced for example
        generate_plots=True,
        hdi_prob_plot=0.9,
        save_plots=True,
        plot_save_path='enhanced_workflow_results',
        variable_names=['GDP', 'Inflation']
    )
    
    if results:
        print("Enhanced workflow completed successfully!")
        
        # Debug the draws if needed
        debug_smoother_draws(
            results['reconstructed_trend_draws'],
            results['reconstructed_stationary_draws'],
            results['component_names']
        )
    
    return results


def quick_test_fixed_smoother_workflow():
    """Test the fixed smoother workflow"""
    print("=== Running Quick Test with FIXED Smoother ===")
    
    num_obs, num_stat = 2, 2
    gpm_file = "smoother_test_fixed_model.gpm"
    create_default_gpm_file_if_needed(gpm_file, num_obs, num_stat)
    
    true_params = { 
        "rho": 0.5, 
        "SHK_TREND1": 0.1, 
        "SHK_TREND2": 0.15, 
        "SHK_STAT1": 0.2, 
        "SHK_STAT2": 0.25,
        "_var_coefficients": jnp.array([[[0.7, 0.1], [0.0, 0.6]]]),
        "_var_innovation_corr_chol": jnp.array([[1.0, 0.0], [0.3, jnp.sqrt(1-0.3**2)]])
    }
    
    sim_y = generate_synthetic_data_for_gpm(gpm_file, true_params, num_steps=100, rng_key_seed=789)
    if sim_y is None: 
        print("Failed to generate synthetic data.")
        return None
    
    # Test with the FIXED workflow
    results = complete_gpm_workflow_with_smoother_fixed(
        data=sim_y, 
        gpm_file=gpm_file,
        num_warmup=50, 
        num_samples=100, 
        num_chains=1, 
        num_extract_draws=25,
        generate_plots=True, 
        use_gamma_init=True,  # Test with gamma-based P0
        gamma_scale_factor=1.0,
        show_plot_info_boxes=False
    )
    
    if results: 
        print(f"\n✓ FIXED smoother test successful!")
        print(f"  Trend draws shape: {results['reconstructed_trend_draws'].shape}")
        print(f"  Stationary draws shape: {results['reconstructed_stationary_draws'].shape}")
        print(f"  Used gamma init: {results['used_gamma_init']}")
        
        # Check if the confidence bands look reasonable now
        if hasattr(results['reconstructed_trend_draws'], 'shape') and results['reconstructed_trend_draws'].shape[0] > 1:
            trend_draws_np = np.asarray(results['reconstructed_trend_draws'])
            
            # Calculate HDI width as a measure of uncertainty
            from reporting_plots import compute_hdi_robust
            hdi_lower, hdi_upper = compute_hdi_robust(trend_draws_np, 0.9)
            hdi_width = hdi_upper - hdi_lower
            
            print(f"\nTrend uncertainty analysis:")
            trend_names = results['component_names']['trends']
            for i, name in enumerate(trend_names):
                if i < hdi_width.shape[1]:
                    mean_width = np.mean(hdi_width[:, i])
                    print(f"  {name}: Mean HDI width = {mean_width:.4f}")
                    
                    # Check if this looks reasonable (not massively inflated)
                    if mean_width > 10:  # Arbitrary threshold
                        print(f"    ⚠ Still appears to have inflated uncertainty")
                    else:
                        print(f"    ✓ Uncertainty appears reasonable")
    else: 
        print("\n✗ FIXED smoother test failed.")
    
    # Cleanup
    if os.path.exists(gpm_file): 
        os.remove(gpm_file)
    
    return results



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
    quick_test_fixed_smoother_workflow()

    
    # Example of using custom_plot_specs
    # Create some dummy data for the custom plot example
    if True: # Set to true to run this example after quick_test
        print("\n=== Running Custom Plot Example ===")
        num_obs_example, num_stat_example = 2, 1
        gpm_file_custom_plot = "custom_plot_model.gpm"
        create_default_gpm_file_if_needed(gpm_file_custom_plot, num_obs_example, num_stat_example)
        
        # Create some dummy observed data as a DataFrame with a DatetimeIndex
        dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=50, freq='QE'))
        obs_data_df = pd.DataFrame(
            np.cumsum(np.random.randn(50, num_obs_example) * 0.2, axis=0) + np.random.randn(50, num_obs_example) * 0.1,
            index=dates,
            columns=[f"OBS{i+1}" for i in range(num_obs_example)]
        )

        # Define a custom plot specification
        # Assumes model has OBS1, OBS2, TREND1, TREND2, STAT1
        custom_specs = [
            { # Plot 1: OBS1 vs its trend and cycle
                "title": "OBS1 Components",
                "series_to_plot": [
                    {'type': 'observed', 'name': 'OBS1', 'label': 'Observed Data 1', 'style': 'k-', 'color':'black'},
                    {'type': 'trend', 'name': 'TREND1', 'label': 'Trend of OBS1', 'show_hdi': True, 'color':'blue'},
                    {'type': 'stationary', 'name': 'STAT1', 'label': 'Cycle of OBS1', 'show_hdi': True, 'color':'green'}
                ]
            },
            { # Plot 2: OBS2 vs its trend
                "title": "OBS2 and its Trend",
                "series_to_plot": [
                    {'type': 'observed', 'name': 'OBS2', 'label': 'Observed Data 2', 'style': '-', 'color':'black'},
                    {'type': 'trend', 'name': 'TREND2', 'label': 'Trend of OBS2', 
                     'show_hdi': True, 'color':'red', 'hdi_alpha': 0.3} # Explicitly set hdi_alpha
                ]
            },
            { # Plot 3: Compare the two trends
                 "title": "Trend Comparison",
                 "series_to_plot": [
                    {'type': 'trend', 'name': 'TREND1', 'label': 'Trend 1', 'show_hdi': True, 'color':'cyan','hdi_alpha': 0.3},
                    {'type': 'trend', 'name': 'TREND2', 'label': 'Trend 2', 'show_hdi': True, 'color':'magenta','hdi_alpha': 0.3}
                 ]
            },
            { # Plot 4: OBS1 vs combined Trend1+Stat1
                "title": "OBS1 vs Fitted (Trend1+Stat1)",
                "series_to_plot": [
                    {'type': 'observed', 'name': 'OBS1', 'label': 'Observed OBS1', 'style': 'k-'},
                    {'type': 'combined', 
                     'components': [{'type':'trend', 'name':'TREND1'}, {'type':'stationary', 'name':'STAT1'}],
                     'label': 'Fitted (TREND1+STAT1)', 'show_hdi': True, 'color':'purple'}
                ]
            }
        ]

        results_custom = complete_gpm_workflow_with_smoother_fixed(
            data=obs_data_df, # Pass DataFrame with DatetimeIndex
            gpm_file=gpm_file_custom_plot,
            num_warmup=20, num_samples=30, num_chains=1, num_extract_draws=10, # Small for quick run
            generate_plots=True,
            show_plot_info_boxes=False,
            custom_plot_specs=custom_specs, # Pass the custom plot specs
            plot_save_path="custom_plot_output", # Optional: save plots
            save_plots=True
        )
        if results_custom:
            print("Custom plot example workflow successful.")
        else:
            print("Custom plot example workflow failed.")
        if os.path.exists(gpm_file_custom_plot): os.remove(gpm_file_custom_plot)


    print("\nEnhanced workflow integration ready!")
    print("Use complete_gpm_workflow_with_smoother() for full workflow")
    print("  - Pass data as Pandas DataFrame with DatetimeIndex for time-axis plots.")
    print("  - Use 'show_plot_info_boxes=False' to hide default plot text boxes.")
    print("  - Use 'custom_plot_specs' for flexible plotting with plot_custom_series_comparison.")
    print("Use plot_existing_smoother_results() for plotting existing results (now supports time_index and show_info_box).")