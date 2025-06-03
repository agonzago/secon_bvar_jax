# clean_gpm_bvar_trends/gpm_bar_smoother.py

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime 
from typing import Dict, List, Optional, Any, Tuple, Union
import numpyro
import arviz as az # Import ArviZ

from pandas import DataFrame
import matplotlib.pyplot as plt

from .gpm_numpyro_models import fit_gpm_numpyro_model, define_gpm_numpyro_model
from .integration_orchestrator import IntegrationOrchestrator, create_integration_orchestrator
from .simulation_smoothing import extract_reconstructed_components_fixed 
from .constants import _DEFAULT_DTYPE
from .gpm_model_parser import ReducedModel, PriorSpec

from .reporting_plots import (
    plot_smoother_results,
    plot_observed_vs_fitted,
    compute_hdi_robust,
    compute_summary_statistics,
    plot_custom_series_comparison 
)

try:
    from .Kalman_filter_jax import simulate_state_space
except ImportError:
    simulate_state_space = None
    print("Warning: simulate_state_space not available from Kalman_filter_jax")

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# --- MODIFIED/ENHANCED Summary Function ---
def _print_model_and_run_settings_summary( 
    gpm_file: str,
    data_file_source_for_summary: Optional[str], 
    parsed_gpm_model: ReducedModel,
    num_warmup: int, num_samples: int, num_chains: int,
    use_gamma_init: bool, gamma_scale_factor: float,
    target_accept_prob: float,
    num_extract_draws: int 
):
    # ... (function body as previously defined - no changes here) ...
    print("\n" + "="*70)
    print("      GPM WORKFLOW CONFIGURATION & MODEL SUMMARY      ")
    print("="*70)
    print(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPM File: {gpm_file}")
    if data_file_source_for_summary:
        print(f"Data Source: {data_file_source_for_summary}")
    else:
        print(f"Data Source: In-memory data (no file path provided)")

    print("\n--- GPM Model Structure ---")
    print(f"  Observed Variables (from GPM): {parsed_gpm_model.gpm_observed_variables_original}")
    print(f"  Trend Variables (from GPM): {parsed_gpm_model.gpm_trend_variables_original}")
    print(f"  Stationary Variables (from GPM): {parsed_gpm_model.gpm_stationary_variables_original}")
    print(f"  Core Variables (for state space): {parsed_gpm_model.core_variables}")
    print(f"  Structural Parameters Declared: {parsed_gpm_model.parameters}")
    print(f"  Trend Shocks Declared: {parsed_gpm_model.trend_shocks}")
    print(f"  Stationary Shocks Declared: {parsed_gpm_model.stationary_shocks}")


    print("\n--- Estimated Parameters (Priors) ---")
    if parsed_gpm_model.estimated_params:
        for name, prior in parsed_gpm_model.estimated_params.items():
            params_str = ", ".join(map(str, prior.params))
            print(f"  {name:<25}: {prior.distribution}({params_str})")
    else:
        print("  No estimated parameters found in GPM.")

    if parsed_gpm_model.var_prior_setup:
        vps = parsed_gpm_model.var_prior_setup
        print("\n--- VAR Prior Setup ---")
        print(f"  Order: {vps.var_order}")
        print(f"  es (A means diag,offdiag): {vps.es}")
        print(f"  fs (A stds diag,offdiag): {vps.fs}")
        print(f"  gs (InvGamma prec shapes): {vps.gs}")
        print(f"  hs (InvGamma prec scales): {vps.hs}")
        print(f"  eta (LKJ concentration): {vps.eta}")

    print("\n--- MCMC Settings ---")
    print(f"  Warmup steps: {num_warmup}")
    print(f"  Sampling steps: {num_samples}")
    print(f"  Chains: {num_chains}")
    print(f"  Target accept probability: {target_accept_prob}")

    print("\n--- P0 Initialization ---")
    print(f"  Use Gamma-based P0: {use_gamma_init}")
    if use_gamma_init:
        print(f"  Gamma P0 scaling factor: {gamma_scale_factor}")
    
    print("\n--- Smoother Settings ---")
    print(f"  Number of draws for smoother: {num_extract_draws}")
    print("="*70 + "\n")

# --- MODIFIED: print_filtered_mcmc_summary to use ArviZ ---
def print_filtered_mcmc_summary(
    mcmc_results: numpyro.infer.MCMC,
    parsed_gpm_model: ReducedModel
):
    print("\n--- Filtered MCMC Summary (Key Model Parameters using ArviZ) ---")
    
    # Define the parameters we want to include in the summary
    params_to_include = set()
    # 1. Structural parameters from the 'parameters' block in GPM
    params_to_include.update(parsed_gpm_model.parameters)
    
    # 2. Shock standard deviations (MCMC names are typically "sigma_SHOCK_NAME")
    for shock_name in parsed_gpm_model.trend_shocks + parsed_gpm_model.stationary_shocks:
        params_to_include.add(f"sigma_{shock_name}")
        
    # 3. Key VAR parameters (if VAR model exists)
    #    az.from_numpyro typically includes deterministic sites in 'posterior' or a similar group.
    if parsed_gpm_model.var_prior_setup:
        params_to_include.add("A_transformed")  # Transformed VAR coefficients
        params_to_include.add("Omega_u_chol")   # Cholesky of VAR innovation correlation
        
    # 4. Initial state mean (if sampled or deterministic)
    params_to_include.add("init_mean_full")

    # Convert NumPyro MCMC object to ArviZ InferenceData
    try:
        idata = az.from_numpyro(mcmc_results)
    except Exception as e:
        print(f"  Error converting NumPyro MCMC to ArviZ InferenceData: {e}")
        print("  Falling back to basic NumPyro summary for all parameters.")
        mcmc_results.print_summary(exclude_deterministic=False)
        print("-" * 70 + "\n")
        return

    # Filter var_names: only include those that are actually present in the InferenceData object
    # This is important because `az.from_numpyro` might not include all theoretically defined
    # deterministic sites if they weren't explicitly saved or have issues.
    # We check against the 'posterior' group, which is where `numpyro.sample` and 
    # often `numpyro.deterministic` sites land.
    
    available_vars_in_posterior = set(idata.posterior.data_vars.keys())
    final_vars_to_summarize = list(params_to_include.intersection(available_vars_in_posterior))

    if not final_vars_to_summarize:
        print("  No relevant model parameters (from params_to_include) found in ArviZ InferenceData posterior group.")
        print("  Displaying basic NumPyro summary for all parameters instead.")
        mcmc_results.print_summary(exclude_deterministic=False)
        print("-" * 70 + "\n")
        return

    try:
        summary_df = az.summary(idata, var_names=final_vars_to_summarize)
        print(summary_df)
    except Exception as e:
        print(f"  Error generating ArviZ summary for selected parameters: {e}")
        print("  Parameters intended for summary:", final_vars_to_summarize)
        print("  Falling back to basic NumPyro summary for all parameters.")
        mcmc_results.print_summary(exclude_deterministic=False)
    
    print("-" * 70 + "\n")


def print_run_report(
    gpm_file: str,
    data_file_source_for_summary: Optional[str],
    parsed_gpm_model: ReducedModel,
    mcmc_results: numpyro.infer.MCMC,
    num_warmup: int, num_samples: int, num_chains: int,
    use_gamma_init: bool, gamma_scale_factor: float,
    target_accept_prob: float,
    num_extract_draws: int
):
    # ... (function body as previously defined - no changes here) ...
    _print_model_and_run_settings_summary(
        gpm_file=gpm_file,
        data_file_source_for_summary=data_file_source_for_summary,
        parsed_gpm_model=parsed_gpm_model,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        use_gamma_init=use_gamma_init,
        gamma_scale_factor=gamma_scale_factor,
        target_accept_prob=target_accept_prob,
        num_extract_draws=num_extract_draws
    )
    print_filtered_mcmc_summary(
        mcmc_results=mcmc_results,
        parsed_gpm_model=parsed_gpm_model
    )

def create_default_gpm_file_if_needed(filename: str, num_obs_vars: int, num_stat_vars: int = 0):
    # ... (function body as previously defined - no changes here) ...
    if os.path.exists(filename): return
    print(f"Creating default gpm file: {filename} with {num_obs_vars} obs vars and {num_stat_vars} stat vars")
    gpm_content = "parameters rho;\n"
    gpm_content += "\nestimated_params;\n"
    gpm_content += "    rho, normal_pdf, 0.5, 0.2;\n"
    for i in range(num_obs_vars): gpm_content += f"    stderr SHK_TREND{i+1}, inv_gamma_pdf, 2.0, 0.02;\n"
    for i in range(num_stat_vars): gpm_content += f"    stderr SHK_STAT{i+1}, inv_gamma_pdf, 2.0, 0.1;\n"
    gpm_content += "end;\n"
    trend_names = [f"TREND{i+1}" for i in range(num_obs_vars)]
    gpm_content += f"\ntrends_vars {', '.join(trend_names)};\n"
    gpm_content += "\ntrend_shocks;\n"
    for i in range(num_obs_vars): gpm_content += f"    var SHK_TREND{i+1};\n"
    gpm_content += "end;\n"
    if num_stat_vars > 0:
        stat_names = [f"STAT{i+1}" for i in range(num_stat_vars)]
        gpm_content += f"\nstationary_variables {', '.join(stat_names)};\n"
        gpm_content += "\nshocks;\n"
        for i in range(num_stat_vars): gpm_content += f"    var SHK_STAT{i+1};\n"
        gpm_content += "end;\n"
    else:
        gpm_content += "\nstationary_variables ;\n\nshocks;\nend;\n"
    gpm_content += "\ntrend_model;\n"
    for i in range(num_obs_vars): gpm_content += f"    TREND{i+1} = TREND{i+1}(-1) + SHK_TREND{i+1};\n"
    gpm_content += "end;\n"
    obs_names = [f"OBS{i+1}" for i in range(num_obs_vars)]
    gpm_content += f"\nvarobs {', '.join(obs_names)};\n"
    gpm_content += "\nmeasurement_equations;\n"
    for i in range(num_obs_vars):
        stat_term = f" + STAT{i+1}" if i < num_stat_vars else ""
        gpm_content += f"    OBS{i+1} = TREND{i+1}{stat_term};\n"
    gpm_content += "end;\n"
    if num_stat_vars > 0:
        gpm_content += """\nvar_prior_setup;
    var_order = 1; es = 0.5,0.1; fs = 0.5,0.5; gs = 3.0,3.0; hs = 1.0,1.0; eta = 2.0; 
end;\n"""
    gpm_content += "\ninitval;\n"
    for i in range(num_obs_vars): gpm_content += f"    TREND{i+1}, normal_pdf, 0, 1;\n"
    gpm_content += "end;\n"
    with open(filename, 'w') as f: f.write(gpm_content)
    print(f"✓ Created default gpm file: {filename}")


def generate_synthetic_data_for_gpm(
    gpm_file_path: str,
    true_params: Dict[str, Any],
    num_steps: int = 150,
    rng_key_seed: int = 42
) -> Optional[jnp.ndarray]:
    # ... (function body as previously defined - no changes here) ...
    if simulate_state_space is None:
        print("ERROR: simulate_state_space not available. Cannot generate synthetic data.")
        return None
    try:
        orchestrator = create_integration_orchestrator(gpm_file_path)
        F_true, Q_true, C_true, H_true = orchestrator.build_ss_from_direct_dict(true_params)
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
        print(f"Error generating synthetic data: {e}"); traceback.print_exc()
        return None

def debug_mcmc_parameter_variation(mcmc_results, num_draws_to_check=5):
    # ... (function body as previously defined - no changes here) ...
    print(f"\n=== DEBUGGING MCMC PARAMETER VARIATION ===")
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    total_draws = list(mcmc_samples.values())[0].shape[0]
    print(f"Total MCMC draws available: {total_draws}")
    key_params = ['sigma_shk_cycle_y_us', 'sigma_shk_trend_y_us', 'init_mean_full'] 
    for param_name in key_params:
        if param_name in mcmc_samples:
            param_array = mcmc_samples[param_name]
            print(f"\nParameter: {param_name} (Shape: {param_array.shape})")
            print(f"  Mean across draws: {jnp.mean(param_array):.6f}")
            print(f"  Std across draws: {jnp.std(param_array):.6f}")
            if param_array.ndim == 1: print(f"  First {num_draws_to_check} draws: {param_array[:num_draws_to_check]}")
            else: print(f"  First draw mean: {jnp.mean(param_array[0]):.6f}; Last draw mean: {jnp.mean(param_array[-1]):.6f}")
    print("=== END MCMC DEBUGGING ===\n")


def complete_gpm_workflow_with_smoother_fixed(
    data: Union[DataFrame, np.ndarray, jnp.ndarray],
    gpm_file: str = 'model_for_smoother.gpm',
    num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2,
    rng_seed_mcmc: int = 0, target_accept_prob: float = 0.85,
    use_gamma_init: bool = True, gamma_scale_factor: float = 1.0,
    num_extract_draws: int = 100, rng_seed_smoother: int = 42,
    generate_plots: bool = True, hdi_prob_plot: float = 0.9,
    save_plots: bool = False, plot_save_path: Optional[str] = None,
    variable_names_override: Optional[List[str]] = None,
    show_plot_info_boxes: bool = False,
    custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
    data_file_source_for_summary: Optional[str] = None 
) -> Optional[Dict[str, Any]]:
    # ... (function body largely as previously defined, ensuring print_run_report is called) ...
    print("=== Starting FIXED GPM Workflow ===")
    y_numpy: np.ndarray; time_index_actual: Optional[Any] = None
    obs_var_names_actual: Optional[List[str]] = variable_names_override

    if isinstance(data, pd.DataFrame):
        y_numpy = data.values.astype(_DEFAULT_DTYPE)
        if obs_var_names_actual is None: obs_var_names_actual = list(data.columns)
        time_index_actual = data.index
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        y_numpy = np.asarray(data, dtype=_DEFAULT_DTYPE)
    else:
        print(f"ERROR: Unsupported data type {type(data)}"); return None

    y_jax = jnp.asarray(y_numpy)
    T_actual, N_actual_obs = y_jax.shape
    
    if obs_var_names_actual is None: 
        obs_var_names_actual = [f"OBS{i+1}" for i in range(N_actual_obs)]
    elif len(obs_var_names_actual) != N_actual_obs:
        print(f"Warning: Mismatch variable_names_override ({len(obs_var_names_actual)}) vs data columns ({N_actual_obs}). Using defaults.")
        obs_var_names_actual = [f"OBS{i+1}" for i in range(N_actual_obs)]

    print(f"Data shape: ({T_actual}, {N_actual_obs})")
    print(f"Observed variable names for plotting: {obs_var_names_actual}")

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
        
        if mcmc_results is None or parsed_gpm_model is None or state_space_builder is None:
            raise RuntimeError("MCMC fitting failed to return all necessary objects.")
    except Exception as e:
        import traceback
        print(f"ERROR during GPM model fitting: {e}"); traceback.print_exc()
        return None
    
    print_run_report(
        gpm_file=gpm_file,
        data_file_source_for_summary=data_file_source_for_summary,
        parsed_gpm_model=parsed_gpm_model,
        mcmc_results=mcmc_results,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        use_gamma_init=use_gamma_init,
        gamma_scale_factor=gamma_scale_factor,
        target_accept_prob=target_accept_prob,
        num_extract_draws=num_extract_draws
    )
    
    print(f"\nExtracting Components via FIXED Simulation Smoother...")
    mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    if not mcmc_samples: print("ERROR: No MCMC samples available."); return None
    
    total_mcmc_samples = list(mcmc_samples.values())[0].shape[0]
    actual_extract_draws = min(num_extract_draws, total_mcmc_samples)
    print(f"Using {actual_extract_draws}/{total_mcmc_samples} available MCMC draws for smoother")
    
    all_trends_draws = jnp.empty((0, T_actual, 0)); all_stationary_draws = jnp.empty((0, T_actual, 0))
    component_names = {'trends': [], 'stationary': []}

    if parsed_gpm_model:
        num_orig_trends = len(parsed_gpm_model.gpm_trend_variables_original)
        num_orig_stat = len(parsed_gpm_model.gpm_stationary_variables_original)
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
                mcmc_output=mcmc_results, y_data=y_jax,
                gpm_model=parsed_gpm_model, 
                ss_builder=state_space_builder,
                num_smooth_draws=actual_extract_draws, rng_key_smooth=rng_key_for_smoother,
                use_gamma_init_for_smoother=use_gamma_init,
                gamma_init_scaling_for_smoother=gamma_scale_factor
            )
            component_names = component_names_from_extract 
            print(f"Successfully extracted components with FIXED smoother:")
            print(f"  Trends: {all_trends_draws.shape}, Stationary: {all_stationary_draws.shape}")
        except Exception as e:
            import traceback
            print(f"ERROR during FIXED component extraction: {e}"); traceback.print_exc()
    else:
        print("Skipping component extraction (no MCMC draws selected or available)")

    if generate_plots:
        print(f"\nGenerating Plots...")
        plot_path_full = os.path.join(plot_save_path, "plot") if save_plots and plot_save_path else None
        if plot_path_full and not os.path.exists(plot_save_path): 
            os.makedirs(plot_save_path, exist_ok=True)
            
        current_trend_names = component_names.get('trends', [])
        current_stationary_names = component_names.get('stationary', [])
        can_plot_components = (hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0) or \
                              (hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0)

        if can_plot_components:
            try:
                print("Creating smoother results plots (trends and stationary)...")
                trend_fig, stat_fig = plot_smoother_results(
                    trend_draws=all_trends_draws, stationary_draws=all_stationary_draws,
                    trend_names=current_trend_names, stationary_names=current_stationary_names,
                    hdi_prob=hdi_prob_plot, save_path=plot_path_full,
                    time_index=time_index_actual, show_info_box=show_plot_info_boxes
                )
                plt.close(trend_fig); plt.close(stat_fig)

                print("Creating observed vs fitted plot...")
                fitted_fig = plot_observed_vs_fitted(
                    observed_data=y_numpy, trend_draws=all_trends_draws, stationary_draws=all_stationary_draws,
                    variable_names=obs_var_names_actual, trend_names=current_trend_names, stationary_names=current_stationary_names,
                    reduced_measurement_equations=parsed_gpm_model.reduced_measurement_equations, 
                    hdi_prob=hdi_prob_plot, save_path=plot_path_full,
                    time_index=time_index_actual, show_info_box=show_plot_info_boxes
                )
                plt.close(fitted_fig)

                if custom_plot_specs:
                    for i, spec_dict in enumerate(custom_plot_specs):
                        plot_title = spec_dict.get("title", f"Custom Plot {i+1}")
                        series_to_draw = spec_dict.get("series_to_plot", [])
                        if series_to_draw:
                            print(f"Creating custom plot: {plot_title}")
                            fig_custom = plot_custom_series_comparison(
                                plot_title=plot_title, series_specs=series_to_draw,
                                observed_data=y_numpy, trend_draws=all_trends_draws, stationary_draws=all_stationary_draws,
                                observed_names=obs_var_names_actual, trend_names=current_trend_names, stationary_names=current_stationary_names,
                                time_index=time_index_actual, hdi_prob=hdi_prob_plot
                            )
                            if plot_path_full: 
                                safe_title = plot_title.lower().replace(' ','_').replace('/','_').replace('(','').replace(')','')
                                fig_custom.savefig(f"{plot_path_full}_custom_{safe_title}.png", dpi=300, bbox_inches='tight')
                            plt.close(fig_custom) 

                print("\n=== SUMMARY STATISTICS (FROM PLOTTING SECTION) ===") 
                if hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0:
                    trend_stats = compute_summary_statistics(all_trends_draws)
                    print(f"\nTrend Components (mean of time series means):")
                    for i_ts, name_ts in enumerate(current_trend_names):
                         if trend_stats['mean'].ndim > 1 and i_ts < trend_stats['mean'].shape[1]:
                            print(f"  {name_ts}: Mean={np.mean(trend_stats['mean'][:, i_ts]):.4f}, Std={np.mean(trend_stats['std'][:, i_ts]):.4f}")
                if hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0:
                    stat_stats = compute_summary_statistics(all_stationary_draws)
                    print(f"\nStationary Components (mean of time series means):")
                    for i_ts, name_ts in enumerate(current_stationary_names):
                         if stat_stats['mean'].ndim > 1 and i_ts < stat_stats['mean'].shape[1]:
                            print(f"  {name_ts}: Mean={np.mean(stat_stats['mean'][:, i_ts]):.4f}, Std={np.mean(stat_stats['std'][:, i_ts]):.4f}")
                print("✓ Plotting completed successfully.") 
            except Exception as e:
                import traceback
                print(f"Warning: Plotting failed: {e}"); traceback.print_exc()
        else:
            print("Skipping plots (no valid component draws available or generate_plots=False)")
    
    results_dict = {
        'mcmc_object': mcmc_results, 'parsed_gpm_model': parsed_gpm_model,
        'state_space_builder': state_space_builder,
        'reconstructed_trend_draws': all_trends_draws, 'reconstructed_stationary_draws': all_stationary_draws,
        'component_names': component_names, 'observed_data_numpy': y_numpy,
        'time_index': time_index_actual, 'observed_variable_names': obs_var_names_actual,
        'fitting_time_seconds': fit_time,
        'trend_summary_stats': compute_summary_statistics(all_trends_draws) if hasattr(all_trends_draws, 'shape') and all_trends_draws.shape[0] > 0 and all_trends_draws.shape[2] > 0 else None,
        'stationary_summary_stats': compute_summary_statistics(all_stationary_draws) if hasattr(all_stationary_draws, 'shape') and all_stationary_draws.shape[0] > 0 and all_stationary_draws.shape[2] > 0 else None,
        'hdi_prob': hdi_prob_plot, 'used_gamma_init': use_gamma_init, 'gamma_scale_factor': gamma_scale_factor
    }
    print("\n=== FIXED Workflow Complete ===")
    return results_dict

# plot_existing_smoother_results, debug_smoother_draws, quick_test_fixed_smoother_workflow, run_workflow_from_csv remain unchanged
def plot_existing_smoother_results( 
    all_trends_draws: jnp.ndarray, all_stationary_draws: jnp.ndarray,
    observed_data: np.ndarray, component_names: Dict[str, List[str]],
    variable_names: Optional[List[str]] = None, hdi_prob: float = 0.9,
    save_plots: bool = False, save_path: Optional[str] = None,
    time_index: Optional[Any] = None, show_info_box: bool = False,
    reduced_measurement_equations: Optional[Dict[str, Any]] = None 
) -> Dict[str, Any]:
    # ... (function body unchanged)
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
        plt.close(trend_fig); plt.close(stat_fig) 
        
        fitted_fig = plot_observed_vs_fitted(
            obs_np, trends_np, stationary_np, 
            variable_names=variable_names,
            trend_names=component_names.get('trends'), 
            stationary_names=component_names.get('stationary'),
            reduced_measurement_equations=reduced_measurement_equations,    
            hdi_prob=hdi_prob, save_path=plot_path_full, 
            time_index=time_index, show_info_box=show_info_box
        )
        results['fitted_figure'] = fitted_fig
        plt.close(fitted_fig) 
        
        if trends_np.shape[0] > 0 and trends_np.shape[2]>0: results['trend_stats'], results['trend_hdi'] = compute_summary_statistics(trends_np), compute_hdi_robust(trends_np, hdi_prob)
        if stationary_np.shape[0] > 0 and stationary_np.shape[2]>0: results['stationary_stats'], results['stationary_hdi'] = compute_summary_statistics(stationary_np), compute_hdi_robust(stationary_np, hdi_prob)
        print("✓ Plotting and analysis complete")
    except Exception as e: import traceback; traceback.print_exc(); results['error'] = str(e)
    return results

def debug_smoother_draws(
    all_trends_draws: jnp.ndarray,
    all_stationary_draws: jnp.ndarray,
    component_names: Dict[str, List[str]]
) -> None:
    # ... (function body unchanged)
    print("\n=== DEBUGGING SMOOTHER DRAWS ===")
    trends_np, stationary_np = np.asarray(all_trends_draws), np.asarray(all_stationary_draws)
    for name, arr in [("Trend", trends_np), ("Stationary", stationary_np)]:
        print(f"\n{name} draws:")
        print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}")
        print(f"  Has NaN: {np.any(np.isnan(arr))}, Has Inf: {np.any(np.isinf(arr))}")
        if arr.size > 0: print(f"  Min: {np.nanmin(arr):.6f}, Max: {np.nanmax(arr):.6f}, Mean: {np.nanmean(arr):.6f}")
    print(f"\nComponent names: Trends: {component_names.get('trends', 'N/A')}, Stationary: {component_names.get('stationary', 'N/A')}")
    if trends_np.shape[0] > 1: print(f"Trend variation (mean std across draws): {np.nanmean(np.nanstd(trends_np, axis=0)):.6f}")
    if stationary_np.shape[0] > 1: print(f"Stationary variation (mean std across draws): {np.nanmean(np.nanstd(stationary_np, axis=0)):.6f}")
    print("=== END DEBUG ===\n")

def quick_test_fixed_smoother_workflow():
    # ... (function body unchanged, ensure data_file_source_for_summary is passed)
    print("=== Running Quick Test with FIXED Smoother ===")
    num_obs, num_stat = 2, 2
    gpm_file = "smoother_test_fixed_model.gpm"
    create_default_gpm_file_if_needed(gpm_file, num_obs, num_stat)
    true_params = { 
        "rho": 0.5, "SHK_TREND1": 0.1, "SHK_TREND2": 0.15, 
        "SHK_STAT1": 0.2, "SHK_STAT2": 0.25, 
        "_var_coefficients": jnp.array([[[0.7, 0.1], [0.0, 0.6]]]),
        "_var_innovation_corr_chol": jnp.array([[1.0, 0.0], [0.3, jnp.sqrt(1-0.3**2)]])
    }
    sim_y = generate_synthetic_data_for_gpm(gpm_file, true_params, num_steps=100, rng_key_seed=789)
    if sim_y is None: print("Failed to generate synthetic data."); return None
    
    results = complete_gpm_workflow_with_smoother_fixed(
        data=sim_y, gpm_file=gpm_file,
        num_warmup=50, num_samples=100, num_chains=1, num_extract_draws=25,
        generate_plots=True, use_gamma_init=True, gamma_scale_factor=1.0,
        show_plot_info_boxes=False,
        data_file_source_for_summary="Synthetic Data (quick_test)" 
    )
    if results: 
        print(f"\n✓ FIXED smoother test successful!")
    else: print("\n✗ FIXED smoother test failed.")
    if os.path.exists(gpm_file): os.remove(gpm_file)
    return results


if __name__ == "__main__":
    quick_test_fixed_smoother_workflow()
    if True: 
        print("\n=== Running Custom Plot Example ===")
        num_obs_example, num_stat_example = 2, 1
        gpm_file_custom_plot = "custom_plot_model.gpm"
        data_file_custom_plot = "custom_plot_data.csv"
        create_default_gpm_file_if_needed(gpm_file_custom_plot, num_obs_example, num_stat_example)
        dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=50, freq='QE'))
        obs_data_df = pd.DataFrame(
            np.cumsum(np.random.randn(50, num_obs_example) * 0.2, axis=0) + np.random.randn(50, num_obs_example) * 0.1,
            index=dates, columns=[f"OBS{i+1}" for i in range(num_obs_example)]
        )
        obs_data_df.to_csv(data_file_custom_plot) 

        custom_specs = [
            {"title": "OBS1 Components", "series_to_plot": [
                {'type': 'observed', 'name': 'OBS1', 'label': 'Observed Data 1', 'style': 'k-'},
                {'type': 'trend', 'name': 'TREND1', 'label': 'Trend OBS1', 'show_hdi': True, 'color':'blue'},
                {'type': 'stationary', 'name': 'STAT1', 'label': 'Cycle OBS1', 'show_hdi': True, 'color':'green'}
            ]},
            {"title": "OBS1 vs Fitted (TREND1+STAT1)", "series_to_plot": [
                {'type': 'observed', 'name': 'OBS1'},
                {'type': 'combined', 'components': [{'type':'trend', 'name':'TREND1'}, {'type':'stationary', 'name':'STAT1'}],
                 'label': 'Fitted (TREND1+STAT1)', 'show_hdi': True, 'color':'purple'}
            ]}
        ]
        results_custom = complete_gpm_workflow_with_smoother_fixed(
            data=obs_data_df, gpm_file=gpm_file_custom_plot,
            num_warmup=20, num_samples=30, num_chains=1, num_extract_draws=10, 
            generate_plots=True, show_plot_info_boxes=False,
            custom_plot_specs=custom_specs, plot_save_path="custom_plot_output", save_plots=True,
            data_file_source_for_summary=data_file_custom_plot 
        )
        print("Custom plot example workflow " + ("successful." if results_custom else "failed."))
        if os.path.exists(gpm_file_custom_plot): os.remove(gpm_file_custom_plot)
        if os.path.exists(data_file_custom_plot): os.remove(data_file_custom_plot)
    print("\nWorkflow enhancements implemented.")