# clean_gpm_bvar_trends/gpm_bar_smoother.py - Simplified and Fixed

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
import arviz as az
import matplotlib.pyplot as plt

from .gpm_numpyro_models import fit_gpm_numpyro_model, define_gpm_numpyro_model
from .integration_orchestrator import IntegrationOrchestrator, create_integration_orchestrator
from .simulation_smoothing import jarocinski_corrected_simulation_smoother, _extract_initial_mean
from .constants import _DEFAULT_DTYPE, _KF_JITTER
from .gpm_model_parser import ReducedModel
from .symbolic_evaluation_utils import (  # ← ADD THESE LINES
    evaluate_coefficient_expression,
    parse_variable_key
)
from .state_space_builder import StateSpaceBuilder
# Import standardized data object
from .common_types import SmootherResults

# Import P0 utilities
try:
    from .P0_utils import (
        _build_gamma_based_p0,
        _create_standard_p0,
        _extract_gamma_matrices_from_params
    )
except ImportError as e:
    _build_gamma_based_p0 = None
    _create_standard_p0 = None
    _extract_gamma_matrices_from_params = None
    print(f"ERROR: Failed to import P0 utilities: {e}")

# Import StateSpaceBuilder
from .symbolic_evaluation_utils import (
    evaluate_coefficient_expression,
    parse_variable_key
)


# Try importing simulate_state_space
try:
    from .Kalman_filter_jax import simulate_state_space
except ImportError:
    simulate_state_space = None
    print("Warning: simulate_state_space not available")

# Simple plotting functions
try:
    from .reporting_plots import (
        compute_hdi_robust,
        compute_summary_statistics,
        plot_time_series_with_uncertainty,
        plot_custom_series_comparison
    )
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import plotting functions: {e}")
    PLOTTING_AVAILABLE = False
    
    # Define dummy functions
    def compute_hdi_robust(draws, hdi_prob=0.9):
        if hasattr(draws, 'shape') and len(draws.shape) > 1:
            return (np.full(draws.shape[1:], np.nan), np.full(draws.shape[1:], np.nan))
        return (np.nan, np.nan)
    
    def compute_summary_statistics(draws):
        if hasattr(draws, 'shape') and len(draws.shape) > 1:
            nan_array = np.full(draws.shape[1:], np.nan)
            return {'mean': nan_array, 'median': nan_array, 'mode': nan_array, 'std': nan_array}
        return {'mean': np.nan, 'median': np.nan, 'mode': np.nan, 'std': np.nan}
    
    def plot_time_series_with_uncertainty(*args, **kwargs):
        print("Plotting disabled - plot_time_series_with_uncertainty skipped")
        return None
    
    
    # def plot_custom_series_comparison(*args, **kwargs):
    #     print("Plotting disabled - plot_custom_series_comparison skipped")
    #     return None

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def _print_model_and_run_settings_summary(
    gpm_file: str,
    data_file_source_for_summary: Optional[str],
    parsed_gpm_model: ReducedModel,
    num_warmup: int, num_samples: int, num_chains: int,
    use_gamma_init: bool, gamma_scale_factor: float,
    target_accept_prob: float,
    num_extract_draws: int
):
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


def print_filtered_mcmc_summary(
    mcmc_results: numpyro.infer.MCMC,
    parsed_gpm_model: ReducedModel
):
    print("\n--- Filtered MCMC Summary (Key Model Parameters using ArviZ) ---")

    params_to_include = set()
    params_to_include.update(parsed_gpm_model.parameters)

    for shock_name in parsed_gpm_model.trend_shocks + parsed_gpm_model.stationary_shocks:
        params_to_include.add(f"sigma_{shock_name}")

    if parsed_gpm_model.var_prior_setup:
        params_to_include.add("A_transformed")
        params_to_include.add("Omega_u_chol")

    params_to_include.add("init_mean_full")

    try:
        idata = az.from_numpyro(mcmc_results)
        available_vars_in_posterior = set(idata.posterior.data_vars.keys())
        final_vars_to_summarize = list(params_to_include.intersection(available_vars_in_posterior))

        if not final_vars_to_summarize:
            print("  No relevant model parameters found in ArviZ InferenceData posterior group.")
            print("  Displaying basic NumPyro summary for all parameters instead.")
            mcmc_results.print_summary(exclude_deterministic=False)
        else:
            summary_df = az.summary(idata, var_names=final_vars_to_summarize)
            print(summary_df)

    except Exception as e_arviz:
        print(f"  Error during ArviZ processing (az.from_numpyro or az.summary): {e_arviz}")
        print("  Attempting basic NumPyro summary for all parameters as fallback.")
        try:
            mcmc_results.print_summary(exclude_deterministic=False)
        except Exception as e_numpyro_summary:
            print(f"    Basic NumPyro summary also failed: {e_numpyro_summary}")
            print(f"    MCMC object type: {type(mcmc_results)}")
    finally:
        print("-" * 70 + "\n")
        return # Ensure it always returns from the function block

    # This part is now effectively dead code due to the return in the finally block,
    # but keeping structure for clarity of what was attempted.
    # available_vars_in_posterior = set(idata.posterior.data_vars.keys()) # This line would be problematic if idata is not defined
    # final_vars_to_summarize = list(params_to_include.intersection(available_vars_in_posterior))

    # if not final_vars_to_summarize:
    #     print("  No relevant model parameters found in ArviZ InferenceData posterior group.")
    #     print("  Displaying basic NumPyro summary for all parameters instead.")
    #     mcmc_results.print_summary(exclude_deterministic=False)
    #     print("-" * 70 + "\n")
    #     return

    # try:
    #     summary_df = az.summary(idata, var_names=final_vars_to_summarize)
    #     print(summary_df)
    # except Exception as e:
    #     print(f"  Error generating ArviZ summary: {e}")
    #     print("  Falling back to basic NumPyro summary.")
    #     mcmc_results.print_summary(exclude_deterministic=False)

    # print("-" * 70 + "\n")


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
    if mcmc_results is not None:
        print_filtered_mcmc_summary(
            mcmc_results=mcmc_results,
            parsed_gpm_model=parsed_gpm_model
        )
    else:
        print("\n--- MCMC Summary ---")
        print("  MCMC results not available at the time of this report section (e.g., called before MCMC run).")
        print("-" * 70 + "\n")


def create_default_gpm_file_if_needed(filename: str, num_obs_vars: int, num_stat_vars: int = 0):
    if os.path.exists(filename): 
        return
    print(f"Creating default gpm file: {filename} with {num_obs_vars} obs vars and {num_stat_vars} stat vars")
    
    gpm_content = "parameters rho;\n"
    gpm_content += "\nestimated_params;\n"
    gpm_content += "    rho, normal_pdf, 0.5, 0.2;\n"
    for i in range(num_obs_vars): 
        gpm_content += f"    stderr SHK_TREND{i+1}, inv_gamma_pdf, 2.0, 0.02;\n"
    for i in range(num_stat_vars): 
        gpm_content += f"    stderr SHK_STAT{i+1}, inv_gamma_pdf, 2.0, 0.1;\n"
    gpm_content += "end;\n"
    
    trend_names = [f"TREND{i+1}" for i in range(num_obs_vars)]
    gpm_content += f"\ntrends_vars {', '.join(trend_names)};\n"
    gpm_content += "\ntrend_shocks;\n"
    for i in range(num_obs_vars): 
        gpm_content += f"    var SHK_TREND{i+1};\n"
    gpm_content += "end;\n"
    
    if num_stat_vars > 0:
        stat_names = [f"STAT{i+1}" for i in range(num_stat_vars)]
        gpm_content += f"\nstationary_variables {', '.join(stat_names)};\n"
        gpm_content += "\nshocks;\n"
        for i in range(num_stat_vars): 
            gpm_content += f"    var SHK_STAT{i+1};\n"
        gpm_content += "end;\n"
    else:
        # If no stationary variables, still need a shocks block, potentially empty or for trend shocks only
        # The original logic for trend shocks is separate. This block is for stationary variable shocks.
        # If num_stat_vars is 0, this path correctly adds empty "shocks;" and "end;"
        gpm_content += "\nshocks;\nend;\n"
        # Also ensure stationary_variables line is added if it wasn't:
        if not "stationary_variables" in gpm_content: # Basic check, could be more robust
             gpm_content += "\nstationary_variables ;\n"
    
    gpm_content += "\ntrend_model;\n"
    for i in range(num_obs_vars): 
        gpm_content += f"    TREND{i+1} = TREND{i+1}(-1) + SHK_TREND{i+1};\n"
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
        print(f"Error generating synthetic data: {e}")
        traceback.print_exc()
        return None


def debug_mcmc_parameter_variation(mcmc_results, num_draws_to_check=5):
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
            if param_array.ndim == 1: 
                print(f"  First {num_draws_to_check} draws: {param_array[:num_draws_to_check]}")
            else: 
                print(f"  First draw mean: {jnp.mean(param_array[0]):.6f}; Last draw mean: {jnp.mean(param_array[-1]):.6f}")
    print("=== END MCMC DEBUGGING ===\n")


def extract_reconstructed_components(
    mcmc_output: numpyro.infer.MCMC,
    y_data: jnp.ndarray,
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    num_smooth_draws: int = 100,
    rng_key_smooth: Optional[jax.Array] = None,
    use_gamma_init_for_smoother: bool = True,
    gamma_init_scaling_for_smoother: float = 1.0,
    hdi_prob: float = 0.9,
    observed_variable_names: Optional[List[str]] = None,
    time_index: Optional[Any] = None,
    # New arguments for P0 overrides for the smoother
    trend_P0_scales_override: Optional[Union[float, Dict[str, float]]] = None,
    stationary_P0_scale_override: Optional[float] = None
) -> SmootherResults:
    """
    Simplified extraction function that returns SmootherResults object.
    """
    print(f"\n=== SIMULATION SMOOTHER (from MCMC draws) ===")
    print(f"Use gamma-based P0: {use_gamma_init_for_smoother}")
    if use_gamma_init_for_smoother:
        print(f"Gamma scaling: {gamma_init_scaling_for_smoother}")
    if trend_P0_scales_override is not None:
        print(f"Smoother Trend P0 Scales Override: {trend_P0_scales_override}")
    if stationary_P0_scale_override is not None:
        print(f"Smoother Stationary P0 Scale Override: {stationary_P0_scale_override}")
    print(f"HDI probability: {hdi_prob}")

    if rng_key_smooth is None:
        rng_key_smooth = random.PRNGKey(0)
    
    if jarocinski_corrected_simulation_smoother is None:
        raise RuntimeError("Simulation smoother function not available.")
    
    if _build_gamma_based_p0 is None or _create_standard_p0 is None:
        raise RuntimeError("P0 building helper functions not available.")

    T_data, N_obs_data = y_data.shape # Moved this line up

    mcmc_samples = mcmc_output.get_samples(group_by_chain=False)

    if not mcmc_samples or not any(hasattr(v, 'shape') and v.shape[0] > 0 for v in mcmc_samples.values()):
        print("Warning: No MCMC samples available.")
        # Return empty SmootherResults
        return _create_empty_smoother_results(T_data, N_obs_data, gpm_model, observed_variable_names, time_index, hdi_prob)

    # T_data, N_obs_data = y_data.shape # Already defined above
    state_dim = ss_builder.state_dim

    first_param_key = list(mcmc_samples.keys())[0]
    total_posterior_draws = mcmc_samples[first_param_key].shape[0]
    actual_num_smooth_draws = min(num_smooth_draws, total_posterior_draws)

    if actual_num_smooth_draws <= 0:
        return _create_empty_smoother_results(T_data, N_obs_data, gpm_model, observed_variable_names, time_index, hdi_prob)

    draw_indices = np.round(np.linspace(0, total_posterior_draws - 1, actual_num_smooth_draws)).astype(int)
    print(f"Processing {actual_num_smooth_draws} draws")

    # NOW ADD THE DEBUG CODE HERE (after draw_indices is defined):
    # if debugging_flag 
    #     print(f"\n=== DEBUGGING MCMC SAMPLES ===")
    #     problematic_params = ['lambda_pi_US', 'lambda_pi_EA', 'lambda_pi_JP']
        
    #     for param_name in problematic_params:
    #         if param_name in mcmc_samples:
    #             param_array = mcmc_samples[param_name]
    #             finite_count = jnp.sum(jnp.isfinite(param_array))
    #             print(f"{param_name}: shape={param_array.shape}, finite={finite_count}/{param_array.shape[0]}")
    #             print(f"  first 3 values: {param_array[:3]}")
    #             print(f"  mean: {jnp.nanmean(param_array):.6f}")
    #         else:
    #             print(f"{param_name}: NOT FOUND in MCMC samples")
        
    #     print(f"Available MCMC parameters: {sorted(list(mcmc_samples.keys()))}")
    #     print(f"=== END MCMC SAMPLES DEBUG ===\n")

    # Process draws
    output_trend_draws_list = []
    output_stationary_draws_list = []
    successful_draws = 0

    # Resolve observed variable names
    obs_var_names_for_results = observed_variable_names if observed_variable_names is not None else [f'OBS{i+1}' for i in range(N_obs_data)]
    if len(obs_var_names_for_results) != N_obs_data:
        obs_var_names_for_results = [f'OBS{i+1}' for i in range(N_obs_data)]

       
    for i_loop, mcmc_draw_idx in enumerate(draw_indices):
        rng_key_smooth, sim_key = random.split(rng_key_smooth)

        try:
            # Extract parameters for current draw
            current_builder_params_draw = ss_builder._extract_params_from_mcmc_draw(mcmc_samples, mcmc_draw_idx)
            # ADD DETAILED DEBUG FOR FIRST FEW DRAWS:
            # if i_loop < 3:  # Only debug first 3 draws to avoid spam
            #     print(f"\n=== DEBUG DRAW {i_loop} (MCMC index {mcmc_draw_idx}) ===")
            #     for param_name in problematic_params:
            #         if param_name in current_builder_params_draw:
            #             val = current_builder_params_draw[param_name]
            #             is_finite = jnp.isfinite(val) if hasattr(val, 'shape') and val.ndim == 0 else np.isfinite(val)
            #             print(f"  {param_name} (extracted): {val} (finite: {is_finite})")
                        
            #             # Compare with raw MCMC
            #             if param_name in mcmc_samples:
            #                 raw_val = mcmc_samples[param_name][mcmc_draw_idx]
            #                 raw_finite = jnp.isfinite(raw_val)
            #                 print(f"  {param_name} (raw MCMC): {raw_val} (finite: {raw_finite})")
            #         else:
            #             print(f"  {param_name}: NOT in extracted parameters")
            #     print(f"=== END DEBUG DRAW {i_loop} ===\n")


            # Build state space matrices
            F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_from_direct_dict(current_builder_params_draw)

            # Get initial mean
            init_mean_for_smoother = _extract_initial_mean(mcmc_samples, mcmc_draw_idx, state_dim)

            # Build P0 (simplified logic)
            # Build P0
            # Get necessary info from ss_builder for P0 functions
            # Correctly derive dynamic_trend_names_list as per the issue
            # Ensure ss_builder.core_var_map and ss_builder.n_dynamic_trends are available and correct.
            # Based on P0_utils, core_var_map is expected from the model_description if not directly on ss_builder.
            # And dynamic_trend_names are specifically the names of the trend variables.
            if not hasattr(ss_builder, 'core_var_map') or not hasattr(ss_builder, 'n_dynamic_trends'):
                raise AttributeError("StateSpaceBuilder instance is missing 'core_var_map' or 'n_dynamic_trends' attributes required for P0 setup.")

            dynamic_trend_names_list = [
                name for name, idx in sorted(ss_builder.core_var_map.items(), key=lambda item: item[1])
                if idx < ss_builder.n_dynamic_trends
            ]
            # core_var_map_for_p0 should be the map used by P0 utils, typically ss_builder.model_description.core_var_map
            # or ss_builder.core_var_map if it's the same. Let's assume ss_builder.core_var_map is the one to use.
            core_var_map_for_p0 = ss_builder.core_var_map

            if (use_gamma_init_for_smoother and ss_builder.n_stationary > 0 and
                ss_builder.var_order > 0 and _extract_gamma_matrices_from_params is not None and
                _build_gamma_based_p0 is not None):
                
                A_trans = current_builder_params_draw.get("_var_coefficients")
                n_stat = ss_builder.n_stationary
                var_start = ss_builder.n_dynamic_trends
                
                gamma_list = None
                if Q_draw is not None and var_start + n_stat <= state_dim and A_trans is not None:
                    Sigma_u = Q_draw[var_start:var_start + n_stat, var_start:var_start + n_stat]
                    gamma_list = _extract_gamma_matrices_from_params(A_trans, Sigma_u, n_stat, ss_builder.var_order)

                if gamma_list is not None:
                    init_cov_for_smoother = _build_gamma_based_p0(
                        state_dim=state_dim,
                        n_dynamic_trends=ss_builder.n_dynamic_trends,
                        gamma_list=gamma_list,
                        n_stationary=n_stat,
                        var_order=ss_builder.var_order,
                        gamma_scaling=gamma_init_scaling_for_smoother,
                        dynamic_trend_names=dynamic_trend_names_list,
                        core_var_map=core_var_map_for_p0,
                        context="mcmc_smoother",
                        trend_P0_scales_override=trend_P0_scales_override, # New arg
                        var_P0_var_scale_override=stationary_P0_scale_override # New arg (maps to var part in P0 func)
                    )
                else:
                    # Fallback to standard P0 if gamma calculation failed
                    if _create_standard_p0 is not None:
                        init_cov_for_smoother = _create_standard_p0(
                            state_dim=state_dim,
                            n_dynamic_trends=ss_builder.n_dynamic_trends,
                            dynamic_trend_names=dynamic_trend_names_list,
                            core_var_map=core_var_map_for_p0,
                            context="mcmc_smoother",
                            trend_P0_scales_override=trend_P0_scales_override, # New arg
                            var_P0_var_scale_override=stationary_P0_scale_override # New arg
                        )
                    else: # Should not happen if imports are correct
                        raise RuntimeError("_create_standard_p0 is not available")
            elif _create_standard_p0 is not None:
                # Standard P0 if not using gamma or conditions not met
                init_cov_for_smoother = _create_standard_p0(
                    state_dim=state_dim,
                    n_dynamic_trends=ss_builder.n_dynamic_trends,
                    dynamic_trend_names=dynamic_trend_names_list,
                    core_var_map=core_var_map_for_p0,
                    context="mcmc_smoother",
                    trend_P0_scales_override=trend_P0_scales_override, # New arg
                    var_P0_var_scale_override=stationary_P0_scale_override # New arg
                )
            else: # Should not happen
                raise RuntimeError("_create_standard_p0 is not available for fallback P0 building.")

            # Check if matrices are finite
            if not (jnp.all(jnp.isfinite(F_draw)) and jnp.all(jnp.isfinite(Q_draw)) and
                    jnp.all(jnp.isfinite(C_draw)) and jnp.all(jnp.isfinite(H_draw)) and
                    jnp.all(jnp.isfinite(init_mean_for_smoother)) and jnp.all(jnp.isfinite(init_cov_for_smoother))):
                continue

            # Regularize Q and get Cholesky
            Q_reg = (Q_draw + Q_draw.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim)
            try:
                R_draw = jnp.linalg.cholesky(Q_reg)
            except:
                R_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_reg), _KF_JITTER)))

            # Run simulation smoother
            core_states_smoothed = jarocinski_corrected_simulation_smoother(
                y_data, F_draw, R_draw, C_draw, H_draw,
                init_mean_for_smoother, init_cov_for_smoother, sim_key
            )

            if not jnp.all(jnp.isfinite(core_states_smoothed)):
                print(f"  Draw {i_loop}: Core states not finite, skipping")
                continue

            # Reconstruct original variables
            trends_draw, stationary_draw = _reconstruct_original_variables(
                core_states_smoothed, gpm_model, ss_builder, current_builder_params_draw, T_data, state_dim
            )
        

            # ADD DETAILED DEBUGGING HERE:
            # print(f"  Draw {i_loop}: Reconstruction completed")
            # print(f"    Trends shape: {trends_draw.shape}")
            # print(f"    Stationary shape: {stationary_draw.shape}")
            # print(f"    Trends finite: {jnp.all(jnp.isfinite(trends_draw))}")
            # print(f"    Stationary finite: {jnp.all(jnp.isfinite(stationary_draw))}")
            
            if not jnp.all(jnp.isfinite(trends_draw)):
                nan_count = jnp.sum(jnp.isnan(trends_draw))
                inf_count = jnp.sum(jnp.isinf(trends_draw))
                print(f"    Trends: {nan_count} NaNs, {inf_count} Infs")
                
                # Check which variables have NaNs
                for var_idx in range(trends_draw.shape[1]):
                    var_finite = jnp.all(jnp.isfinite(trends_draw[:, var_idx]))
                    var_name = gpm_model.gpm_trend_variables_original[var_idx] if var_idx < len(gpm_model.gpm_trend_variables_original) else f"Var{var_idx}"
                    if not var_finite:
                        print(f"      Variable {var_idx} ({var_name}): NOT finite")
            
            if not jnp.all(jnp.isfinite(stationary_draw)):
                nan_count = jnp.sum(jnp.isnan(stationary_draw))
                inf_count = jnp.sum(jnp.isinf(stationary_draw))
                print(f"    Stationary: {nan_count} NaNs, {inf_count} Infs")

            # The existing finite check
            if jnp.all(jnp.isfinite(trends_draw)) and jnp.all(jnp.isfinite(stationary_draw)):
                output_trend_draws_list.append(trends_draw)
                output_stationary_draws_list.append(stationary_draw)
                successful_draws += 1
                #print(f"  Draw {i_loop}: ✓ ACCEPTED")
            else:
                print(f"  Draw {i_loop}: ✗ REJECTED (non-finite values)")

        except Exception as e:
            print(f"  Draw {i_loop}: Failed with error: {e}")
            continue         
        #     if not jnp.all(jnp.isfinite(core_states_smoothed)):
        #         continue

        #     # Reconstruct original variables (simplified)
        #     trends_draw, stationary_draw = _reconstruct_original_variables(
        #         core_states_smoothed, gpm_model, ss_builder, current_builder_params_draw, T_data, state_dim
        #     )

        #     if jnp.all(jnp.isfinite(trends_draw)) and jnp.all(jnp.isfinite(stationary_draw)):
        #         output_trend_draws_list.append(trends_draw)
        #         output_stationary_draws_list.append(stationary_draw)
        #         successful_draws += 1

        # except Exception as e:
        #     print(f"  Draw {i_loop}: Failed with error: {e}")
        #     continue

    print(f"Successfully processed {successful_draws}/{len(draw_indices)} draws")

    # Stack results
    if output_trend_draws_list:
        final_trend_draws = jnp.stack(output_trend_draws_list)
        final_stationary_draws = jnp.stack(output_stationary_draws_list)
    else:
        final_trend_draws = jnp.empty((0, T_data, len(gpm_model.gpm_trend_variables_original)))
        final_stationary_draws = jnp.empty((0, T_data, len(gpm_model.gpm_stationary_variables_original)))

    # Compute statistics
    if PLOTTING_AVAILABLE and final_trend_draws.shape[0] > 0:
        trend_stats = compute_summary_statistics(np.asarray(final_trend_draws))
        stationary_stats = compute_summary_statistics(np.asarray(final_stationary_draws))
        
        trend_hdi_lower, trend_hdi_upper = None, None
        stationary_hdi_lower, stationary_hdi_upper = None, None
        
        if final_trend_draws.shape[0] > 1:
            trend_hdi_lower, trend_hdi_upper = compute_hdi_robust(np.asarray(final_trend_draws), hdi_prob)
            stationary_hdi_lower, stationary_hdi_upper = compute_hdi_robust(np.asarray(final_stationary_draws), hdi_prob)
    else:
        trend_stats = {}
        stationary_stats = {}
        trend_hdi_lower, trend_hdi_upper = None, None
        stationary_hdi_lower, stationary_hdi_upper = None, None

    # Create SmootherResults
    results = SmootherResults(
        observed_data=np.asarray(y_data),
        observed_variable_names=obs_var_names_for_results,
        time_index=time_index,
        trend_draws=np.asarray(final_trend_draws),
        trend_names=list(gpm_model.gpm_trend_variables_original),
        trend_stats=trend_stats,
        trend_hdi_lower=trend_hdi_lower,
        trend_hdi_upper=trend_hdi_upper,
        stationary_draws=np.asarray(final_stationary_draws),
        stationary_names=list(gpm_model.gpm_stationary_variables_original),
        stationary_stats=stationary_stats,
        stationary_hdi_lower=stationary_hdi_lower,
        stationary_hdi_upper=stationary_hdi_upper,
        reduced_measurement_equations=gpm_model.reduced_measurement_equations,
        gpm_model=gpm_model,
        parameters_used=None,
        log_likelihood=None,
        n_draws=final_trend_draws.shape[0],
        hdi_prob=hdi_prob
    )

    print(f"=== END SIMULATION SMOOTHER ===")
    return results


def _create_empty_smoother_results(T_data, N_obs_data, gpm_model, observed_variable_names, time_index, hdi_prob):
    """Helper to create empty SmootherResults"""
    num_orig_trends = len(gpm_model.gpm_trend_variables_original)
    num_orig_stat = len(gpm_model.gpm_stationary_variables_original)
    obs_var_names = observed_variable_names if observed_variable_names is not None else [f'OBS{i+1}' for i in range(N_obs_data)]
    
    return SmootherResults(
        observed_data=np.empty((T_data, N_obs_data)),
        observed_variable_names=obs_var_names,
        time_index=time_index,
        trend_draws=np.empty((0, T_data, num_orig_trends)),
        trend_names=list(gpm_model.gpm_trend_variables_original),
        trend_stats={},
        trend_hdi_lower=None,
        trend_hdi_upper=None,
        stationary_draws=np.empty((0, T_data, num_orig_stat)),
        stationary_names=list(gpm_model.gpm_stationary_variables_original),
        stationary_stats={},
        stationary_hdi_lower=None,
        stationary_hdi_upper=None,
        reduced_measurement_equations=gpm_model.reduced_measurement_equations,
        gpm_model=gpm_model,
        parameters_used=None,
        log_likelihood=None,
        n_draws=0,
        hdi_prob=hdi_prob
    )


# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder,
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     FIXED: Proper reconstruction including non-core trend variables.
#     """
#     from .symbolic_evaluation_utils import SymbolicEvaluationUtils
    
#     # Initialize output arrays
#     reconstructed_trends = jnp.full(
#         (T_data, len(gpm_model.gpm_trend_variables_original)),
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )
#     reconstructed_stationary = jnp.full(
#         (T_data, len(gpm_model.gpm_stationary_variables_original)), 
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )

#     # Get core state values by name
#     core_var_map = ss_builder.core_var_map
#     current_draw_core_state_values = {}
    
#     # Map dynamic trends
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]

#     # FIXED: Reconstruct original trend variables (both core and non-core)
#     utils = SymbolicEvaluationUtils()
    
#     for i, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
#         if orig_trend_name in current_draw_core_state_values:
#             # Core trend - direct mapping
#             reconstructed_trends = reconstructed_trends.at[:, i].set(
#                 current_draw_core_state_values[orig_trend_name]
#             )
#         elif orig_trend_name in gpm_model.non_core_trend_definitions:
#             # Non-core trend - evaluate expression
#             expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
#             reconstructed_value_ts = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
            
#             # Evaluate constant term
#             const_val = utils.evaluate_coefficient_expression(
#                 expr_def.constant_str, current_builder_params_draw
#             )
#             if jnp.isfinite(const_val):
#                 reconstructed_value_ts += const_val
            
#             # Evaluate each term in the expression
#             for var_key, coeff_str in expr_def.terms.items():
#                 var_name, lag = utils._parse_variable_key(var_key)
                
#                 if lag == 0 and var_name in current_draw_core_state_values:
#                     # Evaluate coefficient
#                     coeff_val = utils.evaluate_coefficient_expression(
#                         coeff_str, current_builder_params_draw
#                     )
#                     if jnp.isfinite(coeff_val):
#                         reconstructed_value_ts += coeff_val * current_draw_core_state_values[var_name]
            
#             reconstructed_trends = reconstructed_trends.at[:, i].set(reconstructed_value_ts)

#     # Reconstruct stationary variables (similar logic)
#     for i, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
#         if orig_stat_name in current_draw_core_state_values:
#             reconstructed_stationary = reconstructed_stationary.at[:, i].set(
#                 current_draw_core_state_values[orig_stat_name]
#             )

#     return reconstructed_trends, reconstructed_stationary

# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder,
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Simplified reconstruction of original GPM variables from core states.
#     """
#     # Import utility functions
#     from .symbolic_evaluation_utils import evaluate_coefficient_expression, parse_variable_key
    
#     # Initialize output arrays
#     reconstructed_trends = jnp.full(
#         (T_data, len(gpm_model.gpm_trend_variables_original)),
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )
#     reconstructed_stationary = jnp.full(
#         (T_data, len(gpm_model.gpm_stationary_variables_original)), 
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )

#     # Get core state values by name
#     core_var_map = ss_builder.core_var_map
#     current_draw_core_state_values = {}
    
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]

#     # Reconstruct original trend variables
#     for i, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
#         if orig_trend_name in current_draw_core_state_values:
#             # Core trend - direct mapping
#             reconstructed_trends = reconstructed_trends.at[:, i].set(
#                 current_draw_core_state_values[orig_trend_name]
#             )
#         elif orig_trend_name in gpm_model.non_core_trend_definitions:
#             # Non-core trend - evaluate expression
#             expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
#             reconstructed_value_ts = jnp.full(T_data, 0.0, dtype=_DEFAULT_DTYPE)

#             # Evaluate constant term
#             const_val_eval = evaluate_coefficient_expression(expr_def.constant_str, current_builder_params_draw)
#             if hasattr(const_val_eval, 'ndim') and const_val_eval.ndim == 0:
#                 reconstructed_value_ts += float(const_val_eval)
#             elif isinstance(const_val_eval, (float, int, np.number)):
#                  reconstructed_value_ts += float(const_val_eval)

#             # Evaluate each term in the expression
#             for var_key, coeff_str in expr_def.terms.items():
#                 var_name, lag = parse_variable_key(var_key)
#                 coeff_val_eval = evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
#                 coeff_num = None
#                 if hasattr(coeff_val_eval, 'ndim') and coeff_val_eval.ndim == 0:
#                     coeff_num = float(coeff_val_eval)
#                 elif isinstance(coeff_val_eval, (float, int, np.number)):
#                     coeff_num = float(coeff_val_eval)

#                 if coeff_num is not None:
#                     if lag == 0:
#                         if var_name in current_draw_core_state_values:
#                             reconstructed_value_ts += coeff_num * current_draw_core_state_values[var_name]
#                         elif var_name in current_builder_params_draw:
#                              param_val_eval = evaluate_coefficient_expression(var_name, current_builder_params_draw)
#                              if hasattr(param_val_eval, 'ndim') and param_val_eval.ndim == 0:
#                                   reconstructed_value_ts += coeff_num * float(param_val_eval)
#                              elif isinstance(param_val_eval, (float, int, np.number)):
#                                   reconstructed_value_ts += coeff_num * float(param_val_eval)

#             reconstructed_trends = reconstructed_trends.at[:, i].set(reconstructed_value_ts)

#     # Reconstruct original stationary variables
#     for i, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
#         if orig_stat_name in current_draw_core_state_values and orig_stat_name in gpm_model.stationary_variables:
#              reconstructed_stationary = reconstructed_stationary.at[:, i].set(
#                   current_draw_core_state_values[orig_stat_name]
#               )

#     return reconstructed_trends, reconstructed_stationary

# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder,
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     FIXED: Proper reconstruction including non-core trend variables.
#     """
#     #from .symbolic_evaluation_utils import SymbolicEvaluationUtils
    
#     # Initialize output arrays
#     reconstructed_trends = jnp.full(
#         (T_data, len(gpm_model.gpm_trend_variables_original)),
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )
#     reconstructed_stationary = jnp.full(
#         (T_data, len(gpm_model.gpm_stationary_variables_original)), 
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )

#     # Get core state values by name
#     core_var_map = ss_builder.core_var_map
#     current_draw_core_state_values = {}
    
#     # Map dynamic trends
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]

#     # FIXED: Reconstruct original trend variables (both core and non-core)
#     utils = SymbolicEvaluationUtils()
    
#     for i, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
#         if orig_trend_name in current_draw_core_state_values:
#             # Core trend - direct mapping
#             reconstructed_trends = reconstructed_trends.at[:, i].set(
#                 current_draw_core_state_values[orig_trend_name]
#             )
#         elif orig_trend_name in gpm_model.non_core_trend_definitions:
#             # Non-core trend - evaluate expression
#             expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
#             reconstructed_value_ts = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
            
#             # Evaluate constant term
#             const_val = utils.evaluate_coefficient_expression(
#                 expr_def.constant_str, current_builder_params_draw
#             )
#             if jnp.isfinite(const_val):
#                 reconstructed_value_ts += const_val
            
#             # Evaluate each term in the expression
#             for var_key, coeff_str in expr_def.terms.items():
#                 var_name, lag = utils._parse_variable_key(var_key)
                
#                 if lag == 0 and var_name in current_draw_core_state_values:
#                     # Evaluate coefficient
#                     coeff_val = utils.evaluate_coefficient_expression(
#                         coeff_str, current_builder_params_draw
#                     )
#                     if jnp.isfinite(coeff_val):
#                         reconstructed_value_ts += coeff_val * current_draw_core_state_values[var_name]
            
#             reconstructed_trends = reconstructed_trends.at[:, i].set(reconstructed_value_ts)

#     # Reconstruct stationary variables (similar logic)
#     for i, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
#         if orig_stat_name in current_draw_core_state_values:
#             reconstructed_stationary = reconstructed_stationary.at[:, i].set(
#                 current_draw_core_state_values[orig_stat_name]
#             )

#     return reconstructed_trends, reconstructed_stationary

# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model: ReducedModel,
#     ss_builder: StateSpaceBuilder,
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     FIXED: Proper reconstruction including non-core trend variables.
#     """
#     # Remove this line since you've imported at the top:
#     # from .symbolic_evaluation_utils import evaluate_coefficient_expression, parse_variable_key
    
#     # Initialize output arrays
#     reconstructed_trends = jnp.full(
#         (T_data, len(gpm_model.gpm_trend_variables_original)),
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )
#     reconstructed_stationary = jnp.full(
#         (T_data, len(gpm_model.gpm_stationary_variables_original)), 
#         jnp.nan, dtype=_DEFAULT_DTYPE
#     )

#     # Get core state values by name
#     core_var_map = ss_builder.core_var_map
#     current_draw_core_state_values = {}
    
#     # Map ALL core variables (both dynamic trends and stationary)
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]

#     print(f"    Available core state values: {list(current_draw_core_state_values.keys())}")

#     # Reconstruct original trend variables
#     for i, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
#         print(f"    Processing trend {i}: {orig_trend_name}")
        
#         if orig_trend_name in current_draw_core_state_values:
#             # Core trend - direct mapping
#             print(f"      -> Core trend, direct mapping")
#             reconstructed_trends = reconstructed_trends.at[:, i].set(
#                 current_draw_core_state_values[orig_trend_name]
#             )
#         elif orig_trend_name in gpm_model.non_core_trend_definitions:
#             # Non-core trend - evaluate expression
#             print(f"      -> Non-core trend, evaluating expression")
#             expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
#             print(f"      -> Expression: terms={expr_def.terms}, constant='{expr_def.constant_str}'")
            
#             # Start with zeros
#             reconstructed_value_ts = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)

#             # Evaluate each term in the expression
#             for var_key, coeff_str in expr_def.terms.items():
#                 try:
#                     # Parse variable key using imported function
#                     var_name, lag = parse_variable_key(var_key)
#                     print(f"        -> Processing term: {coeff_str}*{var_name}(-{lag})")
                    
#                     if lag == 0 and var_name in current_draw_core_state_values:
#                         # Evaluate coefficient using imported function
#                         coeff_val_eval = evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
                        
#                         if hasattr(coeff_val_eval, 'ndim') and coeff_val_eval.ndim == 0:
#                             coeff_num = float(coeff_val_eval)
#                         elif isinstance(coeff_val_eval, (float, int)):
#                             coeff_num = float(coeff_val_eval)
#                         else:
#                             print(f"          -> Coefficient not scalar: {type(coeff_val_eval)}")
#                             continue
                            
#                         reconstructed_value_ts += coeff_num * current_draw_core_state_values[var_name]
#                         print(f"          -> Added: {coeff_num} * {var_name}")
#                     else:
#                         if lag != 0:
#                             print(f"          -> Skipping lagged term: {var_name}(-{lag})")
#                         elif var_name not in current_draw_core_state_values:
#                             print(f"          -> Variable {var_name} not in core states")
#                             # Check if it's a parameter
#                             if var_name in current_builder_params_draw:
#                                 coeff_val_eval = evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
#                                 param_val_eval = evaluate_coefficient_expression(var_name, current_builder_params_draw)
#                                 if (hasattr(coeff_val_eval, 'ndim') and coeff_val_eval.ndim == 0 and
#                                     hasattr(param_val_eval, 'ndim') and param_val_eval.ndim == 0):
#                                     reconstructed_value_ts += float(coeff_val_eval) * float(param_val_eval)
#                                     print(f"          -> Added parameter: {coeff_val_eval} * {param_val_eval}")
                                    
#                 except Exception as e:
#                     print(f"        -> Term evaluation failed: {e}")

#             # Handle constant term using imported function
#             if expr_def.constant_str and expr_def.constant_str != "0":
#                 try:
#                     const_val_eval = evaluate_coefficient_expression(expr_def.constant_str, current_builder_params_draw)
#                     if hasattr(const_val_eval, 'ndim') and const_val_eval.ndim == 0:
#                         reconstructed_value_ts += float(const_val_eval)
#                     elif isinstance(const_val_eval, (float, int)):
#                         reconstructed_value_ts += float(const_val_eval)
#                     print(f"        -> Added constant: {const_val_eval}")
#                 except Exception as e:
#                     print(f"        -> Constant evaluation failed: {e}")

#             reconstructed_trends = reconstructed_trends.at[:, i].set(reconstructed_value_ts)
#             print(f"      -> Final values finite: {jnp.all(jnp.isfinite(reconstructed_value_ts))}")
#         else:
#             print(f"      -> NOT FOUND in core states or non-core definitions!")

#     # Reconstruct original stationary variables (simpler - usually just direct mapping)
#     for i, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
#         if orig_stat_name in current_draw_core_state_values:
#             reconstructed_stationary = reconstructed_stationary.at[:, i].set(
#                 current_draw_core_state_values[orig_stat_name]
#             )

#     return reconstructed_trends, reconstructed_stationary

def _reconstruct_original_variables(
    core_states_draw: jnp.ndarray,
    gpm_model: ReducedModel,
    ss_builder: StateSpaceBuilder,
    current_builder_params_draw: Dict[str, Any],
    T_data: int,
    state_dim: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    FIXED: Better parameter handling and NaN detection
    """
    
    # Initialize output arrays
    reconstructed_trends = jnp.full(
        (T_data, len(gpm_model.gpm_trend_variables_original)),
        jnp.nan, dtype=_DEFAULT_DTYPE
    )
    reconstructed_stationary = jnp.full(
        (T_data, len(gpm_model.gpm_stationary_variables_original)), 
        jnp.nan, dtype=_DEFAULT_DTYPE
    )

    # Get core state values by name
    core_var_map = ss_builder.core_var_map
    current_draw_core_state_values = {}
    
    for var_name, state_idx in core_var_map.items():
        if state_idx is not None and state_idx < state_dim:
            current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]

    # print(f"    Available core states: {len(current_draw_core_state_values)}")
    # print(f"    Available parameters: {len(current_builder_params_draw)}")
    
    # # Debug problematic parameters specifically
    # problematic_params = ['lambda_pi_US', 'lambda_pi_EA', 'lambda_pi_JP']
    # print(f"    Checking problematic parameters:")
    # for param in problematic_params:
    #     if param in current_builder_params_draw:
    #         val = current_builder_params_draw[param]
    #         is_finite = jnp.isfinite(val) if hasattr(val, 'shape') and val.ndim == 0 else False
    #         print(f"      {param}: {val} (finite: {is_finite})")
    #     else:
    #         print(f"      {param}: NOT FOUND")

    # Reconstruct original trend variables
    for i, orig_trend_name in enumerate(gpm_model.gpm_trend_variables_original):
        if orig_trend_name in current_draw_core_state_values:
            # Core trend - direct mapping
            reconstructed_trends = reconstructed_trends.at[:, i].set(
                current_draw_core_state_values[orig_trend_name]
            )
        elif orig_trend_name in gpm_model.non_core_trend_definitions:
            # Non-core trend - evaluate expression
            expr_def = gpm_model.non_core_trend_definitions[orig_trend_name]
            reconstructed_value_ts = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)

            # Process each term, skipping NaN parameters
            for var_key, coeff_str in expr_def.terms.items():
                try:
                    var_name, lag = parse_variable_key(var_key)
                    
                    if lag != 0:
                        continue
                    
                    # Case 1: var_name is a state variable
                    if var_name in current_draw_core_state_values:
                        coeff_val_eval = evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
                        if hasattr(coeff_val_eval, 'ndim') and coeff_val_eval.ndim == 0:
                            coeff_num = float(coeff_val_eval)
                        else:
                            continue
                            
                        if jnp.isfinite(coeff_num):
                            reconstructed_value_ts += coeff_num * current_draw_core_state_values[var_name]
                    
                    # Case 2: var_name is a parameter
                    elif var_name in current_builder_params_draw:
                        param_val = current_builder_params_draw[var_name]
                        if hasattr(param_val, 'ndim') and param_val.ndim == 0:
                            param_num = float(param_val)
                        else:
                            continue
                        
                        # SKIP if parameter is NaN
                        if not jnp.isfinite(param_num):
                            print(f"        Skipping NaN parameter {var_name}: {param_num}")
                            continue
                        
                        if coeff_str in current_draw_core_state_values:
                            reconstructed_value_ts += param_num * current_draw_core_state_values[coeff_str]
                                    
                except Exception as e:
                    continue

            reconstructed_trends = reconstructed_trends.at[:, i].set(reconstructed_value_ts)

    # Reconstruct stationary variables (unchanged)
    for i, orig_stat_name in enumerate(gpm_model.gpm_stationary_variables_original):
        if orig_stat_name in current_draw_core_state_values:
            reconstructed_stationary = reconstructed_stationary.at[:, i].set(
                current_draw_core_state_values[orig_stat_name]
            )

    return reconstructed_trends, reconstructed_stationary

def complete_gpm_workflow_with_smoother_fixed(
    data: Union[jnp.ndarray, pd.DataFrame],
    gpm_file: str,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 2,
    use_gamma_init: bool = False,
    gamma_scale_factor: float = 1.0,
    num_extract_draws: int = 100,
    generate_plots: bool = True,
    hdi_prob_plot: float = 0.9,
    show_plot_info_boxes: bool = False,
    plot_save_path: Optional[str] = None,
    save_plots: bool = False,
    custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
    variable_names_override: Optional[List[str]] = None,
    data_file_source_for_summary: Optional[str] = None,
    target_accept_prob: float = 0.85,
    # New arguments for P0 overrides
    mcmc_trend_P0_scales: Optional[Union[float, Dict[str, float]]] = None,
    mcmc_stationary_P0_scale: Optional[float] = None,
    smoother_trend_P0_scales: Optional[Union[float, Dict[str, float]]] = None,
    smoother_stationary_P0_scale: Optional[float] = None
) -> Optional[SmootherResults]:
    """
    Complete GPM workflow with simplified smoother that returns SmootherResults.
    """
    print(f"\n=== COMPLETE GPM WORKFLOW WITH SMOOTHER ===")
    print(f"GPM File: {gpm_file}")
    
    # Convert data to JAX array
    if isinstance(data, pd.DataFrame):
        y_data = jnp.asarray(data.values, dtype=_DEFAULT_DTYPE)
        time_index = data.index
        if variable_names_override is None:
            variable_names_override = list(data.columns)
    else:
        y_data = jnp.asarray(data, dtype=_DEFAULT_DTYPE)
        time_index = None

    try:
        # Parse model
        orchestrator = create_integration_orchestrator(gpm_file, strict_validation=False)
        gpm_model = orchestrator.reduced_model
        ss_builder = orchestrator.ss_builder

        print_run_report(
            gpm_file=gpm_file,
            data_file_source_for_summary=data_file_source_for_summary,
            parsed_gpm_model=gpm_model,
            mcmc_results=None,  # Will be filled after MCMC
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            use_gamma_init=use_gamma_init,
            gamma_scale_factor=gamma_scale_factor,
            target_accept_prob=target_accept_prob,
            num_extract_draws=num_extract_draws
        )

        # Run MCMC
        print(f"\n--- Running MCMC ---")
        start_time = time.time()
        
        mcmc_results, _, _ = fit_gpm_numpyro_model(
            gpm_file_path=gpm_file,
            y_data=y_data,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            use_gamma_init_for_P0=use_gamma_init,
            gamma_init_scaling_for_P0=gamma_scale_factor,
            target_accept_prob=target_accept_prob,
            # Pass MCMC P0 overrides with their original names
            mcmc_trend_P0_scales=mcmc_trend_P0_scales,
            mcmc_stationary_P0_scale=mcmc_stationary_P0_scale
        )
        
        mcmc_results.print_summary()
        # print(f"\n=== QUICK MCMC PARAMETER CHECK ===")
        # mcmc_samples = mcmc_results.get_samples(group_by_chain=False)
    
        # for param in ['lambda_pi_US', 'lambda_pi_EA', 'lambda_pi_JP']:
        #     if param in mcmc_samples:
        #         values = mcmc_samples[param]
        #         finite_count = jnp.sum(jnp.isfinite(values))
        #         print(f"{param}: {finite_count}/{values.shape[0]} finite, sample: {values[0]}")
        #     else:
        #         print(f"{param}: NOT FOUND")
        # print(f"=== END QUICK CHECK ===\n")
        
        # print(f"MCMC completed in {time.time() - start_time:.2f}s.")

        # Extract components using smoother
        print(f"\n--- Extracting Components ---")
        smoother_results = extract_reconstructed_components(
            mcmc_output=mcmc_results,
            y_data=y_data,
            gpm_model=gpm_model,
            ss_builder=ss_builder,
            num_smooth_draws=num_extract_draws,
            use_gamma_init_for_smoother=use_gamma_init,
            gamma_init_scaling_for_smoother=gamma_scale_factor,
            hdi_prob=hdi_prob_plot,
            observed_variable_names=variable_names_override,
            time_index=time_index,
            # Pass smoother P0 overrides
            trend_P0_scales_override=smoother_trend_P0_scales,
            stationary_P0_scale_override=smoother_stationary_P0_scale
        )

        # Generate plots if requested
        if generate_plots and PLOTTING_AVAILABLE and smoother_results.n_draws > 0:
            print(f"\n--- Generating Plots ---")
            
            # Create save directory if needed
            if save_plots and plot_save_path:
                os.makedirs(plot_save_path, exist_ok=True)
                save_prefix = os.path.join(plot_save_path, "plot")
            else:
                save_prefix = None



            # Plot trend components
            trend_fig = plot_time_series_with_uncertainty(
                smoother_results.trend_draws,
                variable_names=smoother_results.trend_names,
                hdi_prob=hdi_prob_plot,
                title_prefix="Trend Components",
                time_index=smoother_results.time_index,
                show_info_box=show_plot_info_boxes
            )
            if trend_fig and save_prefix:
                trend_fig.savefig(f"{save_prefix}_trends.png", dpi=150, bbox_inches='tight')
                plt.close(trend_fig)

            # Plot stationary components
            if smoother_results.stationary_draws.shape[2] > 0:
                stat_fig = plot_time_series_with_uncertainty(
                    smoother_results.stationary_draws,
                    variable_names=smoother_results.stationary_names,
                    hdi_prob=hdi_prob_plot,
                    title_prefix="Stationary Components",
                    time_index=smoother_results.time_index,
                    show_info_box=show_plot_info_boxes
                )
                if stat_fig and save_prefix:
                    stat_fig.savefig(f"{save_prefix}_stationary.png", dpi=150, bbox_inches='tight')
                    plt.close(stat_fig)

            # Custom plots
            if custom_plot_specs:
                for i, spec in enumerate(custom_plot_specs):
                    plot_custom_series_comparison(
                        plot_title=spec.get("title", f"Custom Plot {i+1}"),
                        series_specs=spec.get("series_to_plot", []),
                        results=smoother_results,
                        save_path=f"{save_prefix}_custom_{i}" if save_prefix else None
                    )

        print(f"✓ GPM workflow completed successfully")
        return smoother_results

    except Exception as e:
        import traceback
        print(f"✗ GPM workflow failed: {e}")
        traceback.print_exc()
        return None


def debug_smoother_draws(results: SmootherResults) -> None:
    """Debug function to inspect reconstructed component draws."""
    print("\n=== DEBUGGING SMOOTHER DRAWS ===")
    
    for name, arr, names_list in [
        ("Trend", results.trend_draws, results.trend_names), 
        ("Stationary", results.stationary_draws, results.stationary_names)
    ]:
        print(f"\n{name} draws:")
        print(f"  Shape: {arr.shape}")
        print(f"  Names: {names_list}")
        print(f"  Has NaN: {np.any(np.isnan(arr))}")
        print(f"  Has Inf: {np.any(np.isinf(arr))}")
        if arr.size > 0:
            print(f"  Min: {np.nanmin(arr):.6f}")
            print(f"  Max: {np.nanmax(arr):.6f}")
            print(f"  Mean: {np.nanmean(arr):.6f}")

    print("=== END DEBUG ===\n")


def quick_test_fixed_smoother_workflow():
    """Quick test of the simplified workflow."""
    print("=== Running Quick Test with FIXED Smoother ===")
    
    num_obs, num_stat = 2, 2
    gpm_file = "smoother_test_fixed_model.gpm"
    create_default_gpm_file_if_needed(gpm_file, num_obs, num_stat)
    
    # Generate synthetic data
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

    # Run workflow
    results = complete_gpm_workflow_with_smoother_fixed(
        data=sim_y,
        gpm_file=gpm_file,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        num_extract_draws=25,
        generate_plots=True,
        use_gamma_init=True,
        gamma_scale_factor=1.0,
        show_plot_info_boxes=False,
        data_file_source_for_summary="Synthetic Data (quick_test)",
        plot_save_path="quick_test_plots",
        save_plots=True
    )

    if results:
        print(f"\n✓ FIXED smoother test successful!")
        debug_smoother_draws(results)
    else:
        print(f"\n✗ FIXED smoother test failed.")

    # Clean up
    if os.path.exists(gpm_file):
        os.remove(gpm_file)

    return results


if __name__ == "__main__":
    quick_test_fixed_smoother_workflow()