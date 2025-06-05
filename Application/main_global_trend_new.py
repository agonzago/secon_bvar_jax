# main_global_trend_new.py (Modified)
import sys
import os
import numpy as np
import pandas as pd
import jax

# Ensure the package root is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the main workflow function from your package
# This function internally calls complete_gpm_workflow_with_smoother_fixed or evaluate_gpm_at_parameters
from clean_gpm_bvar_trends import run_complete_gpm_analysis 

import multiprocessing
# JAX Configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# Consider setting XLA_FLAGS outside the script if specific CPU counts are needed for JAX/NumPyro parallelism
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count() or 1}"

def main():
    # --- Data Loading ---
    # Assuming data_m5.csv is in the same directory as this script or a 'data' subdirectory
    data_file_name = "data_m5.csv"
    data_file_path = os.path.join(SCRIPT_DIR, data_file_name) 
    if not os.path.exists(data_file_path):
        data_file_path = os.path.join(SCRIPT_DIR, "data", data_file_name) # Try a 'data' subdirectory
        if not os.path.exists(data_file_path):
            print(f"FATAL ERROR: Data file {data_file_name} not found in {SCRIPT_DIR} or {os.path.join(SCRIPT_DIR, 'data')}")
            sys.exit(1)
    
    print(f"Loading data from: {data_file_path}")
    dta = pd.read_csv(data_file_path)
    dta['Date'] = pd.to_datetime(dta['Date'])
    dta.set_index('Date', inplace=True)
    dta = dta.asfreq('QE') # Ensure quarterly frequency

    # --- Model Configuration ---
    # These must match the 'varobs' block in your GPM file
    observed_vars_model = [
        'y_us', 'y_ea', 'y_jp',
        'pi_us', 'pi_ea', 'pi_jp',
        'r_us', 'r_ea', 'r_jp'
    ]
    data_sub = dta[observed_vars_model].copy()
    data_sub = data_sub.dropna() 
    print(f"Data shape after selecting observed variables and dropping NaNs: {data_sub.shape}")
    if data_sub.empty:
        print("FATAL ERROR: Data is empty after processing. Check observed_vars_model and data content.")
        sys.exit(1)

    gpm_file_name = 'gpm_factor_y_pi_rshort.gpm'
    # Assuming the GPM file is in ../clean_gpm_bvar_trends/models/ relative to this script
    gpm_file_path = os.path.join(PROJECT_ROOT, 'clean_gpm_bvar_trends', 'models', gpm_file_name)

    if not os.path.exists(gpm_file_path):
        print(f"FATAL ERROR: GPM file {gpm_file_path} not found.")
        sys.exit(1)

    output_base_dir = os.path.join(SCRIPT_DIR, "estimation_results")
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Custom Plot Specifications for the Factor Model ---
    custom_plot_specs_factor_model = [
        {
            "title": "US Output vs. Trend & Real Rate",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_us', 'label': 'Observed Output (y_us)', 'style': '-'},
                {'type': 'trend', 'name': 'y_US_trend', 'label': 'Output Trend (y_US_trend)', 'show_hdi': True, 'color': 'blue'},
                {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'Real Rate Trend (rr_US_full_trend)', 'show_hdi': True, 'color': 'green', 'style': '--'}
            ]
        },
        {
            "title": "US Inflation vs. Trend Components",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_us', 'label': 'Observed Inflation (pi_us)', 'style': '-'},
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Full Inflation Trend (pi_US_full_trend)', 'show_hdi': True, 'color': 'red'},
                {'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend (pi_w_trend)', 'show_hdi': True, 'color': 'magenta', 'style': ':'},
                {'type': 'trend', 'name': 'pi_US_dev_trend', 'label': 'US Deviation Trend (pi_US_dev_trend)', 'show_hdi': True, 'color': 'cyan', 'style': '-.'}
            ]
        },
        {
            "title": "US Short Rate vs. Trend Components",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_us', 'label': 'Observed Short Rate (r_us)', 'style': '-'},
                {'type': 'trend', 'name': 'R_US_short_trend', 'label': 'Nominal Short Rate Trend (R_US_short_trend)', 'show_hdi': True, 'color': 'orange'},
                {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'Real Rate Trend (rr_US_full_trend)', 'show_hdi': True, 'color': 'grey', 'style': ':'},
            ]
        },
        {
            "title": "US Real Rate Full Trend Decomposition",
            "series_to_plot": [
                {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'US Full Real Rate Trend', 'show_hdi': True, 'color': 'blue'},
                {'type': 'trend', 'name': 'r_w_trend', 'label': 'World Real Rate Trend', 'show_hdi': True, 'color': 'green', 'style': ':'},
                {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Deviation Trend', 'show_hdi': True, 'color': 'purple', 'style': '-.'}
            ]
        },
        {
            "title": "US Real Rate Deviation Trend Decomposition",
            "series_to_plot": [
                {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Deviation Trend', 'show_hdi': True, 'color': 'purple'},
                {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (r_devs)', 'show_hdi': True, 'color': 'black', 'style': '--'},
                {'type': 'trend', 'name': 'r_US_idio_trend', 'label': 'US Idiosyncratic Trend', 'show_hdi': True, 'color': 'brown', 'style': ':'}
            ]
        },
        # Add similar plots for EA and JP if desired
        {
            "title": "Comparison of Real Rate Deviation Trends (US, EA, Factor)",
            "series_to_plot": [
                {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Real Rate Dev Trend', 'show_hdi': True, 'color': 'blue'},
                {'type': 'trend', 'name': 'r_EA_dev_trend', 'label': 'EA Real Rate Dev Trend', 'show_hdi': True, 'color': 'green'},
                {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (r_devs)', 'show_hdi': True, 'color': 'black', 'style': '--'}
            ]
        }
    ]

    # --- MCMC Estimation ---
    print(f"\n--- Scenario 1: MCMC Estimation with Custom P0 Scales ---")
    mcmc_output_dir = os.path.join(output_base_dir, "mcmc_estimation_factor_model")
    os.makedirs(mcmc_output_dir, exist_ok=True)

    results_mcmc = run_complete_gpm_analysis(
        data=data_sub.copy(),
        gpm_file=gpm_file_path,
        analysis_type="mcmc",
        num_warmup=50,  # Adjust for real runs
        num_samples=10, # Adjust for real runs
        num_chains=2,    
        target_accept_prob=0.85,
        use_gamma_init=True, # Ensure Gamma P0 for stationary components
        gamma_scale_factor=1.0, 
        num_extract_draws=10, # Number of draws for smoother from MCMC posterior
        generate_plots=True, 
        hdi_prob_plot=0.68,
        show_plot_info_boxes=False,
        custom_plot_specs=custom_plot_specs_factor_model, 
        plot_save_path=mcmc_output_dir, # Save plots in the specific MCMC output directory
        save_plots=True,
        variable_names_override=observed_vars_model, # From data loading
        data_file_source_for_summary=data_file_path,
        # P0 Overrides for MCMC estimation phase
        mcmc_trend_P0_scales={"r_w_trend": 1.0, "pi_w_trend": 1.0}, # Example: specific scales for world trends
        # mcmc_trend_P0_scales=1e6, # Alternative: single float for all trends
        mcmc_stationary_P0_scale=1.0, # Scale for the VAR part P0 (if gamma P0 fails or not used)
        # P0 Overrides for Smoother phase (can be different from MCMC)
        smoother_trend_P0_scales=1.0,
        smoother_stationary_P0_scale=1.0
    )

    if results_mcmc:
        print(f"\nMCMC Workflow for {gpm_file_name} successfully completed!")
        # Access results, e.g., results_mcmc.trend_draws, results_mcmc.stationary_draws
    else:
        print(f"\nMCMC Workflow for {gpm_file_name} failed.")

    # --- Fixed Parameter Estimation ---
    print(f"\n--- Scenario 2: Fixed Parameter Estimation ---")
    fixed_params_output_dir = os.path.join(output_base_dir, "fixed_params_estimation_factor_model")
    os.makedirs(fixed_params_output_dir, exist_ok=True)

    # Define the fixed parameter values. These MUST match parameter names in the GPM.
    # And for shocks, use the direct shock name (e.g., "shk_r_w") for std. dev.
    # or provide _var_coefficients, _var_innovation_corr_chol directly.
    fixed_parameter_values = {
        'var_phi_US': 2.0, 'var_phi_EA': 2.1, 'var_phi_JP': 1.9,
        'lambda_pi_US': 1.0, 'lambda_pi_EA': 0.9, 'lambda_pi_JP': 0.8,
        'loading_r_EA_on_factor_r_devs': 0.8, 'loading_r_JP_on_factor_r_devs': 0.7,
        'loading_pi_EA_on_factor_pi_devs': 0.85, 'loading_pi_JP_on_factor_pi_devs': 0.75,
        # Shock standard deviations (these are what _resolve_parameter_value expects)
        'shk_r_w': 0.005, 'shk_pi_w': 0.003,
        'shk_factor_r_devs': 0.01, 'shk_factor_pi_devs': 0.008,
        'shk_r_US_idio': 0.002, 'shk_pi_US_idio': 0.001,
        'shk_r_EA_idio': 0.002, 'shk_pi_EA_idio': 0.001,
        'shk_r_JP_idio': 0.002, 'shk_pi_JP_idio': 0.001,
        'shk_y_US': 0.007, 'shk_y_EA': 0.006, 'shk_y_JP': 0.005,
        # For VAR cycles, you can provide _var_coefficients and _var_innovation_corr_chol
        # or let them default based on var_prior_setup in GPM and shock std devs below.
        # If providing _var_coefficients:
        # num_stat_vars = 9 (cycle_Y_US, ..., cycle_Rshort_JP)
        # A_example = np.eye(num_stat_vars) * 0.8 
        # A_example = A_example.reshape(1, num_stat_vars, num_stat_vars) # Assuming var_order = 1
        # '_var_coefficients': jax.numpy.array(A_example),
        # '_var_innovation_corr_chol': jax.numpy.eye(num_stat_vars),
        # If relying on individual shock std devs for VAR:
        'shk_cycle_Y_US': 0.005, 'shk_cycle_PI_US': 0.003, 'shk_cycle_Rshort_US': 0.002,
        'shk_cycle_Y_EA': 0.005, 'shk_cycle_PI_EA': 0.003, 'shk_cycle_Rshort_EA': 0.002,
        'shk_cycle_Y_JP': 0.005, 'shk_cycle_PI_JP': 0.003, 'shk_cycle_Rshort_JP': 0.002,
    }
    
    results_fixed = run_complete_gpm_analysis(
        data=data_sub.copy(),
        gpm_file=gpm_file_path,
        analysis_type="fixed_params", # Specify fixed parameter analysis
        param_values=fixed_parameter_values,
        num_sim_draws=100, # Number of draws for smoother with fixed params
        plot_results=True,
        plot_default_observed_vs_trend_components=True, # Plot default OvT plots
        custom_plot_specs=custom_plot_specs_factor_model,
        variable_names=observed_vars_model, # From data loading
        use_gamma_init_for_test=True, # Ensure Gamma P0 for stationary components
        gamma_init_scaling=1.0,
        hdi_prob=0.9, # HDI for plots from fixed param simulation draws
        trend_P0_var_scale=1e5, # P0 scale for trend components in fixed param eval
        var_P0_var_scale=10.0,  # P0 scale for VAR components in fixed param eval
        save_plots_path_prefix=os.path.join(fixed_params_output_dir, "fixed_eval_plot"), # Path prefix for saving plots
        show_plot_info_boxes=False,
        # initial_state_prior_overrides can be added here if needed
    )

    if results_fixed:
        print(f"\nFixed Parameter Workflow for {gpm_file_name} successfully completed!")
        print(f"  Log-likelihood: {results_fixed.log_likelihood if results_fixed.log_likelihood is not None else 'N/A'}")
    else:
        print(f"\nFixed Parameter Workflow for {gpm_file_name} failed.")

if __name__ == "__main__":
    main()