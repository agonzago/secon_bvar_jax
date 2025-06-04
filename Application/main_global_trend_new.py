# main_global_trend.py
import sys
import os
import numpy as np
import pandas as pd
# import time # Not strictly needed here anymore unless doing custom timing
# import multiprocessing # Removed as it was not directly used in this script's logic

import multiprocessing # Not strictly needed here


import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # This should be secon_bvar_jax
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clean_gpm_bvar_trends.gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed

import jax
# import jax.numpy as jnp # Not directly used here
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if "XLA_FLAGS" not in os.environ: # Set only if not already set by another module
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# --- Data Loading ---
dta_path = os.path.join(SCRIPT_DIR, "data_m5.csv") 
data_source_file_name = dta_path 

dta = pd.read_csv(data_source_file_name)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE') # Ensure quarterly frequency, might introduce NaNs if data is not perfectly quarterly

# --- Model Configuration ---
# UPDATE observed_vars_model based on the 'varobs' in model_with_trends.gpm
observed_vars_model = [
    'y_us', 'y_ea', 'y_jp',
    'r_us', 'r_ea', 'r_jp',
    'pi_us', 'pi_ea', 'pi_jp'
]
data_sub = dta[observed_vars_model].copy() 
# Handle potential NaNs from asfreq or in original data before passing to model
data_sub = data_sub.dropna() # Or use an imputation strategy
print(f"Data shape after dropping NaNs: {data_sub.shape}")


print(f"\n--- Starting GPM Workflow for Model with Trends ---")

gpm_file_name =  'gpm_factor_y_pi_rshort.gpm' # USE THE NEW GPM FILE
gpm_file_path = os.path.join(SCRIPT_DIR, '..', 'clean_gpm_bvar_trends', 'models', gpm_file_name)

if not os.path.exists(gpm_file_path):
    print(f"FATAL ERROR: {gpm_file_path} not found. Please ensure the GPM file exists at this location.")
    sys.exit(1) # Exit if the main GPM file is missing

# UPDATE custom_plot_specs for the new model's variables
custom_plot_specs_factor_model = [
    {
        "title": "US Real Rate Trend Decomposition",
        "series_to_plot": [
            {'type': 'observed', 'name': 'OBS_Rshort_US', 'label': 'Observed Rshort_US', 'style': 'k--'}, # Proxy
            {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'US Full Real Rate Trend (rr_US_full_trend)', 'show_hdi': True, 'color': 'blue'},
            {'type': 'trend', 'name': 'r_w_trend', 'label': 'World Real Rate Trend (r_w_trend)', 'show_hdi': True, 'color': 'green', 'style': ':'},
            {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Full Deviation Trend (r_US_dev_trend)', 'show_hdi': True, 'color': 'orange', 'style': '-.'}
        ]
    },
    {
        "title": "US Real Rate Deviation Further Decomposed",
        "series_to_plot": [
            {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Full Deviation (r_US_dev_trend)', 'show_hdi': True, 'color': 'orange'},
            {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (factor_r_devs)', 'show_hdi': True, 'color': 'purple', 'style': '--'},
            {'type': 'trend', 'name': 'r_US_idio_trend', 'label': 'US Idiosyncratic (r_US_idio_trend)', 'show_hdi': True, 'color': 'brown', 'style': ':'}
        ]
    },
    {
        "title": "US Inflation Trend Decomposition",
        "series_to_plot": [
            {'type': 'observed', 'name': 'OBS_PI_US', 'label': 'Observed PI_US', 'style': 'k--'},
            {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'US Full Inflation Trend', 'show_hdi': True, 'color': 'red'},
            # To plot lambda_pi_US * pi_w_trend, 'combined' would be needed if lambda_pi_US is estimated.
            # If lambda_pi_US is fixed at 1 in fixed_params, you can directly plot pi_w_trend as proxy.
            {'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend (pi_w_trend)', 'show_hdi': True, 'color': 'magenta', 'style': ':'},
            {'type': 'trend', 'name': 'pi_US_dev_trend', 'label': 'US Full Deviation Trend (pi_US_dev_trend)', 'show_hdi': True, 'color': 'cyan', 'style': '-.'}
        ]
    },
    {
        "title": "US Output Trend vs. Real Rate Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'OBS_Y_US', 'label': 'Observed Y_US', 'style': 'k-'},
            {'type': 'trend', 'name': 'y_US_trend', 'label': 'US Output Trend', 'show_hdi': True, 'color': 'teal'},
            # For secondary axis, plot_custom_series_comparison would need modification
            # For now, just plotting on same axis, scale might differ.
            {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'US Full Real Rate Trend', 'show_hdi': True, 'color': 'grey', 'style':'--'}
        ]
    },
    {
        "title": "Comparison of Real Rate Deviation Trends",
        "series_to_plot": [
            {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Real Rate Dev Trend', 'show_hdi': True, 'color': 'blue'},
            {'type': 'trend', 'name': 'r_EA_dev_trend', 'label': 'EA Real Rate Dev Trend', 'show_hdi': True, 'color': 'green'},
            {'type': 'trend', 'name': 'r_JP_dev_trend', 'label': 'JP Real Rate Dev Trend', 'show_hdi': True, 'color': 'red'},
            {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (r_devs)', 'show_hdi': True, 'color': 'black', 'style': '--'}
        ]
    }
]


results_model = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,
    gpm_file=gpm_file_path,
    num_warmup=20,  # Keep low for quick testing, increase for real runs
    num_samples=5, # Keep low for quick testing
    num_chains=2,    # Keep low for quick testing
    target_accept_prob=0.8,
    use_gamma_init=True, 
    gamma_scale_factor=1.0, 
    num_extract_draws=5, 
    generate_plots=True, 
    hdi_prob_plot=0.9,
    show_plot_info_boxes=False, # Keep this false for cleaner output
    custom_plot_specs=custom_plot_specs_factor_model, 
    plot_save_path="results_model_with_trends", # New save path
    save_plots=True,
    variable_names_override=observed_vars_model,
    data_file_source_for_summary=data_source_file_name 
)

if results_model:
    print(f"\nWorkflow for {gpm_file_name} successfully completed!")
    print(f"Final check - Used gamma init from results: {results_model.get('used_gamma_init')}")
else:
    print(f"\nWorkflow for {gpm_file_name} failed.")