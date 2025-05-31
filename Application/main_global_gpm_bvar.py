# applications/main.py
import sys
import os
import numpy as np
import pandas as pd
# import matplotlib as plt # plt is imported within functions, not needed at top level usually
import time

import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'clean_gpm_bvar_trends'))
from gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed
# from gpm_prior_calibration_example import run_sensitivity_analysis_workflow # If you use it
from constants import _DEFAULT_DTYPE
# from gpm_numpyro_models import fit_gpm_numpyro_model # Already used within workflow


import jax
import jax.numpy as jnp
# Configure JAX
# Note: XLA_FLAGS is typically set as an environment variable *before* Python starts.
# Setting it here might not always have the intended effect depending on the JAX backend initialization.
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" # Consider setting this outside Python
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpyro
# numpyro.set_platform('cpu') # This is generally done by JAX config or at MCMC run time
# numpyro.set_host_device_count(4)  # NumPyro gets this from JAX usually


# --- Your Data Loading ---
dta_path=os.path.join(os.path.dirname(__file__),"data_m5.csv")
data_source = dta_path

dta = pd.read_csv(data_source)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')  
data_sub = dta[['y_us', 'y_ea', 'y_jp', 'r_us', 'r_ea', 'r_jp', ]]

# --- Plotting original data (optional but good practice) ---
import matplotlib.pyplot as plt # Import for the initial plot
data_sub.plot(figsize=(12, 8))
plt.title('US, EA, JP Growth Rates (Observed Data)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# plt.show() # Depending on your environment, you might not need plt.show() if using notebooks


print(f"\nStarting GPM Workflow...")

gpm_file_name = 'gdps_1.gpm' # Assuming this is the GPM file you provided earlier
gpm_file_path = os.path.join(os.path.dirname(__file__), gpm_file_name)

# --- Define Custom Plot Specifications ---
# These names ('trend_y_world', 'trend_y_us', etc.) must match the
# `gpm_trend_variables_original` list that comes out of your `parsed_gpm_model`.
# And 'y_us', 'y_ea', 'y_jp' must match the `obs_var_names_actual` (column names from data_sub).

custom_plot_specifications = [
    {
        "title": "World Trend vs US Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'blue'},
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed US', 'style': 'k--'}
        ]
    },
    {
        "title": "World Trend vs EA Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'green'},
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed EA', 'style': 'k--'}
        ]
    },
    {
        "title": "World Trend vs JP Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'red'},
            {'type': 'observed', 'name': 'y_jp', 'label': 'Observed JP', 'style': 'k--'}
        ]
    },
    {
        "title": "Country-Specific Trends",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_us', 'label': 'US Trend Component', 'show_hdi': True, 'color': 'cyan'},
            {'type': 'trend', 'name': 'trend_y_ea', 'label': 'EA Trend Component', 'show_hdi': True, 'color': 'magenta'},
            {'type': 'trend', 'name': 'trend_y_jp', 'label': 'JP Trend Component', 'show_hdi': True, 'color': 'brown'}
        ]
    },
    {
        "title": "US Trend vs US Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_us_d', 'label': 'US Trend Component', 'show_hdi': True, 'color': 'purple'},
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed US', 'style': 'k--'}
        ]
    },

    {
        "title": "EA Trend vs EA Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_ea_d', 'label': 'EA Trend Component', 'show_hdi': True, 'color': 'orange'},
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed EA', 'style': 'k--'}
        ]
    },
    {
        "title": "JP Trend vs JP Data",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_jp_d', 'label': 'JP Trend Component', 'show_hdi': True, 'color': 'gray'},
            {'type': 'observed', 'name': 'y_jp', 'label': 'Observed JP', 'style': 'k--'}
        ]
    }
]

# --- Run the Workflow with Custom Plots ---
results = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,                     # Pass the DataFrame
    gpm_file = gpm_file_path,
    num_warmup  = 100,                 # Keep low for testing, increase for real runs
    num_samples = 100,                 # Keep low for testing, increase for real runs
    num_chains  = 2,                   # Or 1 for faster testing if CPU limited
    target_accept_prob = 0.85,
    use_gamma_init = True,
    gamma_scale_factor = 1.0,
    num_extract_draws = 50,            # Number of MCMC draws for smoother & plots
    generate_plots = True,
    hdi_prob_plot  = 0.9,
    show_plot_info_boxes = True,      # Optional: hide default info boxes
    custom_plot_specs=custom_plot_specifications, # Pass your custom specs here
    plot_save_path="my_gpm_results_with_custom_plots", # Optional: specify a folder to save plots
    save_plots=True                    # Optional: save the plots
)

if results:
    print("\nWorkflow successfully completed!")
    print("Returned dictionary keys:", results.keys())
    # You can now access, e.g.:
    # trends = results['reconstructed_trend_draws']
    # mcmc_object = results['mcmc_object']
    # parsed_model = results['parsed_gpm_model']
    # print("Component names from results:", results['component_names'])
    # print("Observed variable names from results:", results['observed_variable_names'])
    # print("Reduced MEs from parsed model:", parsed_model.reduced_measurement_equations if parsed_model else "No parsed model")

    # Ensure plots are displayed if not saving or if in an interactive environment
    if results.get('generate_plots', True) and not results.get('save_plots', False):
         plt.show() # Add this if plots are not showing automatically
else:
    print("\nWorkflow failed.")