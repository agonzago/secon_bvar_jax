import sys
import os
import numpy as np
import pandas as pd
import time
import multiprocessing

# Ensure the path to your library is correct
script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(script_dir, '..')
# sys.path.append(os.path.abspath(parent_dir))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'clean_gpm_bvar_trends'))
from gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed
from constants import _DEFAULT_DTYPE

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# --- Data Loading ---
dta_path = os.path.join(script_dir, "data_m5.csv") # Assuming data_m5.csv is in the same directory
data_source = dta_path

dta = pd.read_csv(data_source)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')

# Observed variables for this model

observed_vars_model = ['y_us', 'y_ea', 'y_jp',
    'r_us', 'r_ea', 'r_jp',
    'pi_us', 'pi_ea', 'pi_jp']
data_sub = dta[observed_vars_model].copy() # Ensure we work with a copy for this model

print(f"\n--- Starting GPM Workflow for Global Trend Model ---")

gpm_file_name = 'bvar_global_trend.gpm'
gpm_file_path = os.path.join(script_dir, gpm_file_name) # Assuming GPM file is in the same directory

custom_plot_specs_global_trend = [
    {
        "title": "US Output: Observed vs. Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_us', 'label': 'Trend y_us', 'show_hdi': True, 'color': 'blue'}
        ]
    },
    {
        "title": "EA Output: Observed vs. Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_ea', 'label': 'Trend y_ea (Abs)', 'show_hdi': True, 'color': 'green'},
            {'type': 'trend', 'name': 'rel_trend_y_ea', 'label': 'Trend y_ea (Rel)', 'show_hdi': True, 'color': 'limegreen', 'style':'--'}
        ]
    },
    {
        "title": "US Real Rate Trend vs. Observed Nominal Rate",
        "series_to_plot": [
            {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us (Nominal)', 'style': 'k-'},
            {'type': 'trend', 'name': 'rr_trend_us', 'label': 'Trend rr_us (Real)', 'show_hdi': True, 'color': 'red'}
        ]
    },
    {
        "title": "US Inflation: Observed vs. Trend Component",
        "series_to_plot": [
            {'type': 'observed', 'name': 'pi_us', 'label': 'Observed pi_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_pi_us_comp', 'label': 'Trend pi_us (Core Comp)', 'show_hdi': True, 'color': 'purple'}
        ]
    }
]

results_global_trend = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,
    gpm_file=gpm_file_path,
    num_warmup=50, num_samples=50, num_chains=1, # Low for quick testing
    target_accept_prob=0.8,
    use_gamma_init=True, gamma_scale_factor=1.0,
    num_extract_draws=20,
    generate_plots=True, hdi_prob_plot=0.9,
    show_plot_info_boxes=False,
    custom_plot_specs=custom_plot_specs_global_trend,
    plot_save_path="results_global_trend",
    save_plots=True
)

if results_global_trend:
    print("\nGlobal Trend Model Workflow successfully completed!")
else:
    print("\nGlobal Trend Model Workflow failed.")