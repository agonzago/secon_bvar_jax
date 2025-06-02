import sys
import os
import numpy as np
import pandas as pd
import time
import multiprocessing

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(os.path.abspath(parent_dir))

from gpm_bar_smoother_old import complete_gpm_workflow_with_smoother_fixed
from constants import _DEFAULT_DTYPE

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# --- Data Loading ---
dta_path = os.path.join(script_dir, "data_m5.csv")
data_source = dta_path

dta = pd.read_csv(data_source)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')

# Observed variables for this model
observed_vars_model = ['y_us', 'y_ea', 'y_jp', 'r_us', 'r_ea', 'r_jp', 'pi_us', 'pi_ea', 'pi_jp', 'ltr_us', 'ltr_ea', 'ltr_jp']
data_sub = dta[observed_vars_model].copy()


print(f"\n--- Starting GPM Workflow for Long Rates Model ---")

gpm_file_name = 'model_long_rates.gpm'
gpm_file_path = os.path.join(script_dir, gpm_file_name)


custom_plot_specs_long_rates = [
    {
        "title": "US Output: Observed vs. Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_us', 'label': 'Trend y_us', 'show_hdi': True, 'color': 'blue'}
        ]
    },
    {
        "title": "US Short Rate (rs_trend_us) vs. Observed r_us",
        "series_to_plot": [
            {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'rs_trend_us', 'label': 'Trend rs_us (Short)', 'show_hdi': True, 'color': 'green'}
        ]
    },
    {
        "title": "US Long Rate (rs_long_trend_us) vs. Observed ltr_us",
        "series_to_plot": [
            {'type': 'observed', 'name': 'ltr_us', 'label': 'Observed ltr_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'rs_long_trend_us', 'label': 'Trend rs_long_us', 'show_hdi': True, 'color': 'red'}
        ]
    },
    {
        "title": "US Term Premium Trend",
        "series_to_plot": [
            {'type': 'trend', 'name': 'term_premium_trend_us', 'label': 'US Term Premium Trend', 'show_hdi': True, 'color': 'purple'}
        ]
    },
    {
        "title": "EA Output: Observed vs. Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_ea', 'label': 'Trend y_ea', 'show_hdi': True, 'color': 'cyan'}
        ]
    },
    {
        "title": "EA Long Rate vs. Observed ltr_ea",
        "series_to_plot": [
            {'type': 'observed', 'name': 'ltr_ea', 'label': 'Observed ltr_ea', 'style': 'k-'},
            {'type': 'trend', 'name': 'rs_long_trend_ea', 'label': 'Trend rs_long_ea', 'show_hdi': True, 'color': 'magenta'}
        ]
    }
]

results_long_rates = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,
    gpm_file=gpm_file_path,
    num_warmup=50, 
    num_samples=50, 
    num_chains=1,
    target_accept_prob=0.8,
    use_gamma_init=True, 
    gamma_scale_factor=1.0,
    num_extract_draws=20,
    generate_plots=True, 
    hdi_prob_plot=0.9,
    show_plot_info_boxes=False,
    custom_plot_specs=custom_plot_specs_long_rates,
    plot_save_path="results_long_rates",
    save_plots=True
)

if results_long_rates:
    print("\nLong Rates Model Workflow successfully completed!")
else:
    print("\nLong Rates Model Workflow failed.")

# Display all plots if running in an environment that requires it
import matplotlib.pyplot as plt
plt.show()