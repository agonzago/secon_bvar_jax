import sys
import os
import numpy as np
import pandas as pd
import time
import multiprocessing

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(os.path.abspath(parent_dir))

from clean_gpm_bvar_trends.gpm_bar_smoother_old import complete_gpm_workflow_with_smoother_fixed
from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE

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

print(f"\n--- Starting GPM Workflow for Common Factor Model ---")

gpm_file_name = 'common_factor_bvar_trends.gpm' # Changed from model_common_factor.gpm
gpm_file_path = os.path.join(script_dir, '..', 'clean_gpm_bvar_trends', 'models', gpm_file_name)



custom_plot_specs_common_factor = [
    {
        "title": "Common Output Trend vs. Regional Output",
        "series_to_plot": [
            {'type': 'trend', 'name': 'common_trend_y', 'label': 'Common Output Trend', 'show_hdi': True, 'color': 'black'},
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'r--', 'alpha': 0.7},
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'g--', 'alpha': 0.7},
            {'type': 'observed', 'name': 'y_jp', 'label': 'Observed y_jp', 'style': 'b--', 'alpha': 0.7}
        ]
    },
    {
        "title": "Common Real Rate Trend vs. Regional Rates",
        "series_to_plot": [
            {'type': 'trend', 'name': 'common_trend_r', 'label': 'Common Real Rate Trend', 'show_hdi': True, 'color': 'black'},
            {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us', 'style': 'r--', 'alpha': 0.7},
            {'type': 'observed', 'name': 'r_ea', 'label': 'Observed r_ea', 'style': 'g--', 'alpha': 0.7},
            {'type': 'observed', 'name': 'r_jp', 'label': 'Observed r_jp', 'style': 'b--', 'alpha': 0.7}
        ]
    },
    {
        "title": "US Output: Observed vs. Total Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_us', 'label': 'Total Trend y_us', 'show_hdi': True, 'color': 'blue'}
        ]
    },
    {
        "title": "US Inflation: Observed vs. Trend Component",
        "series_to_plot": [
            {'type': 'observed', 'name': 'pi_us', 'label': 'Observed pi_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_pi_us_comp', 'label': 'Trend pi_us (Core Comp)', 'show_hdi': True, 'color': 'purple'}
        ]
    },
    {
        "title": "US Long Rate: Observed vs. Trend",
        "series_to_plot": [
            {'type': 'observed', 'name': 'ltr_us', 'label': 'Observed ltr_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'rs_long_trend_us', 'label': 'Trend rs_long_us', 'show_hdi': True, 'color': 'orange'}
        ]
    }
]

results_common_factor = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,
    gpm_file=gpm_file_path,
    num_warmup=50, num_samples=50, num_chains=1,
    target_accept_prob=0.8,
    use_gamma_init=True, gamma_scale_factor=1.0,
    num_extract_draws=20,
    generate_plots=True, hdi_prob_plot=0.9,
    show_plot_info_boxes=False,
    custom_plot_specs=custom_plot_specs_common_factor,
    plot_save_path="results_common_factor",
    save_plots=True
)

if results_common_factor:
    print("\nCommon Factor Model Workflow successfully completed!")
else:
    print("\nCommon Factor Model Workflow failed.")