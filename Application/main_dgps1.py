# main_global_trend.py
import sys
import os
import numpy as np
import pandas as pd
# import time # Not strictly needed here anymore unless doing custom timing
import multiprocessing # Not strictly needed here


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..', 'clean_gpm_bvar_trends')
sys.path.append(os.path.abspath(parent_dir))
try:
    from .gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed
    # from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE # Not directly used here
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(script_dir, os.pardir)))
    from .clean_gpm_bvar_trends.gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed
    # from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE


import jax
# import jax.numpy as jnp # Not directly used here
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# --- Data Loading ---
dta_path = os.path.join(script_dir, "data_m5.csv") 
data_source_file_name = dta_path # Store for summary

dta = pd.read_csv(data_source_file_name)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')


observed_vars_model = ['y_us', 'y_ea', 'y_jp'] 
data_sub = dta[observed_vars_model].copy() 

print(f"\n--- Starting GPM Workflow for GDPs Model (using gdps_1.gpm) ---")

gpm_file_name =  'gdps_1.gpm' 
gpm_file_path = os.path.join(script_dir, gpm_file_name) 

if not os.path.exists(gpm_file_path):
    print(f"Warning: {gpm_file_path} not found. Consider placing the correct GPM file or modifying this script.")

custom_plot_specs_for_gdps1 = [
    {
        "title": "US Output (y_us) vs. Trend (trend_y_us_d)",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_us_d', 'label': 'Trend y_us_d (World Trend)', 'show_hdi': True, 'color': 'blue'}
        ]
    },
    {
        "title": "EA Output (y_ea) vs. Trends",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_ea_d', 'label': 'Trend y_ea_d (Absolute)', 'show_hdi': True, 'color': 'green'},
            {'type': 'trend', 'name': 'trend_y_ea', 'label': 'Trend y_ea (Relative Comp)', 'show_hdi': True, 'color': 'limegreen', 'style':'--'}
        ]
    },
    {
        "title": "JP Output (y_jp) vs. Trends",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_jp', 'label': 'Observed y_jp', 'style': 'k-'},
            {'type': 'trend', 'name': 'trend_y_jp_d', 'label': 'Trend y_jp_d (Absolute)', 'show_hdi': True, 'color': 'red'},
            {'type': 'trend', 'name': 'trend_y_jp', 'label': 'Trend y_jp (Relative Comp)', 'show_hdi': True, 'color': 'salmon', 'style':'--'}
        ]
    },
    {
        "title": "World Trend vs Core Country-Specific Trend Components",
        "series_to_plot": [
            {'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color':'black'},
            {'type': 'trend', 'name': 'trend_y_ea', 'label': 'EA Specific Trend Comp', 'show_hdi': True, 'color':'cyan'},
            {'type': 'trend', 'name': 'trend_y_jp', 'label': 'JP Specific Trend Comp', 'show_hdi': True, 'color':'magenta'}
        ]
    },
     { 
        "title": "US Output vs Fitted (trend_y_us_d + cycle_y_us)",
        "series_to_plot": [
            {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
            {'type': 'combined',
             'components': [{'type':'trend', 'name':'trend_y_us_d'}, {'type':'stationary', 'name':'cycle_y_us'}],
             'label': 'Fitted y_us', 'show_hdi': True, 'color':'purple'}
        ]
    }
]


results_gdp_model = complete_gpm_workflow_with_smoother_fixed(
    data=data_sub,
    gpm_file=gpm_file_path,
    num_warmup=50, num_samples=100, 
    num_chains=1, 
    target_accept_prob=0.8,
    use_gamma_init=True, gamma_scale_factor=1.0, 
    num_extract_draws=50, 
    generate_plots=True, hdi_prob_plot=0.9,
    show_plot_info_boxes=False,
    custom_plot_specs=custom_plot_specs_for_gdps1, 
    plot_save_path="results_gdps1_model", 
    save_plots=True,
    variable_names_override=observed_vars_model,
    data_file_source_for_summary=data_source_file_name # Pass the data file path
)

if results_gdp_model:
    print("\nGDPs Model Workflow (gdps_1.gpm) successfully completed!")
    # The comprehensive summary is now printed within the workflow.
    # Additional specific checks can still be done here.
    print(f"Final check - Used gamma init from results: {results_gdp_model.get('used_gamma_init')}")
else:
    print("\nGDPs Model Workflow (gdps_1.gpm) failed.")