import sys
import os
import jax.numpy as jnp # For _var_innovation_corr_chol
import numpy as np # For isnan checks
import matplotlib.pyplot as plt # For direct plt use in sensitivity summary

# --- Path Setup ---
# This script is in Application/examples. We need to go up two levels to secon_bvar_jax
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Adjusted sys.path in Application/examples/gpm_prior_calibration_example.py. Includes: {project_root}")

# --- JAX Configuration (can be handled by importing a config from the library) ---
# Assuming jax_config.py exists in clean_gpm_bvar_trends and configures JAX
try:
    import clean_gpm_bvar_trends.jax_config # This will execute the JAX setup
except ImportError as e:
    print(f"Warning: Could not import jax_config. JAX might not be configured as expected. {e}")
    # Fallback JAX config if needed, though ideally jax_config.py handles it
    import jax
    if "XLA_FLAGS" not in os.environ: os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


# --- Import calibration utilities and constants ---
from clean_gpm_bvar_trends.calibration_utils import (
    PriorCalibrationConfig,
    validate_calibration_config,
    load_data_for_calibration,
    run_mcmc_workflow, # Renamed from run_single_point_evaluation
    run_fixed_parameter_evaluation, # Renamed from run_single_point_evaluation_fixed_params
    run_parameter_sensitivity_workflow,
    plot_sensitivity_study_results,
    PLOTTING_AVAILABLE_UTILS # Import the flag from utils
)
from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE

# --- Main Example Workflow ---
def main_factor_model_calibration():
    print("\n" + "="*70)
    print("      PRIOR CALIBRATION & SENSITIVITY FOR: gpm_factor_y_pi_rshort.gpm      ")
    print("="*70)

    # --- Define paths specific to this example ---
    # Paths are relative to project_root (secon_bvar_jax)
    gpm_file_to_use = os.path.join(project_root, "Application", "Models", "gpm_factor_y_pi_rshort.gpm")
    data_file_to_use = os.path.join(project_root, "Application", "data_m5.csv")

    # Output directory for this specific example (can be inside Application/examples/ or a general output dir)
    example_output_dir = os.path.join(script_dir, "example_factor_y_pi_rshort_calibration_output")
    os.makedirs(example_output_dir, exist_ok=True)

    if not os.path.exists(gpm_file_to_use):
        print(f"FATAL ERROR: GPM file '{gpm_file_to_use}' not found."); sys.exit(1)
    if not os.path.exists(data_file_to_use):
        print(f"FATAL ERROR: Data file '{data_file_to_use}' not found."); sys.exit(1)

    gpm_varobs_names = ['y_us', 'pi_us', 'r_us', 'y_ea', 'pi_ea', 'r_ea', 'y_jp', 'pi_jp', 'r_jp']
    csv_column_names_to_select = gpm_varobs_names # Assuming CSV columns match GPM varobs

    # Fixed parameters for the factor model
    fixed_params_for_run = {
        'var_phi_US': 2.0, 'var_phi_EA': 2.0, 'var_phi_JP': 2.0,
        'lambda_pi_US': 2.0, 'lambda_pi_EA': 2.0, 'lambda_pi_JP': 0.3,
        'loading_r_EA_on_factor_r_devs': 0.8, 'loading_r_JP_on_factor_r_devs': 0.2,
        'loading_pi_EA_on_factor_pi_devs': 0.8, 'loading_pi_JP_on_factor_pi_devs': 0.2,
        'shk_r_w': 0.1, 'shk_pi_w': 0.001,
        'shk_factor_r_devs': 0.05, 'shk_factor_pi_devs': 0.05,
        'shk_r_US_idio': 0.02, 'shk_pi_US_idio': 0.02, 'shk_r_EA_idio': 0.02,
        'shk_pi_EA_idio': 0.02, 'shk_r_JP_idio': 0.02, 'shk_pi_JP_idio': 0.02,
        'shk_y_US': 0.01, 'shk_y_EA': 0.01, 'shk_y_JP': 0.01,
        'shk_cycle_Y_US': 0.1, 'shk_cycle_PI_US': 0.1, 'shk_cycle_Rshort_US': 0.1,
        'shk_cycle_Y_EA': 0.1, 'shk_cycle_PI_EA': 0.1, 'shk_cycle_Rshort_EA': 0.1,
        'shk_cycle_Y_JP': 0.1, 'shk_cycle_PI_JP': 0.1, 'shk_cycle_Rshort_JP': 0.1,
    }
    
    num_stationary_vars = 9
    fixed_params_for_run['_var_innovation_corr_chol'] = jnp.eye(num_stationary_vars, dtype=_DEFAULT_DTYPE)

    # Example: Overriding initial mean for 'r_w_trend' for fixed parameter evaluation
    initial_state_overrides_example = {
        "pi_w_trend": {"mean": 2.0} # Example: try a different initial mean for world real rate trend
        # Add other overrides here as needed, e.g., "pi_w_trend": {"mean": 0.021}
    }


    custom_plots_for_fixed_eval = [
        # --- Output Gap Analysis (Observed Output vs. Output Trend) ---
        {
            "title": "US: Observed Output vs. Estimated Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k-'},
                {'type': 'trend', 'name': 'y_US_trend', 'label': 'Est. Output Trend (y_US_trend)', 'show_hdi': True, 'color': 'blue'}
            ]
        },
        {
            "title": "EA: Observed Output vs. Estimated Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k-'},
                {'type': 'trend', 'name': 'y_EA_trend', 'label': 'Est. Output Trend (y_EA_trend)', 'show_hdi': True, 'color': 'green'}
            ]
        },
        {
            "title": "JP: Observed Output vs. Estimated Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_jp', 'label': 'Observed y_jp', 'style': 'k-'}, # Corrected to y_jp
                {'type': 'trend', 'name': 'y_JP_trend', 'label': 'Est. Output Trend (y_JP_trend)', 'show_hdi': True, 'color': 'red'}
            ]
        },

        # --- Inflation Analysis (Observed Inflation vs. Full Estimated Inflation Trend) ---
        {
            "title": "US: Observed Inflation vs. Est. Full Inflation Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_us', 'label': 'Observed pi_us', 'style': 'k-'},
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Est. Full Infl. Trend (pi_US_full_trend)', 'show_hdi': True, 'color': 'firebrick'}
            ]
        },
        {
            "title": "EA: Observed Inflation vs. Est. Full Inflation Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_ea', 'label': 'Observed pi_ea', 'style': 'k-'},
                {'type': 'trend', 'name': 'pi_EA_full_trend', 'label': 'Est. Full Infl. Trend (pi_EA_full_trend)', 'show_hdi': True, 'color': 'darkorange'}
            ]
        },
        {
            "title": "JP: Observed Inflation vs. Est. Full Inflation Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_jp', 'label': 'Observed pi_jp', 'style': 'k-'},
                {'type': 'trend', 'name': 'pi_JP_full_trend', 'label': 'Est. Full Infl. Trend (pi_JP_full_trend)', 'show_hdi': True, 'color': 'purple'}
            ]
        },

        # --- Short Rate Analysis (Observed Short Rate vs. Full Estimated Nominal Rate Trend) ---
        {
            "title": "US: Observed Short Rate vs. Est. Nominal Rate Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us', 'style': 'k-'},
                {'type': 'trend', 'name': 'R_US_short_trend', 'label': 'Est. Nominal Rate Trend (R_US_short_trend)', 'show_hdi': True, 'color': 'teal'}
            ]
        },
        {
            "title": "EA: Observed Short Rate vs. Est. Nominal Rate Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_ea', 'label': 'Observed r_ea', 'style': 'k-'},
                {'type': 'trend', 'name': 'R_EA_short_trend', 'label': 'Est. Nominal Rate Trend (R_EA_short_trend)', 'show_hdi': True, 'color': 'darkcyan'}
            ]
        },
        {
            "title": "JP: Observed Short Rate vs. Est. Nominal Rate Trend",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_jp', 'label': 'Observed r_jp', 'style': 'k-'},
                {'type': 'trend', 'name': 'R_JP_short_trend', 'label': 'Est. Nominal Rate Trend (R_JP_short_trend)', 'show_hdi': True, 'color': 'cadetblue'}
            ]
        },

        # --- Unobserved World Trends and Factors ---
        {
            "title": "Estimated World Real Rate Trend (r_w_trend)",
            "series_to_plot": [
                {'type': 'trend', 'name': 'r_w_trend', 'label': 'r_w_trend', 'show_hdi': True, 'color': 'navy'}
            ]
        },
        {
            "title": "Estimated World Inflation Trend (pi_w_trend)",
            "series_to_plot": [
                {'type': 'trend', 'name': 'pi_w_trend', 'label': 'pi_w_trend', 'show_hdi': True, 'color': 'maroon'}
            ]
        },
        {
            "title": "Estimated Common Factor for Real Rate Deviations (factor_r_devs)",
            "series_to_plot": [
                {'type': 'trend', 'name': 'factor_r_devs', 'label': 'factor_r_devs', 'show_hdi': True, 'color': 'darkgreen'}
            ]
        },
        {
            "title": "Estimated Common Factor for Inflation Deviations (factor_pi_devs)",
            "series_to_plot": [
                {'type': 'trend', 'name': 'factor_pi_devs', 'label': 'factor_pi_devs', 'show_hdi': True, 'color': 'darkslateblue'}
            ]
        },

        # --- Decomposition of a Country's Full Inflation Trend ---
        # Example for the US (can be replicated for EA, JP)
        # This assumes lambda_pi_US is a fixed parameter passed in param_values or estimated.
        # If lambda_pi_US is estimated, its value will vary with draws, making 'combined' more complex.
        # For fixed param eval, lambda_pi_US is fixed.
        {
            "title": "US: Full Inflation Trend Decomposition",
            "series_to_plot": [
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Est. Full Infl. Trend (US)', 'show_hdi': True, 'color': 'black', 'style': '-'},
                {'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Infl. Trend (scaled by lambda_pi_US)', 'show_hdi': True, 'color': 'blue', 'style': '--'}, # Note: This doesn't automatically scale by lambda_pi_US. plot_custom_series_comparison would need to support coefficients for combined series or you'd pre-calculate lambda*pi_w_trend if lambda is fixed.
                                                                                                                                                                    # For fixed params, lambda_pi_US is fixed.
                {'type': 'trend', 'name': 'pi_US_dev_trend', 'label': 'Est. US Infl. Deviation Trend', 'show_hdi': True, 'color': 'red', 'style': ':'}
            ]
        },
        # To truly plot lambda_pi_US * pi_w_trend, you would need to:
        # 1. Make sure lambda_pi_US is a parameter in your `fixed_params_for_run`.
        # 2. The `plot_custom_series_comparison` function would need enhancement to handle scaling a trend component by a parameter.
        # OR: If `lambda_pi_US` is part of the GPM `parameters` list, and `pi_w_trend_scaled_US = lambda_pi_US * pi_w_trend`
        #     is defined as another non-core trend in the GPM, then you can plot `pi_w_trend_scaled_US` directly.
        #     This is often the cleaner way for GPM-defined relationships.

        # --- Decomposition of a Country's Full Real Rate Trend ---
        # Example for the US
        {
            "title": "US: Full Real Rate Trend Decomposition",
            "series_to_plot": [
                {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'Est. Full Real Rate Trend (US)', 'show_hdi': True, 'color': 'black', 'style': '-'},
                {'type': 'trend', 'name': 'r_w_trend', 'label': 'World Real Rate Trend', 'show_hdi': True, 'color': 'blue', 'style': '--'},
                {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'Est. US Real Rate Deviation Trend', 'show_hdi': True, 'color': 'red', 'style': ':'}
            ]
        },

        # --- Compare Idiosyncratic Components Across Countries (Example for Inflation Deviations) ---
        {
            "title": "Idiosyncratic Inflation Deviation Trends",
            "series_to_plot": [
                {'type': 'trend', 'name': 'pi_US_idio_trend', 'label': 'US Idio Infl. Dev.', 'show_hdi': True, 'color': 'red'},
                {'type': 'trend', 'name': 'pi_EA_idio_trend', 'label': 'EA Idio Infl. Dev.', 'show_hdi': True, 'color': 'green'},
                {'type': 'trend', 'name': 'pi_JP_idio_trend', 'label': 'JP Idio Infl. Dev.', 'show_hdi': True, 'color': 'blue'}
            ]
        }
    ]

    config = PriorCalibrationConfig(
        data_file_path=data_file_to_use,
        gpm_file_path=gpm_file_to_use,
        observed_variable_names=gpm_varobs_names,
        fixed_parameter_values=fixed_params_for_run,
        initial_state_prior_overrides=initial_state_overrides_example, # Pass overrides
        num_mcmc_warmup=5, num_mcmc_samples=10, num_mcmc_chains=1, # Minimal for testing
        num_smoother_draws=5,
        use_gamma_init=True, gamma_scale_factor=1.0,
        generate_plots=True, # User intent
        plot_save_path=example_output_dir, save_plots=True,
        num_smoother_draws_for_fixed_params=10,
        plot_sensitivity_point_results=True,
        custom_plot_specs=custom_plots_for_fixed_eval,
        sensitivity_plot_custom_specs=custom_plots_for_fixed_eval,
        trend_P0_var_scale_fixed_eval = 0.1,
        var_P0_var_scale_fixed_eval = 1.0
    )

    if not validate_calibration_config(config): return

    data_for_run_df, time_idx_plots = load_data_for_calibration(config, csv_columns_to_select=csv_column_names_to_select)
    if data_for_run_df is None: return

    # --- Run Fixed-Parameter Evaluation ---
    print("\n--- Running Single Fixed-Parameter Evaluation (No MCMC) ---")
    fixed_eval_plot_dir = os.path.join(example_output_dir, "fixed_param_eval_plots_factor_model")
    os.makedirs(fixed_eval_plot_dir, exist_ok=True)
    config_fixed_eval = config # Or deepcopy if modifying common attributes like plot_save_path per run
    config_fixed_eval.plot_save_path = fixed_eval_plot_dir # Set specific output for this run type
    
    fixed_eval_results = run_fixed_parameter_evaluation(config_fixed_eval, data_for_run_df, time_idx_plots)
    if fixed_eval_results: print("\nSingle fixed-parameter evaluation successful.")
    else: print("\nSingle fixed-parameter evaluation failed.")

    # --- Optionally Run MCMC Workflow ---
    RUN_MCMC_FLAG = False
    if RUN_MCMC_FLAG:
        print("\n--- Running Main Workflow (MCMC based on GPM priors) ---")
        mcmc_plot_dir = os.path.join(example_output_dir, "mcmc_workflow_factor_model")
        os.makedirs(mcmc_plot_dir, exist_ok=True)
        config_mcmc = config # Or deepcopy
        config_mcmc.plot_save_path = mcmc_plot_dir
        # For MCMC, typically initial_state_prior_overrides is None as priors come from GPM
        config_mcmc.initial_state_prior_overrides = None 
        main_workflow_results = run_mcmc_workflow(config_mcmc, data_for_run_df)
        if main_workflow_results: print("\nMain MCMC workflow results obtained.")

    # --- Optionally Run Sensitivity Analysis ---
    RUN_SENSITIVITY_FLAG = False
    param_to_vary_sensitivity = 'shk_factor_r_devs' # Example
    if RUN_SENSITIVITY_FLAG and param_to_vary_sensitivity in fixed_params_for_run:
        print(f"\n--- Running Sensitivity Analysis for '{param_to_vary_sensitivity}' ---")
        sensitivity_plot_dir = os.path.join(example_output_dir, f"sensitivity_study_plots_{param_to_vary_sensitivity}")
        os.makedirs(sensitivity_plot_dir, exist_ok=True)
        config_sensitivity = config # Or deepcopy
        config_sensitivity.plot_save_path = sensitivity_plot_dir
        # Sensitivity often uses fixed param eval, so initial_state_prior_overrides might be relevant from base_config
        
        sensitivity_study_output = run_parameter_sensitivity_workflow(
            base_config=config_sensitivity, data_df=data_for_run_df,
            time_index_for_plots=time_idx_plots,
            parameter_name_to_vary=param_to_vary_sensitivity,
            values_to_test=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        )
        if sensitivity_study_output and 'error' not in sensitivity_study_output:
            # ... (print summary as before) ...
            if PLOTTING_AVAILABLE_UTILS: # Check if plotting from utils is available
                overall_sens_plot_filename = os.path.join(config_sensitivity.plot_save_path, f"sensitivity_overall_LL_{param_to_vary_sensitivity}.png")
                plot_sensitivity_study_results(sensitivity_study_output, save_path=overall_sens_plot_filename if config_sensitivity.save_plots else None)
        else: print(f"\nSensitivity study for '{param_to_vary_sensitivity}' encountered issues.")
    elif RUN_SENSITIVITY_FLAG:
        print(f"Warning: param_to_vary_sensitivity '{param_to_vary_sensitivity}' not in fixed_params_for_run. Skipping sensitivity.")


    print("\n--- Example Calibration Workflow (for factor model) Finished ---")
    print(f"Plots and any saved data are in: {os.path.abspath(example_output_dir)}")

if __name__ == "__main__":
    main_factor_model_calibration()
