# clean_gpm_bvar_trends/Application/gpm_ai_calibrator.py

import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
import argparse # Import argparse
import json # Import json for initial state overrides
import arviz as az 
# --- Path Setup ---
# (Keep your existing sys.path modification block here)
SCRIPT_DIR_CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))
# Assuming project root is two directories up from the script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_CURRENT_FILE, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"{os.path.basename(__file__)}: Adjusted sys.path. Includes: {PROJECT_ROOT}")


# --- JAX Configuration ---
# The configuration should ideally be done by importing jax_config
# which is imported by other modules like gpm_numpyro_models and Kalman_filter_jax.
# Explicitly importing and calling configure_jax here ensures it runs early.
try:
    from clean_gpm_bvar_trends.jax_config import configure_jax
    configure_jax()
except ImportError:
     print("Warning: clean_gpm_bvar_trends.jax_config not found or configure_jax not callable. Skipping explicit JAX configuration.")

import jax.numpy as jnp # Import JAX after potential config


# --- Core Library Imports ---
# Import the main workflow functions that now return SmootherResults
try:
    from clean_gpm_bvar_trends.gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed # For MCMC mode
    from clean_gpm_bvar_trends.gpm_prior_evaluator import evaluate_gpm_at_parameters, _resolve_parameter_value # For fixed mode
    from clean_gpm_bvar_trends.integration_orchestrator import create_integration_orchestrator # Needed for fixed param extraction
    from clean_gpm_bvar_trends.gpm_model_parser import ReducedModel # Needed for fixed param extraction
    from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE # Needed for fixed param extraction
    # You might still import plotting functions here if you need to call them *after* the workflow
    # For this script's primary purpose (run workflow and save outputs), importing workflow is enough
    # from clean_gpm_bvar_trends.reporting_plots import plot_observed_vs_fitted, plot_smoother_results, plot_custom_series_comparison
    # import arviz as az # Needed if you perform ArviZ analysis here

    MODULES_LOADED = True
    # PLOTTING_AVAILABLE_MAIN flag is less relevant here, plotting is controlled by workflow flags
    print("✓ Core workflow modules loaded successfully")
except ImportError as e:
    print(f"FATAL ERROR ({os.path.basename(__file__)}): Could not import core library modules: {e}")
    MODULES_LOADED = False
    # sys.exit(1) # Exit if core modules cannot be loaded


def extract_fixed_parameters_from_gpm(parsed_gpm_model: ReducedModel) -> Dict[str, Any]:
    """
    Extracts a single set of fixed parameter values from the GPM's prior definitions
    (typically means for normal priors, modes for inverse gamma priors).
    This requires _resolve_parameter_value from gpm_prior_evaluator.
    """
    if not MODULES_LOADED:
        print("ERROR: Cannot extract fixed parameters, core modules not loaded.")
        return {}

    fixed_params = {}

    # Structural parameters
    # Iterate through parameters listed in the GPM (these should have priors in estimated_params)
    for param_name in parsed_gpm_model.parameters:
         # _resolve_parameter_value will look in param_values_input (empty here) and then estimated_params
        try:
            fixed_params[param_name] = _resolve_parameter_value(
                param_name, {}, parsed_gpm_model.estimated_params, is_shock_std_dev=False) # Explicitly not shock
        except ValueError as e:
             print(f"Warning: Could not resolve fixed value for structural parameter '{param_name}': {e}. Skipping.")
             # Depending on requirements, you might want to raise an error here.


    # Shock standard deviations
    all_shocks = parsed_gpm_model.trend_shocks + parsed_gpm_model.stationary_shocks
    for shock_name in all_shocks:
        # _resolve_parameter_value will look for 'shock_name' or 'sigma_shock_name'
        try:
            fixed_params[shock_name] = _resolve_parameter_value(
                shock_name, {}, parsed_gpm_model.estimated_params, is_shock_std_dev=True) # Explicitly is shock
        except ValueError as e:
             print(f"Warning: Could not resolve fixed value for shock '{shock_name}': {e}. Skipping.")
             # Again, might want to raise an error here.


    # VAR innovation correlation Cholesky (default to identity if not otherwise specified)
    # This parameter is not typically defined with a prior for a single point estimate,
    # but is sampled in MCMC as Omega_u_chol. For fixed evaluation, we might need a default
    # if not provided explicitly in --fixed_params.
    # The _build_var_parameters function in gpm_prior_evaluator handles this default.
    # We don't need to add _var_innovation_corr_chol here unless it's part of
    # the parameters you *want* to fix via this function's output.
    # If you need to fix the correlation structure, you would add _var_innovation_corr_chol
    # to the --fixed_params argument when running the script.

    print(f"  Extracted {len(fixed_params)} fixed parameters from GPM priors.")
    return fixed_params


def run_model_and_generate_outputs(
    gpm_file_arg: str,
    output_dir_arg: str,
    data_path_arg: str,
    eval_mode: str = "mcmc",
    initial_state_overrides_fixed: Optional[Dict[str, Dict[str, float]]] = None,
    trend_p0_scale_fixed: float = 1e4,
    var_p0_scale_fixed: float = 1.0,
    mcmc_warmup: int = 50, # Added MCMC config
    mcmc_samples: int = 50, # Added MCMC config
    mcmc_chains: int = 1, # Added MCMC config
    smoother_draws: int = 20, # Added smoother config
    generate_plots_flag: bool = True, # Added plotting flags
    plot_default_ovf_flag: bool = True,
    plot_default_ovt_flag: bool = True,
    show_plot_info_boxes_flag: bool = False,
    custom_plot_specs_list: Optional[List[Dict[str, Any]]] = None, # Added custom plot specs
    save_plots_flag: bool = True # Added save plots flag
):
    print(f"\n=== DEBUG: Starting run_model_and_generate_outputs ===")
    print(f"Mode: {eval_mode}")
    print(f"GPM file: {gpm_file_arg}")
    print(f"Data file: {data_path_arg}")
    print(f"Output dir: {output_dir_arg}")

    if not MODULES_LOADED:
        print("ERROR: Core modules not loaded, cannot proceed.")
        # Return a simple status indication
        return {"status": "failed", "reason": "modules not loaded"}

    # Ensure output directory exists
    os.makedirs(output_dir_arg, exist_ok=True)
    plot_base_save_path = os.path.join(output_dir_arg, "plots") # Base dir for all plots
    os.makedirs(plot_base_save_path, exist_ok=True)

    metrics_output_file = os.path.join(output_dir_arg, "metrics.json")
    run_metrics = {"status": "running", "start_time": time.time()}


    try:
        # --- Data Loading ---
        print(f"Loading data from: {data_path_arg}")
        if not os.path.exists(data_path_arg):
            raise FileNotFoundError(f"Data file {data_path_arg} not found.")

        data_df = pd.read_csv(data_path_arg)
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.set_index('Date', inplace=True)
        data_df = data_df.asfreq('QE') # Ensure consistent frequency

        # --- Parse GPM to get variable names and structure ---
        # This needs to be done early to get expected observed variable names
        # and model structure regardless of eval_mode.
        try:
            orchestrator_for_info = create_integration_orchestrator(gpm_file_arg, strict_validation=False)
            parsed_gpm_model_info = orchestrator_for_info.reduced_model
            expected_observed_names = list(parsed_gpm_model_info.gpm_observed_variables_original)
            print(f"✓ GPM parsed for info. Expected observed variables: {expected_observed_names}")

            # Use the expected observed names to select columns from the data
            missing_cols = [col for col in expected_observed_names if col not in data_df.columns]
            if missing_cols:
                 raise ValueError(f"Data file missing required columns from GPM varobs: {missing_cols}. Available: {data_df.columns.tolist()}")

            data_sub = data_df[expected_observed_names].copy()
            print(f"✓ Data filtered for GPM observed variables. Shape: {data_sub.shape}")

        except Exception as e:
            raise RuntimeError(f"Error parsing GPM or filtering data: {e}") from e


        # --- Run Workflow Based on Mode ---
        smoother_results: Optional[object] = None # Will store the SmootherResults object

        if eval_mode == "fixed":
            print(f"\n  Running in FIXED parameter evaluation mode.")

            # Extract fixed parameters from GPM priors
            fixed_params_for_eval = extract_fixed_parameters_from_gpm(parsed_gpm_model_info)

            # Allow command-line specified fixed parameters to override those from GPM priors
            # This requires adding an argument for fixed parameters to the script's parser.
            # Assuming for now that if fixed_params_for_eval needs overrides, they are
            # handled by the initial_state_prior_overrides_fixed or will be added separately.
            # If you add a --fixed_params_json argument to the script, you would merge that dict here.

            # Call the fixed parameter evaluation workflow
            smoother_results = evaluate_gpm_at_parameters(
                gpm_file_path=gpm_file_arg,
                y=data_sub, # Pass the filtered DataFrame
                param_values=fixed_params_for_eval, # Use the extracted fixed parameters
                initial_state_prior_overrides=initial_state_overrides_fixed, # Pass overrides
                num_sim_draws=smoother_draws if generate_plots_flag else 0, # Simulate draws if plotting is enabled
                plot_results=generate_plots_flag, # Pass main plotting flag
                plot_default_observed_vs_fitted=plot_default_ovf_flag, # Pass default plot flags
                plot_default_observed_vs_trend_components=plot_default_ovt_flag, # Pass default plot flags
                custom_plot_specs=custom_plot_specs_list, # Pass custom plot specs
                variable_names=expected_observed_names, # Explicitly pass observed names
                use_gamma_init_for_test=True, # Assuming gamma init is default for fixed eval test
                gamma_init_scaling=1.0, # Default gamma scale for fixed eval
                trend_P0_var_scale=trend_p0_scale_fixed, # Pass P0 scales
                var_P0_var_scale=var_p0_scale_fixed,
                save_plots_path_prefix=os.path.join(plot_base_save_path, "fixed_param_evaluation", "plot") if save_plots_flag else None, # Construct save prefix
                show_plot_info_boxes=show_plot_info_boxes_flag # Pass info box flag
            )

            run_metrics["eval_mode"] = "fixed"
            if smoother_results and hasattr(smoother_results, 'log_likelihood') and smoother_results.log_likelihood is not None:
                 run_metrics["log_likelihood"] = smoother_results.log_likelihood
                 print(f"  ✓ Fixed parameter evaluation completed. LogLik: {run_metrics['log_likelihood']:.3f}")
                 run_metrics["status"] = "success"
            elif smoother_results: # Results object returned but no loglik (e.g., num_sim_draws=0 or LL calculation failed)
                 run_metrics["status"] = "completed_no_loglik"
                 print("  ✓ Fixed parameter evaluation completed (no loglik available).")
            else: # evaluate_gpm_at_parameters returned None
                 run_metrics["status"] = "failed"
                 run_metrics["error"] = "Fixed parameter evaluation failed during core process."
                 print(f"  ❌ {run_metrics['error']}")


        elif eval_mode == "mcmc":
            print(f"\n  Running in MCMC evaluation mode.")
            print(f"  MCMC settings: {mcmc_warmup} warmup, {mcmc_samples} samples, {mcmc_chains} chains")
            print(f"  Smoother draws: {smoother_draws}")

            # Call the MCMC and smoother workflow
            smoother_results = complete_gpm_workflow_with_smoother_fixed(
                data=data_sub, # Pass the filtered DataFrame
                gpm_file=gpm_file_arg,
                num_warmup=mcmc_warmup,
                num_samples=mcmc_samples,
                num_chains=mcmc_chains,
                target_accept_prob=0.80, # Default target accept prob
                use_gamma_init=True, # Assuming gamma init is default for MCMC
                gamma_scale_factor=1.0, # Default gamma scale for MCMC
                num_extract_draws=smoother_draws,
                generate_plots=generate_plots_flag, # Pass main plotting flag
                plot_default_observed_vs_fitted=plot_default_ovf_flag, # Pass default plot flags
                plot_default_observed_vs_trend_components=plot_default_ovt_flag, # Pass default plot flags
                hdi_prob_plot=0.9, # Default HDI prob for plotting
                save_plots=save_plots_flag, # Pass save plots flag
                plot_save_path=os.path.join(plot_base_save_path, "mcmc_smoother_results", "plot") if save_plots_flag else None, # Construct save prefix
                variable_names_override=expected_observed_names, # Explicitly pass observed names
                show_plot_info_boxes=show_plot_info_boxes_flag, # Pass info box flag
                custom_plot_specs=custom_plot_specs_list, # Pass custom plot specs
                data_file_source_for_summary=data_path_arg # Pass data file path for summary
            )

            run_metrics["eval_mode"] = "mcmc"
            if smoother_results and hasattr(smoother_results, 'mcmc_object') and smoother_results.mcmc_object is not None:
                print("  ✓ MCMC workflow completed successfully.")
                # MCMC summary is printed internally by the workflow.
                # Extract key metrics from the SmootherResults if available
                # (Metrics might need to be added to SmootherResults or extracted from mcmc_object here)
                try:
                    mcmc_summary_az = az.summary(
                        smoother_results.mcmc_object,
                        hdi_prob=0.9,
                        kind='stats',
                        # Attempt to get parameter names from the parsed model within results
                        var_names=[p for p in smoother_results.gpm_model.parameters if not p.startswith("_")] +
                                  [f"sigma_{s}" for s in smoother_results.gpm_model.trend_shocks + smoother_results.gpm_model.stationary_shocks]
                    )
                    run_metrics["mcmc_summary_dict"] = mcmc_summary_az.to_dict()
                    run_metrics["min_ess_bulk"] = float(mcmc_summary_az['ess_bulk'].min()) if 'ess_bulk' in mcmc_summary_az and not mcmc_summary_az['ess_bulk'].empty else None
                    run_metrics["max_rhat"] = float(mcmc_summary_az['r_hat'].max()) if 'r_hat' in mcmc_summary_az and not mcmc_summary_az['r_hat'].empty else None

                    # Loglik estimate from MCMC
                    mcmc_samples_full = smoother_results.mcmc_object.get_samples()
                    if 'loglik' in mcmc_samples_full:
                         run_metrics["log_likelihood_estimate"] = np.mean(mcmc_samples_full['loglik']).item()

                except Exception as e_summary:
                     print(f"  ⚠️  Error generating MCMC summary metrics: {e_summary}")
                     run_metrics["mcmc_summary_metrics_error"] = str(e_summary)


                run_metrics["fitting_time_seconds"] = smoother_results.fitting_time_seconds
                run_metrics["status"] = "success" # Overall success for MCMC workflow

            else:
                run_metrics["status"] = "failed"
                run_metrics["error"] = "MCMC Workflow failed to return MCMC results."
                print(f"  ❌ {run_metrics['error']}")

        else:
            # This case should be caught by argparse
            raise ValueError(f"Invalid eval_mode: {eval_mode}. Choose 'mcmc' or 'fixed'.")

    except FileNotFoundError as e:
        run_metrics["status"] = "failed"
        run_metrics["error"] = f"File not found: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except ValueError as e:
        run_metrics["status"] = "failed"
        run_metrics["error"] = f"Value error during setup or evaluation: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except RuntimeError as e:
        run_metrics["status"] = "failed"
        run_metrics["error"] = f"Runtime error during core process: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except Exception as e:
        import traceback
        run_metrics["status"] = "failed"
        run_metrics["error"] = f"An unexpected error occurred: {type(e).__name__}: {e}"
        run_metrics["traceback"] = traceback.format_exc()
        print(f"❌ {run_metrics['error']}")
        print(f"Traceback: {run_metrics['traceback']}")

    finally:
        # Final timestamp and duration
        run_metrics["end_time"] = time.time()
        run_metrics["duration_seconds"] = run_metrics["end_time"] - run_metrics.get("start_time", run_metrics["end_time"])


        # Save metrics regardless of success
        try:
            with open(metrics_output_file, 'w') as f:
                # Use default=str to handle potential non-JSON serializable objects like NaNs/Infs
                json.dump(run_metrics, f, indent=4, default=str)
            print(f"✓ Metrics saved to {metrics_output_file}")
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")

        # Close all plot figures created during the run
        if plt:
            try:
                plt.close('all')
                print("✓ Closed all plot figures.")
            except Exception as e:
                 print(f"Warning: Error closing plot figures: {e}")


    print("\n--- GPM Model Runner (Corrected Version) Finished ---")
    # You can return the SmootherResults object if needed for external use
    # return smoother_results # Or return the run_metrics dictionary
    return run_metrics # Returning metrics seems more appropriate for a main execution script


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPM Model Runner (MCMC or Fixed Params)")

    # --- File Paths ---
    parser.add_argument("--gpm_file", type=str, required=True,
                        help="Path to the GPM file defining the model.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output (plots, metrics, etc.). Will be created if it doesn't exist.")
    parser.add_argument("--data_file", type=str, required=True,
                        # Default path relative to the script if needed
                        # default=os.path.join(SCRIPT_DIR_CURRENT_FILE, "data", "data_m5.csv"),
                        help="Path to the data CSV file. Must contain a 'Date' column and columns matching GPM varobs.")

    # --- Evaluation Mode ---
    parser.add_argument("--eval_mode", type=str, default="mcmc", choices=["mcmc", "fixed"],
                        help="Evaluation mode: 'mcmc' (MCMC sampling) or 'fixed' (single fixed parameter evaluation).")

    # --- MCMC Configuration (used in 'mcmc' mode) ---
    parser.add_argument("--mcmc_warmup", type=int, default=50,
                        help="Number of MCMC warmup steps per chain.")
    parser.add_argument("--mcmc_samples", type=int, default=50,
                        help="Number of MCMC sampling steps per chain.")
    parser.add_argument("--mcmc_chains", type=int, default=1,
                        help="Number of MCMC chains to run.")
    parser.add_argument("--smoother_draws", type=int, default=20,
                        help="Number of posterior draws to use for simulation smoothing and plotting.")

    # --- Fixed Parameter Configuration (used in 'fixed' mode) ---
    # Note: The fixed parameter *values* themselves are expected to be specified
    # externally (e.g., in a config file) or passed via a dedicated argument
    # like --fixed_params_json (not implemented here but can be added).
    # By default, 'fixed' mode resolves values from GPM priors if not overridden.
    # parser.add_argument("--fixed_params_json", type=str, default=None,
    #                     help="JSON string or path to JSON file specifying fixed parameter values to override GPM priors.")
    parser.add_argument("--initial_state_overrides_json", type=str, default=None,
                        help="JSON string for initial state prior overrides (mean) for fixed evaluation.")
    parser.add_argument("--fixed_trend_p0_scale", type=float, default=1e4,
                        help="Trend P0 variance scale for fixed evaluation mode.")
    parser.add_argument("--fixed_var_p0_scale", type=float, default=1.0,
                        help="VAR P0 fallback variance scale for fixed evaluation mode.")


    # --- Plotting Configuration ---
    parser.add_argument("--generate_plots", action="store_true", default=True,
                        help="Enable/disable generation of all plots.")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Enable/disable saving plots to files.")
    parser.add_argument("--plot_hdi_prob", type=float, default=0.90,
                        help="Probability mass for Highest Density Interval (HDI) in plots.")
    parser.add_argument("--show_plot_info_boxes", action="store_true", default=False,
                        help="Show small info boxes on plots (e.g., N_draws, RMSE).")
    parser.add_argument("--plot_default_observed_vs_fitted", action="store_true", default=True,
                        help="Enable/disable the default 'Observed vs Fitted' plot.")
    parser.add_argument("--plot_default_observed_vs_trend_components", action="store_true", default=True,
                        help="Enable/disable the default 'Observed vs Single Trend Component' plots.")
    # custom_plot_specs argument is harder to pass via command line directly,
    # it's better defined within the script or loaded from a JSON file.
    # For this script, we will define `custom_plot_specifications` directly.
    # parser.add_argument("--custom_plot_specs_json", type=str, default=None,
    #                     help="JSON string or path to JSON file defining custom plots.")


    args = parser.parse_args()

    # --- Validate arguments and setup ---
    if not os.path.exists(args.gpm_file):
        print(f"Error: GPM file not found at {args.gpm_file}")
        sys.exit(1)

    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse initial state overrides if provided
    parsed_initial_state_overrides = None
    if args.initial_state_overrides_json:
        try:
            parsed_initial_state_overrides = json.loads(args.initial_state_overrides_json)
            if not isinstance(parsed_initial_state_overrides, dict):
                print("Warning: --initial_state_overrides_json was not a valid JSON dictionary. Ignoring.")
                parsed_initial_state_overrides = None
        except json.JSONDecodeError as e:
            print(f"Error decoding --initial_state_overrides_json: {e}. Ignoring overrides.")
            parsed_initial_state_overrides = None

    # --- Define Custom Plot Specifications (Defined directly in the script) ---
    # This list defines which series from the SmootherResults should be plotted together.
    # The names ('y_us', 'pi_US_full_trend', etc.) must match the names available
    # in the SmootherResults object (i.e., observed_variable_names, trend_names, stationary_names).
    # 'combined' type uses 'components' list with {'type', 'name'} pairs.
    # Ensure these names match your GPM file (model_common_factor.gpm).

    custom_plot_specifications = [
        {
            "title": "US: Output (Observed, Trend, Cycle)",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k.-', 'show_hdi': False},
                {'type': 'trend', 'name': 'y_US_trend', 'label': 'Trend y_US_trend', 'show_hdi': True, 'color': 'blue'},
                {'type': 'stationary', 'name': 'cycle_Y_US', 'label': 'Cycle cycle_Y_US', 'show_hdi': True, 'color': 'green'}
            ]
        },
        {
            "title": "US: Inflation (Observed, Trend, Cycle)",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_us', 'label': 'Observed pi_us', 'style': 'k.-', 'show_hdi': False},
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Trend pi_US_full_trend', 'show_hdi': True, 'color': 'red'},
                {'type': 'stationary', 'name': 'cycle_PI_US', 'label': 'Cycle cycle_PI_US', 'show_hdi': True, 'color': 'orange'}
            ]
        },
         {
            "title": "US: Short Rate (Observed, Trend, Cycle)",
            "series_to_plot": [
                {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us', 'style': 'k.-', 'show_hdi': False},
                {'type': 'trend', 'name': 'R_US_short_trend', 'label': 'Trend R_US_short_trend', 'show_hdi': True, 'color': 'green'},
                {'type': 'stationary', 'name': 'cycle_Rshort_US', 'label': 'Cycle cycle_Rshort_US', 'show_hdi': True, 'color': 'brown'}
            ]
        },
        # Add EA and JP plots similarly
        {
            "title": "EA: Output (Observed, Trend, Cycle)",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k.-', 'show_hdi': False},
                {'type': 'trend', 'name': 'y_EA_trend', 'label': 'Trend y_EA_trend', 'show_hdi': True, 'color': 'blue'}
                # Add stationary component if needed: {'type': 'stationary', 'name': 'cycle_Y_EA', 'label': 'Cycle cycle_Y_EA', 'show_hdi': True, 'color': 'green'}
            ]
        },
        # ... add more custom plot specs for EA and JP as needed ...
    ]
    # You can load custom specs from a JSON file instead:
    # custom_plot_specifications = []
    # if args.custom_plot_specs_json:
    #     try:
    #         with open(args.custom_plot_specs_json, 'r') as f:
    #             custom_plot_specifications = json.load(f)
    #         print(f"✓ Loaded custom plot specs from {args.custom_plot_specs_json}")
    #     except FileNotFoundError:
    #         print(f"Error: Custom plot specs file not found at {args.custom_plot_specs_json}. Skipping custom plots.")
    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding custom plot specs JSON from {args.custom_plot_specs_json}: {e}. Skipping custom plots.")


    # --- Run the workflow ---
    final_run_metrics = run_model_and_generate_outputs(
        gpm_file_arg=args.gpm_file,
        output_dir_arg=args.output_dir,
        data_path_arg=args.data_file,
        eval_mode=args.eval_mode,
        initial_state_overrides_fixed=parsed_initial_state_overrides,
        trend_p0_scale_fixed=args.fixed_trend_p0_scale,
        var_p0_scale_fixed=args.fixed_var_p0_scale,
        mcmc_warmup=args.mcmc_warmup,
        mcmc_samples=args.mcmc_samples,
        mcmc_chains=args.mcmc_chains,
        smoother_draws=args.smoother_draws,
        generate_plots_flag=args.generate_plots,
        plot_default_ovf_flag=args.plot_default_observed_vs_fitted,
        plot_default_ovt_flag=args.plot_default_observed_vs_trend_components,
        show_plot_info_boxes_flag=args.show_plot_info_boxes,
        custom_plot_specs_list=custom_plot_specifications, # Pass the custom specs list
        save_plots_flag=args.save_plots
    )

    # The workflow function saved the metrics.
    # You can print a final summary based on the returned metrics dictionary
    print("\n" + "="*60)
    print("      FINAL RUN SUMMARY      ")
    print("="*60)
    print(f"Evaluation Mode: {final_run_metrics.get('eval_mode', 'N/A')}")
    print(f"Run Status: {final_run_metrics.get('status', 'Unknown')}")
    if final_run_metrics.get('status') == 'failed':
        print(f"Error: {final_run_metrics.get('error', 'N/A')}")
        if final_run_metrics.get('traceback'):
             print("Traceback:\n", final_run_metrics['traceback'])

    if final_run_metrics.get('eval_mode') == 'fixed' and 'log_likelihood' in final_run_metrics:
         print(f"Log-likelihood: {final_run_metrics['log_likelihood']:.3f}")
    elif final_run_metrics.get('eval_mode') == 'mcmc' and 'log_likelihood_estimate' in final_run_metrics:
         print(f"Estimated Mean Log-likelihood (MCMC): {final_run_metrics['log_likelihood_estimate']:.3f}")
         if 'min_ess_bulk' in final_run_metrics:
             print(f"MCMC Diagnostics: Min ESS_bulk = {final_run_metrics['min_ess_bulk']:.2f}, Max Rhat = {final_run_metrics['max_rhat']:.2f}")

    print(f"Output saved to: {args.output_dir}")
    print("="*60 + "\n")


# These are the optiins for calling this function
#     python path/to/your/Application/gpm_ai_calibrator.py \
#     --gpm_file path/to/your/model_common_factor.gpm \
#     --data_file path/to/your/data_m5.csv \
#     --output_dir results_output \
#     --eval_mode fixed \
#     --generate_plots --save_plots --show_plot_info_boxes \
#     --plot_default_observed_vs_fitted --plot_default_observed_vs_trend_components \
#     --fixed_trend_p0_scale 1e4 --fixed_var_p0_scale 1.0 \
#     # Optional: --initial_state_overrides_json '{"trend_r_world": {"mean": 0.03}}'
#     # Optional: --mcmc_warmup 500 --mcmc_samples 1000 # Only relevant for mcmc mode


#    python path/to/your/Application/gpm_ai_calibrator.py \
#     --gpm_file path/to/your/model_common_factor.gpm \
#     --data_file path/to/your/data_m5.csv \
#     --output_dir results_output_mcmc \
#     --eval_mode mcmc \
#     --generate_plots --save_plots \
#     --mcmc_warmup 500 --mcmc_samples 1000 --mcmc_chains 4 \
#     --smoother_draws 100 \
#     --plot_default_observed_vs_fitted --plot_default_observed_vs_trend_components \
#     --plot_hdi_prob 0.95 