import sys
import os

# --- Path Setup ---
# (Keep your existing sys.path modification block here)
SCRIPT_DIR_CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_CURRENT_FILE, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"{os.path.basename(__file__)}: Adjusted sys.path. Includes: {PROJECT_ROOT}")

# NOW, import other standard libraries and THEN your custom packages/modules
import argparse  # <--- ADD THIS LINE
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any

# --- JAX Configuration ---
from clean_gpm_bvar_trends.jax_config import configure_jax
configure_jax()
import jax.numpy as jnp

# --- Core Library Imports ---
# (Keep your existing core library imports)
try:
    from clean_gpm_bvar_trends.gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed
    from clean_gpm_bvar_trends.gpm_prior_evaluator import evaluate_gpm_at_parameters, _resolve_parameter_value
    from clean_gpm_bvar_trends.integration_orchestrator import create_integration_orchestrator
    from clean_gpm_bvar_trends.gpm_model_parser import ReducedModel
    from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE
    from clean_gpm_bvar_trends.reporting_plots import plot_observed_vs_fitted, plot_smoother_results, plot_custom_series_comparison
    import arviz as az
    MODULES_LOADED = True
    PLOTTING_AVAILABLE_MAIN = True
    print("✓ All modules loaded successfully")
except ImportError as e:
    print(f"FATAL ERROR ({os.path.basename(__file__)}): Could not import core library modules: {e}")
    MODULES_LOADED = False
    PLOTTING_AVAILABLE_MAIN = False
    # Define dummy plotting functions if needed for script to not crash entirely
    def plot_observed_vs_fitted(*args, **kwargs): return None
    def plot_smoother_results(*args, **kwargs): return None, None
    def plot_custom_series_comparison(*args, **kwargs): return None




def extract_fixed_parameters_from_gpm(parsed_gpm_model: ReducedModel) -> Dict[str, Any]:
    """
    Extracts a single set of fixed parameter values from the GPM's prior definitions
    (typically means for normal priors, modes for inverse gamma priors).
    """
    fixed_params = {}
    
    # Structural parameters
    for param_name, prior_spec in parsed_gpm_model.estimated_params.items():
        if param_name in parsed_gpm_model.parameters:
            fixed_params[param_name] = _resolve_parameter_value(
                param_name, {}, parsed_gpm_model.estimated_params, False)

    # Shock standard deviations
    all_shocks = parsed_gpm_model.trend_shocks + parsed_gpm_model.stationary_shocks
    for shock_name in all_shocks:
        fixed_params[shock_name] = _resolve_parameter_value(
            shock_name, {}, parsed_gpm_model.estimated_params, True)

    # VAR innovation correlation Cholesky (default to identity if not otherwise specified)
    if parsed_gpm_model.stationary_variables:
        n_stat = len(parsed_gpm_model.stationary_variables)
        if n_stat > 0:
            fixed_params['_var_innovation_corr_chol'] = jnp.eye(n_stat, dtype=_DEFAULT_DTYPE)
    
    print(f"  Extracted fixed parameters from GPM priors: {list(fixed_params.keys())}")
    return fixed_params


def run_model_and_generate_outputs(
    gpm_file_arg: str,  # CORRECTED: consistent parameter name
    output_dir_arg: str,
    data_path_arg: str,
    eval_mode: str = "mcmc",
    initial_state_overrides_fixed: Optional[Dict[str, Dict[str, float]]] = None,
    trend_p0_scale_fixed: float = 1e4,
    var_p0_scale_fixed: float = 1.0
):
    print(f"\n=== DEBUG: Starting run_model_and_generate_outputs ===")
    print(f"Mode: {eval_mode}")
    print(f"GPM file: {gpm_file_arg}")  # CORRECTED: use consistent parameter name
    print(f"Data file: {data_path_arg}")
    print(f"Output dir: {output_dir_arg}")
    
    if not MODULES_LOADED:
        print("ERROR: Modules not loaded, cannot proceed")
        return
        
    plot_subdir = os.path.join(output_dir_arg, "plots")
    os.makedirs(plot_subdir, exist_ok=True)
    metrics_output_file = os.path.join(output_dir_arg, "metrics.json")
    run_metrics = {}

    try:
        # --- Data Loading ---
        print(f"Loading data from: {data_path_arg}")
        if not os.path.exists(data_path_arg): 
            raise FileNotFoundError(f"Data file {data_path_arg} not found.")
        
        dta = pd.read_csv(data_path_arg)
        dta['Date'] = pd.to_datetime(dta['Date'])
        dta.set_index('Date', inplace=True)
        dta = dta.asfreq('QE')
        
        observed_cols = ['y_us', 'pi_us', 'r_us', 'y_ea', 'pi_ea', 'r_ea', 'y_jp', 'pi_jp', 'r_jp']
        missing_cols = [col for col in observed_cols if col not in dta.columns]
        if missing_cols: 
            raise ValueError(f"Data file missing required columns: {missing_cols}")
        
        data_sub = dta[observed_cols]
        print(f"✓ Data loaded successfully. Shape: {data_sub.shape}")

        # --- Custom Plot Specifications ---
        custom_plot_specifications = [
            {"title": "US: Observed Output vs. Estimated Output Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'y_us', 'label': 'Observed y_us', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'y_US_trend', 'label': 'Est. Trend (y_US_trend)', 'show_hdi': True, 'color': 'blue'}
             ]},
            {"title": "US: Observed Inflation vs. Estimated Inflation Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'pi_us', 'label': 'Observed pi_us', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Est. Trend (pi_US_full_trend)', 'show_hdi': True, 'color': 'red'}
             ]},
            {"title": "US: Observed Short Rate vs. Estimated Rate Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'r_us', 'label': 'Observed r_us', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'R_US_short_trend', 'label': 'Est. Trend (R_US_short_trend)', 'show_hdi': True, 'color': 'green'}
             ]},
            # Add similar plots for EA and JP
            {"title": "EA: Observed Output vs. Estimated Output Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'y_ea', 'label': 'Observed y_ea', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'y_EA_trend', 'label': 'Est. Trend (y_EA_trend)', 'show_hdi': True, 'color': 'darkorange'}
             ]},
            {"title": "EA: Observed Inflation vs. Estimated Inflation Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'pi_ea', 'label': 'Observed pi_ea', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'pi_EA_full_trend', 'label': 'Est. Trend (pi_EA_full_trend)', 'show_hdi': True, 'color': 'purple'}
             ]},
            {"title": "EA: Observed Short Rate vs. Estimated Rate Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'r_ea', 'label': 'Observed r_ea', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'R_EA_short_trend', 'label': 'Est. Trend (R_EA_short_trend)', 'show_hdi': True, 'color': 'brown'}
             ]},
            {"title": "JP: Observed Output vs. Estimated Output Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'y_jp', 'label': 'Observed y_jp', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'y_JP_trend', 'label': 'Est. Trend (y_JP_trend)', 'show_hdi': True, 'color': 'magenta'}
             ]},
            {"title": "JP: Observed Inflation vs. Estimated Inflation Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'pi_jp', 'label': 'Observed pi_jp', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'pi_JP_full_trend', 'label': 'Est. Trend (pi_JP_full_trend)', 'show_hdi': True, 'color': 'cyan'}
             ]},
            {"title": "JP: Observed Short Rate vs. Estimated Rate Trend",
             "series_to_plot": [
                 {'type': 'observed', 'name': 'r_jp', 'label': 'Observed r_jp', 'style': 'k.-', 'color': 'black'},
                 {'type': 'trend', 'name': 'R_JP_short_trend', 'label': 'Est. Trend (R_JP_short_trend)', 'show_hdi': True, 'color': 'lime'}
             ]}
            # You can also keep plots of individual core trends if they are economically meaningful on their own
            # For example, the world trends:
            # {"title": "Estimated World Real Rate Trend (r_w_trend)",
            #  "series_to_plot": [
            #      {'type': 'trend', 'name': 'r_w_trend', 'label': 'r_w_trend', 'show_hdi': True, 'color': 'navy'}
            #  ]},
            # {"title": "Estimated World Inflation Trend (pi_w_trend)",
            #  "series_to_plot": [
            #      {'type': 'trend', 'name': 'pi_w_trend', 'label': 'pi_w_trend', 'show_hdi': True, 'color': 'teal'}
            #  ]},
        ]

        if eval_mode == "fixed":
            print(f"  Running in FIXED parameter evaluation mode.")
            orchestrator = create_integration_orchestrator(gpm_file_arg, strict_validation=True)
            parsed_gpm_model_for_fixed = orchestrator.reduced_model
            
            fixed_params_from_priors = extract_fixed_parameters_from_gpm(parsed_gpm_model_for_fixed)
            
            # results = evaluate_gpm_at_parameters(
            #     gpm_file_path=gpm_file_arg,
            #     y=jnp.array(data_sub.values, dtype=_DEFAULT_DTYPE),
            #     param_values=fixed_params_from_priors,
            #     initial_state_prior_overrides=initial_state_overrides_fixed,
            #     num_sim_draws=50, 
            #     plot_results=PLOTTING_AVAILABLE_MAIN,
            #     variable_names=observed_cols,
            #     trend_P0_var_scale=trend_p0_scale_fixed,
            #     var_P0_var_scale=var_p0_scale_fixed
            # )

            results = evaluate_gpm_at_parameters(
                gpm_file_path=gpm_file_arg,
                y=jnp.array(data_sub.values, dtype=_DEFAULT_DTYPE),
                param_values=fixed_params_from_priors,
                initial_state_prior_overrides=initial_state_overrides_fixed,
                num_sim_draws=1,
                plot_results=False,  # DISABLE default plotting
                variable_names=observed_cols,
                trend_P0_var_scale=trend_p0_scale_fixed,
                var_P0_var_scale=var_p0_scale_fixed
            )            
            
            run_metrics["eval_mode"] = "fixed"
            if results:
                loglik_val = results.get('loglik')
                if loglik_val is not None:
                    run_metrics["log_likelihood"] = float(loglik_val.item()) if hasattr(loglik_val, 'item') else float(loglik_val)
                run_metrics["mcmc_summary_sample"] = "N/A (Fixed Parameter Evaluation)"
                run_metrics["min_ess_bulk"] = None
                run_metrics["max_rhat"] = None
                print("  ✓ Fixed parameter evaluation completed successfully.")

                # Generate custom plots based on fixed parameter results
                # if PLOTTING_AVAILABLE_MAIN and results and results.get('reconstructed_original_trends') is not None:
                #     print("  Generating CUSTOM plots instead of default plots...")
                    
                #     # Generate each custom plot specification
                #     for i, spec in enumerate(custom_plot_specifications):
                #         try:
                #             fig_custom = plot_custom_series_comparison(
                #                 plot_title=spec["title"],
                #                 series_specs=spec["series_to_plot"],
                #                 observed_data=np.asarray(data_sub.values),
                #                 trend_draws=np.asarray(results['reconstructed_original_trends']),
                #                 stationary_draws=np.asarray(results['reconstructed_original_stationary']),
                #                 observed_names=observed_cols,
                #                 trend_names=results['gpm_model'].gpm_trend_variables_original,
                #                 stationary_names=results['gpm_model'].gpm_stationary_variables_original,
                #                 time_index=data_sub.index,
                #                 hdi_prob=0.9
                #             )
                            
                #             if fig_custom:
                #                 # Create safe filename
                #                 safe_title = spec["title"].lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
                #                 plot_filename = f"custom_{i+1:02d}_{safe_title}.png"
                #                 plot_path = os.path.join(plot_subdir, plot_filename)
                                
                #                 fig_custom.savefig(plot_path, dpi=150, bbox_inches='tight')
                #                 plt.close(fig_custom)
                #                 print(f"    ✓ Saved: {plot_filename}")
                                
                #         except Exception as e:
                #             print(f"    ❌ Error generating custom plot {i+1}: {e}")
                if PLOTTING_AVAILABLE_MAIN and results and results.get('reconstructed_original_trends') is not None:
                        print("=== DEBUGGING AVAILABLE NAMES ===")
                        
                        gpm_model = results['gpm_model']
                        trends_data = results['reconstructed_original_trends']
                        stationary_data = results['reconstructed_original_stationary']
                        
                        print(f"Observed columns: {list(data_sub.columns)}")
                        print(f"Trend variables: {gpm_model.gpm_trend_variables_original}")
                        print(f"Stationary variables: {gpm_model.gpm_stationary_variables_original}")
                        
                        # Create simple working plots based on available data
                        obs_cols = list(data_sub.columns)
                        trend_vars = gpm_model.gpm_trend_variables_original
                        
                        # Create as many plots as we have data for
                        num_plots = min(len(obs_cols), len(trend_vars))
                        
                        for i in range(num_plots):
                            try:
                                plot_spec = {
                                    "title": f"{obs_cols[i]} vs {trend_vars[i]}",
                                    "series_to_plot": [
                                        {
                                            'type': 'observed', 
                                            'name': obs_cols[i], 
                                            'label': f'Observed {obs_cols[i]}', 
                                            'style': '-', 
                                            'color': 'black'
                                        },
                                        {
                                            'type': 'trend', 
                                            'name': trend_vars[i], 
                                            'label': f'Trend {trend_vars[i]}', 
                                            'show_hdi': True, 
                                            'style': '-', 
                                            'color': ['blue', 'red', 'green', 'orange', 'purple'][i % 5]
                                        }
                                    ]
                                }
                                
                                print(f"Creating plot: {plot_spec['title']}")
                                
                                fig_custom = plot_custom_series_comparison(
                                    plot_title=plot_spec["title"],
                                    series_specs=plot_spec["series_to_plot"],
                                    observed_data=np.asarray(data_sub.values),
                                    trend_draws=np.asarray(trends_data),
                                    stationary_draws=np.asarray(stationary_data),
                                    observed_names=obs_cols,
                                    trend_names=trend_vars,
                                    stationary_names=gpm_model.gpm_stationary_variables_original,
                                    time_index=data_sub.index,
                                    hdi_prob=0.9
                                )
                                
                                if fig_custom:
                                    plot_filename = f"obs_vs_trend_{i+1}_{obs_cols[i]}_vs_{trend_vars[i]}.png"
                                    plot_path = os.path.join(plot_subdir, plot_filename)
                                    fig_custom.savefig(plot_path, dpi=150, bbox_inches='tight')
                                    plt.close(fig_custom)
                                    print(f"✓ Saved: {plot_filename}")
                                else:
                                    print(f"❌ Plot function returned None")
                                    
                            except Exception as e:
                                print(f"❌ Error creating plot {i+1}: {e}")
                                import traceback
                                traceback.print_exc()            
            else:
                run_metrics["error"] = "Fixed parameter evaluation failed to return results."

        elif eval_mode == "mcmc":
            print(f"  Running in MCMC evaluation mode.")
            num_warmup_calib = 100
            num_samples_calib = 200  
            num_chains_calib = 1
            num_extract_draws_calib = 50

            print(f"  MCMC settings: {num_warmup_calib} warmup, {num_samples_calib} samples, {num_chains_calib} chains")
            
            results = complete_gpm_workflow_with_smoother_fixed(
                data=data_sub, 
                gpm_file=gpm_file_arg,  # CORRECTED: consistent parameter name
                num_warmup=num_warmup_calib, 
                num_samples=num_samples_calib, 
                num_chains=num_chains_calib,
                target_accept_prob=0.80, 
                use_gamma_init=True, 
                gamma_scale_factor=1.0,
                num_extract_draws=num_extract_draws_calib,
                generate_plots=PLOTTING_AVAILABLE_MAIN,
                plot_save_path=plot_subdir, 
                save_plots=True,
                custom_plot_specs=custom_plot_specifications,
                variable_names_override=observed_cols,
                data_file_source_for_summary=data_path_arg
            )
            
            print(f"  DEBUG: MCMC workflow returned: {type(results)}")
            if results:
                print(f"  DEBUG: Results keys: {list(results.keys())}")
                mcmc_obj = results.get('mcmc_object')
                print(f"  DEBUG: MCMC object type: {type(mcmc_obj)}")
                
            run_metrics["eval_mode"] = "mcmc"
            if results and results.get('mcmc_object'):
                print("  ✓ MCMC workflow returned MCMC results. Generating summary...")
                try:
                    mcmc_summary = az.summary(
                        results['mcmc_object'], 
                        hdi_prob=0.9, 
                        kind='stats',
                        var_names=[p for p in results['parsed_gpm_model'].parameters if not p.startswith("_")] + \
                                  [f"sigma_{s}" for s in results['parsed_gpm_model'].trend_shocks + results['parsed_gpm_model'].stationary_shocks]
                    )
                    run_metrics["mcmc_summary_dict"] = mcmc_summary.to_dict()
                    run_metrics["min_ess_bulk"] = float(mcmc_summary['ess_bulk'].min()) if 'ess_bulk' in mcmc_summary and not mcmc_summary['ess_bulk'].empty else None
                    run_metrics["max_rhat"] = float(mcmc_summary['r_hat'].max()) if 'r_hat' in mcmc_summary and not mcmc_summary['r_hat'].empty else None
                    run_metrics["fitting_time_seconds"] = results.get('fitting_time_seconds')
                    if 'loglik' in results['mcmc_object'].get_samples():
                        run_metrics["log_likelihood_estimate"] = np.mean(results['mcmc_object'].get_samples()['loglik']).item()
                    print("  ✓ MCMC workflow completed successfully.")
                except Exception as e:
                    print(f"  ⚠️  Error generating MCMC summary: {e}")
                    run_metrics["mcmc_summary_error"] = str(e)
            else:
                error_msg = "MCMC Workflow did not return expected MCMC results in main_custom_plots."
                run_metrics["error"] = error_msg
                print(f"  ❌ {error_msg}")
                
                # Debug what we actually got
                if results:
                    print(f"  DEBUG: Results is not None, available keys: {list(results.keys())}")
                    for key, value in results.items():
                        print(f"    {key}: {type(value)}")
                else:
                    print(f"  DEBUG: Results is None or empty")
                    
        else:
            raise ValueError(f"Invalid eval_mode: {eval_mode}. Choose 'mcmc' or 'fixed'.")

    except FileNotFoundError as e:
        run_metrics["error"] = f"File not found: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except ValueError as e:
        run_metrics["error"] = f"Value error: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except Exception as e:
        import traceback
        run_metrics["error"] = f"Model execution error: {str(e)}"
        run_metrics["traceback"] = traceback.format_exc()
        print(f"❌ {run_metrics['error']}")
        print(f"Traceback: {run_metrics['traceback']}")
    
    # Save metrics
    try:
        with open(metrics_output_file, 'w') as f:
            json.dump(run_metrics, f, indent=4, default=str)
        print(f"✓ Metrics saved to {metrics_output_file}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
        
    if PLOTTING_AVAILABLE_MAIN and plt: 
        plt.close('all')

    return run_metrics  # Return metrics for debugging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPM Model Runner (MCMC or Fixed Params) - Corrected Version")
    
    parser.add_argument("--gpm_file", type=str, required=True, help="Path to the GPM file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots and metrics.")
    parser.add_argument("--data_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "Application", "data_m5.csv"),
                        help="Path to the data CSV file.")
    parser.add_argument("--eval_mode", type=str, default="mcmc", choices=["mcmc", "fixed"],
                        help="Evaluation mode: 'mcmc' or 'fixed' (parameter evaluation).")
    parser.add_argument("--fixed_trend_p0_scale", type=float, default=1e4,
                        help="Trend P0 variance scale for fixed eval.")
    parser.add_argument("--fixed_var_p0_scale", type=float, default=1.0,
                        help="VAR P0 fallback variance scale for fixed eval.")
    parser.add_argument("--initial_state_overrides_json", type=str, default=None,
                        help="JSON string for initial state prior overrides")

    args = parser.parse_args()

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
    
    print(f"=== Starting GPM Model Runner (Corrected Version) ===")
    print(f"Arguments: {vars(args)}")
    
    # CORRECTED: Use consistent parameter names
    results = run_model_and_generate_outputs(
        gpm_file_arg=args.gpm_file,
        output_dir_arg=args.output_dir,
        data_path_arg=args.data_file,
        eval_mode=args.eval_mode,
        trend_p0_scale_fixed=args.fixed_trend_p0_scale,
        var_p0_scale_fixed=args.fixed_var_p0_scale,
        initial_state_overrides_fixed=parsed_initial_state_overrides
    )
    
    print("--- main_custom_plots.py execution finished ---")
    
    # Print final summary
    if results:
        if "error" in results:
            print(f"\n❌ FINAL RESULT: Failed - {results['error']}")
        else:
            print(f"\n✅ FINAL RESULT: Success")
            if "log_likelihood" in results:
                print(f"   Log-likelihood: {results['log_likelihood']:.3f}")