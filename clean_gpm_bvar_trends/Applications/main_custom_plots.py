import sys
import os

# --- Path Setup ---
SCRIPT_DIR_CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_CURRENT_FILE, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"{os.path.basename(__file__)}: Adjusted sys.path. Includes: {PROJECT_ROOT}")

# Standard library imports
import argparse
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
try:
    from clean_gpm_bvar_trends import (
        run_complete_gpm_analysis,
        run_quick_example,
        SmootherResults
    )
    from clean_gpm_bvar_trends.integration_orchestrator import create_integration_orchestrator
    from clean_gpm_bvar_trends.gpm_prior_evaluator import _resolve_parameter_value
    from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE
    from clean_gpm_bvar_trends.reporting_plots import (
        create_all_standard_plots,
        plot_custom_series_comparison
    )
    MODULES_LOADED = True
    PLOTTING_AVAILABLE_MAIN = True
    print("✓ All modules loaded successfully")
except ImportError as e:
    print(f"FATAL ERROR ({os.path.basename(__file__)}): Could not import core library modules: {e}")
    MODULES_LOADED = False
    PLOTTING_AVAILABLE_MAIN = False
    
    # Define dummy classes/functions for graceful degradation
    class SmootherResults:
        pass
    
    def run_complete_gpm_analysis(*args, **kwargs):
        return None
    
    def plot_custom_series_comparison(*args, **kwargs):
        return None


def extract_fixed_parameters_from_gpm(parsed_gpm_model) -> Dict[str, Any]:
    """
    Extracts fixed parameter values from GPM model's prior definitions.
    Updated for new codebase structure.
    """
    fixed_params = {}
    
    # Structural parameters
    for param_name in parsed_gpm_model.parameters:
        if param_name in parsed_gpm_model.estimated_params:
            try:
                fixed_params[param_name] = _resolve_parameter_value(
                    param_name, {}, parsed_gpm_model.estimated_params, False)
            except Exception as e:
                print(f"Warning: Could not resolve parameter {param_name}: {e}")
    
    # Shock standard deviations
    all_shocks = parsed_gpm_model.trend_shocks + parsed_gpm_model.stationary_shocks
    for shock_name in all_shocks:
        try:
            fixed_params[shock_name] = _resolve_parameter_value(
                shock_name, {}, parsed_gpm_model.estimated_params, True)
        except Exception as e:
            print(f"Warning: Could not resolve shock {shock_name}: {e}")
    
    # VAR innovation correlation Cholesky (default to identity if needed)
    if parsed_gpm_model.stationary_variables:
        n_stat = len(parsed_gpm_model.stationary_variables)
        if n_stat > 0:
            fixed_params['_var_innovation_corr_chol'] = jnp.eye(n_stat, dtype=_DEFAULT_DTYPE)
    
    print(f"  Extracted {len(fixed_params)} fixed parameters from GPM priors")
    return fixed_params


def create_custom_plot_specs() -> List[Dict[str, Any]]:
    """
    Creates custom plot specifications for the GPM model.
    Updated for new codebase structure.
    """
    return [
        {
            "title": "US: Observed Output vs. Estimated Output Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'y_us', 
                    'label': 'Observed y_us', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'y_US_trend', 
                    'label': 'Est. Trend (y_US_trend)', 
                    'show_hdi': True, 
                    'color': 'blue'
                }
            ]
        },
        {
            "title": "US: Observed Inflation vs. Estimated Inflation Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'pi_us', 
                    'label': 'Observed pi_us', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'pi_US_full_trend', 
                    'label': 'Est. Trend (pi_US_full_trend)', 
                    'show_hdi': True, 
                    'color': 'red'
                }
            ]
        },
        {
            "title": "US: Observed Short Rate vs. Estimated Rate Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'r_us', 
                    'label': 'Observed r_us', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'R_US_short_trend', 
                    'label': 'Est. Trend (R_US_short_trend)', 
                    'show_hdi': True, 
                    'color': 'green'
                }
            ]
        },
        # Euro Area plots
        {
            "title": "EA: Observed Output vs. Estimated Output Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'y_ea', 
                    'label': 'Observed y_ea', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'y_EA_trend', 
                    'label': 'Est. Trend (y_EA_trend)', 
                    'show_hdi': True, 
                    'color': 'darkorange'
                }
            ]
        },
        {
            "title": "EA: Observed Inflation vs. Estimated Inflation Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'pi_ea', 
                    'label': 'Observed pi_ea', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'pi_EA_full_trend', 
                    'label': 'Est. Trend (pi_EA_full_trend)', 
                    'show_hdi': True, 
                    'color': 'purple'
                }
            ]
        },
        {
            "title": "EA: Observed Short Rate vs. Estimated Rate Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'r_ea', 
                    'label': 'Observed r_ea', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'R_EA_short_trend', 
                    'label': 'Est. Trend (R_EA_short_trend)', 
                    'show_hdi': True, 
                    'color': 'brown'
                }
            ]
        },
        # Japan plots
        {
            "title": "JP: Observed Output vs. Estimated Output Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'y_jp', 
                    'label': 'Observed y_jp', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'y_JP_trend', 
                    'label': 'Est. Trend (y_JP_trend)', 
                    'show_hdi': True, 
                    'color': 'magenta'
                }
            ]
        },
        {
            "title": "JP: Observed Inflation vs. Estimated Inflation Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'pi_jp', 
                    'label': 'Observed pi_jp', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'pi_JP_full_trend', 
                    'label': 'Est. Trend (pi_JP_full_trend)', 
                    'show_hdi': True, 
                    'color': 'cyan'
                }
            ]
        },
        {
            "title": "JP: Observed Short Rate vs. Estimated Rate Trend",
            "series_to_plot": [
                {
                    'type': 'observed', 
                    'name': 'r_jp', 
                    'label': 'Observed r_jp', 
                    'style': 'k.-', 
                    'color': 'black'
                },
                {
                    'type': 'trend', 
                    'name': 'R_JP_short_trend', 
                    'label': 'Est. Trend (R_JP_short_trend)', 
                    'show_hdi': True, 
                    'color': 'lime'
                }
            ]
        }
    ]


def generate_fallback_plots(results: SmootherResults, plot_subdir: str) -> int:
    """
    Generate fallback plots based on available data in results.
    This works when the custom plot specifications don't match the actual variable names.
    """
    if not PLOTTING_AVAILABLE_MAIN or not results:
        return 0
    
    plots_generated = 0
    
    try:
        # Get available data
        obs_names = results.observed_variable_names
        trend_names = results.trend_names
        
        if not obs_names or not trend_names:
            print("  No variable names available for fallback plots")
            return 0
        
        print(f"  Generating fallback plots for {len(obs_names)} observed vars")
        print(f"  Available observed: {obs_names}")
        print(f"  Available trends: {trend_names}")
        
        # Create simple obs vs trend plots
        num_plots = min(len(obs_names), len(trend_names))
        
        for i in range(num_plots):
            try:
                plot_spec = {
                    "title": f"{obs_names[i]} vs {trend_names[i]}",
                    "series_to_plot": [
                        {
                            'type': 'observed',
                            'name': obs_names[i],
                            'label': f'Observed {obs_names[i]}',
                            'style': '-',
                            'color': 'black'
                        },
                        {
                            'type': 'trend',
                            'name': trend_names[i],
                            'label': f'Trend {trend_names[i]}',
                            'show_hdi': True,
                            'style': '-',
                            'color': ['blue', 'red', 'green', 'orange', 'purple'][i % 5]
                        }
                    ]
                }
                
                fig = plot_custom_series_comparison(
                    plot_title=plot_spec["title"],
                    series_specs=plot_spec["series_to_plot"],
                    results=results,
                    show_info_box=False
                )
                
                if fig:
                    safe_filename = f"fallback_{i+1}_{obs_names[i]}_vs_{trend_names[i]}.png"
                    plot_path = os.path.join(plot_subdir, safe_filename)
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    plots_generated += 1
                    print(f"    ✓ Generated: {safe_filename}")
                else:
                    print(f"    ❌ Plot function returned None for {obs_names[i]}")
                    
            except Exception as e:
                print(f"    ❌ Error creating plot {i+1}: {e}")
        
        return plots_generated
        
    except Exception as e:
        print(f"  ❌ Error in fallback plot generation: {e}")
        return 0


def run_model_and_generate_outputs(
    gpm_file_arg: str,
    output_dir_arg: str,
    data_path_arg: str,
    eval_mode: str = "mcmc",
    initial_state_overrides_fixed: Optional[Dict[str, Dict[str, float]]] = None,
    trend_p0_scale_fixed: float = 1e4,
    var_p0_scale_fixed: float = 1.0
) -> Dict[str, Any]:
    """
    Updated main function using the new codebase API.
    """
    print(f"\n=== Starting Model Run ===")
    print(f"Mode: {eval_mode}")
    print(f"GPM file: {gpm_file_arg}")
    print(f"Data file: {data_path_arg}")
    print(f"Output dir: {output_dir_arg}")
    
    if not MODULES_LOADED:
        return {"error": "Required modules not loaded"}
    
    # Create output directories
    plot_subdir = os.path.join(output_dir_arg, "plots")
    os.makedirs(plot_subdir, exist_ok=True)
    metrics_output_file = os.path.join(output_dir_arg, "metrics.json")
    run_metrics = {}
    
    try:
        # --- Data Loading ---
        print(f"Loading data from: {data_path_arg}")
        if not os.path.exists(data_path_arg):
            raise FileNotFoundError(f"Data file {data_path_arg} not found")
        
        # Load and prepare data
        data_df = pd.read_csv(data_path_arg)
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'])
            data_df.set_index('Date', inplace=True)
            data_df = data_df.asfreq('QE')  # Quarterly frequency
        
        # Check for required columns
        observed_cols = ['y_us', 'pi_us', 'r_us', 'y_ea', 'pi_ea', 'r_ea', 'y_jp', 'pi_jp', 'r_jp']
        available_cols = [col for col in observed_cols if col in data_df.columns]
        
        if not available_cols:
            raise ValueError(f"No required columns found in data. Available: {list(data_df.columns)}")
        
        # Use available columns
        data_sub = data_df[available_cols]
        print(f"✓ Data loaded. Shape: {data_sub.shape}, Columns: {available_cols}")
        
        # --- Custom Plot Specifications ---
        custom_plot_specs = create_custom_plot_specs()
        
        # --- Run Analysis Based on Mode ---
        if eval_mode == "fixed":
            print("Running fixed parameter evaluation...")
            
            # Extract fixed parameters from GPM priors
            orchestrator = create_integration_orchestrator(gpm_file_arg, strict_validation=True)
            parsed_gpm_model = orchestrator.reduced_model
            fixed_params = extract_fixed_parameters_from_gpm(parsed_gpm_model)
            
            if not fixed_params:
                raise ValueError("No fixed parameters extracted from GPM file")
            
            # Run fixed parameter analysis
            results = run_complete_gpm_analysis(
                data=data_sub,
                gpm_file=gpm_file_arg,
                analysis_type="fixed_params",
                param_values=fixed_params,
                initial_state_prior_overrides=initial_state_overrides_fixed,
                num_sim_draws=50,
                plot_results=False,  # We'll handle plotting separately
                trend_P0_var_scale=trend_p0_scale_fixed,
                var_P0_var_scale=var_p0_scale_fixed,
                use_gamma_init_for_test=True
            )
            
            run_metrics["eval_mode"] = "fixed"
            if results:
                run_metrics["log_likelihood"] = results.log_likelihood
                run_metrics["n_draws"] = results.n_draws
                print(f"✓ Fixed evaluation completed. Log-likelihood: {results.log_likelihood}")
            else:
                run_metrics["error"] = "Fixed parameter evaluation returned None"
        
        elif eval_mode == "mcmc":
            print("Running MCMC analysis...")
            
            # MCMC settings
            num_warmup = 100
            num_samples = 200
            num_chains = 1
            num_extract_draws = 50
            
            print(f"  MCMC settings: {num_warmup} warmup, {num_samples} samples, {num_chains} chains")
            
            # Run MCMC analysis
            results = run_complete_gpm_analysis(
                data=data_sub,
                gpm_file=gpm_file_arg,
                analysis_type="mcmc",
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                num_extract_draws=num_extract_draws,
                generate_plots=False,  # We'll handle plotting separately
                use_gamma_init=True,
                gamma_scale_factor=1.0,
                target_accept_prob=0.80,
                variable_names_override=available_cols
            )
            
            run_metrics["eval_mode"] = "mcmc"
            if results:
                run_metrics["log_likelihood"] = results.log_likelihood
                run_metrics["n_draws"] = results.n_draws
                # Add MCMC-specific metrics if available
                print(f"✓ MCMC completed. Log-likelihood: {results.log_likelihood}, Draws: {results.n_draws}")
            else:
                run_metrics["error"] = "MCMC analysis returned None"
        
        else:
            raise ValueError(f"Invalid eval_mode: {eval_mode}")
        
        # --- Generate Plots ---
        if PLOTTING_AVAILABLE_MAIN and results:
            print("Generating plots...")
            plots_generated = 0
            
            # Try custom plots first
            if hasattr(results, 'trend_names') and hasattr(results, 'observed_variable_names'):
                try:
                    # Generate standard plots
                    standard_plots = create_all_standard_plots(
                        results, 
                        save_path_prefix=os.path.join(plot_subdir, "standard"),
                        show_info_boxes=False
                    )
                    plots_generated += len([p for p in standard_plots.values() if p is not None])
                    
                    # Try custom plots
                    for i, spec in enumerate(custom_plot_specs):
                        try:
                            fig = plot_custom_series_comparison(
                                plot_title=spec["title"],
                                series_specs=spec["series_to_plot"],
                                results=results,
                                show_info_box=False
                            )
                            
                            if fig:
                                safe_title = spec["title"].lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
                                plot_filename = f"custom_{i+1:02d}_{safe_title}.png"
                                plot_path = os.path.join(plot_subdir, plot_filename)
                                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                                plt.close(fig)
                                plots_generated += 1
                            
                        except Exception as e:
                            print(f"    ❌ Custom plot {i+1} failed: {e}")
                    
                    # If custom plots failed, try fallback
                    if plots_generated == 0:
                        print("  Custom plots failed, trying fallback approach...")
                        plots_generated = generate_fallback_plots(results, plot_subdir)
                    
                except Exception as e:
                    print(f"  ❌ Plot generation failed: {e}")
                    # Try fallback approach
                    plots_generated = generate_fallback_plots(results, plot_subdir)
            
            run_metrics["plots_generated"] = plots_generated
            print(f"✓ Generated {plots_generated} plots")
        
        else:
            if not results:
                print("⚠️ No results to plot")
            else:
                print("⚠️ Plotting not available")
    
    except FileNotFoundError as e:
        run_metrics["error"] = f"File not found: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except ValueError as e:
        run_metrics["error"] = f"Value error: {str(e)}"
        print(f"❌ {run_metrics['error']}")
    except Exception as e:
        import traceback
        run_metrics["error"] = f"Execution error: {str(e)}"
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
    
    # Clean up plots
    if PLOTTING_AVAILABLE_MAIN and plt:
        plt.close('all')
    
    return run_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPM Model Runner - Updated for New Codebase")
    
    parser.add_argument("--gpm_file", type=str, required=True, 
                        help="Path to the GPM file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save plots and metrics")
    parser.add_argument("--data_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "Application", "data_m5.csv"),
                        help="Path to the data CSV file")
    parser.add_argument("--eval_mode", type=str, default="mcmc", choices=["mcmc", "fixed"],
                        help="Evaluation mode: 'mcmc' or 'fixed'")
    parser.add_argument("--fixed_trend_p0_scale", type=float, default=1e4,
                        help="Trend P0 variance scale for fixed eval")
    parser.add_argument("--fixed_var_p0_scale", type=float, default=1.0,
                        help="VAR P0 fallback variance scale for fixed eval")
    parser.add_argument("--initial_state_overrides_json", type=str, default=None,
                        help="JSON string for initial state prior overrides")
    
    # Test/debug arguments
    parser.add_argument("--test_data_loading", action="store_true",
                        help="Test data loading without running model")
    parser.add_argument("--test_gpm_parsing", action="store_true",
                        help="Test GPM file parsing")

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
    
    # Handle test modes
    if args.test_data_loading:
        print("=== Testing Data Loading ===")
        if not os.path.exists(args.data_file):
            print(f"❌ Data file not found: {args.data_file}")
        else:
            try:
                data_df = pd.read_csv(args.data_file)
                print(f"✓ Data loaded successfully")
                print(f"  Shape: {data_df.shape}")
                print(f"  Columns: {list(data_df.columns)}")
                
                if 'Date' in data_df.columns:
                    data_df['Date'] = pd.to_datetime(data_df['Date'])
                    print(f"  Date range: {data_df['Date'].min()} to {data_df['Date'].max()}")
                
                # Check for missing values
                missing_info = data_df.isnull().sum()
                if missing_info.sum() > 0:
                    print("  Missing values:")
                    for col, count in missing_info.items():
                        if count > 0:
                            print(f"    {col}: {count}")
                else:
                    print("  ✓ No missing values")
                    
            except Exception as e:
                print(f"❌ Error loading data: {e}")
    
    elif args.test_gpm_parsing:
        print("=== Testing GPM File Parsing ===")
        if not os.path.exists(args.gpm_file):
            print(f"❌ GPM file not found: {args.gpm_file}")
        else:
            try:
                from clean_gpm_bvar_trends.integration_orchestrator import create_integration_orchestrator
                
                orchestrator = create_integration_orchestrator(args.gpm_file, strict_validation=True)
                parsed_model = orchestrator.reduced_model
                
                print(f"✓ GPM file parsed successfully")
                print(f"  Parameters: {parsed_model.parameters}")
                print(f"  Trend variables: {parsed_model.gpm_trend_variables_original}")
                print(f"  Stationary variables: {parsed_model.gpm_stationary_variables_original}")
                print(f"  Observed variables: {parsed_model.gpm_observed_variables_original}")
                print(f"  Trend shocks: {parsed_model.trend_shocks}")
                print(f"  Stationary shocks: {parsed_model.stationary_shocks}")
                
                if parsed_model.var_prior_setup:
                    print(f"  VAR setup: order={parsed_model.var_prior_setup.var_order}")
                else:
                    print("  No VAR prior setup")
                    
                # Test parameter extraction
                try:
                    fixed_params = extract_fixed_parameters_from_gpm(parsed_model)
                    print(f"  ✓ Extracted {len(fixed_params)} fixed parameters")
                except Exception as e:
                    print(f"  ⚠️ Parameter extraction warning: {e}")
                    
            except Exception as e:
                print(f"❌ Error parsing GPM: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        # Normal execution
        print(f"=== Starting GPM Model Runner ===")
        print(f"Arguments: {vars(args)}")
        
        if not MODULES_LOADED:
            print("❌ Required modules not loaded. Cannot proceed.")
            sys.exit(1)
        
        results = run_model_and_generate_outputs(
            gpm_file_arg=args.gpm_file,
            output_dir_arg=args.output_dir,
            data_path_arg=args.data_file,
            eval_mode=args.eval_mode,
            trend_p0_scale_fixed=args.fixed_trend_p0_scale,
            var_p0_scale_fixed=args.fixed_var_p0_scale,
            initial_state_overrides_fixed=parsed_initial_state_overrides
        )
        
        print("=== Execution Finished ===")
        
        # Print final summary
        if results:
            if "error" in results:
                print(f"❌ FINAL RESULT: Failed - {results['error']}")
                sys.exit(1)
            else:
                print(f"✅ FINAL RESULT: Success")
                if "log_likelihood" in results:
                    print(f"   Log-likelihood: {results['log_likelihood']:.3f}")
                if "plots_generated" in results:
                    print(f"   Plots generated: {results['plots_generated']}")
        else:
            print("❌ No results returned")
            sys.exit(1)