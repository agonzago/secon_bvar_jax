
# clean_gpm_bvar_trends/calibration_utils.py

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any, Tuple, Union # Added Union
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Imports from within the same package (clean_gpm_bvar_trends)
from .constants import _DEFAULT_DTYPE # Relative import
from .gpm_model_parser import ReducedModel, SymbolicReducerUtils # Relative import, Added SymbolicReducerUtils
from .integration_orchestrator import create_integration_orchestrator # Relative import
from .gpm_prior_evaluator import evaluate_gpm_at_parameters # Relative import
from .gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed # Relative import
from .common_types import SmootherResults # Import SmootherResults

# Conditional import for actual plotting functions
try:
    from .reporting_plots import ( # Relative import
        plot_time_series_with_uncertainty,
        plot_custom_series_comparison
    )
    PLOTTING_AVAILABLE_UTILS = True
    print("✓ Plotting functions imported in calibration_utils.py")
except ImportError as e:
    print(f"⚠️  Warning (calibration_utils.py): Could not import plotting functions: {e}")
    PLOTTING_AVAILABLE_UTILS = False
    # Define dummy plotting functions
    def plot_time_series_with_uncertainty(*args, **kwargs):
        print("Plotting disabled (calibration_utils) - plot_time_series_with_uncertainty skipped")
        if plt and kwargs.get('save_path'): 
             try:
                 fig, ax = plt.subplots()
                 ax.text(0.5, 0.5, "Plotting Disabled", horizontalalignment='center', verticalalignment='center')
                 fig.savefig(f"{kwargs['save_path']}_disabled.png"); plt.close(fig)
             except Exception: pass 
        return None
    def plot_custom_series_comparison(*args, **kwargs):
        print("Plotting disabled (calibration_utils) - plot_custom_series_comparison skipped")
        # Check for 'results' argument to determine if it's the SmootherResults version
        save_path_arg = kwargs.get('save_path')
        title_arg = kwargs.get('plot_title')
        
        if plt and save_path_arg and title_arg:
            try:
                fig, ax = plt.subplots(figsize=kwargs.get('default_fig_size', (12,6))) # default_fig_size not used by SmootherResults version
                ax.text(0.5, 0.5, f"Plotting Disabled: {title_arg}", horizontalalignment='center', verticalalignment='center')
                safe_title = title_arg.lower().replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('=','_').replace(':','_').replace('.','')
                fig.savefig(f"{save_path_arg}_{safe_title}_DISABLED.png", dpi=150, bbox_inches='tight'); plt.close(fig)
            except Exception: pass
        return None


# --- Configuration Class Definition ---
class PriorCalibrationConfig:
    def __init__(self,
                data_file_path: str = 'data/sim_data.csv',
                gpm_file_path: str = 'models/test_model.gpm',
                observed_variable_names: Optional[List[str]] = None,
                fixed_parameter_values: Optional[Dict[str, Any]] = None,
                # MCMC settings
                num_mcmc_warmup: int = 50,
                num_mcmc_samples: int = 100,
                num_mcmc_chains: int = 1,
                # Smoother settings
                num_smoother_draws: int = 50,
                # P0 settings
                use_gamma_init: bool = True,
                gamma_scale_factor: float = 1.0,
                # Plotting settings
                generate_plots: bool = True,
                plot_hdi_prob: float = 0.9,
                show_plot_info_boxes: bool = False,
                plot_save_path: Optional[str] = "prior_calibration_plots",
                save_plots: bool = False,
                custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
                # Settings for fixed param evaluation / sensitivity
                num_smoother_draws_for_fixed_params: int = 0,
                plot_sensitivity_point_results: bool = False,
                sensitivity_plot_custom_specs: Optional[List[Dict[str, Any]]] = None,
                # New: For overriding initval parameters in fixed evaluation
                initial_state_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                trend_P0_var_scale_fixed_eval: float = 1e4,
                var_P0_var_scale_fixed_eval: float = 1.0,
                # New: Control generation of specific default plot types
                plot_default_observed_vs_trend_components: bool = True 
                 ):
        self.data_file_path = data_file_path
        self.gpm_file_path = gpm_file_path
        self.observed_variable_names = observed_variable_names if observed_variable_names is not None else []
        self.fixed_parameter_values = fixed_parameter_values if fixed_parameter_values is not None else {}
        self.num_mcmc_warmup = num_mcmc_warmup
        self.num_mcmc_samples = num_mcmc_samples
        self.num_mcmc_chains = num_mcmc_chains
        self.num_smoother_draws = num_smoother_draws
        self.use_gamma_init = use_gamma_init
        self.gamma_scale_factor = gamma_scale_factor
        self.generate_plots = generate_plots
        self.plot_hdi_prob = plot_hdi_prob
        self.show_plot_info_boxes = show_plot_info_boxes
        self.plot_save_path = plot_save_path
        self.save_plots = save_plots
        self.custom_plot_specs = custom_plot_specs
        self.num_smoother_draws_for_fixed_params = num_smoother_draws_for_fixed_params
        self.plot_sensitivity_point_results = plot_sensitivity_point_results
        self.sensitivity_plot_custom_specs = sensitivity_plot_custom_specs
        self.initial_state_prior_overrides = initial_state_prior_overrides 
        self.trend_P0_var_scale_fixed_eval = trend_P0_var_scale_fixed_eval
        self.var_P0_var_scale_fixed_eval = var_P0_var_scale_fixed_eval
        self.plot_default_observed_vs_trend_components = plot_default_observed_vs_trend_components

# --- Helper Functions (General Utilities) ---
def validate_calibration_config(config: PriorCalibrationConfig) -> bool:
    print("\n--- Validating Prior Calibration Configuration ---"); issues = []
    if not os.path.exists(config.data_file_path): 
        issues.append(f"Data file not found: {config.data_file_path}")

    if not os.path.exists(config.gpm_file_path): 
        issues.append(f"GPM file not found: {config.gpm_file_path}")
    if not isinstance(config.fixed_parameter_values, dict): 
        issues.append("`fixed_parameter_values` must be a dictionary.")
    if not isinstance(config.observed_variable_names, list): 
        issues.append("`observed_variable_names` must be a list.")
    if config.num_smoother_draws_for_fixed_params < 0: 
        issues.append("`num_smoother_draws_for_fixed_params` cannot be negative.")
    
    if issues: print("❌ Configuration Issues Found:"); [print(f"  - {issue}") for issue in issues]; return False

    print("✓ Calibration configuration valid."); return True

def load_data_for_calibration(config: PriorCalibrationConfig, csv_columns_to_select: Optional[List[str]] = None) -> Optional[Tuple[pd.DataFrame, Optional[pd.Index]]]:
    print(f"\n--- Loading Data from: {config.data_file_path} ---")
    try:
        df_loaded = pd.read_csv(config.data_file_path, usecols=['Date'] + csv_columns_to_select if csv_columns_to_select else None)
        time_index_from_data = None
        if 'Date' in df_loaded.columns:
            try:
                df_loaded['Date'] = pd.to_datetime(df_loaded['Date'])
                df_loaded.set_index('Date', inplace=True); time_index_from_data = df_loaded.index
                print("  Data has 'Date' column, set as index.")
            except Exception as e: print(f"  Warning: Could not process 'Date' column: {e}. Using as is.")
        final_df = pd.DataFrame()
        if not config.observed_variable_names:
            config.observed_variable_names = [col for col in df_loaded.columns if col != 'Date']
            print(f"  No 'observed_variable_names' in config, using from loaded data: {config.observed_variable_names}")
            final_df = df_loaded[config.observed_variable_names].copy() if config.observed_variable_names else df_loaded.copy()
        else:
            missing_cols = [col for col in config.observed_variable_names if col not in df_loaded.columns]
            if missing_cols: raise ValueError(f"GPM varobs '{missing_cols}' not in DataFrame. Available: {df_loaded.columns.tolist()}")
            final_df = df_loaded[config.observed_variable_names].copy()
        final_df = final_df.dropna()
        print(f"✓ Data loaded for GPM variables: {config.observed_variable_names}. Shape after dropna: {final_df.shape}")
        if final_df.isnull().values.any(): print("⚠ Warning: Loaded data still contains NaN values AFTER selection and dropna.")
        return final_df, time_index_from_data
    except Exception as e: print(f"✗ Error loading/processing data: {e}"); return None, None

def run_mcmc_workflow(config: PriorCalibrationConfig, data_df: pd.DataFrame) -> Optional[SmootherResults]: # Return type changed to SmootherResults
    print("\n--- Running MCMC Workflow (based on GPM priors) ---")
    print(f"  GPM file: {config.gpm_file_path}")
    print(f"  MCMC settings: {config.num_mcmc_warmup} warmup, {config.num_mcmc_samples} samples, {config.num_mcmc_chains} chains.")
    print(f"  Smoother draws: {config.num_smoother_draws}")
    try:
        start_time = time.time()
        results = complete_gpm_workflow_with_smoother_fixed(
            data=data_df, gpm_file=config.gpm_file_path,
            num_warmup=config.num_mcmc_warmup, num_samples=config.num_mcmc_samples, num_chains=config.num_mcmc_chains,
            use_gamma_init=config.use_gamma_init, gamma_scale_factor=config.gamma_scale_factor,
            num_extract_draws=config.num_smoother_draws,
            generate_plots=config.generate_plots and PLOTTING_AVAILABLE_UTILS, 
            hdi_prob_plot=config.plot_hdi_prob,
            show_plot_info_boxes=config.show_plot_info_boxes,
            plot_save_path=config.plot_save_path, save_plots=config.save_plots,
            custom_plot_specs=config.custom_plot_specs,
            variable_names_override=config.observed_variable_names,
            data_file_source_for_summary=config.data_file_path
        )
        print(f"✓ MCMC workflow evaluation step completed in {time.time() - start_time:.2f}s.")
        return results # Should be SmootherResults
    except Exception as e:
        import traceback
        print(f"✗ MCMC workflow evaluation step failed: {type(e).__name__}: {e}"); traceback.print_exc()
        return None

def _print_fixed_param_evaluation_summary(config: PriorCalibrationConfig, parsed_gpm_model: Optional[ReducedModel], eval_results: SmootherResults): # eval_results is SmootherResults
    print("\n" + "="*60 + "\n  FIXED-PARAMETER EVALUATION SUMMARY  \n" + "="*60)
    print(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPM File: {config.gpm_file_path}; Data File: {config.data_file_path}")
    print("\n--- GPM Model Structure (Brief) ---")
    print(f"  Observed Variables (in config): {config.observed_variable_names}")
    if parsed_gpm_model: print(f"  GPM Varobs: {parsed_gpm_model.gpm_observed_variables_original}; Core Variables: {parsed_gpm_model.core_variables}; Parameters: {parsed_gpm_model.parameters}; Trend Shocks: {parsed_gpm_model.trend_shocks}; Stationary Shocks: {parsed_gpm_model.stationary_shocks}")
    else: print("  (Parsed GPM model details not available for this summary printout)")
    print("\n--- Fixed Parameter Values Used ---")
    if config.fixed_parameter_values:
        for name, value in config.fixed_parameter_values.items():
            val_str = f"{value:.4f}" if isinstance(value, (float, np.float32, np.float64, jax.Array)) and (hasattr(value, 'ndim') and value.ndim == 0) else str(value)
            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.ndim > 0: val_str = f"array, shape: {value.shape}"
            print(f"  {name:<25}: {val_str}")
    else: print("  No fixed parameters specified.")
    if config.initial_state_prior_overrides:
        print("\n--- Initial State Prior Overrides Used ---")
        for name, override_spec in config.initial_state_prior_overrides.items(): print(f"  {name:<25}: {override_spec}")
    print("\n--- Evaluation Settings ---")
    print(f"  Smoother draws for fixed params: {config.num_smoother_draws_for_fixed_params}")
    print(f"  P0 Init (for fixed eval): Gamma-based = {config.use_gamma_init}, Scale = {config.gamma_scale_factor}")
    print("\n--- Evaluation Results ---")
    loglik = eval_results.log_likelihood # Access from SmootherResults
    if loglik is not None and np.isfinite(loglik): print(f"  Log-Likelihood: {loglik:.3f}")
    else: print(f"  Log-Likelihood: N/A or non-finite ({loglik})")
    print("="*60 + "\n")


def run_fixed_parameter_evaluation(config: PriorCalibrationConfig, data_df: pd.DataFrame, time_index_for_plots: Optional[pd.Index]) -> Optional[SmootherResults]: # Return type SmootherResults
    print("\n--- Running Evaluation at FIXED PARAMETERS ---")
    print(f"  GPM file: {config.gpm_file_path}, Fixed params: {config.fixed_parameter_values}, Initial State Overrides: {config.initial_state_prior_overrides}, Smoother draws: {config.num_smoother_draws_for_fixed_params}")
    if not config.fixed_parameter_values: print("✗ Error: `fixed_parameter_values` must be populated."); return None

    parsed_gpm_for_summary = None
    try:
        orchestrator = create_integration_orchestrator(config.gpm_file_path, strict_validation=False)
        parsed_gpm_for_summary = orchestrator.reduced_model
    except Exception as e_parse: print(f"  Warning: Could not parse GPM file for summary: {e_parse}")

    start_time_eval = time.time() # Define start time for duration calculation
    try:
        eval_results = evaluate_gpm_at_parameters( 
            gpm_file_path=config.gpm_file_path,
            y=data_df, 
            param_values=config.fixed_parameter_values,
            initial_state_prior_overrides=config.initial_state_prior_overrides,
            num_sim_draws=config.num_smoother_draws_for_fixed_params, 
            plot_results=config.generate_plots and PLOTTING_AVAILABLE_UTILS, 
            plot_default_observed_vs_trend_components=config.plot_default_observed_vs_trend_components, 
            custom_plot_specs=config.custom_plot_specs, 
            variable_names=config.observed_variable_names, 
            use_gamma_init_for_test=config.use_gamma_init, 
            gamma_init_scaling=config.gamma_scale_factor,
            trend_P0_var_scale=config.trend_P0_var_scale_fixed_eval,
            var_P0_var_scale=config.var_P0_var_scale_fixed_eval,
            save_plots_path_prefix=os.path.join(config.plot_save_path, "fixed_param_evaluation", "plot") if config.save_plots and config.plot_save_path else None,
            show_plot_info_boxes=config.show_plot_info_boxes 
        )
        print(f"✓ Fixed-parameter evaluation completed in {time.time() - start_time_eval:.2f}s.")
        
        gpm_model_from_results = eval_results.gpm_model if eval_results else parsed_gpm_for_summary
        _print_fixed_param_evaluation_summary(config, gpm_model_from_results, eval_results if eval_results else SmootherResults(observed_data=np.array([]), observed_variable_names=[], trend_draws=np.array([]), trend_names=[], trend_stats={}, stationary_draws=np.array([]), stationary_names=[], stationary_stats={}))


        loglik_val = eval_results.log_likelihood if eval_results else None
        if not (loglik_val is not None and np.isfinite(loglik_val)):
            print("  Warning: LogLik not available or non-finite from eval_results."); 
        
        return eval_results 

    except Exception as e: import traceback; print(f"✗ Fixed-parameter evaluation failed: {type(e).__name__}: {e}"); traceback.print_exc(); return None


def run_parameter_sensitivity_workflow(base_config: PriorCalibrationConfig, data_df: pd.DataFrame, time_index_for_plots: Optional[pd.Index], parameter_name_to_vary: str, values_to_test: List[float]) -> Dict[str, Any]:
    print(f"\n--- Sensitivity Study for Parameter: '{parameter_name_to_vary}' ---")
    if parameter_name_to_vary not in base_config.fixed_parameter_values:
        err_msg = f"Parameter '{parameter_name_to_vary}' not in base_config.fixed_parameter_values. Available: {list(base_config.fixed_parameter_values.keys())}"
        print(f"✗ Error: {err_msg}"); return {'error': err_msg, 'parameter_name': parameter_name_to_vary, 'values_tested': values_to_test, 'log_likelihoods': []}
    study_results = {'parameter_name': parameter_name_to_vary, 'values_tested': [], 'log_likelihoods': [], 'run_status': [], 'all_eval_results': []}
    
    # Use the original DataFrame for sensitivity evaluation
    # evaluate_gpm_at_parameters handles DataFrame input directly.
    # y_for_eval = data_df[base_config.observed_variable_names].values 
    # y_jax_for_eval = jnp.asarray(y_for_eval, dtype=_DEFAULT_DTYPE)

    for i, p_val in enumerate(values_to_test):
        print(f"\n  Test {i+1}/{len(values_to_test)}: {parameter_name_to_vary} = {p_val}")
        current_fixed_params = base_config.fixed_parameter_values.copy(); current_fixed_params[parameter_name_to_vary] = p_val
        study_results['values_tested'].append(p_val)
        try:
            # eval_results will be a SmootherResults object
            eval_results_sensitivity_point = evaluate_gpm_at_parameters( 
                gpm_file_path=base_config.gpm_file_path, 
                y=data_df, # Pass DataFrame directly
                param_values=current_fixed_params,
                initial_state_prior_overrides=base_config.initial_state_prior_overrides,
                num_sim_draws=base_config.num_smoother_draws_for_fixed_params if base_config.plot_sensitivity_point_results else 0,
                plot_results=False, 
                use_gamma_init_for_test=base_config.use_gamma_init, 
                gamma_init_scaling=base_config.gamma_scale_factor,
                variable_names=base_config.observed_variable_names,
                trend_P0_var_scale=base_config.trend_P0_var_scale_fixed_eval, 
                var_P0_var_scale=base_config.var_P0_var_scale_fixed_eval,
                hdi_prob=base_config.plot_hdi_prob, # Pass hdi_prob for consistency
                show_plot_info_boxes=base_config.show_plot_info_boxes # Pass info box flag
            )
            study_results['all_eval_results'].append(eval_results_sensitivity_point)

            loglik_val_sens = eval_results_sensitivity_point.log_likelihood if eval_results_sensitivity_point else None
            if loglik_val_sens is not None and np.isfinite(loglik_val_sens):
                loglik_float = float(loglik_val_sens)
                study_results['log_likelihoods'].append(loglik_float)
                study_results['run_status'].append('success')
                print(f"    ✓ LogLik: {loglik_float:.3f}")

                if base_config.plot_sensitivity_point_results and PLOTTING_AVAILABLE_UTILS and eval_results_sensitivity_point and eval_results_sensitivity_point.n_draws > 0:
                    print(f"    Generating plots for sensitivity point {i+1} ({parameter_name_to_vary}={p_val})...")
                    point_save_path_prefix = None
                    if base_config.save_plots and base_config.plot_save_path:
                        sens_point_dir = os.path.join(base_config.plot_save_path, "sensitivity_points", f"{parameter_name_to_vary}_{i+1}")
                        os.makedirs(sens_point_dir, exist_ok=True)
                        point_save_path_prefix = os.path.join(sens_point_dir, "plot") # Prefix for individual plot files

                    # Plot Smoother Results (Trends & Stationary) for this point
                    if callable(plot_time_series_with_uncertainty) and eval_results_sensitivity_point.trend_draws.shape[2] > 0:
                        fig_trends = plot_time_series_with_uncertainty(
                            eval_results_sensitivity_point.trend_draws, 
                            variable_names=eval_results_sensitivity_point.trend_names, 
                            hdi_prob=base_config.plot_hdi_prob, 
                            title_prefix=f"Trend Comp. ({parameter_name_to_vary}={p_val:.4g})", 
                            show_info_box=base_config.show_plot_info_boxes, 
                            time_index=eval_results_sensitivity_point.time_index
                        )
                        if fig_trends and point_save_path_prefix: fig_trends.savefig(f"{point_save_path_prefix}_trends.png", dpi=150, bbox_inches='tight'); plt.close(fig_trends)

                    # Plot Observed vs Trend Component for this point
                    if base_config.plot_default_observed_vs_trend_components:
                        plot_observed_vs_trend_component(
                            config=base_config, 
                            data_df=data_df, 
                            time_index_for_plots=eval_results_sensitivity_point.time_index, 
                            eval_results=eval_results_sensitivity_point # This is SmootherResults
                        )
                        # plot_observed_vs_trend_component handles saving internally if path prefix in eval_results.
                        # We might need to pass point_save_path_prefix to it or modify it.
                        # For now, assuming it plots or the main eval_gpm_at_parameters handles it.
                        # The `plot_observed_vs_trend_component` is typically called from within `evaluate_gpm_at_parameters` if `plot_results` is true.
                        # Here, we are calling it directly for sensitivity points if `plot_sensitivity_point_results` is true.

                    # Plot Custom Specs for this point
                    actual_sensitivity_custom_specs = base_config.sensitivity_plot_custom_specs
                    if actual_sensitivity_custom_specs and callable(plot_custom_series_comparison):
                        for spec_idx, spec_dict_item in enumerate(actual_sensitivity_custom_specs):
                            # CORRECTED CALL: Pass SmootherResults object
                            fig_custom = plot_custom_series_comparison(
                                plot_title=spec_dict_item.get("title", f"Custom Plot {spec_idx+1}") + f" ({parameter_name_to_vary}={p_val:.4g})",
                                series_specs=spec_dict_item.get("series_to_plot", []),
                                results=eval_results_sensitivity_point, # Pass the SmootherResults object
                                save_path=point_save_path_prefix, # Pass the save path prefix
                                show_info_box=base_config.show_plot_info_boxes
                            )
                            # plot_custom_series_comparison handles its own saving and closing
                            # if fig_custom and point_save_path_prefix: # Already handled by the function
                            #    pass
            else:
                study_results['log_likelihoods'].append(np.nan)
                study_results['run_status'].append('failed_or_non_finite_loglik')
                print("    Warning: LogLik N/A or non-finite. Point evaluation skipped.")
        except Exception as e:
            import traceback
            print(f"    ✗ Error during sensitivity evaluation for {parameter_name_to_vary}={p_val}: {type(e).__name__}: {e}")
            study_results['log_likelihoods'].append(np.nan)
            study_results['run_status'].append(f'error: {type(e).__name__}')

    best_idx = np.nanargmax(study_results['log_likelihoods']) if study_results['log_likelihoods'] else -1
    if best_idx != -1 and np.isfinite(study_results['log_likelihoods'][best_idx]):
        study_results['best_parameter_value'] = study_results['values_tested'][best_idx]
        study_results['best_log_likelihood'] = study_results['log_likelihoods'][best_idx]
    else:
        study_results['best_parameter_value'] = np.nan
        study_results['best_log_likelihood'] = np.nan

    print(f"\n--- Sensitivity Study for '{parameter_name_to_vary}' Complete ---")
    if np.isfinite(study_results['best_log_likelihood']):
         print(f"Best value found: {study_results['best_parameter_value']:.4g} (LogLik: {study_results['best_log_likelihood']:.3f})")
    else:
         print("No finite log-likelihoods found in sensitivity study.")

    if base_config.generate_plots and PLOTTING_AVAILABLE_UTILS:
        print("Generating overall sensitivity plot...")
        sensitivity_plot_path_dir = base_config.plot_save_path
        if base_config.save_plots and sensitivity_plot_path_dir:
            os.makedirs(sensitivity_plot_path_dir, exist_ok=True)
            sensitivity_plot_file = os.path.join(sensitivity_plot_path_dir, f"sensitivity_plot_{parameter_name_to_vary}.png")
        else:
            sensitivity_plot_file = None
        plot_sensitivity_study_results(study_results, save_path=sensitivity_plot_file)

    return study_results


def plot_sensitivity_study_results(sensitivity_output: Dict[str, Any], save_path: Optional[str]=None):
    if not PLOTTING_AVAILABLE_UTILS: print("Plotting disabled - plot_sensitivity_study_results skipped."); return
    param_name = sensitivity_output.get('parameter_name', 'Parameter'); param_values = np.asarray(sensitivity_output.get('values_tested', [])); log_likelihoods = np.asarray(sensitivity_output.get('log_likelihoods', []))
    if param_values.ndim > 1 or (param_values.size > 0 and not np.isscalar(param_values[0])): print(f"Cannot plot sensitivity for non-scalar '{param_name}'."); return
    if param_values.size == 0 or log_likelihoods.size == 0 or param_values.size != log_likelihoods.size: print("Insufficient data for sensitivity plot."); return
    finite_mask = np.isfinite(log_likelihoods)
    if not np.any(finite_mask): print("No finite log-likelihoods to plot."); return
    p_plot = param_values[finite_mask]; ll_plot = log_likelihoods[finite_mask]
    if len(p_plot) == 0 : print("All log-likelihoods NaN, cannot plot."); return
    sort_idx = np.argsort(p_plot); p_plot_sorted = p_plot[sort_idx]; ll_plot_sorted = ll_plot[sort_idx]
    fig_sens = plt.figure()
    plt.plot(p_plot_sorted, ll_plot_sorted, marker='o', linestyle='-', color='royalblue')
    plt.xlabel(f"Value of {param_name}"); plt.ylabel("Log-likelihood"); plt.title(f"Sensitivity of Log-likelihood to {param_name}"); plt.grid(True, linestyle=':', alpha=0.7)
    best_val = sensitivity_output.get('best_parameter_value'); best_ll = sensitivity_output.get('best_log_likelihood')
    if best_val is not None and best_ll is not None and np.isfinite(best_ll) and np.isscalar(best_val): plt.scatter([best_val], [best_ll], color='red', s=100, zorder=5, label=f"Best: {best_val:.4g} (LL: {best_ll:.2f})"); plt.legend()
    plt.tight_layout()
    if save_path: 
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        fig_sens.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"Saved sensitivity plot to {save_path}")
    plt.close(fig_sens) 

def plot_observed_vs_trend_component(
    config: PriorCalibrationConfig,
    data_df: pd.DataFrame,
    time_index_for_plots: Optional[pd.Index],
    eval_results: SmootherResults # Changed to SmootherResults
):
    if not config.generate_plots or not PLOTTING_AVAILABLE_UTILS:
        print("Skipping observed vs trend component plots: plotting disabled.")
        return

    if not eval_results or eval_results.n_draws == 0 :
        print("Skipping observed vs trend component plots: no trend draws available in eval_results.")
        return

    gpm_model_eval = eval_results.gpm_model
    if gpm_model_eval is None or not hasattr(gpm_model_eval, 'reduced_measurement_equations'):
        print("Skipping observed vs trend component plots: GPM model or MEs not found in eval_results.")
        return
        
    reduced_meas_eqs = gpm_model_eval.reduced_measurement_equations
    trend_names_gpm = eval_results.trend_names
    obs_var_names_actual = eval_results.observed_variable_names # Use names from SmootherResults for consistency
    
    plot_path_full_prefix = None
    if config.save_plots and config.plot_save_path:
        plot_subdir = os.path.join(config.plot_save_path, "obs_vs_trend_component")
        os.makedirs(plot_subdir, exist_ok=True)
        plot_path_full_prefix = os.path.join(plot_subdir, "plot")


    print("\nGenerating Observed vs. Trend Component Plots...")
    plotted_count = 0
    utils = SymbolicReducerUtils()

    for obs_name in obs_var_names_actual:
        if obs_name not in reduced_meas_eqs:
            continue

        expr = reduced_meas_eqs[obs_name]
        potential_trend_terms_lag0 = []
        for var_key, coeff_str in expr.terms.items():
            var_name, lag = utils._parse_var_key_for_rules(var_key)
            if lag == 0 and var_name in trend_names_gpm:
                potential_trend_terms_lag0.append(var_name)

        trend_component_name_for_plot = None
        # if len(potential_trend_terms_lag0) == 1:
        #      trend_component_name_for_plot = potential_trend_terms_lag0[0]
        # elif len(potential_trend_terms_lag0) > 1:
        #     trend_component_name_for_plot = potential_trend_terms_lag0[0]
        if len(potential_trend_terms_lag0) == 1:
            trend_component_name_for_plot = potential_trend_terms_lag0[0]
        elif len(potential_trend_terms_lag0) > 1:
            # Smart selection: prefer full trends over idio trends
            for trend_name in potential_trend_terms_lag0:
                if 'full_trend' in trend_name or 'short_trend' in trend_name:
                    trend_component_name_for_plot = trend_name
                    break
            else:
                trend_component_name_for_plot = potential_trend_terms_lag0[0]
        if trend_component_name_for_plot:
            series_specs_for_this_plot = [
                {'type': 'observed', 'name': obs_name, 'label': f'Observed {obs_name}', 'style': 'k-'},
                {'type': 'trend', 'name': trend_component_name_for_plot, 'label': f'Trend {trend_component_name_for_plot}', 'show_hdi': True, 'color': 'blue'}
            ]
            plot_title = f"Observed {obs_name} vs. Trend Component {trend_component_name_for_plot}"

            # Call the generic custom plotting function, which now takes SmootherResults
            # The save_path here is a prefix for the filename
            fig_custom = plot_custom_series_comparison(
                plot_title=plot_title, 
                series_specs=series_specs_for_this_plot,
                results=eval_results, # Pass the full SmootherResults object
                save_path=plot_path_full_prefix, # Pass the directory and filename prefix
                show_info_box=config.show_plot_info_boxes
            )
            # plot_custom_series_comparison now handles saving and plt.close()
            if fig_custom is None and plot_path_full_prefix: # Indicates saved and closed
                 plotted_count +=1
            elif fig_custom is not None: # Not saved, or save failed, figure returned
                 plt.close(fig_custom) # Ensure it's closed if not handled by plotting function

    print(f"Generated {plotted_count} Observed vs. Trend Component Plots.")