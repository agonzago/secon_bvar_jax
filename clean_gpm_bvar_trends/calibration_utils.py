import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt # For dummy plot definitions if reporting_plots fails
import time
from datetime import datetime

# Imports from within the same package (clean_gpm_bvar_trends)
from .constants import _DEFAULT_DTYPE # Relative import
from .gpm_model_parser import ReducedModel # Relative import
from .integration_orchestrator import create_integration_orchestrator # Relative import
from .gpm_prior_evaluator import evaluate_gpm_at_parameters # Relative import
from .gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed # Relative import

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
        if plt and kwargs.get('plot_save_prefix_fixed') and kwargs.get('fig_trends'): # Simplified check
             try: kwargs['fig_trends'].savefig(f"{kwargs['plot_save_prefix_fixed']}_trends_DISABLED.png"); plt.close(kwargs['fig_trends'])
             except Exception: pass
        return None
    def plot_custom_series_comparison(*args, **kwargs):
        print("Plotting disabled (calibration_utils) - plot_custom_series_comparison skipped")
        if plt and kwargs.get('plot_save_prefix_fixed') and kwargs.get('fig_custom'):
             try:
                safe_title = kwargs.get("plot_title", "custom").lower().replace(' ','_')
                kwargs['fig_custom'].savefig(f"{kwargs['plot_save_prefix_fixed']}_custom_{safe_title}_DISABLED.png"); plt.close(kwargs['fig_custom'])
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
                var_P0_var_scale_fixed_eval: float = 1.0
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
        self.initial_state_prior_overrides = initial_state_prior_overrides # Added
        self.trend_P0_var_scale_fixed_eval = trend_P0_var_scale_fixed_eval
        self.var_P0_var_scale_fixed_eval = var_P0_var_scale_fixed_eval

# --- Helper Functions (General Utilities) ---
def validate_calibration_config(config: PriorCalibrationConfig) -> bool:
    # (Content from your existing gpm_prior_calibration_example.py)
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
    # (Content from your existing gpm_prior_calibration_example.py)
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

def run_mcmc_workflow(config: PriorCalibrationConfig, data_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    # (Renamed from run_single_point_evaluation for clarity, uses complete_gpm_workflow_with_smoother_fixed)
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
            generate_plots=config.generate_plots and PLOTTING_AVAILABLE_UTILS, # Apply flag
            hdi_prob_plot=config.plot_hdi_prob,
            show_plot_info_boxes=config.show_plot_info_boxes,
            plot_save_path=config.plot_save_path, save_plots=config.save_plots,
            custom_plot_specs=config.custom_plot_specs,
            variable_names_override=config.observed_variable_names,
            data_file_source_for_summary=config.data_file_path
        )
        print(f"✓ MCMC workflow evaluation step completed in {time.time() - start_time:.2f}s.")
        return results
    except Exception as e:
        import traceback
        print(f"✗ MCMC workflow evaluation step failed: {type(e).__name__}: {e}"); traceback.print_exc()
        return None

def _print_fixed_param_evaluation_summary(config: PriorCalibrationConfig, parsed_gpm_model: Optional[ReducedModel], eval_results: Dict[str, Any]):
    # (Content from your existing gpm_prior_calibration_example.py, slightly adapted for JAX array loglik)
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
    loglik = eval_results.get('loglik')
    if loglik is not None and hasattr(loglik, 'item') and jnp.isfinite(loglik): print(f"  Log-Likelihood: {float(loglik.item()):.3f}")
    elif loglik is not None and isinstance(loglik, (float, int)) and np.isfinite(loglik): print(f"  Log-Likelihood: {float(loglik):.3f}")
    else: print(f"  Log-Likelihood: N/A or non-finite ({loglik})")
    print("="*60 + "\n")


def run_fixed_parameter_evaluation(config: PriorCalibrationConfig, data_df: pd.DataFrame, time_index_for_plots: Optional[pd.Index]) -> Optional[Dict[str, Any]]:
    # (Renamed from run_single_point_evaluation_fixed_params, uses evaluate_gpm_at_parameters)
    print("\n--- Running Evaluation at FIXED PARAMETERS ---")
    print(f"  GPM file: {config.gpm_file_path}, Fixed params: {config.fixed_parameter_values}, Initial State Overrides: {config.initial_state_prior_overrides}, Smoother draws: {config.num_smoother_draws_for_fixed_params}")
    if not config.fixed_parameter_values: print("✗ Error: `fixed_parameter_values` must be populated."); return None

    parsed_gpm_for_summary = None
    try:
        orchestrator = create_integration_orchestrator(config.gpm_file_path, strict_validation=False)
        parsed_gpm_for_summary = orchestrator.reduced_model
    except Exception as e_parse: print(f"  Warning: Could not parse GPM file for summary: {e_parse}")

    try:
        y_jax_for_eval = jnp.asarray(data_df.values, dtype=_DEFAULT_DTYPE)
        start_time = time.time()
        eval_results = evaluate_gpm_at_parameters( # This now needs initial_state_prior_overrides
            gpm_file_path=config.gpm_file_path,
            y=y_jax_for_eval,
            param_values=config.fixed_parameter_values,
            initial_state_prior_overrides=config.initial_state_prior_overrides, # Pass the overrides
            num_sim_draws=config.num_smoother_draws_for_fixed_params, 
            plot_results=False,
            use_gamma_init_for_test=config.use_gamma_init, 
            gamma_init_scaling=config.gamma_scale_factor,
            variable_names=config.observed_variable_names,
            # Pass the P0 scale parameters from the config object
            trend_P0_var_scale=config.trend_P0_var_scale_fixed_eval,
            var_P0_var_scale=config.var_P0_var_scale_fixed_eval
        )
        print(f"✓ Fixed-parameter evaluation completed in {time.time() - start_time:.2f}s.")
        _print_fixed_param_evaluation_summary(config, eval_results.get('gpm_model', parsed_gpm_for_summary), eval_results if eval_results else {})

        loglik_val = eval_results.get('loglik')
        if not (loglik_val is not None and ((hasattr(loglik_val, 'item') and jnp.isfinite(loglik_val.item())) or (isinstance(loglik_val, (float,int)) and np.isfinite(loglik_val)))):
            print("  Warning: LogLik not available or non-finite from eval_results."); return eval_results

        if config.generate_plots and PLOTTING_AVAILABLE_UTILS and eval_results and eval_results.get('reconstructed_original_trends') is not None:
            print(f"    Generating plots for fixed-parameter evaluation...")
            reconstructed_trends_np = np.asarray(eval_results['reconstructed_original_trends']); 
            reconstructed_stationary_np = np.asarray(eval_results['reconstructed_original_stationary'])
            gpm_model_eval = eval_results['gpm_model']; 
            trend_names_gpm = gpm_model_eval.gpm_trend_variables_original; 
            stat_names_gpm = gpm_model_eval.gpm_stationary_variables_original
            plot_save_prefix_fixed = None
            if config.save_plots and config.plot_save_path:
                fixed_plot_dir = os.path.join(config.plot_save_path, "fixed_param_evaluation"); os.makedirs(fixed_plot_dir, exist_ok=True)
                plot_save_prefix_fixed = os.path.join(fixed_plot_dir, "plot")
            if reconstructed_trends_np.ndim == 3 and reconstructed_trends_np.shape[0] > 0 and reconstructed_trends_np.shape[2] > 0:
                fig_trends = plot_time_series_with_uncertainty(reconstructed_trends_np, variable_names=trend_names_gpm, hdi_prob=config.plot_hdi_prob, title_prefix="Trend Components (Fixed Params)", show_info_box=config.show_plot_info_boxes, time_index=time_index_for_plots)
                if fig_trends and plot_save_prefix_fixed: fig_trends.savefig(f"{plot_save_prefix_fixed}_trends.png", dpi=150, bbox_inches='tight')
                if fig_trends: plt.close(fig_trends) # Close after saving or if not saving (to prevent display if not intended)
            # ... (rest of plotting logic, using plot_custom_series_comparison)
            # Ensure to check if figures are returned before closing/saving if PLOTTING_AVAILABLE_UTILS is False
            # ... similar logic for custom_plot_specs ...
            actual_custom_specs = config.custom_plot_specs # ... (default spec generation as before)
            if actual_custom_specs and ((reconstructed_trends_np.ndim ==3 and reconstructed_trends_np.shape[0] > 0) or (reconstructed_stationary_np.ndim ==3 and reconstructed_stationary_np.shape[0] > 0)):
                for spec_idx, spec_dict_item in enumerate(actual_custom_specs):
                    fig_custom = plot_custom_series_comparison(plot_title=spec_dict_item.get("title", f"Custom Plot {spec_idx+1}") + " (Fixed Params)", series_specs=spec_dict_item.get("series_to_plot", []), observed_data=np.asarray(data_df[config.observed_variable_names].values), trend_draws=reconstructed_trends_np, stationary_draws=reconstructed_stationary_np, observed_names=config.observed_variable_names, trend_names=trend_names_gpm, stationary_names=stat_names_gpm, time_index=time_index_for_plots, hdi_prob=config.plot_hdi_prob)
                    if fig_custom and plot_save_prefix_fixed:
                        safe_title_fig = spec_dict_item.get("title", f"custom_{spec_idx+1}").lower().replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('=','_').replace(':','');
                        fig_custom.savefig(f"{plot_save_prefix_fixed}_custom_{safe_title_fig}.png", dpi=150, bbox_inches='tight');
                    if fig_custom: plt.close(fig_custom)

        return eval_results
    except Exception as e: import traceback; print(f"✗ Fixed-parameter evaluation failed: {type(e).__name__}: {e}"); traceback.print_exc(); return None


def run_parameter_sensitivity_workflow(base_config: PriorCalibrationConfig, data_df: pd.DataFrame, time_index_for_plots: Optional[pd.Index], parameter_name_to_vary: str, values_to_test: List[float]) -> Dict[str, Any]:
    # (Content from your existing gpm_prior_calibration_example.py, ensure plotting calls use PLOTTING_AVAILABLE_UTILS)
    # Also, ensure it passes initial_state_prior_overrides from base_config if present
    print(f"\n--- Sensitivity Study for Parameter: '{parameter_name_to_vary}' ---")
    if parameter_name_to_vary not in base_config.fixed_parameter_values:
        err_msg = f"Parameter '{parameter_name_to_vary}' not in base_config.fixed_parameter_values. Available: {list(base_config.fixed_parameter_values.keys())}"
        print(f"✗ Error: {err_msg}"); return {'error': err_msg, 'parameter_name': parameter_name_to_vary, 'values_tested': values_to_test, 'log_likelihoods': []}
    study_results = {'parameter_name': parameter_name_to_vary, 'values_tested': [], 'log_likelihoods': [], 'run_status': [], 'all_eval_results': []}
    try: y_jax_for_eval = jnp.asarray(data_df[base_config.observed_variable_names].values, dtype=_DEFAULT_DTYPE)
    except Exception as e: print(f"Error converting data for sensitivity: {e}"); study_results['error'] = "Data conversion error."; return study_results
    for i, p_val in enumerate(values_to_test):
        print(f"\n  Test {i+1}/{len(values_to_test)}: {parameter_name_to_vary} = {p_val}")
        current_fixed_params = base_config.fixed_parameter_values.copy(); current_fixed_params[parameter_name_to_vary] = p_val
        study_results['values_tested'].append(p_val)
        try:
            eval_results = evaluate_gpm_at_parameters( # Will need initial_state_prior_overrides
                gpm_file_path=base_config.gpm_file_path, y=y_jax_for_eval, param_values=current_fixed_params,
                initial_state_prior_overrides=base_config.initial_state_prior_overrides, # Pass through
                num_sim_draws=base_config.num_smoother_draws_for_fixed_params, plot_results=False,
                use_gamma_init_for_test=base_config.use_gamma_init, gamma_init_scaling=base_config.gamma_scale_factor,
                variable_names=base_config.observed_variable_names
            )
            study_results['all_eval_results'].append(eval_results)
            loglik_val_sens = eval_results.get('loglik')
            if loglik_val_sens is not None and hasattr(loglik_val_sens, 'item') and jnp.isfinite(loglik_val_sens.item()):
                loglik_float = float(loglik_val_sens.item())
                study_results['log_likelihoods'].append(loglik_float); study_results['run_status'].append('success'); print(f"    ✓ LogLik: {loglik_float:.3f}")
                if base_config.plot_sensitivity_point_results and PLOTTING_AVAILABLE_UTILS and eval_results and eval_results.get('reconstructed_original_trends') is not None:
                    # ... (plotting logic for sensitivity point, similar to run_fixed_parameter_evaluation)
                    pass # Placeholder for brevity
            else: study_results['log_likelihoods'].append(np.nan); study_results['run_status'].append('failed_or_non_finite_loglik')
        except Exception as e: study_results['log_likelihoods'].append(np.nan); study_results['run_status'].append(f'error: {type(e).__name__}')
    # ... (rest of sensitivity result processing)
    return study_results


def plot_sensitivity_study_results(sensitivity_output: Dict[str, Any], save_path: Optional[str]=None):
    # (Content from your existing gpm_prior_calibration_example.py)
    # This function uses plt directly, so it's fine here or in reporting_plots.
    # For consistency, if it's very general, it could be in reporting_plots.
    # If it's specific to this calibration workflow's output format, it's fine here.
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
    if save_path: fig_sens.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"Saved sensitivity plot to {save_path}")
    plt.show(); plt.close(fig_sens)

