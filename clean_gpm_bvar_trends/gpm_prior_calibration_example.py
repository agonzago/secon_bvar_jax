# clean_gpm_bvar_trends/gpm_prior_calibration_example.py

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import time

# Ensure imports from your project structure are correct
from gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed # Core workflow
from gpm_prior_evaluator import evaluate_gpm_at_parameters # For sensitivity
from constants import _DEFAULT_DTYPE

# Configure JAX (consistent with other modules)
if "XLA_FLAGS" not in os.environ: # Set only if not already set externally
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1" # Default to 1 for calibration, can be overridden
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

class PriorCalibrationConfig:
    """Configuration for prior calibration/evaluation workflow."""
    def __init__(self,
                 data_file_path: str = 'data/sim_data.csv',
                 gpm_file_path: str = 'models/test_model.gpm',
                 observed_variable_names: Optional[List[str]] = None, # Will default from data if None
                 fixed_parameter_values: Optional[Dict[str, Any]] = None,
                 num_mcmc_warmup: int = 50, # Keep low for single evaluation
                 num_mcmc_samples: int = 100, # Keep low for single evaluation
                 num_mcmc_chains: int = 1,
                 num_smoother_draws: int = 50,
                 use_gamma_init: bool = True,
                 gamma_scale_factor: float = 1.0,
                 generate_plots: bool = True,
                 plot_hdi_prob: float = 0.9,
                 show_plot_info_boxes: bool = False,
                 plot_save_path: Optional[str] = "prior_calibration_plots",
                 save_plots: bool = False,
                 custom_plot_specs: Optional[List[Dict[str, Any]]] = None
                 ):
        self.data_file_path = data_file_path
        self.gpm_file_path = gpm_file_path
        self.observed_variable_names = observed_variable_names if observed_variable_names is not None else []
        self.fixed_parameter_values = fixed_parameter_values if fixed_parameter_values is not None else {}

        # Workflow execution parameters
        self.num_mcmc_warmup = num_mcmc_warmup
        self.num_mcmc_samples = num_mcmc_samples
        self.num_mcmc_chains = num_mcmc_chains
        self.num_smoother_draws = num_smoother_draws
        self.use_gamma_init = use_gamma_init
        self.gamma_scale_factor = gamma_scale_factor
        
        # Plotting
        self.generate_plots = generate_plots
        self.plot_hdi_prob = plot_hdi_prob
        self.show_plot_info_boxes = show_plot_info_boxes
        self.plot_save_path = plot_save_path
        self.save_plots = save_plots
        self.custom_plot_specs = custom_plot_specs


def validate_calibration_config(config: PriorCalibrationConfig) -> bool:
    """Validates the prior calibration configuration."""
    print("\n--- Validating Prior Calibration Configuration ---")
    issues = []
    if not os.path.exists(config.data_file_path):
        issues.append(f"Data file not found: {config.data_file_path}")
    if not os.path.exists(config.gpm_file_path):
        issues.append(f"GPM file not found: {config.gpm_file_path}")
    if not isinstance(config.fixed_parameter_values, dict): # Can be empty if all from prior
        issues.append("`fixed_parameter_values` must be a dictionary.")
    if not isinstance(config.observed_variable_names, list):
        issues.append("`observed_variable_names` must be a list (can be empty to infer from data).")

    if issues:
        print("❌ Configuration Issues Found:")
        for issue in issues: print(f"  - {issue}")
        return False
    print("✓ Calibration configuration valid.")
    return True

def load_data_for_calibration(config: PriorCalibrationConfig) -> Optional[pd.DataFrame]:
    """Loads data and selects observed variables."""
    print(f"\n--- Loading Data from: {config.data_file_path} ---")
    try:
        dta = pd.read_csv(config.data_file_path)
        if 'Date' in dta.columns:
            try:
                dta['Date'] = pd.to_datetime(dta['Date'])
                dta.set_index('Date', inplace=True)
                print("  Data has 'Date' column, set as index.")
            except Exception as e:
                print(f"  Warning: Could not process 'Date' column: {e}. Using as is.")

        if not config.observed_variable_names: # If empty, use all columns
            config.observed_variable_names = dta.columns.tolist()
            print(f"  No 'observed_variable_names' specified, using all columns from data: {config.observed_variable_names}")
        else: # Check if specified columns exist
            missing_cols = [col for col in config.observed_variable_names if col not in dta.columns]
            if missing_cols:
                raise ValueError(f"Specified observed_variable_names not found in data: {missing_cols}. Available: {dta.columns.tolist()}")
            dta = dta[config.observed_variable_names]

        print(f"✓ Data loaded for variables: {config.observed_variable_names}. Shape: {dta.shape}")
        if dta.isnull().values.any():
            print("⚠ Warning: Loaded data contains NaN values.")
        return dta
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def run_single_point_evaluation(config: PriorCalibrationConfig,
                                data_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Runs the GPM workflow for a single set of fixed parameters (or priors if params are empty).
    This function uses the full workflow, which implies MCMC estimation based on GPM priors.
    `config.fixed_parameter_values` is NOT used to fix MCMC parameters in this function.
    """
    print("\n--- Running Single Evaluation via Full Workflow (MCMC based on GPM priors) ---")
    print(f"  GPM file: {config.gpm_file_path}")
    print(f"  MCMC settings: {config.num_mcmc_warmup} warmup, {config.num_mcmc_samples} samples, {config.num_mcmc_chains} chains.")
    print(f"  Smoother draws: {config.num_smoother_draws}")

    try:
        start_time = time.time()
        results = complete_gpm_workflow_with_smoother_fixed(
            data=data_df, 
            gpm_file=config.gpm_file_path,
            num_warmup=config.num_mcmc_warmup,
            num_samples=config.num_mcmc_samples,
            num_chains=config.num_mcmc_chains,
            use_gamma_init=config.use_gamma_init,
            gamma_scale_factor=config.gamma_scale_factor,
            num_extract_draws=config.num_smoother_draws,
            generate_plots=config.generate_plots,
            hdi_prob_plot=config.plot_hdi_prob,
            show_plot_info_boxes=config.show_plot_info_boxes,
            plot_save_path=config.plot_save_path,
            save_plots=config.save_plots,
            custom_plot_specs=config.custom_plot_specs,
            variable_names_override=config.observed_variable_names 
        )
        print(f"✓ Workflow evaluation step completed in {time.time() - start_time:.2f}s.")
        if results and results.get('mcmc_object') and hasattr(results['mcmc_object'], 'get_samples'):
            mcmc_samples = results['mcmc_object'].get_samples()
            if 'potential_energy' in mcmc_samples: # NumPyro specific
                 print("  MCMC Mean Log posterior (potential_energy): ", np.mean(mcmc_samples['potential_energy']))
            elif 'lp__' in mcmc_samples: # Stan specific
                 print("  MCMC Mean Log posterior (lp__): ", np.mean(mcmc_samples['lp__']))
        return results
    except Exception as e:
        import traceback
        print(f"✗ Workflow evaluation step failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

def run_parameter_sensitivity_workflow(
    base_config: PriorCalibrationConfig,
    data_df: pd.DataFrame,
    parameter_name_to_vary: str,
    values_to_test: List[float]
) -> Dict[str, Any]:
    """
    Performs a sensitivity analysis by running direct evaluations at fixed parameter points.
    Uses `evaluate_gpm_at_parameters`.
    `base_config.fixed_parameter_values` must contain ALL necessary fixed values for the GPM,
    and this function will vary one of them.
    """
    print(f"\n--- Sensitivity Study for Parameter: '{parameter_name_to_vary}' ---")
    print("    (Note: This uses direct evaluation at fixed parameters via evaluate_gpm_at_parameters)")

    if parameter_name_to_vary not in base_config.fixed_parameter_values:
        err_msg = (f"Parameter '{parameter_name_to_vary}' not in base_config.fixed_parameter_values "
                   f"which is required for this sensitivity type. Available: {list(base_config.fixed_parameter_values.keys())}")
        print(f"✗ Error: {err_msg}")
        return {'error': err_msg, 'parameter_name': parameter_name_to_vary, 'values_tested': values_to_test, 'log_likelihoods': []}

    study_results = {
        'parameter_name': parameter_name_to_vary,
        'values_tested': [],
        'log_likelihoods': [],
        'run_status': []
    }
    
    try:
        y_jax_for_eval = jnp.asarray(data_df[base_config.observed_variable_names].values, dtype=_DEFAULT_DTYPE)
    except Exception as e:
        print(f"Error converting data for sensitivity evaluation: {e}")
        study_results['error'] = "Data conversion error for evaluation."
        return study_results

    for i, p_val in enumerate(values_to_test):
        print(f"  Test {i+1}/{len(values_to_test)}: {parameter_name_to_vary} = {p_val}")
        
        current_fixed_params = base_config.fixed_parameter_values.copy()
        current_fixed_params[parameter_name_to_vary] = p_val
        study_results['values_tested'].append(p_val)

        try:
            eval_results = evaluate_gpm_at_parameters(
                gpm_file_path=base_config.gpm_file_path,
                y=y_jax_for_eval,
                param_values=current_fixed_params,
                num_sim_draws=0, 
                plot_results=False,
                use_gamma_init_for_test=base_config.use_gamma_init,
                gamma_init_scaling=base_config.gamma_scale_factor,
                variable_names=base_config.observed_variable_names # Pass observed names
            )
            if eval_results and 'loglik' in eval_results and jnp.isfinite(eval_results['loglik']):
                loglik_val = float(eval_results['loglik'])
                study_results['log_likelihoods'].append(loglik_val)
                study_results['run_status'].append('success')
                print(f"    ✓ LogLik: {loglik_val:.3f}")
            else:
                study_results['log_likelihoods'].append(np.nan)
                study_results['run_status'].append('failed_or_non_finite_loglik')
                print(f"    ✗ Failed or non-finite LogLik. Eval results: {eval_results.get('loglik', 'N/A') if eval_results else 'None'}")
        except Exception as e:
            study_results['log_likelihoods'].append(np.nan)
            study_results['run_status'].append(f'error: {type(e).__name__}')
            print(f"    ✗ Error during evaluation for {parameter_name_to_vary}={p_val}: {e}")

    valid_runs = [(ll, val) for ll, val, status in zip(study_results['log_likelihoods'], study_results['values_tested'], study_results['run_status']) if status == 'success' and np.isfinite(ll)]
    if valid_runs:
        best_ll, best_val = max(valid_runs, key=lambda item: item[0])
        study_results['best_parameter_value'] = best_val
        study_results['best_log_likelihood'] = best_ll
        print(f"\n  Best value for '{parameter_name_to_vary}': {best_val} -> LogLik: {best_ll:.3f}")
    else:
        study_results['best_parameter_value'] = None
        study_results['best_log_likelihood'] = np.nan
        print(f"\n  No successful runs with finite log-likelihood for '{parameter_name_to_vary}'.")
        
    return study_results

def plot_sensitivity_study_results(sensitivity_output: Dict[str, Any]):
    """Plots sensitivity analysis results if matplotlib is available."""
    param_name = sensitivity_output.get('parameter_name', 'Parameter')
    param_values = np.asarray(sensitivity_output.get('values_tested', []))
    log_likelihoods = np.asarray(sensitivity_output.get('log_likelihoods', []))

    if param_values.ndim > 1 or (param_values.size > 0 and not np.isscalar(param_values[0])):
        print(f"Cannot plot sensitivity for non-scalar parameter '{param_name}'. Values: {param_values}")
        return
    if param_values.size == 0 or log_likelihoods.size == 0 or param_values.size != log_likelihoods.size:
        print("Insufficient or mismatched data for sensitivity plot.")
        return

    finite_mask = np.isfinite(log_likelihoods)
    if not np.any(finite_mask):
        print("No finite log-likelihoods to plot for sensitivity.")
        return
        
    p_plot = param_values[finite_mask]
    ll_plot = log_likelihoods[finite_mask]
    
    if len(p_plot) == 0 :
        print("All log-likelihoods were NaN, cannot plot.")
        return

    sort_idx = np.argsort(p_plot)
    p_plot_sorted = p_plot[sort_idx]
    ll_plot_sorted = ll_plot[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(p_plot_sorted, ll_plot_sorted, marker='o', linestyle='-', color='royalblue')
    plt.xlabel(f"Value of {param_name}")
    plt.ylabel("Log-likelihood")
    plt.title(f"Sensitivity of Log-likelihood to {param_name}")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    best_val = sensitivity_output.get('best_parameter_value')
    best_ll = sensitivity_output.get('best_log_likelihood')
    if best_val is not None and best_ll is not None and np.isfinite(best_ll) and np.isscalar(best_val):
        plt.scatter([best_val], [best_ll], color='red', s=100, zorder=5, label=f"Best: {best_val:.4g} (LL: {best_ll:.2f})")
        plt.legend()
    plt.tight_layout()
    plt.show()


def example_calibration_workflow():
    """Example demonstrating the refactored prior calibration workflow with a 2-variable model."""
    print("\n" + "="*70)
    print("      EXAMPLE: PRIOR CALIBRATION & SENSITIVITY (2-VARIABLE MODEL)      ")
    print("="*70)

    example_dir = "example_2var_calibration_run"
    os.makedirs(example_dir, exist_ok=True)
    
    # Define content for a 2-variable GPM model
    gpm_2var_content = """
parameters ; // No structural parameters directly defined here for this example

estimated_params;
    stderr shk_trend_y_world, inv_gamma_pdf, 2.5, 0.025;
    stderr shk_trend_y_ea, inv_gamma_pdf, 1.5, 0.25;
    stderr shk_cycle_y_us, inv_gamma_pdf, 2.5, 0.5;
    stderr shk_cycle_y_ea, inv_gamma_pdf, 3.5, 0.5;
end;

trends_vars trend_y_world, trend_y_ea, trend_y_us_d, trend_y_ea_d;
stationary_variables cycle_y_us, cycle_y_ea;

trend_shocks; 
    var shk_trend_y_world; 
    var shk_trend_y_ea; 
end;
shocks;
 var shk_cycle_y_us;
 var shk_cycle_y_ea; 
end; // For stationary variables

trend_model;
    trend_y_world = trend_y_world(-1) + shk_trend_y_world;
    trend_y_ea = trend_y_ea(-1) + shk_trend_y_ea;
    trend_y_us_d = trend_y_world; // US trend linked to world trend
    trend_y_ea_d = trend_y_world + trend_y_ea; // EA trend linked to world and EA specific
end;

varobs y_us, y_ea;

measurement_equations;
    y_us = trend_y_us_d + cycle_y_us;
    y_ea = trend_y_ea_d + cycle_y_ea;
end;

var_prior_setup; 
    var_order=1; 
    es=0.5,0.1;      // Priors for VAR coefficients (mean_diag, mean_offdiag)
    fs=0.2,0.2;      // Priors for VAR coefficients (std_diag, std_offdiag)
    gs=3.0,3.0;      // Priors for VAR innovation precision (Gamma shape)
    hs=1.0,1.0;      // Priors for VAR innovation precision (Gamma rate)
    eta=2.0;         // LKJ prior for VAR innovation correlation
end;

initval; 
    trend_y_world, normal_pdf, 0, 1; 
    trend_y_ea, normal_pdf, 0, 1; 
    // trend_y_us_d and trend_y_ea_d are non-core, no initval needed
    // cycle_y_us, cycle_y_ea default to 0 mean for P0 unless specified
end;
"""
    gpm_example_path = os.path.join(example_dir, "gpm_2country_growth_test.gpm")
    with open(gpm_example_path, "w") as f: f.write(gpm_2var_content)

    # Generate 2D synthetic data
    data_example_path = os.path.join(example_dir, "example_2var_data.csv")
    T_example = 100
    y_us_example = np.cumsum(np.random.randn(T_example) * 0.1) + np.random.randn(T_example) * 0.2
    y_ea_example = np.cumsum(np.random.randn(T_example) * 0.15) + np.random.randn(T_example) * 0.25
    pd.DataFrame({'y_us': y_us_example, 'y_ea': y_ea_example}).to_csv(data_example_path, index=False)

    # 1. Configure the evaluation
    config = PriorCalibrationConfig(
        data_file_path=data_example_path,
        gpm_file_path=gpm_example_path,
        observed_variable_names=['y_us', 'y_ea'],
        fixed_parameter_values={
            # Shock standard deviations (builder names). These are what evaluate_gpm_at_parameters will use.
            'shk_trend_y_world': 0.08,
            'shk_trend_y_ea': 0.12,
            'shk_cycle_y_us': 0.3,
            'shk_cycle_y_ea': 0.35,
            # VAR innovation Cholesky factor. Needed by evaluate_gpm_at_parameters for Sigma_u.
            '_var_innovation_corr_chol': jnp.array([[1.0, 0.0], [0.4, jnp.sqrt(1-0.4**2)]], dtype=_DEFAULT_DTYPE)
            # Note: _var_coefficients are determined by GPM 'es' priors within evaluate_gpm_at_parameters.
            # No structural parameters in this GPM's 'parameters;' block.
        },
        num_mcmc_warmup=50,
        num_mcmc_samples=100,
        num_mcmc_chains=1,
        num_smoother_draws=20,
        generate_plots=True,
        show_plot_info_boxes=True, # Set to True to see info boxes on plots
        plot_save_path=example_dir,
        save_plots=True
    )

    if not validate_calibration_config(config): return
    data_for_run = load_data_for_calibration(config)
    if data_for_run is None: return

    print("\n--- Running Main Workflow (MCMC based on GPM priors) ---")
    main_workflow_results = run_single_point_evaluation(config, data_for_run)
    if main_workflow_results:
        print("\nMain workflow results obtained. Access 'main_workflow_results' dictionary.")
        if main_workflow_results.get('mcmc_object') and hasattr(main_workflow_results['mcmc_object'], 'print_summary'):
             main_workflow_results['mcmc_object'].print_summary(exclude_deterministic=False)

    # 5. Run Sensitivity Analysis for a shock standard deviation
    param_to_vary_sensitivity = 'shk_trend_y_world'
    print(f"\n--- Running Sensitivity Analysis for '{param_to_vary_sensitivity}' (direct evaluation) ---")
    sensitivity_study_output = run_parameter_sensitivity_workflow(
        base_config=config,
        data_df=data_for_run,
        parameter_name_to_vary=param_to_vary_sensitivity,
        values_to_test=[0.01, 0.05, 0.08, 0.1, 0.15, 0.20]
    )

    if sensitivity_study_output and 'error' not in sensitivity_study_output:
        print(f"\nSensitivity study results for '{param_to_vary_sensitivity}':")
        for val, ll, status in zip(sensitivity_study_output['values_tested'],
                                   sensitivity_study_output['log_likelihoods'],
                                   sensitivity_study_output['run_status']):
            ll_str = f"{ll:.3f}" if not np.isnan(ll) else "NaN"
            print(f"  Value: {val:.3f}, LogLik: {ll_str}, Status: {status}")
        if sensitivity_study_output.get('best_parameter_value') is not None:
             print(f"  Suggested best '{param_to_vary_sensitivity}': {sensitivity_study_output['best_parameter_value']:.3f}")
        if config.generate_plots:
            plot_sensitivity_study_results(sensitivity_study_output)
    else:
        print(f"\nSensitivity study for '{param_to_vary_sensitivity}' encountered issues or produced no valid results.")

    print("\n--- Example 2-Variable Workflow Finished ---")
    print(f"Plots and any saved data are in: {os.path.abspath(example_dir)}")

if __name__ == "__main__":
    example_calibration_workflow()