# clean_gpm_bvar_trends/gpm_prior_calibration_example.py
# Refactored to use the updated gpm_prior_evaluator and follow stricter error handling.

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt # Keep for sensitivity plot
import time

from gpm_prior_evaluator import evaluate_gpm_at_parameters # Core evaluator
from constants import _DEFAULT_DTYPE # Assuming this is your float type

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

class PriorCalibrationConfig:
    """Configuration for prior calibration workflow."""
    def __init__(self):
        self.data_file_path: str = 'data/sim_data.csv'
        self.gpm_file_path: str = 'models/test_model.gpm'
        self.observed_variable_names: List[str] = ['OBS1', 'OBS2']
        self.fixed_parameter_values: Dict[str, Any] = { # Allow Any for _var_innovation_corr_chol
            'sigma_SHK_TREND1': 0.15, 'SHK_TREND2': 0.20,
            'sigma_SHK_STAT1': 0.30, 'SHK_STAT2': 0.40,
            'rho': 0.5,
        }
        self.num_evaluation_sim_draws: int = 100
        self.evaluation_rng_seed: int = 123
        self.use_gamma_init_for_evaluation: bool = True
        self.gamma_init_scaling_for_evaluation: float = 1.0
        self.generate_plots: bool = True
        self.plot_hdi_prob: float = 0.9

def validate_configuration(config: PriorCalibrationConfig) -> bool:
    """Validates the configuration."""
    print("\n--- Validating Configuration ---")
    issues = []
    if not os.path.exists(config.data_file_path): issues.append(f"Data file not found: {config.data_file_path}")
    if not os.path.exists(config.gpm_file_path): issues.append(f"GPM file not found: {config.gpm_file_path}")
    if not isinstance(config.fixed_parameter_values, dict) or not config.fixed_parameter_values:
        issues.append("`fixed_parameter_values` must be a non-empty dictionary.")
    if not isinstance(config.observed_variable_names, list) or not config.observed_variable_names:
        issues.append("`observed_variable_names` must be a non-empty list.")

    if issues:
        print("❌ Configuration Issues Found:")
        for issue in issues: print(f"  - {issue}")
        return False
    print("✓ Configuration valid.")
    return True

def load_and_prepare_data(config: PriorCalibrationConfig) -> Optional[jnp.ndarray]:
    """Loads and prepares data."""
    print(f"\n--- Loading Data from: {config.data_file_path} ---")
    try:
        dta = pd.read_csv(config.data_file_path)
        if len(config.observed_variable_names) != dta.shape[1]:
            raise ValueError(f"Mismatch: {len(config.observed_variable_names)} specified obs vars, CSV has {dta.shape[1]} columns.")
        # Ensure columns used are those specified, GPM drives actual variable use.
        y_jax = jnp.asarray(dta[config.observed_variable_names].values, dtype=_DEFAULT_DTYPE)
        print(f"✓ Data loaded: Shape {y_jax.shape}, Type {y_jax.dtype}")
        if jnp.any(jnp.isnan(y_jax)) or jnp.any(jnp.isinf(y_jax)):
            print("⚠ Warning: Data contains NaN/Inf values.")
        return y_jax
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def run_prior_evaluation_workflow_step(config: PriorCalibrationConfig, y_data: jnp.ndarray) -> Optional[Dict[str, Any]]:
    """Runs the core prior evaluation using fixed parameters from config."""
    print("\n--- Running Single Prior Evaluation Step ---")
    try:
        start_time = time.time()
        results = evaluate_gpm_at_parameters(
            gpm_file_path=config.gpm_file_path, y=y_data,
            param_values=config.fixed_parameter_values,
            num_sim_draws=config.num_evaluation_sim_draws,
            rng_key=jax.random.PRNGKey(config.evaluation_rng_seed),
            plot_results=config.generate_plots,
            variable_names=config.observed_variable_names,
            use_gamma_init_for_test=config.use_gamma_init_for_evaluation,
            gamma_init_scaling=config.gamma_init_scaling_for_evaluation,
            hdi_prob=config.plot_hdi_prob
        )
        print(f"✓ Evaluation step completed in {time.time() - start_time:.2f}s.")
        return results
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"✗ Evaluation step failed: {type(e).__name__}: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        import traceback
        print(f"✗ Unexpected error in evaluation step: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def analyze_evaluation_results(results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes and reports on the evaluation results."""
    if not results:
        print("No evaluation results to analyze.")
        return {'status': 'failed_evaluation'}

    print("\n--- Analyzing Evaluation Results ---")
    analysis = {}
    loglik = results.get('loglik', jnp.array(-jnp.inf))
    analysis['loglik'] = float(loglik)
    print(f"  Log-likelihood: {analysis['loglik']:.3f}")
    if not jnp.isfinite(loglik) or loglik < -1e9: # Arbitrary low threshold
        print("  ⚠ Log-likelihood is non-finite or extremely low.")
        analysis['loglik_status'] = 'poor'
    else:
        analysis['loglik_status'] = 'ok'

    sim_draws = results.get('sim_draws_core_state')
    if sim_draws is not None and sim_draws.shape[0] > 0:
        analysis['num_sim_draws_completed'] = sim_draws.shape[0]
        print(f"  Simulation draws completed: {sim_draws.shape[0]}")
        # Basic check on reconstructed components
        recon_trends = results.get('reconstructed_original_trends', jnp.empty((0,)))
        if recon_trends.shape[0] == sim_draws.shape[0]:
            print(f"  Reconstructed original trends shape: {recon_trends.shape}")
        else:
            print("  ⚠ Mismatch or missing reconstructed original trends.")
    else:
        analysis['num_sim_draws_completed'] = 0
        print("  No successful simulation draws for detailed component analysis based on draws.")
    
    # Further analysis on matrix properties could be added here if desired
    # For example, checking condition numbers of Q, H, P0 if they are returned.
    # The evaluator already checks for PSD and finite values.
    print("--- Analysis Complete ---")
    return analysis

def main_prior_calibration_workflow(config: Optional[PriorCalibrationConfig] = None) -> Optional[Dict[str, Any]]:
    """Main workflow: config -> load data -> evaluate -> analyze."""
    print("\n" + "="*70)
    print("      MAIN PRIOR CALIBRATION WORKFLOW      ")
    print("="*70)

    active_config = config if config is not None else PriorCalibrationConfig()
    if not validate_configuration(active_config): return None
    y_data = load_and_prepare_data(active_config)
    if y_data is None: return None

    evaluation_results = run_prior_evaluation_workflow_step(active_config, y_data)
    if evaluation_results:
        analysis_summary = analyze_evaluation_results(evaluation_results)
        evaluation_results['analysis_summary'] = analysis_summary
        print("\n✓ Workflow completed successfully.")
        return evaluation_results
    else:
        print("\n✗ Workflow failed during prior evaluation step.")
        return None

def create_parameter_sensitivity_study(
    base_config: PriorCalibrationConfig,
    y_data: jnp.ndarray,
    param_to_vary: str,
    param_values_to_test: List[Any] # Allow Any for complex params like Cholesky factors
) -> Dict[str, Any]:
    """Runs sensitivity analysis by varying one parameter."""
    print(f"\n--- Sensitivity Study for Parameter: '{param_to_vary}' ---")

    if param_to_vary not in base_config.fixed_parameter_values:
        err_msg = f"Parameter '{param_to_vary}' not in base config's fixed_parameter_values. Available: {list(base_config.fixed_parameter_values.keys())}"
        print(f"✗ Error: {err_msg}")
        return {'error': err_msg, 'param_name': param_to_vary, 'param_values': param_values_to_test, 'log_likelihoods': []}

    study_results = {
        'param_name': param_to_vary, 'param_values': param_values_to_test,
        'log_likelihoods': [], 'run_status': []
    }

    for i, p_val in enumerate(param_values_to_test):
        print(f"  Test {i+1}/{len(param_values_to_test)}: {param_to_vary} = {p_val}")
        current_config = PriorCalibrationConfig() # Create a fresh config instance
        # Copy all attributes from base_config
        for attr, value in base_config.__dict__.items():
            if attr == 'fixed_parameter_values': # Deep copy for dicts
                setattr(current_config, attr, value.copy())
            elif isinstance(value, list): # Shallow copy for lists
                setattr(current_config, attr, value[:])
            else:
                setattr(current_config, attr, value)
        
        # Override the parameter to vary
        current_config.fixed_parameter_values[param_to_vary] = p_val
        # Modify settings for speed in sensitivity
        current_config.generate_plots = False
        current_config.num_evaluation_sim_draws = 0 # No sim draws needed for LL
        current_config.evaluation_rng_seed += i # Vary seed slightly

        eval_res = run_prior_evaluation_workflow_step(current_config, y_data)
        if eval_res and 'loglik' in eval_res and jnp.isfinite(eval_res['loglik']):
            study_results['log_likelihoods'].append(float(eval_res['loglik']))
            study_results['run_status'].append('success')
            print(f"    ✓ LogLik: {eval_res['loglik']:.3f}")
        else:
            study_results['log_likelihoods'].append(np.nan) # Use NaN for failed/non-finite LL
            study_results['run_status'].append('failed')
            print(f"    ✗ Failed or non-finite LogLik.")
            
    # Determine best value if any successful runs
    valid_lls = [(ll, val) for ll, val, status in zip(study_results['log_likelihoods'], study_results['param_values'], study_results['run_status']) if status == 'success' and np.isfinite(ll)]
    if valid_lls:
        best_ll, best_val = max(valid_lls, key=lambda item: item[0])
        study_results['best_param_value'] = best_val
        study_results['best_loglik'] = best_ll
        print(f"  Best value for '{param_to_vary}': {best_val} -> LogLik: {best_ll:.3f}")
    else:
        study_results['best_param_value'] = None
        study_results['best_loglik'] = np.nan
        print(f"  No successful runs with finite log-likelihood for '{param_to_vary}'.")
        
    return study_results

def plot_sensitivity_results(sensitivity_output: Dict[str, Any]):
    """Plots sensitivity analysis results if matplotlib is available."""
    if not plt: 
        print("Matplotlib not available, skipping sensitivity plot.")
        return

    param_name = sensitivity_output.get('param_name', 'Parameter')
    param_vals = np.asarray(sensitivity_output.get('param_values', []))
    logliks = np.asarray(sensitivity_output.get('log_likelihoods', []))

    if param_vals.ndim > 1 or (param_vals.size > 0 and not np.isscalar(param_vals[0])):
        print(f"Cannot plot sensitivity for non-scalar parameter '{param_name}'. Values: {param_vals}")
        return
    if param_vals.size == 0 or logliks.size == 0 or param_vals.size != logliks.size:
        print("Insufficient data for sensitivity plot.")
        return

    finite_mask = np.isfinite(logliks)
    if not np.any(finite_mask):
        print("No finite log-likelihoods to plot for sensitivity.")
        return
        
    p_plot = param_vals[finite_mask]
    ll_plot = logliks[finite_mask]
    
    sort_idx = np.argsort(p_plot)
    p_plot_sorted = p_plot[sort_idx]
    ll_plot_sorted = ll_plot[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(p_plot_sorted, ll_plot_sorted, marker='o', linestyle='-', color='royalblue')
    plt.xlabel(f"Value of {param_name}")
    plt.ylabel("Log-likelihood")
    plt.title(f"Sensitivity of Log-likelihood to {param_name}")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    best_val = sensitivity_output.get('best_param_value')
    best_ll = sensitivity_output.get('best_loglik')
    if best_val is not None and best_ll is not None and np.isfinite(best_ll) and np.isscalar(best_val):
        plt.scatter([best_val], [best_ll], color='red', s=100, zorder=5, label=f"Best: {best_val:.4g} (LL: {best_ll:.2f})")
        plt.legend()
    plt.tight_layout()
    plt.show()


def run_sensitivity_analysis_workflow(
    config: Optional[PriorCalibrationConfig] = None,
    param_to_study: Optional[str] = None,
    param_values_range: Optional[List[Any]] = None
) -> Optional[Dict[str, Any]]:
    """Orchestrates a parameter sensitivity study."""
    print("\n" + "="*70)
    print("      SENSITIVITY ANALYSIS WORKFLOW      ")
    print("="*70)
    
    active_config = config if config is not None else PriorCalibrationConfig()
    if not validate_configuration(active_config): return None
    y_data = load_and_prepare_data(active_config)
    if y_data is None: return None

    param_names_in_config = list(active_config.fixed_parameter_values.keys())
    if not param_names_in_config:
        print("✗ No parameters in `fixed_parameter_values` to study.")
        return None

    if param_to_study is None: # Default to first parameter if none specified
        param_to_study = param_names_in_config[0]
        print(f"  No 'param_to_study' provided, defaulting to '{param_to_study}'.")
    
    if param_values_range is None:
        base_val = active_config.fixed_parameter_values.get(param_to_study)
        if base_val is None or not np.isscalar(base_val):
            print(f"✗ Cannot create default range for non-scalar or missing base parameter '{param_to_study}'. Provide 'param_values_range'.")
            return None
        # Create a simple default range for scalar parameters
        spread = max(abs(float(base_val)) * 0.5, 0.1) if float(base_val) != 0 else 0.5
        param_values_range = np.linspace(float(base_val) - spread, float(base_val) + spread, 7).tolist()
        print(f"  No 'param_values_range' provided. Defaulting to range for '{param_to_study}': {[f'{x:.3g}' for x in param_values_range]}")

    sensitivity_output = create_parameter_sensitivity_study(active_config, y_data, param_to_study, param_values_range)
    
    if 'error' not in sensitivity_output and sensitivity_output.get('log_likelihoods'):
        plot_sensitivity_results(sensitivity_output)
        print("\n✓ Sensitivity analysis workflow completed.")
    else:
        print("\n✗ Sensitivity analysis workflow failed or produced no usable results.")
    return sensitivity_output


def test_prior_calibration_workflow_integration():
    """Comprehensive integration test for the entire prior calibration workflow."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Prior Calibration Workflow")
    print("="*70)

    test_config = PriorCalibrationConfig()
    test_dir = "temp_workflow_test_files"
    os.makedirs(test_dir, exist_ok=True)
    test_config.data_file_path = os.path.join(test_dir, 'test_workflow_data.csv')
    test_config.gpm_file_path = os.path.join(test_dir, 'test_workflow_model.gpm')
    test_config.observed_variable_names = ['OBS_X', 'OBS_Y']
    
    # FIXED GPM content - proper formatting and structure
    gpm_content_workflow_test = """
parameters rho_param;

estimated_params;
    rho_param, normal_pdf, 0.7, 0.1;
    stderr TREND_SHK_X, inv_gamma_pdf, 2, 0.02;
    sigma_TREND_SHK_Y, normal_pdf, 0.15, 0.05;
    STAT_SHK_1, inv_gamma_pdf, 3, 0.03; 
end;

trends_vars TREND_X, TREND_Y;

stationary_variables STAT_1; 

trend_shocks; 
    var TREND_SHK_X; 
    var TREND_SHK_Y; 
end;

shocks; 
    var STAT_SHK_1; 
end;

trend_model;
    TREND_X = rho_param * TREND_X(-1) + TREND_SHK_X;
    TREND_Y = TREND_Y(-1) + TREND_SHK_Y;
end;

varobs OBS_X, OBS_Y;

measurement_equations; 
    OBS_X = TREND_X + STAT_1; 
    OBS_Y = TREND_Y; 
end;

var_prior_setup; 
    var_order=1; 
    es=0.9,0; 
    fs=0.1,0.1; 
    gs=1,1; 
    hs=1,1; 
    eta=1; 
end;

initval;
    TREND_X, normal_pdf, 0, 1; 
    TREND_Y, normal_pdf, 0, 1;
    STAT_1, normal_pdf, 0, 0.5;
end;
"""
    with open(test_config.gpm_file_path, "w") as f: 
        f.write(gpm_content_workflow_test)

    np.random.seed(777)
    T_data, N_obs = 50, 2
    y_wf_data = np.cumsum(np.random.randn(T_data, N_obs) * 0.1, axis=0) + np.random.randn(T_data, N_obs)*0.05
    pd.DataFrame(y_wf_data, columns=test_config.observed_variable_names).to_csv(test_config.data_file_path, index=False)

    test_config.fixed_parameter_values = {
        'rho_param': 0.75, # Override prior
        'TREND_SHK_X': 0.04, # Override prior (stderr name)
        'sigma_TREND_SHK_Y': 0.12, # Override prior (sigma_ name)
        # STAT_SHK_1 will use its prior mode: 0.03 / (3+1) = 0.0075
         '_var_innovation_corr_chol': jnp.eye(1, dtype=_DEFAULT_DTYPE) # For 1 stationary var
    }
    test_config.num_evaluation_sim_draws = 10
    test_config.generate_plots = False # Disable plots for automated test runs

    print("--- Running Main Workflow Test ---")
    main_results = main_prior_calibration_workflow(test_config)
    assert main_results is not None and 'analysis_summary' in main_results and main_results['analysis_summary'].get('loglik_status') == 'ok', "Main workflow test failed."
    print("✓ Main workflow test passed.")

    print("\n--- Running Sensitivity Analysis Test (on rho_param) ---")
    sensitivity_results = run_sensitivity_analysis_workflow(
        config=test_config,
        param_to_study='rho_param',
        param_values_range=[0.6, 0.7, 0.8, 0.9]
    )
    assert sensitivity_results is not None and 'error' not in sensitivity_results and sensitivity_results.get('best_loglik') is not None, "Sensitivity workflow test for rho_param failed."
    print("✓ Sensitivity workflow test for rho_param passed.")
    
    # Clean up
    if os.path.exists(test_config.data_file_path): os.remove(test_config.data_file_path)
    if os.path.exists(test_config.gpm_file_path): os.remove(test_config.gpm_file_path)
    if os.path.exists(test_dir): 
        try: os.rmdir(test_dir) # Only if empty
        except OSError: pass # Ignore if not empty (e.g. plots saved by user)
    
    print("\n" + "="*70)
    print("✅ ALL WORKFLOW INTEGRATION TESTS COMPLETED.")
    print("="*70)

if __name__ == "__main__":
    # Run the integration test for the example workflow
    test_prior_calibration_workflow_integration()

    # --- Example of running user-defined workflow (uncomment and modify) ---
    # print("\n" + "="*70)
    # print("      RUNNING USER-DEFINED WORKFLOW      ")
    # print("="*70)
    # user_config = PriorCalibrationConfig()
    # # REQUIRED: Update these paths to your actual files
    # user_config.data_file_path = 'path/to/your/data.csv'
    # user_config.gpm_file_path = 'path/to/your/model.gpm'
    # # REQUIRED: Update with your observed variable names (must match CSV columns and GPM varobs)
    # user_config.observed_variable_names = ['MY_OBS_VAR1', 'MY_OBS_VAR2']
    # # REQUIRED: Provide fixed values for ALL parameters needed by your GPM
    # # This includes structural params and ALL shock standard deviations (trend and stationary)
    # user_config.fixed_parameter_values = {
    # 'my_structural_param': 0.9,
    # 'my_trend_shock_std': 0.1, # Or 'sigma_my_trend_shock_std'
    # 'my_stationary_shock_std': 0.5, # Or 'sigma_my_stationary_shock_std'
    # # If your model has a VAR component and use_gamma_init_for_evaluation is True,
    # # you might want to specify the Cholesky of the VAR innovation correlation matrix:
    # # '_var_innovation_corr_chol': jnp.eye(N_STATIONARY_VARS_HERE) # Example for identity
    # }
    # user_config.num_evaluation_sim_draws = 500 # Increase for smoother component estimates
    # user_config.generate_plots = True # Enable plots
    # user_config.use_gamma_init_for_evaluation = True # Recommended
    # user_config.gamma_init_scaling_for_evaluation = 1.0 # Adjust as needed

    # main_results = main_prior_calibration_workflow(user_config)
    # if main_results:
    # print("\nUser workflow finished. Access 'main_results' dictionary for details.")

    # # Example Sensitivity Analysis (Uncomment and modify)
    # # param_to_study_user = 'my_structural_param' # Choose a parameter from your fixed_parameter_values
    # # values_to_test_user = [0.8, 0.85, 0.9, 0.95] # Define a range of values
    # # sensitivity_results_user = run_sensitivity_analysis_workflow(user_config, param_to_study_user, values_to_test_user)
    # # if sensitivity_results_user:
    # # print(f"\nSensitivity analysis for '{param_to_study_user}' complete.")