# main_global_trend_fixed.py - FIXED PARAMETER VERSION
import sys
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the FIXED PARAMETER evaluator (not the MCMC workflow)
from clean_gpm_bvar_trends.gpm_prior_evaluator import evaluate_gpm_at_parameters

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# ============================================================================
# PRE-SAMPLE CALIBRATION TOOL (same as before)
# ============================================================================

def split_data_for_presample(data, split_ratio=0.15, method='first'):
    """Split data into pre-sample and main sample."""
    n_total = len(data)
    n_presample = int(n_total * split_ratio)
    n_main = n_total - n_presample
    
    print(f"\n=== DATA SPLITTING ===")
    print(f"Total observations: {n_total}")
    print(f"Pre-sample size: {n_presample} ({split_ratio*100:.1f}%)")
    print(f"Main sample size: {n_main} ({(1-split_ratio)*100:.1f}%)")
    
    if method == 'first':
        presample_data = data.iloc[:n_presample].copy()
        main_data = data.iloc[n_presample:].copy()
        print(f"Pre-sample period: {presample_data.index[0]} to {presample_data.index[-1]}")
        print(f"Main sample period: {main_data.index[0]} to {main_data.index[-1]}")
        
    elif method == 'last':
        presample_data = data.iloc[-n_presample:].copy()
        main_data = data.iloc[:-n_presample].copy()
        print(f"Pre-sample period: {presample_data.index[0]} to {presample_data.index[-1]}")
        print(f"Main sample period: {main_data.index[0]} to {main_data.index[-1]}")
        
    elif method == 'random':
        np.random.seed(42)
        presample_indices = np.random.choice(n_total, n_presample, replace=False)
        main_indices = np.setdiff1d(np.arange(n_total), presample_indices)
        
        presample_data = data.iloc[presample_indices].copy()
        main_data = data.iloc[main_indices].copy()
        print(f"Random split - Pre-sample: {n_presample} observations")
        print(f"Random split - Main sample: {n_main} observations")
    
    split_info = {
        'method': method,
        'split_ratio': split_ratio,
        'n_presample': n_presample,
        'n_main': n_main,
        'presample_period': (presample_data.index[0], presample_data.index[-1]),
        'main_period': (main_data.index[0], main_data.index[-1])
    }
    
    return presample_data, main_data, split_info


def get_observable_ranges_for_initial_conditions(presample_data, observed_vars):
    """Extract ranges and statistics from pre-sample for setting initial conditions."""
    print(f"\n=== OBSERVABLE RANGES FROM PRE-SAMPLE ===")
    
    ranges_info = {}
    
    for var in observed_vars:
        if var in presample_data.columns:
            data_series = presample_data[var].dropna()
            
            if len(data_series) > 0:
                ranges_info[var] = {
                    'min': float(data_series.min()),
                    'max': float(data_series.max()),
                    'mean': float(data_series.mean()),
                    'median': float(data_series.median()),
                    'std': float(data_series.std()),
                    'q25': float(data_series.quantile(0.25)),
                    'q75': float(data_series.quantile(0.75)),
                    'range': float(data_series.max() - data_series.min()),
                    'n_obs': len(data_series)
                }
                
                print(f"{var:>8}: [{data_series.min():6.3f}, {data_series.max():6.3f}], "
                      f"mean={data_series.mean():6.3f}, std={data_series.std():6.3f}")
            else:
                print(f"{var:>8}: No valid observations")
                ranges_info[var] = None
        else:
            print(f"{var:>8}: Variable not found in data")
            ranges_info[var] = None
    
    return ranges_info


def suggest_initial_conditions_from_ranges(ranges_info, trend_variables, scale_factor=1.0):
    """Suggest initial conditions for trends based on observable ranges."""
    print(f"\n=== SUGGESTED INITIAL CONDITIONS ===")
    
    initial_conditions = {}
    
    for trend_var in trend_variables:
        matched_obs = None
        
        # Direct matching patterns
        if 'y_US' in trend_var and 'y_us' in ranges_info:
            matched_obs = 'y_us'
        elif 'y_EA' in trend_var and 'y_ea' in ranges_info:
            matched_obs = 'y_ea'
        elif 'y_JP' in trend_var and 'y_jp' in ranges_info:
            matched_obs = 'y_jp'
        elif 'pi_US' in trend_var and 'pi_us' in ranges_info:
            matched_obs = 'pi_us'
        elif 'pi_EA' in trend_var and 'pi_ea' in ranges_info:
            matched_obs = 'pi_ea'
        elif 'pi_JP' in trend_var and 'pi_jp' in ranges_info:
            matched_obs = 'pi_jp'
        elif 'r_US' in trend_var or 'R_US' in trend_var and 'r_us' in ranges_info:
            matched_obs = 'r_us'
        elif 'r_EA' in trend_var or 'R_EA' in trend_var and 'r_ea' in ranges_info:
            matched_obs = 'r_ea'
        elif 'r_JP' in trend_var or 'R_JP' in trend_var and 'r_jp' in ranges_info:
            matched_obs = 'r_jp'
        
        if matched_obs and ranges_info[matched_obs] is not None:
            obs_info = ranges_info[matched_obs]
            
            # Use median as initial mean, scaled std as initial variance
            initial_mean = obs_info['median']
            initial_var = (obs_info['std'] * scale_factor) ** 2
            
            initial_conditions[trend_var] = {
                'mean': initial_mean,
                'variance': initial_var
            }
            
            print(f"{trend_var:>20} -> {matched_obs}: mean={initial_mean:6.3f}, var={initial_var:6.3f}")
        else:
            print(f"{trend_var:>20} -> No matching observable found")
    
    return initial_conditions


def run_presample_parameter_test(presample_data, gpm_file_path, param_config, 
                                initial_conditions=None, test_name="Test"):
    """Run fixed-parameter evaluation on pre-sample data."""
    print(f"\n=== RUNNING PRE-SAMPLE TEST: {test_name} ===")
    print(f"Parameter configuration:")
    for param, value in param_config.items():
        print(f"  {param}: {value}")
    
    try:
        # Run fixed-parameter evaluation
        results = evaluate_gpm_at_parameters(
            gpm_file_path=gpm_file_path,
            y=presample_data,
            param_values=param_config,
            initial_state_prior_overrides=initial_conditions,
            num_sim_draws=0,  # Don't need simulation draws for calibration
            plot_results=False,  # Don't generate plots for calibration
            use_gamma_init_for_test=True,
            gamma_init_scaling=1.0
        )
        
        if results and hasattr(results, 'log_likelihood'):
            loglik = results.log_likelihood
            print(f"✓ {test_name} completed - Log-likelihood: {loglik:.4f}")
            
            return {
                'test_name': test_name,
                'parameters': param_config,
                'log_likelihood': loglik,
                'success': True,
                'results': results
            }
        else:
            print(f"✗ {test_name} failed - No valid results")
            return {
                'test_name': test_name,
                'parameters': param_config,
                'log_likelihood': np.nan,
                'success': False,
                'results': None
            }
            
    except Exception as e:
        print(f"✗ {test_name} failed with error: {e}")
        return {
            'test_name': test_name,
            'parameters': param_config,
            'log_likelihood': np.nan,
            'success': False,
            'error': str(e),
            'results': None
        }


# ============================================================================
# MAIN WORKFLOW - FIXED PARAMETER VERSION
# ============================================================================

# --- Data Loading ---
dta_path = os.path.join(SCRIPT_DIR, "data_m5.csv") 
data_source_file_name = dta_path 

dta = pd.read_csv(data_source_file_name)
dta['Date'] = pd.to_datetime(dta['Date'])
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')

# --- Model Configuration ---
observed_vars_model = [
    'y_us', 'y_ea', 'y_jp',
    'r_us', 'r_ea', 'r_jp',
    'pi_us', 'pi_ea', 'pi_jp'
]
data_sub = dta[observed_vars_model].copy() 
data_sub = data_sub.dropna()
print(f"Data shape after dropping NaNs: {data_sub.shape}")

gpm_file_name = 'gpm_factor_y_pi_rshort.gpm'
gmp_file_path = os.path.join(SCRIPT_DIR, '..', 'clean_gpm_bvar_trends', 'models', gpm_file_name)

if not os.path.exists(gmp_file_path):
    print(f"FATAL ERROR: {gmp_file_path} not found.")
    sys.exit(1)

# ============================================================================
# DEFINE FIXED PARAMETER VALUES
# ============================================================================

# You need to provide the parameter values you want to test
# These should match the parameters defined in your GPM file
fixed_param_values = {
    # Example parameter values - REPLACE WITH YOUR ACTUAL PARAMETERS
    'var_phi_US': 2.0,           # Risk aversion parameter for US
    'var_phi_EA': 2.0,           # Risk aversion parameter for EA  
    'var_phi_JP': 2.0,           # Risk aversion parameter for JP
    'lambda_pi_US': 1.0,         # Inflation factor loading for US
    'lambda_pi_EA': 1.0,         # Inflation factor loading for EA
    'lambda_pi_JP': 1.0,         # Inflation factor loading for JP
    
    # Shock standard deviations - you may need to add more based on your GPM file
    'SHK_TREND1': 0.1,          # Example trend shock std dev
    'SHK_TREND2': 0.1,          # Example trend shock std dev
    'shk_stat1': 0.2,           # Example stationary shock std dev
    'shk_stat2': 0.2,           # Example stationary shock std dev
    
    # # VAR parameters (if you have stationary variables)
    # '_var_coefficients': jnp.array([[[0.7, 0.1], [0.05, 0.6]]], dtype=jnp.float64),  # Example VAR(1) coefficients
    # '_var_innovation_corr_chol': jnp.array([[1.0, 0.0], [0.3, 0.9]], dtype=jnp.float64)  # Example correlation matrix
}

# Optional: Override initial conditions for trend variables
initial_state_overrides = {
    'r_w_trend': {'mean': 2.0, 'variance': 1.0},
    'pi_w_trend': {'mean': 2.0, 'variance': 1.0},
    'y_US_trend': {'mean': 2.0, 'variance': 1.0},
    'y_EA_trend': {'mean': 2.0, 'variance': 1.0},
    'y_JP_trend': {'mean': 2.0, 'variance': 1.0},
    # Add more trend variables as needed
}

# Your existing custom plot specs
custom_plot_specs_factor_model = [
    {
        "title": "US Real Rate Trend Decomposition",
        "series_to_plot": [
            {'type': 'observed', 'name': 'r_us', 'label': 'Observed Rshort_US', 'style': '--'},
            {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'US Full Real Rate Trend', 'show_hdi': True, 'color': 'blue'},
            {'type': 'trend', 'name': 'r_w_trend', 'label': 'World Real Rate Trend', 'show_hdi': True, 'color': 'green', 'style': ':'},
            {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Full Deviation Trend', 'show_hdi': True, 'color': 'orange', 'style': '-.'}
        ]
    },
    {
        "title": "US Real Rate Deviation Further Decomposed",
        "series_to_plot": [
            {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Full Deviation', 'show_hdi': True, 'color': 'orange'},
            {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor', 'show_hdi': True, 'color': 'purple', 'style': '--'},
            {'type': 'trend', 'name': 'r_US_idio_trend', 'label': 'US Idiosyncratic', 'show_hdi': True, 'color': 'brown', 'style': ':'}
        ]
    },
    {
        "title": "US Inflation Trend Decomposition",
        "series_to_plot": [
            {'type': 'observed', 'name': 'pi_us', 'label': 'Observed PI_US', 'style': '--'},
            {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'US Full Inflation Trend', 'show_hdi': True, 'color': 'red'},
            {'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend', 'show_hdi': True, 'color': 'magenta', 'style': ':'},
            {'type': 'trend', 'name': 'pi_US_dev_trend', 'label': 'US Full Deviation Trend', 'show_hdi': True, 'color': 'cyan', 'style': '-.'}
        ]
    }
]

print(f"\n--- Starting FIXED PARAMETER Evaluation for Model with Trends ---")
print(f"Using parameter values:")
for param, value in fixed_param_values.items():
    if isinstance(value, jnp.ndarray):
        print(f"  {param}: JAX array with shape {value.shape}")
    else:
        print(f"  {param}: {value}")

# ============================================================================
# RUN FIXED PARAMETER EVALUATION
# ============================================================================

try:
    results_model = evaluate_gpm_at_parameters(
        gpm_file_path=gmp_file_path,
        y=data_sub,  # Your data
        param_values=fixed_param_values,  # Fixed parameter values
        initial_state_prior_overrides=initial_state_overrides,  # Optional
        num_sim_draws=10,  # Number of simulation draws for plotting
        plot_results=True,  # Generate plots
        plot_default_observed_vs_trend_components=True,  # Plot observed vs trends
        custom_plot_specs=custom_plot_specs_factor_model,  # Your custom plots
        variable_names=observed_vars_model,  # Variable names
        use_gamma_init_for_test=True,  # Use gamma-based P0 initialization
        gamma_init_scaling=1.0,
        hdi_prob=0.9,
        trend_P0_var_scale=0.01,  # Scale for trend P0 variance
        var_P0_var_scale=1.0,    # Scale for VAR P0 variance
        save_plots_path_prefix="results_model_with_trends/fixed_eval_plot",  # Save plots
        show_plot_info_boxes=False
    )

    if results_model:
        print(f"\n✓ Fixed parameter evaluation for {gpm_file_name} successfully completed!")
        print(f"  Log-likelihood: {results_model.log_likelihood:.4f}")
        print(f"  Number of simulation draws: {results_model.n_draws}")
        print(f"  Plots saved to: results_model_with_trends/")
        
        # You can access other results
        if hasattr(results_model, 'trend_draws') and results_model.trend_draws is not None:
            print(f"  Trend draws shape: {results_model.trend_draws.shape}")
            print(f"  Trend variable names: {results_model.trend_names}")
        
        if hasattr(results_model, 'stationary_draws') and results_model.stationary_draws is not None:
            print(f"  Stationary draws shape: {results_model.stationary_draws.shape}")
            print(f"  Stationary variable names: {results_model.stationary_names}")
            
    else:
        print(f"\n✗ Fixed parameter evaluation for {gpm_file_name} failed.")

except Exception as e:
    import traceback
    print(f"\n✗ Fixed parameter evaluation failed with error: {e}")
    traceback.print_exc()

print(f"\n--- Fixed Parameter Evaluation Complete ---")