# main_global_trend_fixed.py - FIXED PARAMETER VERSION
import sys
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Union 
import sys
import os

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from sklearn.decomposition import PCA
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

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


def discounted_least_squares(y: np.ndarray, discount_factor: float = 0.98) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Discounted Least Squares to extract trend and cycle components.
    
    Args:
        y: Time series data
        discount_factor: Smoothing parameter (higher = smoother trend)
        
    Returns:
        trend: Extracted trend component
        cycle: Cycle component (y - trend)
    """
    y = np.asarray(y)
    T = len(y)
    trend = np.zeros(T)
    
    # Initialize with first observation
    trend[0] = y[0]
    
    # Recursive updating with discount factor
    for t in range(1, T):
        # Exponentially weighted moving average
        weights = discount_factor ** np.arange(t, -1, -1)
        weights = weights / np.sum(weights)
        trend[t] = np.sum(weights * y[:t+1])
    
    cycle = y - trend
    return trend, cycle

def fit_ar1_to_cycle(cycle: np.ndarray) -> Dict[str, float]:
    """
    Fit AR(1) model to cycle component and return diagnostics.
    
    Args:
        cycle: Cycle component from DLS
        
    Returns:
        Dictionary with AR(1) statistics
    """
    cycle = cycle[~np.isnan(cycle)]  # Remove any NaNs
    T = len(cycle)
    
    if T < 3:
        return {'residual_variance': np.var(cycle), 'ar_coef': 0.0}
    
    # Prepare AR(1) regression: cycle_t = φ * cycle_{t-1} + ε_t
    y_ar = cycle[1:]
    x_ar = cycle[:-1]
    
    if len(x_ar) == 0:
        return {'residual_variance': np.var(cycle), 'ar_coef': 0.0}
    
    # Add constant
    x_ar_const = sm.add_constant(x_ar)
    
    try:
        ar_model = OLS(y_ar, x_ar_const).fit()
        residuals = ar_model.resid
        residual_var = np.var(residuals)
        ar_coef = ar_model.params[1] if len(ar_model.params) > 1 else 0.0
        
        return {
            'residual_variance': residual_var,
            'ar_coef': ar_coef,
            'residuals': residuals
        }
    except:
        return {'residual_variance': np.var(cycle), 'ar_coef': 0.0}

def extract_principal_component(trend_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract first principal component from matrix of trends.
    
    Args:
        trend_matrix: T x N matrix where columns are country trends
        
    Returns:
        Dictionary with PC results
    """
    # Remove any NaN rows
    valid_rows = ~np.isnan(trend_matrix).any(axis=1)
    clean_matrix = trend_matrix[valid_rows]
    
    if clean_matrix.shape[0] < 2 or clean_matrix.shape[1] < 2:
        # Fallback to simple average if insufficient data
        pc_trend = np.nanmean(trend_matrix, axis=1)
        return {
            'pc_trend': pc_trend,
            'explained_variance': 1.0,
            'innovation_variance': np.var(np.diff(pc_trend)) if len(pc_trend) > 1 else 1.0
        }
    
    # Store original means before centering
    original_means = np.nanmean(clean_matrix, axis=0)
    overall_mean = np.nanmean(original_means)  # Average across countries
    
    # Apply PCA (this centers the data automatically)
    pca = PCA(n_components=1)
    pc_scores = pca.fit_transform(clean_matrix)
    
    # Reconstruct PC trend and add back the overall mean level
    pc_trend_centered = np.full(trend_matrix.shape[0], np.nan)
    pc_trend_centered[valid_rows] = pc_scores.flatten()
    
    # Fill NaN values with interpolation or forward fill
    if np.isnan(pc_trend_centered).any():
        pc_trend_centered = pd.Series(pc_trend_centered).fillna(method='ffill').fillna(method='bfill').values
    
    # Add back the overall level (PC + mean level)
    pc_trend = pc_trend_centered + overall_mean
    
    # Innovation variance (for shock priors)
    pc_innovations = np.diff(pc_trend)
    innovation_var = np.var(pc_innovations) if len(pc_innovations) > 0 else 1.0
    
    return {
        'pc_trend': pc_trend,
        'explained_variance': pca.explained_variance_ratio_[0],
        'innovation_variance': innovation_var,
        'loadings': pca.components_[0],
        'overall_mean': overall_mean
    }

def estimate_alpha_method_of_moments(variances: np.ndarray, min_alpha: float = 2.1) -> float:
    """
    Estimate alpha parameter for inverse gamma using method of moments.
    
    Args:
        variances: Array of variance estimates
        min_alpha: Minimum alpha to ensure finite moments
        
    Returns:
        Estimated alpha parameter
    """
    variances = variances[variances > 0]  # Remove zeros/negatives
    
    if len(variances) < 2:
        return 2.3  # Default fallback
    
    sample_mean = np.mean(variances)
    sample_var = np.var(variances)
    
    if sample_var <= 0 or sample_mean <= 0:
        return 2.3  # Default fallback
    
    # Method of moments: α = 2 + μ²/σ²
    alpha_hat = 2 + (sample_mean**2 / sample_var)
    alpha_hat = max(alpha_hat, min_alpha)
    
    return alpha_hat

def get_scalar(value):
    """Helper function to extract scalar from potentially nested structures."""
    if hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return float(value[0])
    else:
        return float(value)

def suggested_priors_gpm(data: pd.DataFrame,
                        sample_split: float = 0.15,
                        smoothing: float = 0.98,
                        min_alpha: float = 2.1) -> Dict[str, Dict[str, float]]:
    """
    Generate DLS-based priors for GPM model following Canova's methodology.
    
    Expected data columns: ['y_us', 'pi_us', 'r_us', 'y_ea', 'pi_ea', 'r_ea', 'y_jp', 'pi_jp', 'r_jp']
    
    Args:
        data: DataFrame with GPM observable variables
        sample_split: Fraction of data to use for training (from beginning)
        smoothing: DLS discount factor (higher = smoother trends)
        min_alpha: Minimum alpha for inverse gamma distributions
        
    Returns:
        Dictionary with prior specifications for GPM model
    """
    
    # Data validation
    expected_cols = ['y_us', 'pi_us', 'r_us', 'y_ea', 'pi_ea', 'r_ea', 'y_jp', 'pi_jp', 'r_jp']
    missing_cols = [col for col in expected_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Split data for training
    train_data = data.iloc[:int(sample_split * len(data))][expected_cols].copy()
    
    print("="*80)
    print("GPM DLS-BASED PRIOR SETUP")
    print("="*80)
    print(f"Training sample: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Sample size: {train_data.shape[0]} observations")
    print(f"DLS smoothing parameter: {smoothing}")
    print("="*80)
    
    # Storage for results
    results = {
        'world_trends': {},
        'idiosyncratic_trends': {},
        'output_trends': {},
        'cycles': {},
        'initial_conditions': {},
        'alpha_estimates': {}
    }
    
    # Step 1: Apply DLS to all variables
    print("\n1. APPLYING DLS TO ALL VARIABLES")
    print("-" * 50)
    
    dls_trends = {}
    dls_cycles = {}
    
    for col in train_data.columns:
        trend, cycle = discounted_least_squares(train_data[col].values, smoothing)
        dls_trends[col] = trend
        dls_cycles[col] = cycle
        
        print(f"{col:10s}: Trend var={np.var(trend):.6f}, Cycle var={np.var(cycle):.6f}")
    
    # Also show computed real rates
    print("\nComputed real rates:")
    real_r_us = dls_trends['r_us'] - dls_trends['pi_us']
    real_r_ea = dls_trends['r_ea'] - dls_trends['pi_ea'] 
    real_r_jp = dls_trends['r_jp'] - dls_trends['pi_jp']
    
    print(f"{'real_r_us':<10s}: Mean={np.mean(real_r_us):.6f}, Var={np.var(real_r_us):.6f}")
    print(f"{'real_r_ea':<10s}: Mean={np.mean(real_r_ea):.6f}, Var={np.var(real_r_ea):.6f}")
    print(f"{'real_r_jp':<10s}: Mean={np.mean(real_r_jp):.6f}, Var={np.var(real_r_jp):.6f}")
    
    # Step 2: Extract world trends via Principal Components
    print("\n2. EXTRACTING WORLD TRENDS (PRINCIPAL COMPONENTS)")
    print("-" * 50)
    
    # Real rate world trend (using real rates)
    r_trends = np.column_stack([real_r_us, real_r_ea, real_r_jp])
    r_world_pc = extract_principal_component(r_trends)
    
    # Inflation world trend  
    pi_trends = np.column_stack([dls_trends['pi_us'], dls_trends['pi_ea'], dls_trends['pi_jp']])
    pi_world_pc = extract_principal_component(pi_trends)
    
    print(f"Real rate world trend - Mean: {np.mean(r_world_pc['pc_trend']):.6f}, Explained variance: {r_world_pc['explained_variance']:.3f}")
    print(f"Inflation world trend - Mean: {np.mean(pi_world_pc['pc_trend']):.6f}, Explained variance: {pi_world_pc['explained_variance']:.3f}")
    
    results['world_trends'] = {
        'r_w_trend': {
            'mean': np.mean(r_world_pc['pc_trend']),
            'variance': np.var(r_world_pc['pc_trend']),
            'innovation_variance': r_world_pc['innovation_variance'],
            'explained_variance': r_world_pc['explained_variance']
        },
        'pi_w_trend': {
            'mean': np.mean(pi_world_pc['pc_trend']),
            'variance': np.var(pi_world_pc['pc_trend']),
            'innovation_variance': pi_world_pc['innovation_variance'],
            'explained_variance': pi_world_pc['explained_variance']
        }
    }
    
    # Step 3: Compute idiosyncratic trends (country - world)
    print("\n3. COMPUTING IDIOSYNCRATIC TRENDS")
    print("-" * 50)
    
    countries = ['US', 'EA', 'JP']
    idio_variances = []
    
    # Recompute real rates for idiosyncratic calculation
    real_rates = {
        'US': real_r_us,
        'EA': real_r_ea, 
        'JP': real_r_jp
    }
    
    for i, country in enumerate(countries):
        # Real rate idiosyncratic (using real rates)
        r_idio = real_rates[country] - r_world_pc['pc_trend']
        pi_idio = dls_trends[f'pi_{country.lower()}'] - pi_world_pc['pc_trend']
        
        r_idio_var = np.var(r_idio)
        pi_idio_var = np.var(pi_idio)
        
        idio_variances.extend([r_idio_var, pi_idio_var])
        
        results['idiosyncratic_trends'][f'r_{country}_idio'] = {
            'mean': np.mean(r_idio),
            'variance': r_idio_var
        }
        results['idiosyncratic_trends'][f'pi_{country}_idio'] = {
            'mean': np.mean(pi_idio),
            'variance': pi_idio_var
        }
        
        print(f"{country} real rate idio var: {r_idio_var:.6f}")
        print(f"{country} inflation idio var: {pi_idio_var:.6f}")
    
    # Step 4: Output trends (direct DLS)
    print("\n4. OUTPUT TRENDS")
    print("-" * 50)
    
    output_variances = []
    for country in countries:
        y_col = f'y_{country.lower()}'
        y_trend = dls_trends[y_col]
        y_innovation_var = np.var(np.diff(y_trend))
        
        output_variances.append(y_innovation_var)
        
        results['output_trends'][f'y_{country}'] = {
            'mean': np.mean(y_trend),
            'variance': np.var(y_trend),
            'innovation_variance': y_innovation_var
        }
        
        print(f"{country} output trend innovation var: {y_innovation_var:.6f}")
    
    # Step 5: Cycle Analysis (AR(1) fits)
    print("\n5. CYCLE ANALYSIS (AR(1) MODELS)")
    print("-" * 50)
    
    cycle_variances = []
    for col in train_data.columns:
        cycle = dls_cycles[col]
        ar_results = fit_ar1_to_cycle(cycle)
        
        cycle_var = ar_results['residual_variance']
        cycle_variances.append(cycle_var)
        
        results['cycles'][f'cycle_{col.upper()}'] = {
            'variance': cycle_var,
            'ar_coefficient': ar_results.get('ar_coef', 0.0)
        }
        
        print(f"{col:10s}: AR(1) residual var={cycle_var:.6f}, φ={ar_results.get('ar_coef', 0.0):.3f}")
    
    # Step 6: Initial Conditions (HAC-based)
    print("\n6. INITIAL CONDITIONS (HAC ESTIMATION)")
    print("-" * 50)
    
    for col in train_data.columns:
        y_data = train_data[col].dropna().values
        if len(y_data) < 2:
            continue
            
        x_const = sm.add_constant(np.ones(len(y_data)))
        
        try:
            model = OLS(y_data, x_const).fit()
            hac_cov = cov_hac(model, use_correction=True)
            
            initial_mean = model.params[0]
            initial_var = hac_cov[0, 0]  # HAC variance of constant term
            
            results['initial_conditions'][col] = {
                'mean': initial_mean,
                'variance': initial_var
            }
            
            print(f"{col:10s}: Initial mean={initial_mean:.6f}, HAC var={initial_var:.6f}")
            
        except Exception as e:
            print(f"{col:10s}: HAC estimation failed, using sample moments")
            results['initial_conditions'][col] = {
                'mean': np.mean(y_data),
                'variance': np.var(y_data) / len(y_data)
            }
    
    # Add initial conditions for world trends (PC means and variances)
    results['initial_conditions']['r_w_trend'] = {
        'mean': results['world_trends']['r_w_trend']['mean'],
        'variance': results['world_trends']['r_w_trend']['variance']
    }
    results['initial_conditions']['pi_w_trend'] = {
        'mean': results['world_trends']['pi_w_trend']['mean'], 
        'variance': results['world_trends']['pi_w_trend']['variance']
    }
    
    print(f"{'r_w_trend':<10s}: Initial mean={results['initial_conditions']['r_w_trend']['mean']:.6f}, var={results['initial_conditions']['r_w_trend']['variance']:.6f}")
    print(f"{'pi_w_trend':<10s}: Initial mean={results['initial_conditions']['pi_w_trend']['mean']:.6f}, var={results['initial_conditions']['pi_w_trend']['variance']:.6f}")
    
    # Step 7: Estimate Alpha Parameters
    print("\n7. ALPHA ESTIMATION (METHOD OF MOMENTS)")
    print("-" * 50)
    
    # Separate alphas for trends and cycles
    trend_variances = []
    trend_variances.extend([results['world_trends']['r_w_trend']['innovation_variance'],
                           results['world_trends']['pi_w_trend']['innovation_variance']])
    trend_variances.extend(idio_variances)
    trend_variances.extend(output_variances)
    
    alpha_trend = estimate_alpha_method_of_moments(np.array(trend_variances), min_alpha)
    alpha_cycle = estimate_alpha_method_of_moments(np.array(cycle_variances), min_alpha)
    
    results['alpha_estimates'] = {
        'alpha_trend': alpha_trend,
        'alpha_cycle': alpha_cycle
    }
    
    print(f"Estimated α for trends: {alpha_trend:.3f}")
    print(f"Estimated α for cycles: {alpha_cycle:.3f}")
    
    # Step 8: Generate Inverse Gamma Prior Specifications
    print("\n8. INVERSE GAMMA PRIOR SPECIFICATIONS")
    print("-" * 50)
    
    def ig_prior_params(variance, alpha):
        """Convert variance to inverse gamma (α, β) parameters."""
        beta = variance * (alpha - 1)
        return alpha, beta
    
    # Print formatted results
    print(f"\n{'Parameter':<25} {'Component':<10} {'Alpha':<8} {'Beta':<12} {'Mean':<12} {'Mode':<12}")
    print("-" * 85)
    
    # World trends
    for param in ['r_w_trend', 'pi_w_trend']:
        var = results['world_trends'][param]['innovation_variance']
        alpha, beta = ig_prior_params(var, alpha_trend)
        mean = beta / (alpha - 1)
        mode = beta / (alpha + 1)
        print(f"stderr shk_{param:<12} {'Trend':<10} {alpha:<8.2f} {beta:<12.6f} {mean:<12.6f} {mode:<12.6f}")
    
    # Idiosyncratic trends
    for param, data in results['idiosyncratic_trends'].items():
        var = data['variance']
        alpha, beta = ig_prior_params(var, alpha_trend)
        mean = beta / (alpha - 1)
        mode = beta / (alpha + 1)
        print(f"stderr shk_{param:<12} {'Trend':<10} {alpha:<8.2f} {beta:<12.6f} {mean:<12.6f} {mode:<12.6f}")
    
    # Output trends
    for param, data in results['output_trends'].items():
        var = data['innovation_variance']
        alpha, beta = ig_prior_params(var, alpha_trend)
        mean = beta / (alpha - 1)
        mode = beta / (alpha + 1)
        print(f"stderr shk_{param:<12} {'Trend':<10} {alpha:<8.2f} {beta:<12.6f} {mean:<12.6f} {mode:<12.6f}")
    
    # Cycle shocks
    for param, data in results['cycles'].items():
        var = data['variance']
        alpha, beta = ig_prior_params(var, alpha_cycle)
        mean = beta / (alpha - 1)
        mode = beta / (alpha + 1)
        param_clean = param.replace('CYCLE_', '').lower()
        print(f"stderr shk_cycle_{param_clean:<7} {'Cycle':<10} {alpha:<8.2f} {beta:<12.6f} {mean:<12.6f} {mode:<12.6f}")
    
    print("\n" + "="*80)
    
    return results

def get_scalar(value):
    """Helper function to extract scalar from potentially nested structures."""
    if hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return float(value[0])
    else:
        return float(value)
    
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


# def run_presample_parameter_test(presample_data, gpm_file_path, param_config, 
#                                 initial_conditions=None, test_name="Test"):
#     """Run fixed-parameter evaluation on pre-sample data."""
#     print(f"\n=== RUNNING PRE-SAMPLE TEST: {test_name} ===")
#     print(f"Parameter configuration:")
#     for param, value in param_config.items():
#         print(f"  {param}: {value}")
    
#     try:
#         # Run fixed-parameter evaluation
#         results = evaluate_gpm_at_parameters(
#             gpm_file_path=gpm_file_path,
#             y=presample_data,
#             param_values=param_config,
#             initial_state_prior_overrides=initial_conditions,
#             num_sim_draws=0,  # Don't need simulation draws for calibration
#             plot_results=False,  # Don't generate plots for calibration
#             use_gamma_init_for_test=True,
#             gamma_init_scaling=1.0
#         )
        
#         if results and hasattr(results, 'log_likelihood'):
#             loglik = results.log_likelihood
#             print(f"✓ {test_name} completed - Log-likelihood: {loglik:.4f}")
            
#             return {
#                 'test_name': test_name,
#                 'parameters': param_config,
#                 'log_likelihood': loglik,
#                 'success': True,
#                 'results': results
#             }
#         else:
#             print(f"✗ {test_name} failed - No valid results")
#             return {
#                 'test_name': test_name,
#                 'parameters': param_config,
#                 'log_likelihood': np.nan,
#                 'success': False,
#                 'results': None
#             }
            
#     except Exception as e:
#         print(f"✗ {test_name} failed with error: {e}")
#         return {
#             'test_name': test_name,
#             'parameters': param_config,
#             'log_likelihood': np.nan,
#             'success': False,
#             'error': str(e),
#             'results': None
#         }


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

#data_sub = data_sub.dropna()
print(f"Data shape after dropping NaNs: {data_sub.shape}")

gpm_file_name = 'gpm_factor_y_pi_rshort.gpm'
gmp_file_path = os.path.join(SCRIPT_DIR, '..', 'clean_gpm_bvar_trends', 'models', gpm_file_name)

# Get the suggested prior parameters
priors_parameters = suggested_priors_gpm(data_sub)



run_model=False
if run_model:
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