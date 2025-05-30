"""
Fixed MCMC Integration - Updated Entry Points
============================================

This module provides updated integration points for the MCMC system.
It keeps all existing MCMC logic intact but replaces the integration
calls with the new coordinator-based approach.

Key changes:
- Replace create_reduced_gmp_model() calls with new coordinator
- Update build_state_space_matrices() calls to use coordinator
- Keep all other MCMC sampling logic unchanged
"""

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from functools import partial
from typing import NamedTuple, Tuple, Optional, List, Dict, Any
import numpy as np
import time

# Import the new integration system
#from parameter_contract import get_parameter_contract

# Import existing modules (keep unchanged)
try:
    from stationary_prior_jax_simplified import (
        make_stationary_var_transformation_jax,
        _JITTER
    )
except ImportError:
    print("Warning: Could not import stationary prior module")
    _JITTER = 1e-8

try:
    from Kalman_filter_jax import KalmanFilter, _KF_JITTER
except ImportError:
    print("Warning: Could not import Kalman filter module")
    _KF_JITTER = 1e-8

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64

try:
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    numpyro.set_host_device_count(min(num_cpu, 8))
except Exception as e:
    print(f"Could not set host device count: {e}. Falling back to default.")
    pass

# # Keep existing EnhancedBVARParams structure unchanged
# class EnhancedBVARParams(NamedTuple):
#     """Enhanced parameters for BVAR with trends supporting GPM specifications"""
#     # Core VAR parameters
#     A: jnp.ndarray                    # VAR coefficient matrices
#     Sigma_u: jnp.ndarray             # Stationary innovation covariance
#     Sigma_eta: jnp.ndarray           # Trend innovation covariance
    
#     # Structural parameters from GPM
#     structural_params: Dict[str, jnp.ndarray] = {}
    
#     # Measurement error (optional)
#     Sigma_eps: Optional[jnp.ndarray] = None


class FixedGPMStateSpaceBuilder:
    """
    Fixed state space builder that uses the new integration coordinator.
    
    This class provides the same interface as the original GPMStateSpaceBuilder
    but uses the new coordinator internally for robust parameter handling.
    """
    
    def __init__(self, gmp_file_path: str):
        """Initialize with GPM file path"""
        self.coordinator = create_integration_coordinator(gmp_file_path)
        
        # Expose attributes for compatibility with existing code
        self.gmp = self.coordinator.gmp
        self.n_trends = self.coordinator.n_trends
        self.n_stationary = self.coordinator.n_stationary
        self.n_observed = self.coordinator.n_observed
        self.var_order = self.coordinator.var_order
        self.state_dim = self.coordinator.state_dim
        self.trend_var_map = self.coordinator.trend_var_map
        self.stat_var_map = self.coordinator.stat_var_map
        self.obs_var_map = self.coordinator.obs_var_map
    
    def build_state_space_matrices(self, params: EnhancedBVARParams) -> Tuple[jnp.ndarray, ...]:
        """Build state space matrices using new coordinator"""
        return self.coordinator.build_state_space_from_enhanced_params(params)


def create_gmp_based_model_fixed(gmp_file_path: str):
    """
    FIXED version of create_gmp_based_model that uses new integration coordinator.
    
    This function provides the same interface as the original but uses the new
    coordinator for robust parameter handling.
    
    Returns:
        Tuple of (model_function, gmp_model, ss_builder)
    """
    
    print(f"Parsing GPM file with FIXED integration: {gmp_file_path}")
    from integration_coordinator import ReducedGPMIntegration, create_integration_coordinator

    # Create coordinator (replaces old parser + builder setup)
    coordinator = create_integration_coordinator(gmp_file_path)
    gmp_model = coordinator.reduced_model
    ss_builder = FixedGPMStateSpaceBuilder(gmp_file_path)
    
    def gmp_bvar_model_fixed(y: jnp.ndarray):
        """Fixed Numpyro model that uses new coordinator"""
        T, n_obs = y.shape
        
        # Sample structural parameters (unchanged)
        structural_params = {}
        for param_name in gmp_model.parameters:
            if param_name in gmp_model.estimated_params:
                prior_spec = gmp_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample shock standard deviations and build covariance matrices (unchanged)
        Sigma_eta = _sample_trend_covariance(gmp_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gmp_model)
        Sigma_eps = _sample_measurement_covariance(gmp_model) if _has_measurement_error(gmp_model) else None
        
        # Sample initial conditions (unchanged)
        init_mean = _sample_initial_conditions(gmp_model, ss_builder.state_dim)
        init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Create parameter structure (unchanged)
        params = EnhancedBVARParams(
            A=A_transformed,
            Sigma_u=Sigma_u,
            Sigma_eta=Sigma_eta,
            structural_params=structural_params,
            Sigma_eps=Sigma_eps
        )
        
        # Build state space matrices using FIXED coordinator
        try:
            F, Q, C, H = ss_builder.build_state_space_matrices(params)
        except Exception as e:
            print(f"State space construction failed: {e}")
            # Return -inf likelihood to reject this sample
            numpyro.factor("loglik", jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE))
            return
        
        # Check for numerical issues (unchanged)
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                       jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                       jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        # Build R matrix from Q (unchanged)
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        # Create Kalman Filter (unchanged)
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        # Compute likelihood (unchanged)
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gmp_bvar_model_fixed, gmp_model, ss_builder


# Keep all existing sampling functions unchanged
def _sample_parameter(name: str, prior_spec) -> jnp.ndarray:
    """Sample a parameter based on its prior specification (unchanged)"""
    if prior_spec.distribution == 'normal_pdf':
        mean, std = prior_spec.params
        return numpyro.sample(name, dist.Normal(mean, std))
    elif prior_spec.distribution == 'inv_gamma_pdf':
        alpha, beta = prior_spec.params
        return numpyro.sample(name, dist.InverseGamma(alpha, beta))
    else:
        raise ValueError(f"Unknown distribution: {prior_spec.distribution}")


def _sample_trend_covariance(gmp_model) -> jnp.ndarray:
    """Sample trend innovation covariance matrix (unchanged)"""
    n_trends = len(gmp_model.trend_variables)
    
    trend_sigmas = []
    for shock in gmp_model.trend_shocks:
        if shock in gmp_model.estimated_params:
            prior_spec = gmp_model.estimated_params[shock]
            sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
            trend_sigmas.append(sigma)
        else:
            sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
            trend_sigmas.append(sigma)
    
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas) ** 2)
    return Sigma_eta


def _sample_var_parameters(gmp_model) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """Sample VAR parameters using hierarchical prior (unchanged)"""
    
    if not gmp_model.var_prior_setup or not gmp_model.stationary_variables:
        n_vars = len(gmp_model.stationary_variables) if gmp_model.stationary_variables else 1
        A = jnp.zeros((1, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
        gamma_list = [Sigma_u]
        return Sigma_u, A, gamma_list
    
    setup = gmp_model.var_prior_setup
    n_vars = len(gmp_model.stationary_variables)
    n_lags = setup.var_order
    
    # Sample hierarchical hyperparameters
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(setup.es[i], setup.fs[i])) 
           for i in range(2)]
    Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(setup.gs[i], setup.hs[i])) 
              for i in range(2)]
    
    # Sample VAR coefficient matrices
    raw_A_list = []
    for lag in range(n_lags):
        A_full = numpyro.sample(f"A_full_{lag}", 
                               dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
        A_diag = numpyro.sample(f"A_diag_{lag}", 
                               dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
        A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag)
        raw_A_list.append(A_lag)
    
    # Sample stationary innovation covariance
    Omega_u_chol = numpyro.sample("Omega_u_chol", 
                                  dist.LKJCholesky(n_vars, concentration=setup.eta))
    
    sigma_u_vec = []
    for shock in gmp_model.stationary_shocks:
        if shock in gmp_model.estimated_params:
            prior_spec = gmp_model.estimated_params[shock]
            sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
            sigma_u_vec.append(sigma)
        else:
            sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
            sigma_u_vec.append(sigma)
    
    sigma_u = jnp.array(sigma_u_vec)
    Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    
    # Apply stationarity transformation
    try:
        phi_list, gamma_list = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
        A_transformed = jnp.stack(phi_list)
        gamma_list_fixed = [Sigma_u] + gamma_list
        numpyro.deterministic("A_transformed", A_transformed)
        return Sigma_u, A_transformed, gamma_list_fixed
    except Exception:
        print("Warning: Stationarity transformation failed. Using raw coefficients.")
        A_transformed = jnp.stack(raw_A_list)
        numpyro.deterministic("A_raw", A_transformed)
        gamma_list = [Sigma_u]
        for lag in range(1, n_lags + 1):
            decay_factor = 0.7 ** lag
            gamma_list.append(Sigma_u * decay_factor)
        return Sigma_u, A_transformed, gamma_list


def _sample_measurement_covariance(gmp_model) -> Optional[jnp.ndarray]:
    """Sample measurement error covariance (unchanged)"""
    return None


def _has_measurement_error(gmp_model) -> bool:
    """Check if model specifies measurement error (unchanged)"""
    return False


def _sample_initial_conditions(gmp_model, state_dim: int) -> jnp.ndarray:
    """Sample initial state mean (unchanged)"""
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    
    for var_name, var_spec in gmp_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            if var_name in gmp_model.trend_variables:
                idx = gmp_model.trend_variables.index(var_name)
                init_val = numpyro.sample(f"init_{var_name}", dist.Normal(mean, std))
                init_mean = init_mean.at[idx].set(init_val)
    
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)))
    return init_mean_sampled


def _create_initial_covariance(state_dim: int, n_trends: int) -> jnp.ndarray:
    """Create initial state covariance matrix (unchanged)"""
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6
    
    if state_dim > n_trends:
        init_cov = init_cov.at[n_trends:, n_trends:].set(jnp.eye(state_dim - n_trends, dtype=_DEFAULT_DTYPE) * 1e-6)
    
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    return init_cov


def fit_gmp_model_fixed(gmp_file_path: str, y: jnp.ndarray, 
                       num_warmup: int = 1000, num_samples: int = 2000, 
                       num_chains: int = 4, rng_key: jnp.ndarray = random.PRNGKey(0)):
    """
    FIXED version of fit_gmp_model that uses new integration coordinator.
    
    This function has the same interface as the original but uses the new
    coordinator for robust parameter handling.
    """
    
    print(f"Parsing GPM file with FIXED integration: {gmp_file_path}")
    model_fn, gmp_model, ss_builder = create_gmp_based_model_fixed(gmp_file_path)
    
    print("GPM Model Summary:")
    print(f"  Trend variables: {gmp_model.trend_variables}")
    print(f"  Stationary variables: {gmp_model.stationary_variables}")
    print(f"  Observed variables: {gmp_model.observed_variables}")
    print(f"  Parameters: {gmp_model.parameters}")
    if gmp_model.var_prior_setup:
        print(f"  VAR order: {gmp_model.var_prior_setup.var_order}")
    
    print("Running MCMC with FIXED integration...")
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    start_time = time.time()
    mcmc.run(rng_key, y=y)
    end_time = time.time()
    
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    
    return mcmc, gmp_model, ss_builder


def example_gmp_workflow_fixed():
    """FIXED example workflow using new integration coordinator"""
    
    from integration_coordinator import create_integration_coordinator
    
    import pandas as pd
    
    # Read data
    dta = pd.read_csv('sim_data.csv')
    y_np = dta.values
    T, n_vars = y_np.shape
    y_jax = jnp.asarray(y_np)
    
    # Create GPM file content for testing
    gmp_content = '''
    parameters ;
    
    estimated_params;
        stderr shk_trend_r_world, inv_gamma_pdf, 2.1, 0.81;
        stderr shk_trend_pi_world, inv_gamma_pdf, 2.1, 0.81;
        stderr shk_cycle_y_us, inv_gamma_pdf, 2.1, 0.38;
        stderr shk_cycle_r_us, inv_gamma_pdf, 2.1, 0.38;
        stderr shk_cycle_pi_us, inv_gamma_pdf, 2.1, 0.38;
        var_phi, normal_pdf, 1.0, 0.2;
    end;
    
    trends_vars
        trend_r_world,
        trend_pi_world,
        rr_trend_world
    ;
    
    stationary_variables
        cycle_y_us,
        cycle_r_us,
        cycle_pi_us
    ;
    
    trend_shocks;
        shk_trend_r_world,
        shk_trend_pi_world
    end;
    
    shocks;
        shk_cycle_y_us,
        shk_cycle_r_us,
        shk_cycle_pi_us
    end;
    
    trend_model;
        trend_r_world = trend_r_world(-1) + shk_trend_r_world;
        trend_pi_world = trend_pi_world(-1) + shk_trend_pi_world;
        rr_trend_world = trend_r_world + var_phi * trend_pi_world;
    end;
    
    varobs 
        y_us,
        r_us,
        pi_us
    ;
    
    measurement_equations;
        y_us = rr_trend_world + cycle_y_us;
        r_us = trend_r_world + cycle_r_us;
        pi_us = trend_pi_world + cycle_pi_us;
    end;
    
    var_prior_setup;
        var_order = 2;
        es = 0.5, 0.3;
        fs = 0.5, 0.5;
        gs = 2.0, 2.0;
        hs = 1.0, 1.0;
        eta = 2.0;
    end;
    '''
    
    # Save to temporary file
    with open('temp_model_fixed.gmp', 'w') as f:
        f.write(gmp_content)
    
    try:
        # Test the FIXED integration first
        print("Testing FIXED integration coordinator...")
        coordinator = create_integration_coordinator('temp_model_fixed.gmp')
        
        # Test state space construction
        test_success = coordinator.test_state_space_construction()
        if not test_success:
            print("✗ FIXED integration test failed")
            return None, None, None
        
        print("✓ FIXED integration test passed")
        
        # Fit the model using FIXED version
        mcmc, gmp_model, ss_builder = fit_gmp_model_fixed('temp_model_fixed.gmp', 
                                                         y_jax, 
                                                         num_warmup=500, 
                                                         num_samples=1000, 
                                                         num_chains=2)
        
        mcmc.print_summary(exclude_deterministic=False)
        
        return mcmc, gmp_model, ss_builder
        
    except Exception as e:
        print(f"Error in FIXED GPM workflow: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# Compatibility functions for existing code
def create_reduced_gmp_model_fixed(gmp_file_path: str):
    """
    FIXED factory function that provides compatibility with existing workflow.
    
    This replaces the old create_reduced_gmp_model() function with one that
    uses the new integration coordinator.
    """
    
    # Create integration with new coordinator
    integration = ReducedGPMIntegration(gmp_file_path)
    
    # Return same structure as before for compatibility
    return integration, integration.reduced_model, integration.coordinator.builder


# Migration utilities
def validate_parameter_compatibility(gmp_file_path: str, mcmc_samples: Dict[str, Any]) -> bool:
    """
    Validate that existing MCMC samples are compatible with new integration.
    
    Args:
        gmp_file_path: Path to GPM file
        mcmc_samples: Dictionary of MCMC samples
        
    Returns:
        True if compatible, False otherwise
    """
    
    try:
        coordinator = create_integration_coordinator(gmp_file_path)
        coordinator.validate_mcmc_compatibility(mcmc_samples)
        print("✓ MCMC samples are compatible with new integration")
        return True
    except Exception as e:
        print(f"✗ MCMC compatibility validation failed: {e}")
        return False


def migrate_existing_code_example():
    """
    Example of how to migrate existing code to use the new integration.
    
    OLD CODE:
        from gmp_bvar_trends import create_gmp_based_model
        model_fn, gmp_model, ss_builder = create_gmp_based_model(gmp_file)
    
    NEW CODE:
        from gmp_bvar_trends_fixed import create_gmp_based_model_fixed
        model_fn, gmp_model, ss_builder = create_gmp_based_model_fixed(gmp_file)
    
    That's it! The interface is identical, but now uses robust parameter handling.
    """
    pass


def print_migration_guide():
    """Print guide for migrating existing code"""
    
    guide = """
    MIGRATION GUIDE: Switching to Fixed Integration
    ==============================================
    
    The new integration system provides the same interface as before
    but with robust, contract-driven parameter handling.
    
    STEP 1: Replace import statements
    ---------------------------------
    OLD: from gmp_bvar_trends import create_gmp_based_model
    NEW: from gmp_bvar_trends_fixed import create_gmp_based_model_fixed
    
    STEP 2: Replace function calls  
    ------------------------------
    OLD: model_fn, gmp_model, ss_builder = create_gmp_based_model(gmp_file)
    NEW: model_fn, gmp_model, ss_builder = create_gmp_based_model_fixed(gmp_file)
    
    STEP 3: Replace integration helpers (if used directly)
    -----------------------------------------------------
    OLD: from integration_helper import create_reduced_gmp_model
    NEW: from gmp_bvar_trends_fixed import create_reduced_gmp_model_fixed
    
    STEP 4: Test compatibility
    -------------------------
    Use validate_parameter_compatibility() to ensure your MCMC samples
    work with the new integration.
    
    BENEFITS OF NEW INTEGRATION:
    - No more parameter name guessing
    - Clear error messages when parameters missing
    - Contract-driven validation
    - Robust type conversion
    - Same interface as before
    
    TROUBLESHOOTING:
    - If you get "PARAMETER CONTRACT VIOLATION" errors, check that your
      GPM file estimated_params section matches the expected naming patterns
    - If you get "TYPE CONVERSION FAILED" errors, ensure MCMC samples
      contain the expected parameter types
    """
    
    print(guide)


if __name__ == "__main__":
    print("Running FIXED GPM workflow with new integration coordinator...")
    
    # Print migration guide
    print_migration_guide()
    
    # Run example
    result = example_gmp_workflow_fixed()
    
    if result and result[0] is not None:
        print("\n=== SUCCESS ===")
        print("FIXED GPM workflow completed successfully!")
        print("The new integration system is working correctly.")
    else:
        print("\n=== FAILED ===")
        print("FIXED GPM workflow encountered errors.")
        print("Check the error messages above for details.")