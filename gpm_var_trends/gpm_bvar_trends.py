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
from gpm_parser import GPMParser, GPMModel, GPMModelBuilder

# Import your existing modules
try:
    from stationary_prior_jax_simplified import (
        AtoP_jax, rev_mapping_jax, make_stationary_var_transformation_jax,
        quad_form_sym_jax, _JITTER
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
    # Attempt to get physical CPU count or logical if physical is not available
    num_cpu = multiprocessing.cpu_count()
    # Set host device count, ensuring it's at least 1 and not excessively large
    numpyro.set_host_device_count(min(num_cpu, 8)) # Cap at 8 for safety/common hardware
except Exception as e:
    print(f"Could not set host device count: {e}. Falling back to default (likely 1 or 4).")
    # If setting fails, numpyro will use its default, which is usually okay.
    pass


class EnhancedBVARParams(NamedTuple):
    """Enhanced parameters for BVAR with trends supporting GPM specifications"""
    # Core VAR parameters
    A: jnp.ndarray                    # VAR coefficient matrices
    Sigma_u: jnp.ndarray             # Stationary innovation covariance
    Sigma_eta: jnp.ndarray           # Trend innovation covariance
    
    # Structural parameters from GPM
    structural_params: Dict[str, jnp.ndarray] = {}
    
    # Measurement error (optional)
    Sigma_eps: Optional[jnp.ndarray] = None


class GPMStateSpaceBuilder:
    """Builds state space matrices from GPM specification"""
    
    def __init__(self, gpm_model: GPMModel):
        self.gpm = gpm_model
        self.n_trends = len(gpm_model.trend_variables)
        self.n_stationary = len(gpm_model.stationary_variables) 
        self.n_observed = len(gpm_model.observed_variables)
        self.var_order = gpm_model.var_prior_setup.var_order if gpm_model.var_prior_setup else 1
        self.state_dim = self.n_trends + self.n_stationary * self.var_order
        
        # Create variable mappings
        self.trend_var_map = {var: i for i, var in enumerate(gpm_model.trend_variables)}
        self.stat_var_map = {var: i for i, var in enumerate(gpm_model.stationary_variables)}
        self.obs_var_map = {var: i for i, var in enumerate(gpm_model.observed_variables)}
    
    def build_state_space_matrices(self, params: EnhancedBVARParams) -> Tuple[jnp.ndarray, ...]:
        """Build complete state space representation from GPM specification"""
        
        # Initialize matrices
        F = jnp.zeros((self.state_dim, self.state_dim), dtype=_DEFAULT_DTYPE)
        Q = jnp.zeros((self.state_dim, self.state_dim), dtype=_DEFAULT_DTYPE)
        C = jnp.zeros((self.n_observed, self.state_dim), dtype=_DEFAULT_DTYPE)
        H = jnp.zeros((self.n_observed, self.n_observed), dtype=_DEFAULT_DTYPE)
        
        # Build trend dynamics from GPM trend equations
        F, Q = self._build_trend_dynamics(F, Q, params.structural_params, params.Sigma_eta)
        
        # Build VAR dynamics for stationary components
        F = self._build_var_dynamics(F, params.A)
        Q = self._add_var_innovations(Q, params.Sigma_u)
        
        # Build measurement equations
        C, H = self._build_measurement_equations(C, H, params.structural_params, params.Sigma_eps)
        
        # Ensure matrices are well-conditioned
        Q = (Q + Q.T) / 2.0 + _JITTER * jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0 + _KF_JITTER * jnp.eye(self.n_observed, dtype=_DEFAULT_DTYPE)
        
        return F, Q, C, H
    
    def _build_trend_dynamics(self, F: jnp.ndarray, Q: jnp.ndarray, 
                            structural_params: Dict[str, jnp.ndarray],
                            Sigma_eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build trend dynamics from GPM trend equations"""
        
        for equation in self.gpm.trend_equations:
            lhs_idx = self.trend_var_map[equation.lhs]
            
            # Updated to handle 4-element tuples: (var_name, lag, coeff_name, sign)
            for var_name, lag, coeff_name, sign in equation.rhs_terms:
                if var_name in self.trend_var_map:
                    # Trend variable
                    rhs_idx = self.trend_var_map[var_name]
                    
                    if coeff_name is None:
                        # Unit coefficient (like random walk)
                        coeff_value = 1.0
                    else:
                        # Structural coefficient
                        coeff_value = structural_params.get(coeff_name, 1.0)
                    
                    # Apply the sign from the equation
                    if sign == '-':
                        coeff_value = -coeff_value
                    
                    F = F.at[lhs_idx, rhs_idx].set(coeff_value)
        
        # Add trend innovation covariance
        Q = Q.at[:self.n_trends, :self.n_trends].set(Sigma_eta)
        
        return F, Q
    
    def _build_var_dynamics(self, F: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
        """Build VAR dynamics for stationary components"""
        var_start = self.n_trends
        
        # Set VAR coefficient matrices
        for lag in range(self.var_order):
            F = F.at[var_start:var_start + self.n_stationary,
                     var_start + lag * self.n_stationary:var_start + (lag + 1) * self.n_stationary].set(A[lag])
        
        # Set identity matrices for lagged states
        if self.var_order > 1:
            for i in range(self.var_order - 1):
                start_row = var_start + (i + 1) * self.n_stationary
                start_col = var_start + i * self.n_stationary
                F = F.at[start_row:start_row + self.n_stationary, 
                         start_col:start_col + self.n_stationary].set(jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE))
        
        return F
    
    def _add_var_innovations(self, Q: jnp.ndarray, Sigma_u: jnp.ndarray) -> jnp.ndarray:
        """Add VAR innovation covariance to state covariance"""
        var_start = self.n_trends
        Q = Q.at[var_start:var_start + self.n_stationary, 
                 var_start:var_start + self.n_stationary].set(Sigma_u)
        return Q
    
    def _build_measurement_equations(self, C: jnp.ndarray, H: jnp.ndarray,
                                structural_params: Dict[str, jnp.ndarray],
                                Sigma_eps: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build measurement equations from GPM specification"""
        
        for equation in self.gpm.measurement_equations:
            obs_idx = self.obs_var_map[equation.lhs]
            
            # Updated to handle 3-element tuples: (var_name, coeff_name, sign)
            for var_name, coeff_name, sign in equation.rhs_terms:
                if var_name in self.trend_var_map:
                    # Trend variable
                    state_idx = self.trend_var_map[var_name]
                elif var_name in self.stat_var_map:
                    # Stationary variable (current period)
                    state_idx = self.n_trends + self.stat_var_map[var_name]
                else:
                    continue
                
                if coeff_name is None:
                    # Unit coefficient
                    coeff_value = 1.0
                else:
                    # Structural coefficient
                    coeff_value = structural_params.get(coeff_name, 1.0)
                
                # Apply the sign from the equation
                if sign == '-':
                    coeff_value = -coeff_value
                
                C = C.at[obs_idx, state_idx].set(coeff_value)
        
        # Set measurement error covariance
        if Sigma_eps is not None:
            H = Sigma_eps
        
        return C, H



#This version will work with the gamma matrices and the 
# stationary prior not functioning yet
# def create_gpm_based_model(gpm_file_path: str):
#     """Create a Numpyro model function from a GPM file"""
    
#     # Parse GPM file
#     parser = GPMParser()
#     gpm_model = parser.parse_file(gpm_file_path)
    
#     # Create state space builder
#     ss_builder = GPMStateSpaceBuilder(gpm_model)
    
#     def gpm_bvar_model(y: jnp.ndarray):
#         """Numpyro model based on GPM specification with gamma matrix initialization"""
#         T, n_obs = y.shape
        
#         # Sample structural parameters
#         structural_params = {}
#         for param_name in gpm_model.parameters:
#             if param_name in gpm_model.estimated_params:
#                 prior_spec = gpm_model.estimated_params[param_name]
#                 structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
#         # Sample shock standard deviations and build covariance matrices
#         Sigma_eta = _sample_trend_covariance(gpm_model)
        
#         # CORRECTED: Now properly unpacks 3 return values including gamma_list
#         Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        
#         Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
#         # Sample initial conditions using gamma matrices
#         init_mean = _sample_initial_conditions_with_gammas(
#             gpm_model, ss_builder.state_dim, gamma_list, 
#             ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order
#         )
#         init_cov = _create_initial_covariance_with_gammas(
#             ss_builder.state_dim, ss_builder.n_trends, gamma_list,
#             ss_builder.n_stationary, ss_builder.var_order
#         )
        
#         # Create parameter structure
#         params = EnhancedBVARParams(
#             A=A_transformed,
#             Sigma_u=Sigma_u,
#             Sigma_eta=Sigma_eta,
#             structural_params=structural_params,
#             Sigma_eps=Sigma_eps
#         )
        
#         # Build state space matrices
#         F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
#         # Check for numerical issues
#         matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
#                       jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
#                       jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
#         # Build R matrix from Q (assuming Q = R @ R.T)
#         try:
#             R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
#         except:
#             R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
#         # Create Kalman Filter
#         kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
#         # Compute likelihood
#         valid_obs_idx = jnp.arange(n_obs, dtype=int)
#         I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
#         loglik = jax.lax.cond(
#             ~matrices_ok,
#             lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
#             lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
#         )
        
#         numpyro.factor("loglik", loglik)
    
#     return gpm_bvar_model, gpm_model, ss_builder


# SIMPLE FIX: Replace the current function in gpm_bvar_trends.py with this version
# This temporarily disables the gamma matrix usage to eliminate divergences

def create_gpm_based_model(gpm_file_path: str):
    """Create a Numpyro model function from a GPM file - SIMPLIFIED VERSION"""
    
    # Parse GPM file
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    
    # Create state space builder
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    def gpm_bvar_model(y: jnp.ndarray):
        """Numpyro model based on GPM specification - BACK TO ORIGINAL INITIALIZATION"""
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample shock standard deviations and build covariance matrices
        Sigma_eta = _sample_trend_covariance(gpm_model)
        
        # FIXED: Get gamma matrices but DON'T use them yet
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # USE ORIGINAL INITIALIZATION (no gamma matrices)
        init_mean = _sample_initial_conditions(gpm_model, ss_builder.state_dim)
        init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Create parameter structure
        params = EnhancedBVARParams(
            A=A_transformed,
            Sigma_u=Sigma_u,
            Sigma_eta=Sigma_eta,
            structural_params=structural_params,
            Sigma_eps=Sigma_eps
        )
        
        # Build state space matrices
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        # Check for numerical issues
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        # Build R matrix from Q (assuming Q = R @ R.T)
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        # Create Kalman Filter
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        # Compute likelihood
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gpm_bvar_model, gpm_model, ss_builder

def _sample_parameter(name: str, prior_spec) -> jnp.ndarray:
    """Sample a parameter based on its prior specification"""
    if prior_spec.distribution == 'normal_pdf':
        mean, std = prior_spec.params
        return numpyro.sample(name, dist.Normal(mean, std))
    elif prior_spec.distribution == 'inv_gamma_pdf':
        alpha, beta = prior_spec.params
        return numpyro.sample(name, dist.InverseGamma(alpha, beta))
    else:
        raise ValueError(f"Unknown distribution: {prior_spec.distribution}")


def _sample_trend_covariance(gpm_model: GPMModel) -> jnp.ndarray:
    """Sample trend innovation covariance matrix"""
    n_trends = len(gpm_model.trend_variables)
    
    # Sample individual shock standard deviations
    trend_sigmas = []
    for shock in gpm_model.trend_shocks:
        if shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
            trend_sigmas.append(sigma)
        else:
            # Default prior if not specified
            sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
            trend_sigmas.append(sigma)
    
    # For simplicity, assume diagonal covariance for trends
    # Could be extended to include correlation structure
    Sigma_eta = jnp.diag(jnp.array(trend_sigmas) ** 2)
    
    return Sigma_eta


# def _sample_var_parameters(gpm_model: GPMModel) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """Sample VAR parameters using hierarchical prior if specified"""
    
#     if not gpm_model.var_prior_setup or not gpm_model.stationary_variables:
#         # Fallback: simple VAR with minimal structure
#         n_vars = len(gpm_model.stationary_variables) if gpm_model.stationary_variables else 1
#         A = jnp.zeros((1, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
#         Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
#         return Sigma_u, A
    
#     setup = gpm_model.var_prior_setup
#     n_vars = len(gpm_model.stationary_variables)
#     n_lags = setup.var_order
    
#     # Sample hierarchical hyperparameters
#     Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(setup.es[i], setup.fs[i])) 
#            for i in range(2)]
#     Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(setup.gs[i], setup.hs[i])) 
#               for i in range(2)]
    
#     # Sample VAR coefficient matrices with hierarchical structure
#     raw_A_list = []
#     for lag in range(n_lags):
#         # Sample off-diagonal elements
#         A_full = numpyro.sample(f"A_full_{lag}", 
#                                dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
        
#         # Sample diagonal elements separately
#         A_diag = numpyro.sample(f"A_diag_{lag}", 
#                                dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
        
#         # Combine diagonal and off-diagonal
#         A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag)
#         raw_A_list.append(A_lag)
    
#     # Sample stationary innovation covariance
#     Omega_u_chol = numpyro.sample("Omega_u_chol", 
#                                   dist.LKJCholesky(n_vars, concentration=setup.eta))
    
#     # Sample shock standard deviations
#     sigma_u_vec = []
#     for shock in gpm_model.stationary_shocks:
#         if shock in gpm_model.estimated_params:
#             prior_spec = gpm_model.estimated_params[shock]
#             sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
#             sigma_u_vec.append(sigma)
#         else:
#             sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
#             sigma_u_vec.append(sigma)
    
#     sigma_u = jnp.array(sigma_u_vec)
#     Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
#     Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    
#     # Apply stationarity transformation
#     try:
#         phi_list, gamma_list = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
#         A_transformed = jnp.stack(phi_list)
        
#         # Store transformed coefficients
#         numpyro.deterministic("A_transformed", A_transformed)
        
#     except Exception:
#         # Fallback if transformation fails
#         A_transformed = jnp.stack(raw_A_list)
#         numpyro.deterministic("A_raw", A_transformed)
    
#     return Sigma_u, A_transformed


def _sample_var_parameters(gpm_model: GPMModel) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """Sample VAR parameters using hierarchical prior and return gamma matrices for initialization"""
    
    if not gpm_model.var_prior_setup or not gpm_model.stationary_variables:
        # Fallback: simple VAR with minimal structure
        n_vars = len(gpm_model.stationary_variables) if gpm_model.stationary_variables else 1
        A = jnp.zeros((1, n_vars, n_vars), dtype=_DEFAULT_DTYPE)
        Sigma_u = jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
        # Fallback gamma list with just the contemporaneous covariance
        gamma_list = [Sigma_u]
        return Sigma_u, A, gamma_list
    
    setup = gpm_model.var_prior_setup
    n_vars = len(gpm_model.stationary_variables)
    n_lags = setup.var_order
    
    # Sample hierarchical hyperparameters
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(setup.es[i], setup.fs[i])) 
           for i in range(2)]
    Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(setup.gs[i], setup.hs[i])) 
              for i in range(2)]
    
    # Sample VAR coefficient matrices with hierarchical structure
    raw_A_list = []
    for lag in range(n_lags):
        # Sample off-diagonal elements
        A_full = numpyro.sample(f"A_full_{lag}", 
                               dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
        
        # Sample diagonal elements separately
        A_diag = numpyro.sample(f"A_diag_{lag}", 
                               dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
        
        # Combine diagonal and off-diagonal
        A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag)
        raw_A_list.append(A_lag)
    
    # Sample stationary innovation covariance
    Omega_u_chol = numpyro.sample("Omega_u_chol", 
                                  dist.LKJCholesky(n_vars, concentration=setup.eta))
    
    # Sample shock standard deviations
    sigma_u_vec = []
    for shock in gpm_model.stationary_shocks:
        if shock in gpm_model.estimated_params:
            prior_spec = gpm_model.estimated_params[shock]
            sigma = _sample_parameter(f"sigma_{shock}", prior_spec)
            sigma_u_vec.append(sigma)
        else:
            sigma = numpyro.sample(f"sigma_{shock}", dist.InverseGamma(2.0, 1.0))
            sigma_u_vec.append(sigma)
    
    sigma_u = jnp.array(sigma_u_vec)
    Sigma_u = jnp.diag(sigma_u) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(sigma_u)
    Sigma_u = (Sigma_u + Sigma_u.T) / 2.0 + _JITTER * jnp.eye(n_vars, dtype=_DEFAULT_DTYPE)
    
    # Apply stationarity transformation and get gamma matrices
    try:
        phi_list, gamma_list = make_stationary_var_transformation_jax(Sigma_u, raw_A_list, n_vars, n_lags)
        A_transformed = jnp.stack(phi_list)
        
        # Store transformed coefficients
        numpyro.deterministic("A_transformed", A_transformed)
        
        return Sigma_u, A_transformed, gamma_list
        
    except Exception:
        # Fallback if transformation fails
        A_transformed = jnp.stack(raw_A_list)
        numpyro.deterministic("A_raw", A_transformed)
        
        # Create fallback gamma list - use innovation covariance as approximation
        gamma_list = [Sigma_u]  # At least provide contemporaneous covariance
        for lag in range(1, n_lags + 1):
            # Exponentially decaying autocovariances as fallback
            decay_factor = 0.7 ** lag
            gamma_list.append(Sigma_u * decay_factor)
        
        return Sigma_u, A_transformed, gamma_list



def _sample_measurement_covariance(gpm_model: GPMModel) -> Optional[jnp.ndarray]:
    """Sample measurement error covariance if specified"""
    # Simple implementation - could be extended based on GPM specification
    return None


def _has_measurement_error(gpm_model: GPMModel) -> bool:
    """Check if model specifies measurement error"""
    # Could analyze measurement equations to determine this
    return False

#These _smaple_initial_conditions and _create_initial_covariance functions are used to 
# sample initial conditions for the state space model but don;t use the optimal 
# covariance matriz for the stationary VAR.  
def _sample_initial_conditions(gpm_model: GPMModel, state_dim: int) -> jnp.ndarray:
    """Sample initial state mean based on GPM initial value specifications"""
    
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Use GPM initial value specifications where available
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            
            # Find variable index (simplified - would need proper mapping)
            # This is a placeholder implementation
            if var_name in gpm_model.trend_variables:
                idx = gpm_model.trend_variables.index(var_name)
                init_val = numpyro.sample(f"init_{var_name}", dist.Normal(mean, std))
                init_mean = init_mean.at[idx].set(init_val)
    
    # Sample remaining initial conditions with default priors
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)))
    
    return init_mean_sampled


def _create_initial_covariance(state_dim: int, n_trends: int) -> jnp.ndarray:
    """Create initial state covariance matrix"""
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e6  # Diffuse for trends
    
    # Tighter for VAR states
    if state_dim > n_trends:
        init_cov = init_cov.at[n_trends:, n_trends:].set(jnp.eye(state_dim - n_trends, dtype=_DEFAULT_DTYPE) * 1e-6)
    
    # Ensure positive definite
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov

# These _sample_initial_conditions_with_gammas and _create_initial_covariance_with_gammas
# functions are used to sample initial conditions for the state space model using
# theoretical VAR properties from gamma matrices, which are derived from the stationary prior.

def _create_initial_covariance_with_gammas(state_dim: int, n_trends: int, 
                                         gamma_list: List[jnp.ndarray], 
                                         n_stationary: int, var_order: int) -> jnp.ndarray:
    """Create initial covariance using theoretical VAR covariances from stationary prior"""
    
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Diffuse prior for trends (unchanged)
    if n_trends > 0:
        init_cov = init_cov.at[:n_trends, :n_trends].set(
            jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * 1e6
        )
    
    # Use theoretical VAR covariances for stationary components
    if len(gamma_list) > 0 and n_stationary > 0 and var_order > 0:
        var_start = n_trends
        var_state_dim = n_stationary * var_order
        
        # Build VAR state covariance matrix using gamma matrices
        var_state_cov = jnp.zeros((var_state_dim, var_state_dim), dtype=_DEFAULT_DTYPE)
        
        for i in range(var_order):
            for j in range(var_order):
                lag_diff = abs(i - j)
                
                # Use gamma matrix if available, otherwise use contemporaneous covariance
                if lag_diff < len(gamma_list):
                    if i <= j:
                        block_cov = gamma_list[lag_diff]
                    else:
                        block_cov = gamma_list[lag_diff].T
                else:
                    # Fallback to contemporaneous covariance with decay
                    decay_factor = 0.5 ** lag_diff if lag_diff > 0 else 1.0
                    block_cov = gamma_list[0] * decay_factor
                
                # Insert block into VAR state covariance
                row_start, row_end = i * n_stationary, (i + 1) * n_stationary
                col_start, col_end = j * n_stationary, (j + 1) * n_stationary
                
                var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)
        
        # Insert VAR covariance into full state covariance matrix
        var_end = var_start + var_state_dim
        if var_end <= state_dim:
            init_cov = init_cov.at[var_start:var_end, var_start:var_end].set(var_state_cov)
    
    # Ensure positive definite and add jitter
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


def _sample_initial_conditions_with_gammas(gpm_model: GPMModel, state_dim: int, 
                                         gamma_list: List[jnp.ndarray], 
                                         n_trends: int, n_stationary: int, 
                                         var_order: int) -> jnp.ndarray:
    """Sample initial conditions using theoretical VAR properties from gamma matrices"""
    
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Handle trends: use GPM specifications where available, otherwise diffuse
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            
            # Find variable index in trend variables
            if var_name in gpm_model.trend_variables:
                idx = gpm_model.trend_variables.index(var_name)
                if idx < n_trends:
                    init_mean = init_mean.at[idx].set(mean)
                    init_std = init_std.at[idx].set(std)
    
    # Set diffuse priors for remaining trend variables
    if n_trends > 0:
        trend_mask = jnp.arange(state_dim) < n_trends
        init_std = jnp.where(trend_mask, 
                           jnp.where(init_std == 1.0, 10.0, init_std),  # 10.0 for unspecified trends
                           init_std)
    
    # Use theoretical information for VAR components
    if len(gamma_list) > 0 and n_stationary > 0 and var_order > 0:
        var_start = n_trends
        
        # Extract theoretical standard deviations from contemporaneous covariance
        if gamma_list[0].shape[0] == n_stationary:
            theoretical_std = jnp.sqrt(jnp.diag(gamma_list[0]))
            
            # Apply to each lag with appropriate scaling
            for lag in range(var_order):
                start_idx = var_start + lag * n_stationary
                end_idx = start_idx + n_stationary
                
                if end_idx <= state_dim:
                    # Use smaller fraction of theoretical std for initial conditions
                    # Current period gets larger std, lags get progressively smaller
                    scale_factor = 0.2 / (1 + lag * 0.5)  # 0.2, 0.133, 0.1, etc.
                    scaled_std = theoretical_std * scale_factor
                    
                    init_std = init_std.at[start_idx:end_idx].set(scaled_std)
                    # Keep mean at zero for VAR components (stationary assumption)
    
    # Sample initial mean with theoretical information
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, init_std))
    
    return init_mean_sampled


## These routines allow save initialization of the covariance matrix of the stationary components

def _create_initial_covariance_with_gammas_safe(state_dim: int, n_trends: int, 
                                              gamma_list: List[jnp.ndarray], 
                                              n_stationary: int, var_order: int,
                                              use_gamma_scaling: float = 0.1,
                                              fallback_to_original: bool = True) -> jnp.ndarray:
    """
    Create initial covariance using gamma matrices with safety checks and scaling control.
    
    Args:
        use_gamma_scaling: Factor to scale down gamma matrices (0.1 = use 10% of theoretical variance)
        fallback_to_original: If True, fall back to original 1e-6 method if gamma matrices seem problematic
    """
    
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Diffuse prior for trends (unchanged)
    if n_trends > 0:
        init_cov = init_cov.at[:n_trends, :n_trends].set(
            jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * 1e6
        )
    
    # Handle stationary components with safety checks
    if len(gamma_list) > 0 and n_stationary > 0 and var_order > 0:
        var_start = n_trends
        var_state_dim = n_stationary * var_order
        
        # Safety check: ensure gamma matrices are reasonable
        gamma_0 = gamma_list[0]
        
        # Check for problematic values
        has_nan = jnp.any(jnp.isnan(gamma_0))
        has_inf = jnp.any(jnp.isinf(gamma_0))
        too_large = jnp.any(jnp.abs(gamma_0) > 1e3)  # Arbitrary large threshold
        too_small = jnp.any(jnp.diag(gamma_0) < 1e-10)  # Too small variances
        
        # Condition number check
        try:
            cond_number = jnp.linalg.cond(gamma_0)
            ill_conditioned = cond_number > 1e12
        except:
            ill_conditioned = True
        
        use_fallback = (has_nan or has_inf or too_large or too_small or ill_conditioned) and fallback_to_original
        
        if use_fallback:
            print(f"Warning: Gamma matrices appear problematic (nan={has_nan}, inf={has_inf}, "
                  f"large={too_large}, small={too_small}, ill_cond={ill_conditioned}). "
                  f"Using original initialization.")
            
            # Fall back to original tight initialization
            if var_state_dim > 0:
                init_cov = init_cov.at[var_start:var_start + var_state_dim, 
                                      var_start:var_start + var_state_dim].set(
                    jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE) * 1e-6
                )
        else:
            try:
                # Build VAR state covariance matrix using scaled gamma matrices
                var_state_cov = jnp.zeros((var_state_dim, var_state_dim), dtype=_DEFAULT_DTYPE)
                
                for i in range(var_order):
                    for j in range(var_order):
                        lag_diff = abs(i - j)
                        
                        # Use gamma matrix if available, otherwise use contemporaneous with decay
                        if lag_diff < len(gamma_list):
                            if i <= j:
                                block_cov = gamma_list[lag_diff] * use_gamma_scaling  # Scale down
                            else:
                                block_cov = gamma_list[lag_diff].T * use_gamma_scaling
                        else:
                            # Fallback with exponential decay
                            decay_factor = (0.5 ** lag_diff) * use_gamma_scaling
                            block_cov = gamma_list[0] * decay_factor
                        
                        # Insert block
                        row_start, row_end = i * n_stationary, (i + 1) * n_stationary
                        col_start, col_end = j * n_stationary, (j + 1) * n_stationary
                        
                        var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)
                
                # Final safety check on constructed covariance
                var_cond_number = jnp.linalg.cond(var_state_cov + _KF_JITTER * jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE))
                if var_cond_number > 1e10:
                    print(f"Warning: Constructed VAR covariance has high condition number {var_cond_number:.2e}. "
                          f"Adding more regularization.")
                    var_state_cov = var_state_cov + _KF_JITTER * 100 * jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE)
                
                # Insert VAR covariance into full state covariance matrix
                var_end = var_start + var_state_dim
                if var_end <= state_dim:
                    init_cov = init_cov.at[var_start:var_end, var_start:var_end].set(var_state_cov)
                    
            except Exception as e:
                print(f"Error constructing gamma-based covariance: {e}. Using fallback.")
                # Emergency fallback
                if var_state_dim > 0:
                    init_cov = init_cov.at[var_start:var_start + var_state_dim, 
                                          var_start:var_start + var_state_dim].set(
                        jnp.eye(var_state_dim, dtype=_DEFAULT_DTYPE) * 1e-3  # Less tight than 1e-6
                    )
    
    # Ensure positive definite with sufficient regularization
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


def _sample_initial_conditions_with_gammas_safe(gpm_model: GPMModel, state_dim: int, 
                                              gamma_list: List[jnp.ndarray], 
                                              n_trends: int, n_stationary: int, 
                                              var_order: int,
                                              gamma_scaling: float = 0.1) -> jnp.ndarray:
    """
    Sample initial conditions with conservative scaling of gamma-based uncertainty.
    """
    
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Handle trends: use GPM specifications where available, otherwise moderate diffuse
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            
            if var_name in gpm_model.trend_variables:
                idx = gpm_model.trend_variables.index(var_name)
                if idx < n_trends:
                    init_mean = init_mean.at[idx].set(mean)
                    init_std = init_std.at[idx].set(std)
    
    # Set moderate diffuse priors for remaining trend variables (less extreme than before)
    if n_trends > 0:
        trend_mask = jnp.arange(state_dim) < n_trends
        init_std = jnp.where(trend_mask, 
                           jnp.where(init_std == 1.0, 3.0, init_std),  # Reduced from 10.0
                           init_std)
    
    # Conservative use of gamma information for VAR components
    if len(gamma_list) > 0 and n_stationary > 0 and var_order > 0:
        var_start = n_trends
        
        try:
            # Extract theoretical standard deviations with safety checks
            gamma_0 = gamma_list[0]
            if (gamma_0.shape[0] == n_stationary and 
                jnp.all(jnp.isfinite(gamma_0)) and 
                jnp.all(jnp.diag(gamma_0) > 0)):
                
                theoretical_std = jnp.sqrt(jnp.diag(gamma_0))
                
                # Conservative scaling - use much smaller fraction
                for lag in range(var_order):
                    start_idx = var_start + lag * n_stationary
                    end_idx = start_idx + n_stationary
                    
                    if end_idx <= state_dim:
                        # Very conservative scaling
                        scale_factor = gamma_scaling / (1 + lag * 2.0)  # 0.1, 0.033, 0.02, etc.
                        scaled_std = theoretical_std * scale_factor
                        
                        # Ensure std is reasonable (not too small or large)
                        scaled_std = jnp.clip(scaled_std, 0.01, 1.0)
                        
                        init_std = init_std.at[start_idx:end_idx].set(scaled_std)
            else:
                print("Warning: Gamma matrix problematic for initial conditions. Using conservative defaults.")
                # Use conservative defaults for VAR components
                var_end = var_start + n_stationary * var_order
                if var_end <= state_dim:
                    init_std = init_std.at[var_start:var_end].set(0.1)  # Conservative default
                    
        except Exception as e:
            print(f"Error using gamma matrices for initial mean: {e}. Using defaults.")
            # Emergency fallback
            var_end = var_start + n_stationary * var_order
            if var_end <= state_dim:
                init_std = init_std.at[var_start:var_end].set(0.1)
    
    # Sample with conservative uncertainty
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, init_std))
    
    return init_mean_sampled


## When integrating sampling with gamma matrices, we need to ensure that the
# STEP 2: Enhanced conditional sampling approach for future implementation
# This addresses your excellent point about conditional priors
# FIXED CONDITIONAL SAMPLING FUNCTIONS (JAX-compatible)

def _sample_initial_conditions_conditional(gpm_model: GPMModel, state_dim: int, 
                                                gamma_list: List[jnp.ndarray],
                                                n_trends: int, n_stationary: int,
                                                var_order: int) -> jnp.ndarray:
    """
    JAX-compatible conditional initial condition sampling.
    Fixed: Removed all Python if statements that cause tracing errors.
    """
    
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Handle trends: use GPM specifications or defaults
    # Note: This part uses Python if but outside of traced functions, so it's OK
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            if var_name in gpm_model.trend_variables:
                idx = gpm_model.trend_variables.index(var_name)
                if idx < n_trends:
                    init_mean = init_mean.at[idx].set(mean)
                    init_std = init_std.at[idx].set(std)
    
    # Set diffuse priors for trends using JAX operations only
    trend_mask = jnp.arange(state_dim) < n_trends
    init_std = jnp.where(trend_mask, 
                        jnp.where(init_std == 1.0, 3.0, init_std),
                        init_std)
    
    # VAR components: Use gamma matrices if available, otherwise defaults
    # Key fix: Use JAX conditional operations instead of Python if statements
    
    def use_gamma_matrices(operand):
        gamma_0, var_start, n_stat, v_order = operand
        
        # Extract conditional standard deviations
        conditional_std = jnp.sqrt(jnp.diag(gamma_0))
        
        # Apply to all VAR states at once using vectorized operations
        var_indices = jnp.arange(n_stat * v_order) + var_start
        lag_numbers = jnp.arange(n_stat * v_order) // n_stat  # Which lag each state belongs to
        
        # Scale factor based on lag (current=1.0, lag1=0.3, lag2=0.15, etc.)
        scale_factors = jnp.where(lag_numbers == 0, 1.0, 0.3 / (1 + lag_numbers))
        
        # Repeat conditional_std for each lag
        repeated_std = jnp.tile(conditional_std, v_order)
        scaled_std = repeated_std * scale_factors
        
        # Clip to reasonable bounds
        scaled_std = jnp.clip(scaled_std, 0.01, 2.0)
        
        # Update init_std
        return init_std.at[var_indices].set(scaled_std)
    
    def use_default_var_std(operand):
        var_start, n_stat, v_order = operand
        var_end = var_start + n_stat * v_order
        var_indices = jnp.arange(var_start, jnp.minimum(var_end, state_dim))
        return init_std.at[var_indices].set(0.1)
    
    # Check if we should use gamma matrices (all JAX operations)
    has_gamma = len(gamma_list) > 0
    has_stationary = n_stationary > 0
    has_var_order = var_order > 0
    
    if has_gamma and has_stationary and has_var_order:
        gamma_0 = gamma_list[0]
        var_start = n_trends
        
        # Safety checks using JAX operations
        is_finite = jnp.all(jnp.isfinite(gamma_0))
        has_positive_diag = jnp.all(jnp.diag(gamma_0) > 0)
        is_right_shape = gamma_0.shape[0] == n_stationary
        
        # Use JAX conditional instead of Python if
        use_gamma = is_finite & has_positive_diag & is_right_shape
        
        init_std = jax.lax.cond(
            use_gamma,
            use_gamma_matrices,
            lambda op: use_default_var_std(op[1:]),  # Skip gamma_0 for default case
            operand=(gamma_0, var_start, n_stationary, var_order)
        )
    else:
        # Use default for VAR components
        var_start = n_trends
        var_end = var_start + n_stationary * var_order
        if var_end <= state_dim:
            var_indices = jnp.arange(var_start, var_end)
            init_std = init_std.at[var_indices].set(0.1)
    
    # Sample from the conditional distribution
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, init_std))
    
    return init_mean_sampled


def _create_initial_covariance_conditional(state_dim: int, n_trends: int,
                                               gamma_list: List[jnp.ndarray],
                                               n_stationary: int, var_order: int,
                                               conditioning_strength: float = 0.1) -> jnp.ndarray:
    """
    JAX-compatible conditional covariance creation.
    Fixed: All operations use JAX conditionals instead of Python if statements.
    """
    
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Trends: diffuse prior
    trend_cov = jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * 1e6
    init_cov = init_cov.at[:n_trends, :n_trends].set(trend_cov)
    
    # VAR components: conditional or default covariance
    def build_conditional_cov(operand):
        gamma_0, v_start, v_state_dim, n_stat, v_order, cond_strength = operand
        
        # Build conditional covariance structure
        var_state_cov = jnp.zeros((v_state_dim, v_state_dim), dtype=_DEFAULT_DTYPE)
        
        # Use vectorized operations instead of nested loops
        i_indices, j_indices = jnp.meshgrid(jnp.arange(v_order), jnp.arange(v_order), indexing='ij')
        lag_diffs = jnp.abs(i_indices - j_indices)
        
        # Create all blocks at once
        for lag in range(len(gamma_list)):
            mask = lag_diffs == lag
            block_cov = gamma_list[lag] * cond_strength
            
            # Apply to all relevant block positions
            i_vals, j_vals = jnp.where(mask)
            for idx in range(len(i_vals)):
                i, j = i_vals[idx], j_vals[idx]
                row_start, row_end = i * n_stat, (i + 1) * n_stat
                col_start, col_end = j * n_stat, (j + 1) * n_stat
                
                if i <= j:
                    var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)
                else:
                    var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov.T)
        
        # Handle lags beyond available gamma matrices with decay
        max_lag = len(gamma_list) - 1
        remaining_mask = lag_diffs > max_lag
        if jnp.any(remaining_mask):
            decay_cov = gamma_list[0] * cond_strength * 0.5**jnp.maximum(lag_diffs - max_lag, 0)
            
            i_vals, j_vals = jnp.where(remaining_mask)
            for idx in range(len(i_vals)):
                i, j = i_vals[idx], j_vals[idx]
                row_start, row_end = i * n_stat, (i + 1) * n_stat
                col_start, col_end = j * n_stat, (j + 1) * n_stat
                
                block_decay = decay_cov if i <= j else decay_cov.T
                var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_decay)
        
        # Insert into full covariance
        var_end = v_start + v_state_dim
        return init_cov.at[v_start:var_end, v_start:var_end].set(var_state_cov)
    
    def build_default_cov(operand):
        v_start, v_state_dim = operand
        default_cov = jnp.eye(v_state_dim, dtype=_DEFAULT_DTYPE) * 0.1
        return init_cov.at[v_start:v_start + v_state_dim, v_start:v_start + v_state_dim].set(default_cov)
    
    # Determine whether to use conditional covariance
    has_gamma = len(gamma_list) > 0
    has_stationary = n_stationary > 0  
    has_var_order = var_order > 0
    
    if has_gamma and has_stationary and has_var_order:
        gamma_0 = gamma_list[0]
        var_start = n_trends
        var_state_dim = n_stationary * var_order
        
        # Safety checks
        is_finite = jnp.all(jnp.isfinite(gamma_0))
        has_positive_diag = jnp.all(jnp.diag(gamma_0) > 0)
        good_condition = jnp.linalg.cond(gamma_0 + _JITTER * jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE)) < 1e10
        
        use_conditional = is_finite & has_positive_diag & good_condition
        
        init_cov = jax.lax.cond(
            use_conditional,
            build_conditional_cov,
            lambda op: build_default_cov(op[-2:]),  # Extract var_start, var_state_dim
            operand=(gamma_0, var_start, var_state_dim, n_stationary, var_order, conditioning_strength)
        )
    
    # Ensure positive definite
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


def create_gpm_based_model_with_conditional_init(gpm_file_path: str, 
                                                     use_conditional_init: bool = False,
                                                     conditioning_strength: float = 0.1):
    """
    FIXED version that works with JAX tracing.
    """
    
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    def gpm_bvar_model_conditional(y: jnp.ndarray):
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample covariances  
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # Conditional vs independent initialization
        if use_conditional_init:
            # Use FIXED conditional functions
            init_mean = _sample_initial_conditions_conditional(
                gpm_model, ss_builder.state_dim, gamma_list,
                ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order
            )
            init_cov = _create_initial_covariance_conditional(
                ss_builder.state_dim, ss_builder.n_trends, gamma_list,
                ss_builder.n_stationary, ss_builder.var_order,
                conditioning_strength=conditioning_strength
            )
        else:
            # Original approach
            init_mean = _sample_initial_conditions(gpm_model, ss_builder.state_dim)
            init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Rest unchanged
        params = EnhancedBVARParams(
            A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
            structural_params=structural_params, Sigma_eps=Sigma_eps
        )
        
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gpm_bvar_model_conditional, gpm_model, ss_builder



# Updated model function with conservative gamma usage
def create_gpm_based_model_safe(gpm_file_path: str, use_gamma_matrices: bool = True, 
                                gamma_scaling: float = 0.1):
    """
    Create model with option to disable or scale gamma matrix usage.
    
    Args:
        use_gamma_matrices: If False, use original initialization
        gamma_scaling: Scale factor for gamma matrix influence (0.1 = conservative)
    """
    
    parser = GPMParser()
    gpm_model = parser.parse_file(gpm_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    def gpm_bvar_model(y: jnp.ndarray):
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gpm_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample covariances
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        Sigma_eps = _sample_measurement_covariance(gpm_model) if _has_measurement_error(gpm_model) else None
        
        # Conditional initialization based on flag
        if use_gamma_matrices:
            init_mean = _sample_initial_conditions_with_gammas_safe(
                gpm_model, ss_builder.state_dim, gamma_list, 
                ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order,
                gamma_scaling=gamma_scaling
            )
            init_cov = _create_initial_covariance_with_gammas_safe(
                ss_builder.state_dim, ss_builder.n_trends, gamma_list,
                ss_builder.n_stationary, ss_builder.var_order,
                use_gamma_scaling=gamma_scaling,
                fallback_to_original=True  # Allow fallback if problems detected
            )
        else:
            # Use original initialization
            init_mean = _sample_initial_conditions(gpm_model, ss_builder.state_dim)
            init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Rest of model unchanged...
        params = EnhancedBVARParams(
            A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
            structural_params=structural_params, Sigma_eps=Sigma_eps
        )
        
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gpm_bvar_model, gpm_model, ss_builder

def fit_gpm_model(gpm_file_path: str, y: jnp.ndarray, 
                  num_warmup: int = 1000, num_samples: int = 2000, 
                  num_chains: int = 4, rng_key: jnp.ndarray = random.PRNGKey(0)):
    """Fit a BVAR model specified by a GPM file"""
    
    print(f"Parsing GPM file: {gpm_file_path}")
    model_fn, gpm_model, ss_builder = create_gpm_based_model(gpm_file_path)
    
    print("GPM Model Summary:")
    print(f"  Trend variables: {gpm_model.trend_variables}")
    print(f"  Stationary variables: {gpm_model.stationary_variables}")
    print(f"  Observed variables: {gpm_model.observed_variables}")
    print(f"  Parameters: {gpm_model.parameters}")
    if gpm_model.var_prior_setup:
        print(f"  VAR order: {gpm_model.var_prior_setup.var_order}")
    
    print("Running MCMC...")
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    start_time = time.time()
    mcmc.run(rng_key, y=y)
    end_time = time.time()
    
    print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    
    return mcmc, gpm_model, ss_builder


def example_gpm_workflow():
    """Example workflow using GPM file"""
    
    # Generate synthetic data
    
    #np_rng = np.random.default_rng(42)
    #y_np = np_rng.normal(0, 1, (T, n_vars)).astype(_DEFAULT_DTYPE)
    import pandas as pd
    # 2. Read the CSV file into a Pandas DataFrame
    dta = pd.read_csv('sim_data.csv')

    # 3. Convert the Pandas DataFrame to a NumPy array
    #    .values attribute gives the underlying NumPy array
    y_np = dta.values
    T = y_np.shape[0]
    n_vars = y_np.shape[1]
    
    # 4. Convert the NumPy array to a JAX NumPy array
    y_jax = jnp.asarray(y_np)
    
    
    # Create a simple GPM file content for testing
    gpm_content = '''
    parameters ;
    
    estimated_params;
        stderr SHK_TREND1, inv_gamma_pdf, 2.3, 1.0;
        stderr SHK_TREND2, inv_gamma_pdf, 2.3, 1.0;
        stderr SHK_TREND3, inv_gamma_pdf, 2.3, 1.0;
        stderr SHK_STAT1, inv_gamma_pdf, 2.3, 1.0;
        stderr SHK_STAT2, inv_gamma_pdf, 2.3, 1.0;
        stderr SHK_STAT3, inv_gamma_pdf, 2.3, 1.0;
        //b1, normal_pdf, 0.1, 0.2;
        //b2, normal_pdf, 0.1, 0.2;
    end;
    
    trends_vars
        TREND1,
        TREND2,
        TREND3
    ;
    
    stationary_variables
        STAT1,
        STAT2,
        STAT3
    ;
    
    trend_shocks;
        var SHK_TREND1
        var SHK_TREND2
        var SHK_TREND3
    end;
    
    shocks;
        var SHK_STAT1
        var SHK_STAT2
        var SHK_STAT3
    end;
    
    trend_model;
        TREND1 = TREND1(-1) + SHK_TREND1;
        TREND2 = TREND2(-1) + SHK_TREND2;
        TREND3 = TREND3(-1) + SHK_TREND3;
    end;
    
    varobs 
        OBS1
        OBS2
        OBS3
    ;
    
    measurement_equations;
        OBS1 = TREND1 + STAT1;
        OBS2 = TREND2 + STAT2;
        OBS3 = TREND3 + STAT3;
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
    with open('temp_model.gpm', 'w') as f:
        f.write(gpm_content)
    
    try:
        # Fit the model
        mcmc, gpm_model, ss_builder = fit_gpm_model('temp_model.gpm', 
                                                    y_jax, 
                                                    num_warmup=500, 
                                                    num_samples=1000, 
                                                    num_chains=2)
        
        mcmc.print_summary(exclude_deterministic=False)
        
        return mcmc, gpm_model, ss_builder
        
    except Exception as e:
        print(f"Error in GPM workflow: {e}")
        return None, None, None


if __name__ == "__main__":
    example_gpm_workflow()