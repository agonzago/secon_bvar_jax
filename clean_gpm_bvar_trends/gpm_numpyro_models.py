# clean_gpm_bvar_trends/gpm_numpyro_models.py

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax # For lax.cond
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from typing import Tuple, Optional, List, Dict, Any as TypingAny
import time 
import numpy as np 

# Local application/library specific imports
from common_types import EnhancedBVARParams
from integration_orchestrator import IntegrationOrchestrator 
from gpm_model_parser import ReducedModel, VarPriorSetup, PriorSpec, VariableSpec # Added VariableSpec
from state_space_builder import StateSpaceBuilder 

try:
    from stationary_prior_jax_simplified import make_stationary_var_transformation_jax, _JITTER as _SP_JITTER
except ImportError:
    make_stationary_var_transformation_jax = None; _SP_JITTER = 1e-8
try:
    from Kalman_filter_jax import KalmanFilter, _KF_JITTER
except ImportError:
    KalmanFilter = None; _KF_JITTER = 1e-8
from constants import _DEFAULT_DTYPE

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# # --- Core NumPyro Model Definition Logic ---

def define_gpm_numpyro_model(
    gpm_file_path: str,
    use_gamma_init_for_P0: bool = False,
    gamma_init_scaling_for_P0: float = 0.1
) -> Tuple[TypingAny, ReducedModel, StateSpaceBuilder]:
    orchestrator = IntegrationOrchestrator(gpm_file_path=gpm_file_path)
    reduced_model: ReducedModel = orchestrator.reduced_model
    ss_builder: StateSpaceBuilder = orchestrator.ss_builder

    def gpm_bvar_numpyro_model(y_data: jnp.ndarray):
        T_obs, n_obs_data = y_data.shape

        # 1. Sample Structural Parameters
        structural_params_draw: Dict[str, jnp.ndarray] = {}
        for param_name in reduced_model.parameters:
            if param_name in reduced_model.estimated_params:
                prior_spec = reduced_model.estimated_params[param_name]
                structural_params_draw[param_name] = _sample_parameter_numpyro(param_name, prior_spec)
            else:
                # If a parameter is in `parameters` list but not `estimated_params`, it's an error
                # unless your GPM system allows fixing parameters outside of this sampling.
                # For now, assume all listed `parameters` need priors in `estimated_params`.
                raise ValueError(f"GPM Error: Parameter '{param_name}' is declared but has no prior in 'estimated_params'.")

        # 2. Sample Shock Standard Deviations
        # Trend shock sigmas
        trend_shock_std_devs_draw: Dict[str, jnp.ndarray] = {} # Key: shock_builder_name
        for shock_builder_name in reduced_model.trend_shocks: # These are declared shocks
            mcmc_sigma_name = f"sigma_{shock_builder_name}"
            prior_to_use = reduced_model.estimated_params.get(shock_builder_name) or \
                           reduced_model.estimated_params.get(mcmc_sigma_name)
            if prior_to_use:
                trend_shock_std_devs_draw[shock_builder_name] = _sample_parameter_numpyro(mcmc_sigma_name, prior_to_use)
            else:
                raise ValueError(f"GPM Error: Prior for declared trend shock '{shock_builder_name}' (or '{mcmc_sigma_name}') not in 'estimated_params'.")

        # VAR Component Sampling (A_raw, Omega_u_chol, stationary shock sigmas)
        A_transformed_draw: Optional[jnp.ndarray] = None
        Sigma_u_draw_for_ebp: Optional[jnp.ndarray] = None
        gamma_list_for_P0: List[jnp.ndarray] = []
        
        if reduced_model.var_prior_setup:
            if not reduced_model.stationary_variables:
                raise ValueError("GPM Error: `var_prior_setup` is present, but no `stationary_variables` are defined.")
            
            n_stat_vars = len(reduced_model.stationary_variables)
            var_order_model = reduced_model.var_prior_setup.var_order

            # _sample_raw_var_coeffs_and_omega_chol now raises error for n_stat_vars=0 or 1 if LKJ is implied
            A_raw_list_draw, Omega_u_chol_draw = _sample_raw_var_coeffs_and_omega_chol(
                reduced_model.var_prior_setup, n_stat_vars
            )

            stat_shock_std_devs_draw: Dict[str, jnp.ndarray] = {}
            if not reduced_model.stationary_shocks:
                raise ValueError("GPM Error: `var_prior_setup` & `stationary_variables` present, but no stationary `shocks` declared.")
            if len(reduced_model.stationary_shocks) != n_stat_vars:
                raise ValueError(f"GPM Error: Mismatch between #stationary_variables ({n_stat_vars}) & #declared stationary shocks ({len(reduced_model.stationary_shocks)}).")

            for shock_builder_name in reduced_model.stationary_shocks:
                mcmc_sigma_name = f"sigma_{shock_builder_name}"
                prior_to_use = reduced_model.estimated_params.get(shock_builder_name) or \
                               reduced_model.estimated_params.get(mcmc_sigma_name)
                if prior_to_use:
                    stat_shock_std_devs_draw[shock_builder_name] = _sample_parameter_numpyro(mcmc_sigma_name, prior_to_use)
                else:
                    raise ValueError(f"GPM Error: Prior for stationary shock '{shock_builder_name}' (or '{mcmc_sigma_name}') not in 'estimated_params'.")
            
            sigma_u_vec_list = [stat_shock_std_devs_draw[sh_name] for sh_name in reduced_model.stationary_shocks] # Order matters
            sigma_u_vec = jnp.array(sigma_u_vec_list, dtype=_DEFAULT_DTYPE)

            # Omega_u_chol_draw would be valid (n_stat_vars >= 2) or error would have been raised
            Sigma_u_draw_for_ebp = jnp.diag(sigma_u_vec) @ Omega_u_chol_draw @ Omega_u_chol_draw.T @ jnp.diag(sigma_u_vec)
            Sigma_u_draw_for_ebp = (Sigma_u_draw_for_ebp + Sigma_u_draw_for_ebp.T) / 2.0 + \
                               _SP_JITTER * jnp.eye(n_stat_vars, dtype=_DEFAULT_DTYPE)

            if make_stationary_var_transformation_jax is not None:
                try:
                    phi_list_draw, gamma_list_for_P0_temp = make_stationary_var_transformation_jax(
                        Sigma_u_draw_for_ebp, A_raw_list_draw, n_stat_vars, var_order_model
                    )
                    A_transformed_draw = jnp.stack(phi_list_draw)
                    numpyro.deterministic("A_transformed", A_transformed_draw) # Store transformed
                    gamma_list_for_P0 =  gamma_list_for_P0_temp
                except Exception as e_transform:
                    raise RuntimeError(f"Stationarity transformation failed in MCMC: {e_transform}. Check VAR params/priors.") from e_transform
            else: # No transformation function available
                A_transformed_draw = jnp.stack(A_raw_list_draw) # Use raw
                numpyro.deterministic("A_raw", A_transformed_draw) # Store raw
                gamma_list_for_P0 = [Sigma_u_draw_for_ebp]
        
        elif reduced_model.stationary_variables: # Stationary vars exist, but no var_prior_setup
            raise ValueError("GPM Error: `stationary_variables` defined, but no `var_prior_setup` block to specify their dynamics/priors.")
        else: # No stationary variables and no var_prior_setup: OK
            Sigma_u_draw_for_ebp = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
            A_transformed_draw = jnp.empty((0,0,0), dtype=_DEFAULT_DTYPE) # (order, n_stat, n_stat)
            gamma_list_for_P0 = []

        # Form Sigma_eta_draw (diagonal covariance for DYNAMIC trend innovations)
        num_dynamic_core_trends = ss_builder.n_core - ss_builder.n_stationary
        sigma_eta_diag_values = jnp.zeros(num_dynamic_core_trends, dtype=_DEFAULT_DTYPE)
        
        dynamic_core_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
        
        for idx_dynamic_trend, core_trend_name in enumerate(dynamic_core_trend_names):
            core_eq = next((eq for eq in reduced_model.core_equations if eq.lhs == core_trend_name), None)
            if core_eq and core_eq.shock: # This dynamic trend's equation specifies a shock
                if core_eq.shock not in reduced_model.trend_shocks:
                    raise ValueError(f"GPM Error: Shock '{core_eq.shock}' in eq for '{core_trend_name}' not in 'trend_shocks' block.")
                if core_eq.shock not in trend_shock_std_devs_draw:
                    # This should not happen if previous sampling loop for trend_shock_std_devs_draw is correct
                    raise ValueError(f"Logic Error: Sampled std. dev. for declared trend shock '{core_eq.shock}' is missing.")
                
                sigma_val_sq = trend_shock_std_devs_draw[core_eq.shock] ** 2
                sigma_eta_diag_values = sigma_eta_diag_values.at[idx_dynamic_trend].set(sigma_val_sq)
            # If core_eq.shock is None, its variance in sigma_eta_diag_values remains 0 (deterministic RW component)
        Sigma_eta_draw_for_ebp = jnp.diag(sigma_eta_diag_values)


        # 3. Measurement Error (Sigma_eps)
        Sigma_eps_draw = None # Assuming no explicit measurement error sampling for now
        # if _has_measurement_error_numpyro(reduced_model):
        #     Sigma_eps_draw = _sample_measurement_covariance_numpyro(reduced_model)

        # 4. Package parameters for StateSpaceBuilder
        current_draw_bvar_params = EnhancedBVARParams(
            A=A_transformed_draw,
            Sigma_u=Sigma_u_draw_for_ebp,
            Sigma_eta=Sigma_eta_draw_for_ebp,
            structural_params=structural_params_draw,
            Sigma_eps=Sigma_eps_draw
        )

        # 5. Build State Space Matrices for the Current Draw
        F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_from_enhanced_bvar(
            current_draw_bvar_params
        )

        # 6. Sample Initial Conditions (P0)
        if use_gamma_init_for_P0 and gamma_list_for_P0:
            init_mean_draw = _sample_initial_conditions_gamma_based(
                reduced_model, ss_builder, gamma_list_for_P0, gamma_init_scaling_for_P0
            )
            init_cov_draw = _create_initial_covariance_gamma_based(
                ss_builder.state_dim, num_dynamic_core_trends, gamma_list_for_P0,
                ss_builder.n_stationary, ss_builder.var_order, gamma_init_scaling_for_P0
            )
        else:
            init_mean_draw = _sample_initial_conditions_standard(reduced_model, ss_builder)
            init_cov_draw = _create_initial_covariance_standard(ss_builder.state_dim, num_dynamic_core_trends)
        
        

        # 7. Kalman Filter Likelihood
        matrices_ok = (
            jnp.all(jnp.isfinite(F_draw)) & jnp.all(jnp.isfinite(Q_draw)) &
            jnp.all(jnp.isfinite(C_draw)) & jnp.all(jnp.isfinite(H_draw)) &
            jnp.all(jnp.isfinite(init_mean_draw)) & jnp.all(jnp.isfinite(init_cov_draw))
        )
        Q_draw_reg = (Q_draw + Q_draw.T) / 2.0 + _SP_JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        try: R_draw = jnp.linalg.cholesky(Q_draw_reg)
        except Exception: R_draw = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(Q_draw_reg), _SP_JITTER)))

        log_likelihood_val = jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)
        if KalmanFilter is not None:
            kf_instance = KalmanFilter(T=F_draw, R=R_draw, C=C_draw, H=H_draw, init_x=init_mean_draw, init_P=init_cov_draw)
            valid_obs_idx_static = jnp.arange(n_obs_data, dtype=jnp.int32)
            I_obs_static = jnp.eye(n_obs_data, dtype=_DEFAULT_DTYPE)
            log_likelihood_val = lax.cond(
                matrices_ok,
                lambda: kf_instance.log_likelihood(y_data, valid_obs_idx_static, n_obs_data, C_draw, H_draw, I_obs_static),
                lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)
            )
        numpyro.factor("loglik", log_likelihood_val)

    return gpm_bvar_numpyro_model, reduced_model, ss_builder




# --- Helper functions for sampling within NumPyro model ---
def _sample_parameter_numpyro(name: str, prior_spec: PriorSpec) -> jnp.ndarray:
    if prior_spec.distribution == 'normal_pdf': return numpyro.sample(name, dist.Normal(prior_spec.params[0], prior_spec.params[1]))
    elif prior_spec.distribution == 'inv_gamma_pdf': return numpyro.sample(name, dist.InverseGamma(prior_spec.params[0], prior_spec.params[1]))
    raise ValueError(f"Unsupported prior: {prior_spec.distribution} for {name}")

def _sample_raw_var_coeffs_and_omega_chol(var_prior_setup: VarPriorSetup, n_vars: int) -> Tuple[List[jnp.ndarray], Optional[jnp.ndarray]]:
    n_lags = var_prior_setup.var_order
    if n_vars == 0: # No stationary variables, implies var_prior_setup should not have been called
         raise ValueError("GPM Error: _sample_raw_var_coeffs_and_omega_chol called with n_vars=0 but var_prior_setup exists.")

    es_len = len(var_prior_setup.es); fs_len = len(var_prior_setup.fs)
    Amu_params = [(var_prior_setup.es[i] if i < es_len else 0.0, var_prior_setup.fs[i] if i < fs_len else 1.0) for i in range(2)]
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(mean, std)) for i, (mean, std) in enumerate(Amu_params)]
    gs_len = len(var_prior_setup.gs); hs_len = len(var_prior_setup.hs)
    Aomega_params = [(var_prior_setup.gs[i] if i < gs_len else 1.0, var_prior_setup.hs[i] if i < hs_len else 1.0) for i in range(2)]
    Aom = [numpyro.sample(f"Aomega_{i}", dist.Gamma(shape, rate)) for i, (shape, rate) in enumerate(Aomega_params)]
    
    aom_diag_eff = Aom[0] if len(Aom) > 0 else jnp.array(1.0); Aom_diag_sqrt_inv = 1.0 / jnp.sqrt(jnp.maximum(aom_diag_eff, _SP_JITTER))
    aom_offdiag_eff = Aom[1] if len(Aom) > 1 else jnp.array(1.0); Aom_offdiag_sqrt_inv = 1.0 / jnp.sqrt(jnp.maximum(aom_offdiag_eff, _SP_JITTER))
    A_std_diag = Aom_diag_sqrt_inv; A_std_offdiag = Aom_offdiag_sqrt_inv
    A_mean_diag = Amu[0] if len(Amu) > 0 else jnp.array(0.0); A_mean_offdiag = Amu[1] if len(Amu) > 1 else jnp.array(0.0)

    raw_A_list = []
    for lag in range(n_lags):
        A_full = numpyro.sample(f"A_full_{lag}", dist.Normal(A_mean_offdiag, A_std_offdiag).expand([n_vars, n_vars]))
        A_diag_el = numpyro.sample(f"A_diag_{lag}", dist.Normal(A_mean_diag, A_std_diag).expand([n_vars]))
        raw_A_list.append(A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag_el))
    
    if n_vars >= 2:
        Omega_u_chol = numpyro.sample("Omega_u_chol", dist.LKJCholesky(n_vars, concentration=var_prior_setup.eta))
    elif n_vars == 1: # Should ideally be handled by GPM validation: VAR setup for 1 var is just an AR.
        # This means the "correlation" matrix is just [[1]]. Its Cholesky is [[1]].
        # This path assumes if var_prior_setup is given for n_vars=1, it's for a univariate AR process.
        Omega_u_chol = jnp.array([[1.0]], dtype=_DEFAULT_DTYPE) 
        # No LKJ sampling for d=1. If strict error is preferred, raise ValueError here.
    else: # n_vars must be 0 if not >=1, which was already caught. Defensive.
        Omega_u_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    return raw_A_list, Omega_u_chol

def _has_measurement_error_numpyro(reduced_model: ReducedModel) -> bool: return False
def _sample_measurement_covariance_numpyro(reduced_model: ReducedModel) -> Optional[jnp.ndarray]: return None



def _sample_initial_conditions_gamma_based(
    reduced_model: ReducedModel, ss_builder: StateSpaceBuilder, 
    gamma_list_draw: List[jnp.ndarray], gamma_scaling: float
) -> jnp.ndarray:
    """
    FIXED: JAX-compatible gamma-based P0 initialization 
    """
    state_dim = ss_builder.state_dim
    init_mean_base = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std_for_sampling = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)  # Start with 1.0, not NaN

    n_dynamic_trends = ss_builder.n_dynamic_trends
    n_stationary = ss_builder.n_stationary
    var_order = ss_builder.var_order
    
    # 1. Dynamic Trend Part - Set from initval entries
    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    
    for trend_name in dynamic_trend_names:
        state_vector_idx = ss_builder.core_var_map.get(trend_name)
        
        if state_vector_idx is None or state_vector_idx >= state_dim:
            raise AssertionError(f"Logic error: Dynamic trend '{trend_name}' index issue. Got {state_vector_idx}, max allowed {state_dim-1}")

        if trend_name in reduced_model.initial_values:
            var_spec = reduced_model.initial_values[trend_name]
            if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
                mean_val, std_val_prior = var_spec.init_params[:2]
                init_mean_base = init_mean_base.at[state_vector_idx].set(mean_val)
                init_std_for_sampling = init_std_for_sampling.at[state_vector_idx].set(std_val_prior)
            else:
                raise ValueError(f"GPM Error: `initval` for dynamic trend '{trend_name}' requires 'normal_pdf' (mean, std).")
        else:
            raise ValueError(f"GPM Error: Dynamic trend '{trend_name}' must have an 'initval' entry for P0 mean sampling.")

    # 2. Stationary VAR Part - Use gamma-based standard deviations
    if n_stationary > 0 and var_order > 0:
        if not gamma_list_draw or gamma_list_draw[0] is None:
            raise ValueError("GPM Error: Gamma-P0 selected, but gamma_list_draw[0] (Sigma_u) unavailable from VAR sampling.")
        
        gamma_0 = gamma_list_draw[0]
        if gamma_0.shape != (n_stationary, n_stationary):
            raise ValueError(f"GPM Error: Sigma_u shape {gamma_0.shape} != expected ({n_stationary}, {n_stationary}) for P0 gamma init.")
        if gamma_scaling <= 0: 
            raise ValueError("gamma_scaling for P0 must be positive.")
        
        # Theoretical standard deviations from VAR unconditional covariance
        theoretical_std_stat = jnp.sqrt(jnp.maximum(jnp.diag(gamma_0), 1e-9)) * jnp.sqrt(gamma_scaling)

        # Set standard deviations for all VAR state components
        for lag in range(var_order):
            stat_block_start_idx = n_dynamic_trends + lag * n_stationary
            stat_block_end_idx = stat_block_start_idx + n_stationary
            
            # Scale std dev by lag (current period has full std, lags have reduced std)
            current_lag_std = jnp.clip(theoretical_std_stat / (float(lag) + 1.0), 0.01, 5.0)
            
            if stat_block_end_idx <= state_dim and current_lag_std.shape == (n_stationary,):
                init_std_for_sampling = init_std_for_sampling.at[stat_block_start_idx:stat_block_end_idx].set(current_lag_std)
            else:
                raise RuntimeError(f"Logic error assigning P0 gamma std dev for stationary lag {lag}. Block end {stat_block_end_idx} > state_dim {state_dim}")
        
        # Allow initval to override means for current period (lag 0) of stationary vars
        for i_stat_in_block, stat_var_name in enumerate(reduced_model.stationary_variables):
            if stat_var_name in reduced_model.initial_values:
                var_spec = reduced_model.initial_values[stat_var_name]
                if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
                    mean_val, _ = var_spec.init_params[:2]  # std from initval is ignored for gamma P0
                    idx_for_this_stat_lag0 = n_dynamic_trends + i_stat_in_block  # Current period index
                    if idx_for_this_stat_lag0 < state_dim:
                         init_mean_base = init_mean_base.at[idx_for_this_stat_lag0].set(mean_val)

    # 3. NO NaN VALIDATION INSIDE JAX FUNCTION
    # Instead, we do static validation outside the JAX-traced function
    # All components should now have valid std devs (either from initval for trends, or from gamma for VAR states)
    
    return numpyro.sample("init_mean_full", dist.Normal(init_mean_base, init_std_for_sampling).to_event(1))


def _sample_initial_conditions_standard(
    reduced_model: ReducedModel, ss_builder: StateSpaceBuilder
) -> jnp.ndarray:
    """
    FIXED: JAX-compatible standard P0 initialization
    """
    state_dim = ss_builder.state_dim
    init_mean_base = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std_for_sampling = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)  # Default std of 1

    n_dynamic_trends = ss_builder.n_dynamic_trends

    # Set means and sampling stds from GPM 'initial_values'
    for var_name_in_gpm, var_spec in reduced_model.initial_values.items():
        if var_name_in_gpm in ss_builder.core_var_map:
            state_idx = ss_builder.core_var_map[var_name_in_gpm]
            if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
                mean_val, std_val_prior = var_spec.init_params[:2]
                init_mean_base = init_mean_base.at[state_idx].set(mean_val)
                init_std_for_sampling = init_std_for_sampling.at[state_idx].set(std_val_prior)
            else:
                raise ValueError(f"GPM Error: 'initval' for core variable '{var_name_in_gpm}' requires 'normal_pdf' with mean and std.")
    
    # Check for dynamic trends without initval (this check happens at model definition time, not during tracing)
    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    for dt_name in dynamic_trend_names:
        if dt_name not in reduced_model.initial_values:
             raise ValueError(f"GPM Error: Dynamic core trend '{dt_name}' must have an 'initval' entry for its P0 mean sampling distribution.")

    # For stationary VAR states, apply default sampling std if not set by initval
    var_block_start_idx = n_dynamic_trends
    for i in range(ss_builder.n_stationary * ss_builder.var_order):
        current_stat_state_idx = var_block_start_idx + i
        if current_stat_state_idx < state_dim:
            # Check if this wasn't set by initval (still has default value of 1.0)
            # We use a JAX-compatible approach: always set for VAR components
            init_std_for_sampling = init_std_for_sampling.at[current_stat_state_idx].set(0.5)

    return numpyro.sample("init_mean_full", dist.Normal(init_mean_base, init_std_for_sampling).to_event(1))


def _validate_p0_setup_before_model(reduced_model: ReducedModel, ss_builder: StateSpaceBuilder, use_gamma_init: bool):
    """
    ADDED: Static validation of P0 setup before running JAX-traced model
    This function runs BEFORE model compilation to catch configuration errors
    """
    state_dim = ss_builder.state_dim
    n_dynamic_trends = ss_builder.n_dynamic_trends
    n_stationary = ss_builder.n_stationary
    var_order = ss_builder.var_order
    
    print(f"Validating P0 setup: state_dim={state_dim}, n_dynamic_trends={n_dynamic_trends}, n_stationary={n_stationary}, var_order={var_order}")
    
    # Check 1: All dynamic trends must have initval entries
    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    missing_initvals = []
    
    for trend_name in dynamic_trend_names:
        if trend_name not in reduced_model.initial_values:
            missing_initvals.append(trend_name)
        else:
            var_spec = reduced_model.initial_values[trend_name]
            if var_spec.init_dist != 'normal_pdf' or len(var_spec.init_params) < 2:
                missing_initvals.append(f"{trend_name} (malformed - needs normal_pdf with mean,std)")
    
    if missing_initvals:
        raise ValueError(f"P0 Validation Error: Dynamic trends missing proper 'initval' entries: {missing_initvals}")
    
    # Check 2: If using gamma init, VAR setup must be complete
    if use_gamma_init and n_stationary > 0:
        if not reduced_model.var_prior_setup:
            raise ValueError("P0 Validation Error: Gamma init requested but no 'var_prior_setup' found")
        if not reduced_model.stationary_variables:
            raise ValueError("P0 Validation Error: Gamma init requested but no stationary variables defined")
        if not reduced_model.stationary_shocks:
            raise ValueError("P0 Validation Error: Gamma init requested but no stationary shocks defined")
    
    # Check 3: State vector indexing makes sense
    expected_state_dim = n_dynamic_trends + n_stationary * var_order
    if state_dim != expected_state_dim:
        raise ValueError(f"P0 Validation Error: State dimension mismatch. Expected {expected_state_dim}, got {state_dim}")
    
    # Check 4: core_var_map indices are within bounds
    for var_name, idx in ss_builder.core_var_map.items():
        if idx >= state_dim:
            raise ValueError(f"P0 Validation Error: Variable '{var_name}' mapped to index {idx} >= state_dim {state_dim}")
    
    print("✓ P0 setup validation passed")


def _create_initial_covariance_gamma_based(
    state_dim: int, n_dynamic_trends: int, gamma_list_draw: List[jnp.ndarray],
    n_stationary: int, var_order: int, gamma_scaling: float
) -> jnp.ndarray:
    """
    FIXED: Gamma-based initial covariance with correct indexing
    """
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Large variance for dynamic trends (diffuse prior)
    if n_dynamic_trends > 0: 
        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(jnp.eye(n_dynamic_trends) * 1e4)
    
    # VAR block covariance based on gamma matrices
    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order
    
    if n_stationary > 0 and var_order > 0 and gamma_list_draw and gamma_list_draw[0] is not None:
        var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
        g0 = gamma_list_draw[0]
        
        if g0.shape == (n_stationary, n_stationary):
            for r_idx in range(var_order):
                for c_idx in range(var_order):
                    lag_d = abs(r_idx - c_idx)
                    
                    # Get gamma matrix for this lag difference
                    if lag_d < len(gamma_list_draw) and gamma_list_draw[lag_d] is not None and \
                       gamma_list_draw[lag_d].shape == (n_stationary, n_stationary):
                        blk_unscaled = gamma_list_draw[lag_d]
                    else:
                        # Fallback: exponential decay
                        blk_unscaled = g0 * (0.5**lag_d)
                    
                    curr_blk = blk_unscaled * gamma_scaling
                    if r_idx > c_idx: 
                        curr_blk = curr_blk.T
                    
                    # Insert block into VAR covariance
                    r_s, r_e = r_idx*n_stationary, (r_idx+1)*n_stationary
                    c_s, c_e = c_idx*n_stationary, (c_idx+1)*n_stationary
                    
                    if r_e <= var_state_total_dim and c_e <= var_state_total_dim:
                        var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
            
            # Insert VAR block into full covariance matrix
            if var_start_idx + var_state_total_dim <= state_dim:
                init_cov = init_cov.at[var_start_idx : var_start_idx+var_state_total_dim, 
                                      var_start_idx : var_start_idx+var_state_total_dim].set(var_block_cov)
    
    elif var_state_total_dim > 0 and var_start_idx + var_state_total_dim <= state_dim:
        # Fallback for VAR states if gamma matrices not available
        init_cov = init_cov.at[var_start_idx:var_start_idx+var_state_total_dim, 
                              var_start_idx:var_start_idx+var_state_total_dim].set(jnp.eye(var_state_total_dim)*0.1)
    
    # Ensure positive definite
    return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)


def _create_initial_covariance_standard(state_dim: int, n_dynamic_trends: int) -> jnp.ndarray:
    """
    FIXED: Standard initial covariance with correct indexing
    """
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * 1e4
    
    # More informative prior for non-trend states (VAR states)
    if state_dim > n_dynamic_trends:
        init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
            jnp.eye(state_dim - n_dynamic_trends) * 1.0
        )
    
    return (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)

# --- Main Fitting Function ---
def fit_gpm_numpyro_model(
    gpm_file_path: str, y_data: jnp.ndarray,
    num_warmup: int = 1000, num_samples: int = 2000, num_chains: int = 2, 
    rng_key_seed: int = 0, use_gamma_init_for_P0: bool = False,
    gamma_init_scaling_for_P0: float = 0.01, target_accept_prob: float = 0.85,
    max_tree_depth: int = 10, dense_mass: bool = False 
) -> Tuple[numpyro.infer.MCMC, ReducedModel, StateSpaceBuilder]:
    # (As before)
    print(f"--- Fitting GPM Model: {gpm_file_path} ---")
    model_function, reduced_model, ss_builder = define_gpm_numpyro_model(
        gpm_file_path, use_gamma_init_for_P0, gamma_init_scaling_for_P0)
    kernel_settings = {"target_accept_prob": target_accept_prob, "max_tree_depth": max_tree_depth}
    if dense_mass: kernel_settings["dense_mass"] = True
    kernel = NUTS(model_function, **kernel_settings)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
    rng_key = random.PRNGKey(rng_key_seed); start_time = time.time()
    mcmc.run(rng_key, y_data=y_data)
    end_time = time.time(); print(f"MCMC completed in {end_time - start_time:.2f}s.")
    return mcmc, reduced_model, ss_builder

# --- Example Workflow ---
def _example_gpm_fitting_workflow(): # As before
    import pandas as pd; import os
    example_gpm_file = "example_gpm_numpyro_model.gpm"
    gpm_content = """
parameters rho; 

estimated_params; 
    stderr SHK_TREND1, inv_gamma_pdf, 2.3, 0.5; 
    stderr SKK_TREND2, inv_gamma_pdf, 2.3, 0.5; 
    #var_phi, normal_pdf, 0.8, 0.1;  // var_phi not in parameters block for this example
    stderr shk_stat1, inv_gamma_pdf, 2.3, 1.5; 
    stderr shk_stat2, inv_gamma_pdf, 2.3, 1.5; 
    rho, normal_pdf, 0.5, 0.1;
end;

trends_vars TREND1, TREND2;

stationary_variables stat1, stat2;

trend_shocks; 
    var SHK_TREND1; 
    var SKK_TREND2; 
end;

shocks; 
    var shk_stat1; 
    var shk_stat2; 
end;

trend_model;     
    TREND2 = TREND2(-1) + SKK_TREND2; 
    TREND1 = TREND1(-1) + rho*TREND2(-1) + SHK_TREND1; 
end; 

varobs OBS1, OBS2;

measurement_equations; 
    OBS1 = TREND1 + stat1;
    OBS2 = TREND2 + stat2;
end;

var_prior_setup; 
    var_order = 1; 
    es = 0.7,0.1;
    fs=0.5,0.5; 
    gs=3,2; 
    hs=1,0.5; 
    eta=2; 
end;

initval; 
    TREND1, normal_pdf, 0, 1; 
    TREND2, normal_pdf, 0, 1; 
end;
""" 
   
    with open(example_gpm_file, "w") as f: f.write(gpm_content)
    T_data, n_obs_actual = 100, 2 # Two obs vars
    key_data_sim = random.PRNGKey(456) 
    # Simulate some data that might roughly fit this structure
    y_trend1_sim = jnp.cumsum(random.normal(key_data_sim, (T_data,)) * 0.1)
    key_data_sim, sub_key = random.split(key_data_sim)
    y_trend2_sim = jnp.cumsum(random.normal(sub_key, (T_data,)) * 0.15)
    key_data_sim, sub_key1, sub_key2 = random.split(key_data_sim, 3)
    y_stat1_sim = random.normal(sub_key1, (T_data,)) * 0.5
    y_stat2_sim = random.normal(sub_key2, (T_data,)) * 0.4
    obs1_sim = y_trend1_sim + y_stat1_sim
    obs2_sim = y_trend2_sim + y_stat2_sim
    y_synthetic_data = jnp.stack([obs1_sim, obs2_sim], axis=-1)

    print(f"\n--- Running Example GPM Fitting Workflow ---")
    try:
        mcmc_obj, _, _ = fit_gpm_numpyro_model(
            gpm_file_path=example_gpm_file, y_data=y_synthetic_data,
            num_warmup=50, num_samples=100, num_chains=1, 
            use_gamma_init_for_P0=True, # Test standard P0
            gamma_init_scaling_for_P0 = 1.0,
            target_accept_prob=0.9
        )
        print("\n--- MCMC Summary ---"); mcmc_obj.print_summary(exclude_deterministic=False)
    except Exception as e: import traceback; print(f"Error in example: {e}"); traceback.print_exc()
    finally: 
        if os.path.exists(example_gpm_file): os.remove(example_gpm_file)

if __name__ == "__main__":
    _example_gpm_fitting_workflow()