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
import multiprocessing # For numpyro.set_host_device_count, if used

# Local application/library specific imports
from .common_types import EnhancedBVARParams
from .integration_orchestrator import IntegrationOrchestrator # Ensure it's not creating circular imports
from .gpm_model_parser import ReducedModel, VarPriorSetup, PriorSpec, VariableSpec
from .state_space_builder import StateSpaceBuilder

try:
    from .stationary_prior_jax_simplified import make_stationary_var_transformation_jax, _JITTER as _SP_JITTER
except ImportError:
    make_stationary_var_transformation_jax = None 
    _SP_JITTER = 1e-8
try:
    from .Kalman_filter_jax import KalmanFilter
except ImportError:
    KalmanFilter = None
    # If KalmanFilter is None, _KF_JITTER might not be defined from there.
    # It should be available from constants.py
from .constants import _DEFAULT_DTYPE, _KF_JITTER # Make sure _KF_JITTER is imported

# --- JAX Configuration (Ideally from jax_config.py, but shown here for completeness if not using that file) ---
# try:
#     from .jax_config import configure_jax # Assumes jax_config.py is in the same package
#     # configure_jax() # Call it if jax_config.py defines it and calls it upon import
# except ImportError:
#     print("Warning: jax_config.py not found or configure_jax not callable. Setting JAX config locally.")
# if "XLA_FLAGS" not in os.environ and multiprocessing.cpu_count() is not None: # Be careful with cpu_count() if it returns None
#     # This is an example; often not needed or set outside Python
#     #os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
#     pass

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if multiprocessing.cpu_count() is not None:
    try:
        numpyro.set_host_device_count(multiprocessing.cpu_count())
    except RuntimeError: # If called after JAX backend is already initialized in a certain way
        pass

# # --- Core NumPyro Model Definition Logic ---

# --- Helper functions for sampling within NumPyro model ---
def _sample_parameter_numpyro(name: str, prior_spec: PriorSpec) -> jnp.ndarray:
    if prior_spec.distribution == 'normal_pdf':
        return numpyro.sample(name, dist.Normal(prior_spec.params[0], prior_spec.params[1]))
    elif prior_spec.distribution == 'inv_gamma_pdf':
        # InverseGamma(concentration, rate) maps to InverseGamma(alpha, beta)
        # Ensure alpha > 0, beta > 0 for a proper distribution
        alpha = jnp.maximum(prior_spec.params[0], 1e-6) # Ensure alpha is positive
        beta = jnp.maximum(prior_spec.params[1], 1e-6)  # Ensure beta is positive
        return numpyro.sample(name, dist.InverseGamma(alpha, beta))
    raise ValueError(f"Unsupported prior: {prior_spec.distribution} for {name}")

def _sample_raw_var_coeffs_and_omega_chol(var_prior_setup: VarPriorSetup, n_vars: int) -> Tuple[List[jnp.ndarray], Optional[jnp.ndarray]]:
    n_lags = var_prior_setup.var_order
    if n_vars == 0:
         raise ValueError("GPM Error: _sample_raw_var_coeffs_and_omega_chol called with n_vars=0 but var_prior_setup exists.")

    es_len = len(var_prior_setup.es); fs_len = len(var_prior_setup.fs)
    Amu_params = [(var_prior_setup.es[i] if i < es_len else 0.0, var_prior_setup.fs[i] if i < fs_len else 1.0) for i in range(2)]
    Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(mean, std)) for i, (mean, std) in enumerate(Amu_params)]

    gs_len = len(var_prior_setup.gs); hs_len = len(var_prior_setup.hs)
    Aomega_params = [(var_prior_setup.gs[i] if i < gs_len else 1.0, var_prior_setup.hs[i] if i < hs_len else 1.0) for i in range(2)]
    Aom = [numpyro.sample(f"Aomega_{i}", dist.Gamma(jnp.maximum(shape, 1e-6), jnp.maximum(rate, 1e-6))) for i, (shape, rate) in enumerate(Aomega_params)] # Ensure positive params

    aom_diag_eff = Aom[0] if len(Aom) > 0 else jnp.array(1.0); Aom_diag_sqrt_inv = 1.0 / jnp.sqrt(jnp.maximum(aom_diag_eff, _SP_JITTER))
    aom_offdiag_eff = Aom[1] if len(Aom) > 1 else jnp.array(1.0); Aom_offdiag_sqrt_inv = 1.0 / jnp.sqrt(jnp.maximum(aom_offdiag_eff, _SP_JITTER))
    A_std_diag = Aom_diag_sqrt_inv; A_std_offdiag = Aom_offdiag_sqrt_inv
    A_mean_diag = Amu[0] if len(Amu) > 0 else jnp.array(0.0); A_mean_offdiag = Amu[1] if len(Amu) > 1 else jnp.array(0.0)

    raw_A_list = []
    for lag in range(n_lags):
        A_full = numpyro.sample(f"A_full_{lag}", dist.Normal(A_mean_offdiag, A_std_offdiag).expand([n_vars, n_vars]).to_event(2))
        A_diag_el = numpyro.sample(f"A_diag_{lag}", dist.Normal(A_mean_diag, A_std_diag).expand([n_vars]).to_event(1))
        raw_A_list.append(A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag_el))

    if n_vars >= 2:
        Omega_u_chol = numpyro.sample("Omega_u_chol", dist.LKJCholesky(n_vars, concentration=jnp.maximum(var_prior_setup.eta, 1e-6))) # Ensure positive conc.
    elif n_vars == 1:
        Omega_u_chol = jnp.array([[1.0]], dtype=_DEFAULT_DTYPE)
    else: # Should be caught by n_vars == 0 check above
        Omega_u_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    return raw_A_list, Omega_u_chol

def _has_measurement_error_numpyro(reduced_model: ReducedModel) -> bool: return False # Placeholder
def _sample_measurement_covariance_numpyro(reduced_model: ReducedModel) -> Optional[jnp.ndarray]: return None # Placeholder


def _sample_initial_conditions_gamma_based(
    reduced_model: ReducedModel, ss_builder: StateSpaceBuilder,
    gamma_list_draw: List[jnp.ndarray], # Can be empty if transformation failed
    gamma_init_scaling: float
) -> jnp.ndarray:
    state_dim = ss_builder.state_dim
    init_mean_base = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std_for_sampling = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE) * 0.5 # Default std

    n_dynamic_trends = ss_builder.n_dynamic_trends
    n_stationary = ss_builder.n_stationary
    var_order = ss_builder.var_order

    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    for trend_name in dynamic_trend_names:
        var_spec = reduced_model.initial_values[trend_name]
        mean_val, std_val_prior = var_spec.init_params[:2]
        state_vector_idx = ss_builder.core_var_map[trend_name]
        init_mean_base = init_mean_base.at[state_vector_idx].set(mean_val)
        init_std_for_sampling = init_std_for_sampling.at[state_vector_idx].set(jnp.maximum(std_val_prior, 1e-6)) # Ensure positive std

    if n_stationary > 0 and var_order > 0:
        gamma_0_available_for_x0_std = (
            gamma_list_draw and len(gamma_list_draw) > 0 and
            gamma_list_draw[0] is not None and
            hasattr(gamma_list_draw[0], 'shape') and
            gamma_list_draw[0].shape == (n_stationary, n_stationary)
        )

        if gamma_0_available_for_x0_std:
            gamma_0 = gamma_list_draw[0]
            diag_gamma_0 = jnp.diag(gamma_0)
            safe_diag_values = jnp.where(jnp.isfinite(diag_gamma_0) & (diag_gamma_0 > _SP_JITTER), diag_gamma_0, _SP_JITTER)
            theoretical_std_stat = jnp.sqrt(safe_diag_values) * jnp.sqrt(jnp.maximum(gamma_init_scaling, 1e-6))
        else:
            theoretical_std_stat = jnp.ones(n_stationary, dtype=_DEFAULT_DTYPE) * 0.5

        for lag in range(var_order):
            stat_block_start_idx = n_dynamic_trends + lag * n_stationary
            stat_block_end_idx = stat_block_start_idx + n_stationary
            current_lag_std = jnp.clip(theoretical_std_stat / (float(lag) + 1.0), 0.01, 10.0) # Wider clip for std
            init_std_for_sampling = init_std_for_sampling.at[stat_block_start_idx:stat_block_end_idx].set(current_lag_std)

        for i_stat_in_block, stat_var_name in enumerate(reduced_model.stationary_variables):
            if stat_var_name in reduced_model.initial_values:
                var_spec_stat = reduced_model.initial_values[stat_var_name]
                if var_spec_stat.init_dist == 'normal_pdf' and len(var_spec_stat.init_params) >= 1:
                    mean_val_stat = var_spec_stat.init_params[0]
                    idx_for_this_stat_lag0 = n_dynamic_trends + i_stat_in_block
                    init_mean_base = init_mean_base.at[idx_for_this_stat_lag0].set(mean_val_stat)

    return numpyro.sample("init_mean_full", dist.Normal(init_mean_base, init_std_for_sampling).to_event(1))


def _sample_initial_conditions_standard(
    reduced_model: ReducedModel, ss_builder: StateSpaceBuilder
) -> jnp.ndarray:
    state_dim = ss_builder.state_dim
    init_mean_base = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std_for_sampling = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE) * 0.5

    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    for trend_name in dynamic_trend_names:
        var_spec = reduced_model.initial_values[trend_name]
        mean_val, std_val_prior = var_spec.init_params[:2]
        state_idx = ss_builder.core_var_map[trend_name]
        init_mean_base = init_mean_base.at[state_idx].set(mean_val)
        init_std_for_sampling = init_std_for_sampling.at[state_idx].set(jnp.maximum(std_val_prior, 1e-6))

    n_dynamic_trends = ss_builder.n_dynamic_trends
    for i_stat_in_block, stat_var_name in enumerate(reduced_model.stationary_variables):
        idx_for_this_stat_lag0 = n_dynamic_trends + i_stat_in_block
        if stat_var_name in reduced_model.initial_values:
            var_spec_stat = reduced_model.initial_values[stat_var_name]
            if var_spec_stat.init_dist == 'normal_pdf' and len(var_spec_stat.init_params) >= 1:
                init_mean_base = init_mean_base.at[idx_for_this_stat_lag0].set(var_spec_stat.init_params[0])
                if len(var_spec_stat.init_params) >=2: # If std is also provided for stationary in initval
                     # This would apply to all lags unless overridden by gamma_based logic if that path was taken
                     # For standard P0, we can allow initval to specify std for lag 0 states
                     for lag in range(ss_builder.var_order):
                        idx_this_stat_this_lag = n_dynamic_trends + lag * ss_builder.n_stationary + i_stat_in_block
                        init_std_for_sampling = init_std_for_sampling.at[idx_this_stat_this_lag].set(jnp.maximum(var_spec_stat.init_params[1], 1e-6))


    return numpyro.sample("init_mean_full", dist.Normal(init_mean_base, init_std_for_sampling).to_event(1))


def _create_initial_covariance_gamma_based(
    state_dim: int, n_dynamic_trends: int, gamma_list_draw: List[jnp.ndarray],
    n_stationary: int, var_order: int, gamma_scaling: float
) -> jnp.ndarray:
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    trend_var_scale_mcmc = 1e6
    var_fallback_scale_mcmc = 4.0

    if n_dynamic_trends > 0:
        init_cov = init_cov.at[:n_dynamic_trends, :n_dynamic_trends].set(
            jnp.eye(n_dynamic_trends, dtype=_DEFAULT_DTYPE) * trend_var_scale_mcmc
        )

    var_start_idx = n_dynamic_trends
    var_state_total_dim = n_stationary * var_order

    # This function is called if define_gpm_numpyro_model decided gamma_list_for_P0 is usable.
    # It assumes gamma_list_draw is a list of JAX arrays with correct shapes (possibly NaNs).
    var_block_cov = jnp.zeros((var_state_total_dim, var_state_total_dim), dtype=_DEFAULT_DTYPE)
    for r_idx in range(var_order):
        for c_idx in range(var_order):
            lag_d = abs(r_idx - c_idx)
            blk_unscaled = gamma_list_draw[lag_d]
            curr_blk = blk_unscaled * gamma_scaling
            if r_idx > c_idx: curr_blk = curr_blk.T
            r_s, r_e = r_idx * n_stationary, (r_idx + 1) * n_stationary
            c_s, c_e = c_idx * n_stationary, (c_idx + 1) * n_stationary
            var_block_cov = var_block_cov.at[r_s:r_e, c_s:c_e].set(curr_blk)
    
    init_cov = init_cov.at[var_start_idx : var_start_idx + var_state_total_dim,
                          var_start_idx : var_start_idx + var_state_total_dim].set(var_block_cov)
    
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    return init_cov


def _create_initial_covariance_standard(state_dim: int, n_dynamic_trends: int) -> jnp.ndarray:
    trend_var_scale_mcmc = 1e6
    var_fallback_scale_mcmc = 4.0
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE) * trend_var_scale_mcmc
    if state_dim > n_dynamic_trends:
        init_cov = init_cov.at[n_dynamic_trends:, n_dynamic_trends:].set(
            jnp.eye(state_dim - n_dynamic_trends, dtype=_DEFAULT_DTYPE) * var_fallback_scale_mcmc
        )
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * 10 * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    return init_cov


def _validate_p0_setup_before_model(reduced_model: ReducedModel, ss_builder: StateSpaceBuilder, use_gamma_init: bool):
    """Static validation of P0 setup before running JAX-traced model."""
    # ... (content as provided previously, ensuring it raises ValueErrors for config issues) ...
    state_dim = ss_builder.state_dim; n_dynamic_trends = ss_builder.n_dynamic_trends
    n_stationary = ss_builder.n_stationary; var_order = ss_builder.var_order
    dynamic_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
    missing_initvals = []
    for trend_name in dynamic_trend_names:
        if trend_name not in reduced_model.initial_values: missing_initvals.append(trend_name)
        else:
            var_spec = reduced_model.initial_values[trend_name]
            if var_spec.init_dist != 'normal_pdf' or len(var_spec.init_params) < 2:
                missing_initvals.append(f"{trend_name} (malformed initval)")
    if missing_initvals: raise ValueError(f"P0 Validation (MCMC): Dynamic trends missing proper 'initval': {missing_initvals}")
    if use_gamma_init and n_stationary > 0:
        if not reduced_model.var_prior_setup: raise ValueError("P0 Validation (MCMC): Gamma init needs 'var_prior_setup'")
        if not reduced_model.stationary_shocks: raise ValueError("P0 Validation (MCMC): Gamma init needs stationary shocks")
    expected_state_dim = n_dynamic_trends + n_stationary * var_order
    if state_dim != expected_state_dim: raise ValueError(f"P0 Validation (MCMC): State dim mismatch. Expected {expected_state_dim}, got {state_dim}")
    for var_name, idx in ss_builder.core_var_map.items():
        if idx >= state_dim: raise ValueError(f"P0 Validation (MCMC): Var '{var_name}' index {idx} >= state_dim {state_dim}")
    print("✓ P0 setup validation passed for MCMC model.")


def define_gpm_numpyro_model(
    gpm_file_path: str,
    use_gamma_init_for_P0: bool = False, # This flag controls user's INTENT for gamma P0
    gamma_init_scaling_for_P0: float = 1.0 # Changed default to 1.0
) -> Tuple[TypingAny, ReducedModel, StateSpaceBuilder]:
    orchestrator = IntegrationOrchestrator(gpm_file_path=gpm_file_path)
    reduced_model: ReducedModel = orchestrator.reduced_model
    ss_builder: StateSpaceBuilder = orchestrator.ss_builder
    _validate_p0_setup_before_model(reduced_model, ss_builder, use_gamma_init_for_P0)

    def gpm_bvar_numpyro_model(y_data: jnp.ndarray):
        T_obs, n_obs_data = y_data.shape
        structural_params_draw = {
            p_name: _sample_parameter_numpyro(p_name, reduced_model.estimated_params[p_name])
            for p_name in reduced_model.parameters
        }
        trend_shock_std_devs_draw = {
            sh_name: _sample_parameter_numpyro(
                f"sigma_{sh_name}",
                reduced_model.estimated_params.get(sh_name) or reduced_model.estimated_params[f"sigma_{sh_name}"]
            ) for sh_name in reduced_model.trend_shocks
        }

        A_transformed_draw = None; Sigma_u_draw_for_ebp = None
        gamma_list_for_P0: List[jnp.ndarray] = []
        n_stat_vars = len(reduced_model.stationary_variables)
        var_order_model = reduced_model.var_prior_setup.var_order if reduced_model.var_prior_setup else 0
        num_dynamic_core_trends = ss_builder.n_core - ss_builder.n_stationary


        if reduced_model.var_prior_setup and n_stat_vars > 0 and var_order_model > 0:
            A_raw_list_draw, Omega_u_chol_draw = _sample_raw_var_coeffs_and_omega_chol(
                reduced_model.var_prior_setup, n_stat_vars)
            stat_shock_std_devs_draw = {
                sh_name: _sample_parameter_numpyro(
                    f"sigma_{sh_name}",
                    reduced_model.estimated_params.get(sh_name) or reduced_model.estimated_params[f"sigma_{sh_name}"]
                ) for sh_name in reduced_model.stationary_shocks
            }
            sigma_u_vec = jnp.array([stat_shock_std_devs_draw[sh_name] for sh_name in reduced_model.stationary_shocks])
            Sigma_u_draw_for_ebp = jnp.diag(sigma_u_vec) @ Omega_u_chol_draw @ Omega_u_chol_draw.T @ jnp.diag(sigma_u_vec)
            Sigma_u_draw_for_ebp = (Sigma_u_draw_for_ebp + Sigma_u_draw_for_ebp.T) / 2.0 + _SP_JITTER * jnp.eye(n_stat_vars)

            if make_stationary_var_transformation_jax is not None:
                # This call happens with JAX tracers as inputs.
                # It's assumed to be JAX-compatible and return JAX arrays (possibly with NaNs).
                phi_list_draw_temp, gamma_list_for_P0_temp = make_stationary_var_transformation_jax(
                    Sigma_u_draw_for_ebp, A_raw_list_draw, n_stat_vars, var_order_model
                )
                # Structural check on output list lengths (Python-level)
                if (isinstance(phi_list_draw_temp, list) and len(phi_list_draw_temp) == var_order_model and
                    isinstance(gamma_list_for_P0_temp, list) and len(gamma_list_for_P0_temp) == var_order_model):
                    A_transformed_draw = jnp.stack(phi_list_draw_temp)
                    numpyro.deterministic("A_transformed", A_transformed_draw)
                    gamma_list_for_P0 = gamma_list_for_P0_temp # Will be list of JAX arrays
                else: # Should not happen if make_stationary_var_transformation_jax is robust
                    A_transformed_draw = jnp.stack(A_raw_list_draw)
                    numpyro.deterministic("A_raw_fallback_struct_err", A_transformed_draw)
                    gamma_list_for_P0 = [] # Explicitly empty
                    numpyro.factor("loglik_transform_struct_penalty", -1e12)
            else: # No transformation function
                A_transformed_draw = jnp.stack(A_raw_list_draw)
                numpyro.deterministic("A_raw_no_transform_func", A_transformed_draw)
                gamma_list_for_P0 = []
        elif reduced_model.stationary_variables: # Has stationary vars but no VAR setup or trivial VAR
             Sigma_u_draw_for_ebp = jnp.empty((0,0), dtype=_DEFAULT_DTYPE) # Or handle as error
             A_transformed_draw = jnp.empty((0,0,0), dtype=_DEFAULT_DTYPE)
             gamma_list_for_P0 = []
        else: # No stationary variables
            Sigma_u_draw_for_ebp = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
            A_transformed_draw = jnp.empty((0,0,0), dtype=_DEFAULT_DTYPE)
            gamma_list_for_P0 = []

        sigma_eta_diag_values = jnp.zeros(num_dynamic_core_trends, dtype=_DEFAULT_DTYPE)
        dynamic_core_trend_names = [cv for cv in reduced_model.core_variables if cv not in reduced_model.stationary_variables]
        for idx_dynamic_trend, core_trend_name in enumerate(dynamic_core_trend_names):
            core_eq = next((eq for eq in reduced_model.core_equations if eq.lhs == core_trend_name), None)
            if core_eq and core_eq.shock:
                sigma_val_sq = trend_shock_std_devs_draw[core_eq.shock] ** 2
                sigma_eta_diag_values = sigma_eta_diag_values.at[idx_dynamic_trend].set(sigma_val_sq)
        Sigma_eta_draw_for_ebp = jnp.diag(sigma_eta_diag_values)
        Sigma_eps_draw = None

        current_draw_bvar_params = EnhancedBVARParams(A=A_transformed_draw, Sigma_u=Sigma_u_draw_for_ebp, Sigma_eta=Sigma_eta_draw_for_ebp, structural_params=structural_params_draw, Sigma_eps=Sigma_eps_draw)
        F_draw, Q_draw, C_draw, H_draw = ss_builder.build_state_space_from_enhanced_bvar(current_draw_bvar_params)

        # P0 Initialization: Decision to use gamma path is static
        if use_gamma_init_for_P0 and n_stat_vars > 0 and var_order_model > 0 and gamma_list_for_P0: # Check non-empty list
            init_mean_draw = _sample_initial_conditions_gamma_based(
                reduced_model, ss_builder, gamma_list_for_P0, gamma_init_scaling_for_P0
            )
            init_cov_draw = _create_initial_covariance_gamma_based(
                ss_builder.state_dim, num_dynamic_core_trends, gamma_list_for_P0,
                n_stat_vars, var_order_model, gamma_init_scaling_for_P0
            )
        else:
            init_mean_draw = _sample_initial_conditions_standard(reduced_model, ss_builder)
            init_cov_draw = _create_initial_covariance_standard(
                ss_builder.state_dim, num_dynamic_core_trends
            )

        # Kalman Filter Likelihood
        matrices_ok_pred = (jnp.all(jnp.isfinite(F_draw)) & jnp.all(jnp.isfinite(Q_draw)) &
                           jnp.all(jnp.isfinite(C_draw)) & jnp.all(jnp.isfinite(H_draw)) &
                           jnp.all(jnp.isfinite(init_mean_draw)) & jnp.all(jnp.isfinite(init_cov_draw)))

        # Regularize Q_draw to ensure Cholesky works if possible
        Q_draw_reg = (Q_draw + Q_draw.T) / 2.0 + _SP_JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)

        # R_draw must be computed using JAX operations; cannot use try-except for Cholesky failure here
        # Attempt Cholesky, if it would fail (e.g., not PSD due to NaNs from bad params),
        # the likelihood should reflect this.
        # A robust way for KF is often to pass Q and let KF handle its Cholesky or stabilization.
        # For now, let's compute R_draw. If it fails (e.g. produces NaNs), matrices_ok_pred should catch it.
        # Alternatively, if KF can take Q directly, that's safer. Assuming KF takes R (Cholesky of Q).
        R_draw = jnp.linalg.cholesky(Q_draw_reg) # This will produce NaNs if Q_draw_reg is not PSD from bad params

        matrices_ok = matrices_ok_pred & jnp.all(jnp.isfinite(R_draw)) # Add R_draw check

        def compute_loglik_branch():
            if KalmanFilter is None: return jnp.array(-1e12, dtype=_DEFAULT_DTYPE)
            kf_instance = KalmanFilter(T=F_draw, R=R_draw, C=C_draw, H=H_draw, init_x=init_mean_draw, init_P=init_cov_draw)
            valid_obs_idx_static = jnp.arange(n_obs_data, dtype=jnp.int32)
            I_obs_static = jnp.eye(n_obs_data, dtype=_DEFAULT_DTYPE)
            ll = kf_instance.log_likelihood(y_data, valid_obs_idx_static, n_obs_data, C_draw, H_draw, I_obs_static)
            return jnp.where(jnp.isfinite(ll), ll, jnp.array(-1e12, dtype=_DEFAULT_DTYPE))

        def return_bad_loglik_branch():
            return jnp.array(-1e12, dtype=_DEFAULT_DTYPE)

        log_likelihood_val = lax.cond(matrices_ok, compute_loglik_branch, return_bad_loglik_branch)
        log_likelihood_val = jnp.asarray(log_likelihood_val).reshape(()) # Ensure scalar

        numpyro.factor("loglik", log_likelihood_val)

    return gpm_bvar_numpyro_model, reduced_model, ss_builder

# --- Main Fitting Function ---
def fit_gpm_numpyro_model(
    gpm_file_path: str, y_data: jnp.ndarray,
    num_warmup: int = 1000, num_samples: int = 2000, num_chains: int = 2, 
    rng_key_seed: int = 0, use_gamma_init_for_P0: bool = False,
    gamma_init_scaling_for_P0: float = 0.01, target_accept_prob: float = 0.85,
    max_tree_depth: int = 10, dense_mass: bool = False 
) -> Tuple[numpyro.infer.MCMC, ReducedModel, StateSpaceBuilder]:
    

    print(f"--- Fitting GPM Model: {gpm_file_path} ---")
    
    model_function, reduced_model, ss_builder = define_gpm_numpyro_model(
        gpm_file_path, use_gamma_init_for_P0, gamma_init_scaling_for_P0)
    
    numpyro.set_host_device_count(num_chains)
    
    kernel_settings = {"target_accept_prob": target_accept_prob, "max_tree_depth": max_tree_depth}
    
    if dense_mass: kernel_settings["dense_mass"] = True
    
    kernel = NUTS(model_function, **kernel_settings)
    mcmc = MCMC(kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples, 
                num_chains=num_chains, 
                chain_method='parallel',
                progress_bar=True)
    rng_key = random.PRNGKey(rng_key_seed); 
    
    start_time = time.time()
    mcmc.run(rng_key, y_data=y_data)
    end_time = time.time(); 
    
    print(f"MCMC completed in {end_time - start_time:.2f}s.")
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