# clean_gpm_bvar_trends/integration_orchestrator.py

from typing import Dict, Any, Tuple, List, Optional
import jax.numpy as jnp # For type hints if needed, and direct use in __main__ example

# Assuming these are in the same package/directory relative to this file
from gpm_model_parser import GPMModelParser, ReducedModel # Ensure class names match
from state_space_builder import StateSpaceBuilder
from common_types import EnhancedBVARParams
from dynamic_parameter_contract import DynamicParameterContract, create_dynamic_parameter_contract 
from constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER # Import constants

class IntegrationOrchestrator:
    """
    Orchestrates the process of parsing a GPM file and using the
    StateSpaceBuilder to construct state-space matrices from parameter inputs.
    """
    def __init__(self, gpm_file_path: str, 
                 contract: Optional[DynamicParameterContract] = None, 
                 strict_validation: bool = True):
        self.gpm_file_path = gpm_file_path

        # Parse the GPM model first
        parser = GPMModelParser()
        self.reduced_model: ReducedModel = parser.parse_file(self.gpm_file_path)
        
        # Create dynamic contract from the parsed model
        self.contract = contract if contract is not None else create_dynamic_parameter_contract(self.reduced_model)
        
        # Build state space builder with the dynamic contract
        self.ss_builder = StateSpaceBuilder(self.reduced_model, self.contract)
        
        # Validate if requested
        if strict_validation:
            self._validate_gpm_model(strict=True)
        
        # Set up convenience attributes
        self.gpm = self.reduced_model
        self.n_trends = self.ss_builder.n_trends
        self.n_stationary = self.ss_builder.n_stationary
        self.n_observed = self.ss_builder.n_observed
        self.var_order = self.ss_builder.var_order
        self.state_dim = self.ss_builder.state_dim
        
        self.trend_var_map = self.ss_builder.core_var_map
        self.stat_var_map = self.ss_builder.stat_var_map
        self.obs_var_map = self.ss_builder.obs_var_map

        print(f"IntegrationOrchestrator initialized with dynamic contract for GPM: {gpm_file_path}")

    def _validate_gpm_model(self, strict: bool = True):
        """Enhanced validation that checks contract consistency"""
        errors = []
        warnings = []
        
        # Standard GPM validation (existing code)
        # ... keep existing validation logic ...
        
        # NEW: Validate contract consistency
        print("\n--- Validating Dynamic Parameter Contract Consistency ---")
        
        # Check that all trend shocks in equations are declared
        used_trend_shocks = set()
        for eq in self.reduced_model.core_equations:
            if eq.shock:
                used_trend_shocks.add(eq.shock)
        
        declared_trend_shocks = set(self.reduced_model.trend_shocks)
        
        # Shocks used but not declared
        undeclared_shocks = used_trend_shocks - declared_trend_shocks
        if undeclared_shocks:
            errors.append(f"Trend shocks used in equations but not declared: {undeclared_shocks}")
        
        # Shocks declared but not used
        unused_shocks = declared_trend_shocks - used_trend_shocks
        if unused_shocks:
            warnings.append(f"Declared trend shocks not used in any equation: {unused_shocks}")
        
        # Check that all parameters have priors
        missing_priors = []
        for param_name in self.reduced_model.parameters:
            if param_name not in self.reduced_model.estimated_params:
                missing_priors.append(param_name)
        
        for shock_name in self.reduced_model.trend_shocks + self.reduced_model.stationary_shocks:
            if shock_name not in self.reduced_model.estimated_params and f"sigma_{shock_name}" not in self.reduced_model.estimated_params:
                missing_priors.append(f"sigma_{shock_name}")
        
        if missing_priors:
            errors.append(f"Parameters/shocks without priors in estimated_params: {missing_priors}")
        
        # Report results
        if warnings:
            print("⚠️  CONTRACT VALIDATION WARNINGS:")
            for warning in warnings:
                print(f"    - {warning}")
            print()
        
        if errors:
            error_msg = "🚫 CONTRACT VALIDATION ERRORS:\n" + "\n".join(f"    - {error}" for error in errors)
            error_msg += "\n\n💡 SUGGESTIONS:"
            error_msg += "\n    - Ensure all shocks used in equations are declared in shock blocks"
            error_msg += "\n    - Add missing priors to estimated_params block"
            error_msg += "\n    - Check shock naming consistency"
            
            if strict:
                raise ValueError(error_msg)
            else:
                print(error_msg)
        else:
            print("✅ Contract validation passed!")

    
    
    def validate_mcmc_requirements(self, use_gamma_init: bool = False):
        """Validate requirements for MCMC sampling"""
        errors = []
        
        # Check P0 requirements
        if use_gamma_init:
            if not self.reduced_model.stationary_variables:
                errors.append("Gamma P0 initialization requested but no stationary variables for VAR covariances")
            if not self.reduced_model.var_prior_setup:
                errors.append("Gamma P0 initialization requested but no VAR prior setup")
        
        # Check all estimated parameters have valid priors
        for param_name, prior_spec in self.reduced_model.estimated_params.items():
            if prior_spec.distribution not in ['normal_pdf', 'inv_gamma_pdf']:
                errors.append(f"Parameter '{param_name}' has unsupported prior '{prior_spec.distribution}'")
            
            if len(prior_spec.params) < 2:
                errors.append(f"Parameter '{param_name}' prior needs at least 2 parameters")
        
        if errors:
            raise ValueError(f"MCMC REQUIREMENTS VALIDATION FAILED:\n" + "\n".join(f"    - {e}" for e in errors))
    
  
    
    def build_ss_from_mcmc_sample(self,
                                  mcmc_samples_dict: Dict[str, jnp.ndarray],
                                  sample_idx: int) -> Tuple[jnp.ndarray, ...]:
        return self.ss_builder.build_state_space_from_mcmc_sample(
            mcmc_samples_dict, sample_idx
        )

    def build_ss_from_enhanced_bvar(self,
                                    bvar_params: EnhancedBVARParams) -> Tuple[jnp.ndarray, ...]:
        return self.ss_builder.build_state_space_from_enhanced_bvar(bvar_params)

    def build_ss_from_direct_dict(self,
                                  direct_params: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        return self.ss_builder.build_state_space_from_direct_dict(direct_params)

    def build_state_space_matrices(self, params_input: Any, sample_idx: Optional[int] = None) -> Tuple[jnp.ndarray, ...]:
        if isinstance(params_input, EnhancedBVARParams):
            return self.build_ss_from_enhanced_bvar(params_input)
        elif isinstance(params_input, dict):
            if sample_idx is not None:
                is_likely_mcmc_output = all(isinstance(v, jnp.ndarray) and v.ndim > 0 for v in params_input.values())
                if is_likely_mcmc_output:
                     return self.build_ss_from_mcmc_sample(params_input, sample_idx)
                else:
                     return self.build_ss_from_direct_dict(params_input)
            else:
                return self.build_ss_from_direct_dict(params_input)
        else:
            raise TypeError(f"Unsupported parameter type for build_state_space_matrices: {type(params_input)}")

    def get_variable_names(self) -> Dict[str, List[str]]:
        return {
            'trend_variables': list(self.ss_builder.core_var_map.keys()),
            'stationary_variables': list(self.ss_builder.stat_var_map.keys()),
            'observed_variables': list(self.ss_builder.obs_var_map.keys()),
            'parameters': self.reduced_model.parameters
        }

    def get_model_summary(self) -> str:
        summary = [
            "INTEGRATION ORCHESTRATOR MODEL SUMMARY", "=" * 50,
            f"GPM File: {self.gpm_file_path}",
            f"Core Variables (Trends): {self.n_trends} {list(self.reduced_model.core_variables)}",
            f"Stationary Variables: {self.n_stationary} {self.reduced_model.stationary_variables}",
            f"Observed Variables: {self.n_observed} {list(self.reduced_model.reduced_measurement_equations.keys())}",
            f"State Dimension: {self.state_dim}", 
            f"VAR Order: {self.var_order}",
            f"Parameters defined in GPM: {self.reduced_model.parameters}", 
            "=" * 50
        ]
        return "\n".join(summary)

    def test_state_space_construction(self, test_params: Optional[Dict[str, Any]] = None) -> bool:
        if test_params is None:
            test_params = {}
            required_builder_param_names = set()
            if self.contract:
                for mcmc_name in self.contract.get_required_mcmc_parameters():
                    try:
                        required_builder_param_names.add(self.contract.get_builder_name(mcmc_name))
                    except ValueError:
                        pass
            
            for param_name_builder in required_builder_param_names:
                 if "var_phi" in param_name_builder:
                     test_params[param_name_builder] = 0.5
                 else:
                     test_params[param_name_builder] = 0.1

            if self.n_stationary > 0 and self.var_order > 0:
                test_params['_var_coefficients'] = jnp.zeros((self.var_order, self.n_stationary, self.n_stationary), dtype=_DEFAULT_DTYPE)
                test_params['_var_coefficients'] = test_params['_var_coefficients'].at[0].set(jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE)*0.7)
                test_params['_var_innovation_corr_chol'] = jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE)
            elif self.n_stationary > 0:
                 test_params['_var_innovation_corr_chol'] = jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE)

        try:
            F, Q, C, H = self.build_ss_from_direct_dict(test_params)
            
            if not (F.shape == (self.state_dim, self.state_dim) and
                    Q.shape == (self.state_dim, self.state_dim) and
                    C.shape == (self.n_observed, self.state_dim) and
                    H.shape == (self.n_observed, self.n_observed)):
                print(f"✗ ERROR: Matrix shape mismatch in orchestrator test.")
                return False

            if not (jnp.all(jnp.isfinite(F)) and jnp.all(jnp.isfinite(Q)) and
                    jnp.all(jnp.isfinite(C)) and jnp.all(jnp.isfinite(H))):
                print(f"✗ ERROR: Matrices contain non-finite values in orchestrator test.")
                return False
            
            return True
            
        except Exception as e:
            import traceback
            print(f"✗ Orchestrator: State space construction test FAILED: {e}")
            return False

def create_integration_orchestrator(gpm_file_path: str, strict_validation: bool = True) -> IntegrationOrchestrator:
    return IntegrationOrchestrator(gpm_file_path, strict_validation=strict_validation)

class ReducedGPMIntegration:
    def __init__(self, gpm_file_path: str):
        self.orchestrator = IntegrationOrchestrator(gpm_file_path)
        self.reduced_model = self.orchestrator.reduced_model
        self.gpm = self.orchestrator.gpm
        self.n_trends = self.orchestrator.n_trends
        self.n_stationary = self.orchestrator.n_stationary
        self.n_observed = self.orchestrator.n_observed
        self.var_order = self.orchestrator.var_order
        self.state_dim = self.orchestrator.state_dim
        self.trend_var_map = self.orchestrator.trend_var_map
        self.stat_var_map = self.orchestrator.stat_var_map
        self.obs_var_map = self.orchestrator.obs_var_map
    
    def build_state_space_matrices(self, params_input: Any, sample_idx: Optional[int]=None) -> Tuple[jnp.ndarray,...]:
        return self.orchestrator.build_state_space_matrices(params_input, sample_idx)
    
    def get_variable_names(self) -> Dict[str, List[str]]:
        return self.orchestrator.get_variable_names()

def create_reduced_gpm_model(gpm_file_path: str):
    integration_wrapper = ReducedGPMIntegration(gpm_file_path)
    return integration_wrapper, integration_wrapper.reduced_model, integration_wrapper.orchestrator.ss_builder

if __name__ == "__main__":
    import os # For __main__ block
    dummy_gpm_content = """
parameters var_phi;
estimated_params;
    stderr SHK_TREND1, inv_gamma_pdf, 2.0, 0.5;
    var_phi, normal_pdf, 1.0, 0.2;
    stderr shk_stat1, inv_gamma_pdf, 2.0, 0.5;
end;
trends_vars TREND1;
stationary_variables stat1;
trend_shocks; var SHK_TREND1; end;
shocks; var shk_stat1; end;
trend_model; TREND1 = TREND1(-1) + SHK_TREND1; end;
varobs OBS1;
measurement_equations; OBS1 = TREND1 + stat1 + var_phi * TREND1; end;
var_prior_setup; var_order = 1; es = 0.5,0.1; fs=1,1; gs=1,1; hs=1,1; eta=1; end;
    """
    gpm_test_file = "dummy_orchestrator_test.gpm"
    with open(gpm_test_file, "w") as f:
        f.write(dummy_gpm_content)

    try:
        print("--- Testing IntegrationOrchestrator ---")
        orchestrator = create_integration_orchestrator(gpm_test_file)
        print(orchestrator.get_model_summary())

        # Test with direct_dict. Keys here should be what StateSpaceBuilder expects
        # after MCMC names are translated by its _standardize_direct_params.
        # So, for shocks, it's "SHK_TREND1" (builder name), not "sigma_SHK_TREND1" (MCMC name).
        test_direct_params = {
            "var_phi": jnp.array(0.9),
            "SHK_TREND1": jnp.array(0.15), # Builder name for shock (std.dev.)
            "shk_stat1": jnp.array(0.25),   # Builder name for shock (std.dev.)
            "_var_coefficients": jnp.array([[[0.6]]], dtype=_DEFAULT_DTYPE),
            "_var_innovation_corr_chol": jnp.array([[1.0]], dtype=_DEFAULT_DTYPE)
        }
        print(f"\nTesting build_ss_from_direct_dict with (builder-friendly keys): {test_direct_params}")
        if orchestrator.test_state_space_construction(test_direct_params):
             print("✓ Direct dictionary test passed via orchestrator.test_state_space_construction.")
        else:
             print("✗ Direct dictionary test FAILED via orchestrator.test_state_space_construction.")

        mcmc_output_example = {
            "var_phi": jnp.array([0.8, 0.9, 1.0]),
            "sigma_SHK_TREND1": jnp.array([0.1, 0.15, 0.2]), # MCMC name
            "sigma_shk_stat1": jnp.array([0.2, 0.25, 0.3]),   # MCMC name
            "A_transformed": jnp.array([[[[0.5]]], [[[0.6]]], [[[0.7]]]], dtype=_DEFAULT_DTYPE),
            "Omega_u_chol": jnp.array([[[1.0]], [[1.0]], [[1.0]]], dtype=_DEFAULT_DTYPE)
        }
        sample_to_test = 1
        print(f"\nTesting build_ss_from_mcmc_sample for sample_idx {sample_to_test}")
        F, Q, C, H = orchestrator.build_ss_from_mcmc_sample(mcmc_output_example, sample_to_test)
        if F is not None and jnp.all(jnp.isfinite(F)):
            print(f"✓ MCMC sample processing via orchestrator OK for sample {sample_to_test}.")
            # print(f"  F[0,0]={F[0,0]}, Q[0,0]={Q[0,0]}, C[0,0]={C[0,0]}, H[0,0]={H[0,0]}")
        else:
            print(f"✗ MCMC sample processing FAILED for sample {sample_to_test}.")
    finally:
        if os.path.exists(gpm_test_file):
            os.remove(gpm_test_file)