"""
Dynamic Parameter Contract - Built from Parsed GPM Model
========================================================

This module creates parameter contracts dynamically from the parsed GPM model,
eliminating hardcoded parameter lists and ensuring consistency.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

from .gpm_model_parser import ReducedModel

class ParameterType(Enum):
    """Types of parameters in the system"""
    STRUCTURAL = "structural"
    TREND_SHOCK = "trend_shock" 
    STATIONARY_SHOCK = "stationary_shock"
    VAR_COEFFICIENT = "var_coefficient"
    VAR_COVARIANCE = "var_covariance"

@dataclass
class ParameterSpec:
    """Specification for a single parameter"""
    mcmc_name: str          # Name as produced by MCMC sampler
    builder_name: str       # Name expected by state space builder  
    param_type: ParameterType
    required: bool = True
    description: str = ""

class DynamicParameterContract:
    """
    Builds parameter contract dynamically from parsed GPM model.
    
    This ensures the contract matches exactly what's in the GPM file,
    eliminating hardcoded parameter lists and sync issues.
    """
    
    def __init__(self, gpm_model: ReducedModel):
        self.gpm_model = gpm_model
        self._contract = self._build_dynamic_contract()
        self._mcmc_to_builder = {spec.mcmc_name: spec.builder_name for spec in self._contract}
        self._builder_to_mcmc = {spec.builder_name: spec.mcmc_name for spec in self._contract}
        self._parameter_types = {spec.mcmc_name: spec.param_type for spec in self._contract}
    
    def _build_dynamic_contract(self) -> List[ParameterSpec]:
        """Build contract dynamically from the parsed GPM model"""
        
        contract = []
        
        # ================================================================
        # STRUCTURAL PARAMETERS (from GPM parameters block)
        # ================================================================
        for param_name in self.gpm_model.parameters:
            contract.append(ParameterSpec(
                mcmc_name=param_name,
                builder_name=param_name, 
                param_type=ParameterType.STRUCTURAL,
                description=f"Structural parameter {param_name} from GPM"
            ))
        
        # ================================================================
        # TREND SHOCK STANDARD DEVIATIONS (from GPM trend_shocks block)
        # ================================================================
        for shock_name in self.gpm_model.trend_shocks:
            contract.append(ParameterSpec(
                mcmc_name=f"sigma_{shock_name}",
                builder_name=shock_name,
                param_type=ParameterType.TREND_SHOCK,
                description=f"Standard deviation of trend shock {shock_name}"
            ))
        
        # ================================================================
        # STATIONARY SHOCK STANDARD DEVIATIONS (from GPM stationary_shocks)
        # ================================================================
        for shock_name in self.gpm_model.stationary_shocks:
            contract.append(ParameterSpec(
                mcmc_name=f"sigma_{shock_name}",
                builder_name=shock_name,
                param_type=ParameterType.STATIONARY_SHOCK,
                description=f"Standard deviation of stationary shock {shock_name}"
            ))
        
        # ================================================================
        # VAR SYSTEM PARAMETERS (if VAR setup exists)
        # ================================================================
        if self.gpm_model.var_prior_setup is not None:
            contract.append(ParameterSpec(
                mcmc_name="A_transformed", 
                builder_name="_var_coefficients",
                param_type=ParameterType.VAR_COEFFICIENT,
                description="VAR coefficient matrices from MCMC"
            ))
            
            contract.append(ParameterSpec(
                mcmc_name="Omega_u_chol",
                builder_name="_var_innovation_corr_chol", 
                param_type=ParameterType.VAR_COVARIANCE,
                description="VAR innovation correlation Cholesky from MCMC"
            ))
        
        return contract
    
    def get_builder_name(self, mcmc_name: str) -> str:
        """Convert MCMC parameter name to builder expected name"""
        if mcmc_name not in self._mcmc_to_builder:
            # Provide detailed error with actual available parameters
            available_mcmc = sorted(self._mcmc_to_builder.keys())
            raise ValueError(
                f"PARAMETER CONTRACT VIOLATION: MCMC parameter '{mcmc_name}' not found in dynamic contract.\n"
                f"Available MCMC parameters: {available_mcmc}\n"
                f"GPM model trend shocks: {self.gpm_model.trend_shocks}\n"
                f"GPM model stationary shocks: {self.gpm_model.stationary_shocks}\n"
                f"GPM model parameters: {self.gpm_model.parameters}\n"
                f"Context: Converting MCMC output to builder input\n"
                f"Solution: Check if parameter is properly declared in GPM file"
            )
        
        return self._mcmc_to_builder[mcmc_name]
    
    def get_mcmc_name(self, builder_name: str) -> str:
        """Convert builder parameter name to MCMC expected name"""
        if builder_name not in self._builder_to_mcmc:
            available_builder = sorted(self._builder_to_mcmc.keys())
            raise ValueError(
                f"PARAMETER CONTRACT VIOLATION: Builder parameter '{builder_name}' not found in dynamic contract.\n"
                f"Available builder parameters: {available_builder}\n"
                f"Context: Converting builder needs to MCMC parameter names\n"
                f"Solution: Check if parameter mapping is correct"
            )
        
        return self._builder_to_mcmc[builder_name]
    
    def get_parameter_type(self, mcmc_name: str) -> ParameterType:
        """Get the type of a parameter"""
        if mcmc_name not in self._parameter_types:
            raise ValueError(f"Parameter '{mcmc_name}' not found in dynamic contract")
        return self._parameter_types[mcmc_name]
    
    def get_required_mcmc_parameters(self) -> Set[str]:
        """Get set of all required MCMC parameter names"""
        return {spec.mcmc_name for spec in self._contract if spec.required}
    
    def get_required_builder_parameters(self) -> Set[str]:
        """Get set of all required builder parameter names"""  
        return {spec.builder_name for spec in self._contract if spec.required}
    
    def validate_mcmc_parameters(self, mcmc_params: Dict[str, any]) -> None:
        """Validate that MCMC parameters match contract expectations"""
        provided_params = set(mcmc_params.keys())
        required_params = self.get_required_mcmc_parameters()
        
        # Check for missing required parameters
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(
                f"MCMC PARAMETER VALIDATION FAILED: Missing required parameters.\n"
                f"Missing: {sorted(missing_params)}\n"
                f"Provided: {sorted(provided_params)}\n"  
                f"Required: {sorted(required_params)}\n"
                f"Context: Validating MCMC output against GPM model requirements\n"
                f"Solution: Ensure MCMC sampler produces all required parameters from GPM"
            )
        
        # Check for unexpected parameters (warnings only)
        # Allow special MCMC parameters that might not be in GPM
        allowed_extra = {
            "A_raw", "A_diag_0", "A_full_0", "Amu_0", "Amu_1", "Aomega_0", "Aomega_1",
            "init_mean_full", "_var_innovation_cov", "_trend_innovation_cov_full"
        }
        unexpected_params = provided_params - required_params - allowed_extra
        if unexpected_params:
            print(f"INFO: Additional MCMC parameters not in GPM contract: {sorted(unexpected_params)}")
    
    def get_contract_summary(self) -> str:
        """Get human-readable summary of the dynamic parameter contract"""
        
        summary = [
            "DYNAMIC PARAMETER CONTRACT SUMMARY", 
            "=" * 50, 
            f"Built from GPM model with:",
            f"  - {len(self.gpm_model.parameters)} structural parameters",
            f"  - {len(self.gpm_model.trend_shocks)} trend shocks", 
            f"  - {len(self.gpm_model.stationary_shocks)} stationary shocks",
            f"  - VAR setup: {'Yes' if self.gpm_model.var_prior_setup else 'No'}",
            ""
        ]
        
        by_type = {}
        for spec in self._contract:
            if spec.param_type not in by_type:
                by_type[spec.param_type] = []
            by_type[spec.param_type].append(spec)
        
        for param_type, specs in by_type.items():
            summary.append(f"{param_type.value.upper()} PARAMETERS:")
            for spec in specs:
                summary.append(f"  {spec.mcmc_name} → {spec.builder_name}")
            summary.append("")
        
        return "\n".join(summary)


def create_dynamic_parameter_contract(gpm_model: ReducedModel) -> DynamicParameterContract:
    """Factory function to create a dynamic parameter contract from GPM model"""
    return DynamicParameterContract(gpm_model)


# For backward compatibility - update existing functions
def get_parameter_contract(gpm_model: Optional[ReducedModel] = None) -> DynamicParameterContract:
    """Get parameter contract - now requires GPM model for dynamic generation"""
    if gpm_model is None:
        raise ValueError(
            "Dynamic parameter contract requires a parsed GPM model. "
            "Pass the ReducedModel from GPMModelParser to get_parameter_contract()"
        )
    return create_dynamic_parameter_contract(gpm_model)


def mcmc_to_builder_name(mcmc_name: str, gpm_model: ReducedModel) -> str:
    """Convert MCMC parameter name to builder name"""
    contract = get_parameter_contract(gpm_model)
    return contract.get_builder_name(mcmc_name)


def builder_to_mcmc_name(builder_name: str, gpm_model: ReducedModel) -> str:
    """Convert builder parameter name to MCMC name"""
    contract = get_parameter_contract(gpm_model)
    return contract.get_mcmc_name(builder_name)


if __name__ == "__main__":
    # Example usage - this would normally use a real parsed GPM model
    print("Dynamic Parameter Contract - requires parsed GPM model for testing")
    print("See integration_orchestrator.py for usage examples")


# """
# Parameter Contract - Single Source of Truth for Parameter Naming
# ================================================================

# This module defines the exact parameter naming conventions and mappings
# between different components of the system. NO GUESSING, NO FALLBACKS.
# """

# from typing import Dict, List, Set, Optional
# from dataclasses import dataclass
# from enum import Enum

# class ParameterType(Enum):
#     """Types of parameters in the system"""
#     STRUCTURAL = "structural"
#     TREND_SHOCK = "trend_shock" 
#     STATIONARY_SHOCK = "stationary_shock"
#     VAR_COEFFICIENT = "var_coefficient"
#     VAR_COVARIANCE = "var_covariance"

# @dataclass
# class ParameterSpec:
#     """Specification for a single parameter"""
#     mcmc_name: str          # Name as produced by MCMC sampler
#     builder_name: str       # Name expected by state space builder  
#     param_type: ParameterType
#     required: bool = True
#     description: str = ""

# class ParameterContract:
#     """
#     Defines the exact contract between MCMC sampler and state space builder.
    
#     This is the SINGLE SOURCE OF TRUTH for parameter naming conventions.
#     Any changes to parameter names must be made here and only here.
#     """
    
#     def __init__(self):
#         self._contract = self._build_contract()
#         self._mcmc_to_builder = {spec.mcmc_name: spec.builder_name for spec in self._contract}
#         self._builder_to_mcmc = {spec.builder_name: spec.mcmc_name for spec in self._contract}
#         self._parameter_types = {spec.mcmc_name: spec.param_type for spec in self._contract}
    
#     def _build_contract(self) -> List[ParameterSpec]:
#         """Build the complete parameter contract based on GPM file analysis"""
        
#         contract = []
        
#         # ================================================================
#         # STRUCTURAL PARAMETERS
#         # ================================================================
#         contract.append(ParameterSpec(
#             mcmc_name="var_phi",
#             builder_name="var_phi", 
#             param_type=ParameterType.STRUCTURAL,
#             description="Coefficient of relative risk aversion"
#         ))
        
#         # ================================================================
#         # TREND SHOCK STANDARD DEVIATIONS
#         # ================================================================
#         # Pattern: MCMC produces "sigma_shk_trend_X", builder expects "shk_trend_X"
        
#         trend_shocks = [
#             # World level shocks
#             "shk_trend_r_world",
#             "shk_trend_pi_world", 
#             "shk_sp_trend_world",
#             "shk_theta_world",
            
#             # Country-specific trend shocks
#             "shk_trend_r_us",
#             "shk_trend_pi_us",
#             "shk_trend_r_ea", 
#             "shk_trend_pi_ea",
#             "shk_trend_r_jp",
#             "shk_trend_pi_jp",
            
#             # Country-specific risk premium shocks
#             "shk_sp_trend_us",
#             "shk_sp_trend_ea",
#             "shk_sp_trend_jp",
            
#             # Country-specific productivity/preference shocks
#             "shk_theta_us",
#             "shk_theta_ea", 
#             "shk_theta_jp",
            
#             # Additional shocks from GPM file (mixed naming patterns)
#             "SHK_L_GDP_TREND",
#             "SHK_G_TREND", 
#             "SHK_PI_TREND",
#             "SHK_RR_TREND",
#             "SHK_L_GDP_GAP",
#             "SHK_DLA_CPI",
#             "SHK_RS"
#         ]
        
#         for shock in trend_shocks:
#             contract.append(ParameterSpec(
#                 mcmc_name=f"sigma_{shock}",
#                 builder_name=shock,
#                 param_type=ParameterType.TREND_SHOCK,
#                 description=f"Standard deviation of {shock}"
#             ))
        
#         # ================================================================
#         # STATIONARY SHOCK STANDARD DEVIATIONS  
#         # ================================================================
#         # Pattern: MCMC produces "sigma_shk_cycle_X", builder expects "shk_cycle_X"
        
#         stationary_shocks = [
#             "shk_cycle_y_us",
#             "shk_cycle_y_ea", 
#             "shk_cycle_y_jp",
#             "shk_cycle_r_us",
#             "shk_cycle_r_ea",
#             "shk_cycle_r_jp",
#             "shk_cycle_pi_us",
#             "shk_cycle_pi_ea",
#             "shk_cycle_pi_jp"
#         ]
        
#         for shock in stationary_shocks:
#             contract.append(ParameterSpec(
#                 mcmc_name=f"sigma_{shock}",
#                 builder_name=shock,
#                 param_type=ParameterType.STATIONARY_SHOCK,
#                 description=f"Standard deviation of {shock}"
#             ))
        
#         # ================================================================
#         # VAR SYSTEM PARAMETERS
#         # ================================================================
#         # These are handled specially as they come from covariance matrices
        
#         contract.append(ParameterSpec(
#             mcmc_name="_var_coefficients", 
#             builder_name="_var_coefficients",
#             param_type=ParameterType.VAR_COEFFICIENT,
#             description="VAR coefficient matrices from MCMC"
#         ))
        
#         contract.append(ParameterSpec(
#             mcmc_name="_var_innovation_cov",
#             builder_name="_var_innovation_cov", 
#             param_type=ParameterType.VAR_COVARIANCE,
#             description="VAR innovation covariance from MCMC"
#         ))
        
#         return contract
    
#     def get_builder_name(self, mcmc_name: str) -> str:
#         """
#         Convert MCMC parameter name to builder expected name.
        
#         Args:
#             mcmc_name: Parameter name as produced by MCMC
            
#         Returns:
#             Builder expected parameter name
            
#         Raises:
#             ValueError: If parameter name not found in contract
#         """
#         if mcmc_name not in self._mcmc_to_builder:
#             raise ValueError(
#                 f"PARAMETER CONTRACT VIOLATION: MCMC parameter '{mcmc_name}' not found in contract.\n"
#                 f"Expected MCMC parameters: {sorted(self._mcmc_to_builder.keys())}\n"
#                 f"Context: Converting MCMC output to builder input\n"
#                 f"Solution: Add missing parameter to ParameterContract._build_contract()"
#             )
        
#         return self._mcmc_to_builder[mcmc_name]
    
#     def get_mcmc_name(self, builder_name: str) -> str:
#         """
#         Convert builder parameter name to MCMC expected name.
        
#         Args:
#             builder_name: Parameter name expected by builder
            
#         Returns:
#             MCMC parameter name
            
#         Raises:
#             ValueError: If parameter name not found in contract
#         """
#         if builder_name not in self._builder_to_mcmc:
#             raise ValueError(
#                 f"PARAMETER CONTRACT VIOLATION: Builder parameter '{builder_name}' not found in contract.\n"
#                 f"Expected builder parameters: {sorted(self._builder_to_mcmc.keys())}\n"
#                 f"Context: Converting builder needs to MCMC parameter names\n"
#                 f"Solution: Add missing parameter to ParameterContract._build_contract()"
#             )
        
#         return self._builder_to_mcmc[builder_name]
    
#     def get_parameter_type(self, mcmc_name: str) -> ParameterType:
#         """Get the type of a parameter"""
#         if mcmc_name not in self._parameter_types:
#             raise ValueError(f"Parameter '{mcmc_name}' not found in contract")
#         return self._parameter_types[mcmc_name]
    
#     def get_required_mcmc_parameters(self) -> Set[str]:
#         """Get set of all required MCMC parameter names"""
#         return {spec.mcmc_name for spec in self._contract if spec.required}
    
#     def get_required_builder_parameters(self) -> Set[str]:
#         """Get set of all required builder parameter names"""  
#         return {spec.builder_name for spec in self._contract if spec.required}
    
#     def validate_mcmc_parameters(self, mcmc_params: Dict[str, any]) -> None:
#         """
#         Validate that MCMC parameters match contract expectations.
        
#         Args:
#             mcmc_params: Dictionary of parameters from MCMC
            
#         Raises:
#             ValueError: If required parameters are missing or unexpected parameters found
#         """
#         provided_params = set(mcmc_params.keys())
#         required_params = self.get_required_mcmc_parameters()
        
#         # Check for missing required parameters
#         missing_params = required_params - provided_params
#         if missing_params:
#             raise ValueError(
#                 f"MCMC PARAMETER VALIDATION FAILED: Missing required parameters.\n"
#                 f"Missing: {sorted(missing_params)}\n"
#                 f"Provided: {sorted(provided_params)}\n"  
#                 f"Required: {sorted(required_params)}\n"
#                 f"Context: Validating MCMC output before conversion\n"
#                 f"Solution: Ensure MCMC sampler produces all required parameters"
#             )
        
#         # Check for unexpected parameters (optional - can be warnings)
#         unexpected_params = provided_params - required_params - {
#             # Allow these special parameters that might come from MCMC
#             "A_transformed", "A_raw", "Omega_u_chol", "init_mean_full"
#         }
#         if unexpected_params:
#             print(f"WARNING: Unexpected MCMC parameters (will be ignored): {sorted(unexpected_params)}")
    
#     def validate_builder_parameters(self, builder_params: Dict[str, any]) -> None:
#         """
#         Validate that builder parameters match contract expectations.
        
#         Args:
#             builder_params: Dictionary of parameters for builder
            
#         Raises:
#             ValueError: If required parameters are missing
#         """
#         provided_params = set(builder_params.keys())
#         required_params = self.get_required_builder_parameters()
        
#         missing_params = required_params - provided_params
#         if missing_params:
#             raise ValueError(
#                 f"BUILDER PARAMETER VALIDATION FAILED: Missing required parameters.\n"
#                 f"Missing: {sorted(missing_params)}\n"
#                 f"Provided: {sorted(provided_params)}\n"
#                 f"Required: {sorted(required_params)}\n"
#                 f"Context: Validating parameters before state space construction\n"
#                 f"Solution: Ensure parameter conversion provides all required parameters"
#             )
    
#     def get_contract_summary(self) -> str:
#         """Get human-readable summary of the parameter contract"""
        
#         summary = ["PARAMETER CONTRACT SUMMARY", "=" * 50, ""]
        
#         by_type = {}
#         for spec in self._contract:
#             if spec.param_type not in by_type:
#                 by_type[spec.param_type] = []
#             by_type[spec.param_type].append(spec)
        
#         for param_type, specs in by_type.items():
#             summary.append(f"{param_type.value.upper()} PARAMETERS:")
#             for spec in specs:
#                 summary.append(f"  {spec.mcmc_name} → {spec.builder_name}")
#             summary.append("")
        
#         return "\n".join(summary)


# # Global contract instance - single source of truth
# PARAMETER_CONTRACT = ParameterContract()


# def get_parameter_contract() -> ParameterContract:
#     """Get the global parameter contract instance"""
#     return PARAMETER_CONTRACT


# # Convenience functions for direct use
# def mcmc_to_builder_name(mcmc_name: str) -> str:
#     """Convert MCMC parameter name to builder name"""
#     return PARAMETER_CONTRACT.get_builder_name(mcmc_name)


# def builder_to_mcmc_name(builder_name: str) -> str:
#     """Convert builder parameter name to MCMC name"""
#     return PARAMETER_CONTRACT.get_mcmc_name(builder_name)


# def validate_mcmc_output(mcmc_params: Dict[str, any]) -> None:
#     """Validate MCMC parameter dictionary"""
#     PARAMETER_CONTRACT.validate_mcmc_parameters(mcmc_params)


# def validate_builder_input(builder_params: Dict[str, any]) -> None:
#     """Validate builder parameter dictionary"""
#     PARAMETER_CONTRACT.validate_builder_parameters(builder_params)


# if __name__ == "__main__":
#     # Print contract summary for debugging
#     print(PARAMETER_CONTRACT.get_contract_summary())