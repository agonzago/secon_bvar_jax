"""
Integration Coordinator - Orchestrate the Complete Pipeline
==========================================================

This module provides the single entry point for converting MCMC output
to state space matrices. It orchestrates the entire pipeline:

MCMC Output → Parameter Transformation → State Space Construction

Replaces the complex integration_helper.py with a clean, contract-driven approach.
"""

import jax.numpy as jnp
from typing import Dict, Any, Tuple, List, Optional
from parameter_contract import get_parameter_contract
from mcmc_adapter import create_mcmc_adapter

# Handle import paths flexibly
try:
    from reduced_gpm_parser import ReducedModel, ReducedGPMParser
    from reduced_state_space_builder import ReducedStateSpaceBuilder
except ImportError:
    try:
        from reduced_gpm_parser import ReducedModel, ReducedGPMParser
        from reduced_state_space_builder import ReducedStateSpaceBuilder
    except ImportError:
        print("Warning: Could not import reduced parser modules. Ensure correct paths.")

class IntegrationCoordinator:
    """
    Coordinates the complete pipeline from MCMC output to state space matrices.
    
    This class provides a clean interface that replaces the complex integration 
    helper with explicit, contract-driven parameter transformation.
    """
    
    def __init__(self, gmp_file_path: str):
        """
        Initialize coordinator with GPM model.
        
        Args:
            gmp_file_path: Path to GPM file
        """
        self.gmp_file_path = gmp_file_path
        self.contract = get_parameter_contract()
        self.adapter = create_mcmc_adapter()
        
        # Parse and setup the reduced model
        self._setup_reduced_model()
        
        print(f"IntegrationCoordinator initialized:")
        print(f"  GPM file: {gmp_file_path}")
        print(f"  Core trends: {self.n_trends}")
        print(f"  Stationary variables: {self.n_stationary}")
        print(f"  State dimension: {self.state_dim}")
    
    def _setup_reduced_model(self):
        """Setup the reduced model and state space builder"""
        
        try:
            # Parse the model
            parser = ReducedGPMParser()
            self.reduced_model = parser.parse_file(self.gmp_file_path)
            
            # Add compatibility attributes
            self.reduced_model.trend_variables = self.reduced_model.core_variables
            self.reduced_model.observed_variables = list(self.reduced_model.reduced_measurement_equations.keys())
            
            # Extract shocks from equations
            trend_shocks = []
            for eq in self.reduced_model.core_equations:
                if eq.shock:
                    trend_shocks.append(eq.shock)
            self.reduced_model.trend_shocks = trend_shocks
            
            # Add stationary shocks
            if not hasattr(self.reduced_model, 'stationary_shocks'):
                self.reduced_model.stationary_shocks = [f"shk_cycle_{var}" for var in self.reduced_model.stationary_variables]
            
            # Add missing attributes
            if not hasattr(self.reduced_model, 'initial_values'):
                self.reduced_model.initial_values = {}
            if not hasattr(self.reduced_model, 'trend_equations'):
                self.reduced_model.trend_equations = self.reduced_model.core_equations
            if not hasattr(self.reduced_model, 'measurement_equations'):
                self.reduced_model.measurement_equations = []
            
            # Create state space builder
            self.builder = ReducedStateSpaceBuilder(self.reduced_model)
            
            # Store dimensions for compatibility
            self.n_trends = len(self.reduced_model.core_variables)
            self.n_stationary = len(self.reduced_model.stationary_variables)
            self.n_observed = len(self.reduced_model.reduced_measurement_equations)
            self.var_order = self.reduced_model.var_prior_setup.var_order if self.reduced_model.var_prior_setup else 1
            self.state_dim = self.builder.state_dim
            
            # Create variable mappings for compatibility
            self.trend_var_map = {var: i for i, var in enumerate(self.reduced_model.core_variables)}
            self.stat_var_map = {var: i for i, var in enumerate(self.reduced_model.stationary_variables)}
            self.obs_var_map = {var: i for i, var in enumerate(self.reduced_model.reduced_measurement_equations.keys())}
            
            # Alias for compatibility with existing code
            self.gmp = self.reduced_model
            
        except Exception as e:
            raise ValueError(
                f"MODEL SETUP FAILED: Could not setup reduced model\n"
                f"GPM file: {self.gmp_file_path}\n"
                f"Error: {e}\n"
                f"Context: Initializing integration coordinator\n"
                f"Solution: Ensure GPM file exists and is valid"
            )
    
    def build_state_space_from_mcmc_sample(self, mcmc_samples: Dict[str, Any], 
                                         sample_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build state space matrices from single MCMC sample.
        
        Args:
            mcmc_samples: Dictionary of MCMC samples
            sample_idx: Index of sample to use
            
        Returns:
            Tuple of (F, Q, C, H) state space matrices
        """
        
        try:
            # Step 1: Transform MCMC sample to standardized parameters
            standardized_params = self.adapter.transform_mcmc_output(mcmc_samples, sample_idx)
            
            # Step 2: Build state space matrices
            return self.builder.build_state_space_matrices(standardized_params)
            
        except Exception as e:
            raise ValueError(
                f"STATE SPACE CONSTRUCTION FAILED: Could not build matrices from MCMC sample {sample_idx}\n"
                f"Error: {e}\n"
                f"Context: Converting MCMC sample to state space matrices\n"
                f"Available MCMC parameters: {sorted(mcmc_samples.keys())}\n"
                f"Solution: Check parameter contract and MCMC output consistency"
            )
    
    def build_state_space_from_enhanced_params(self, params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build state space matrices from EnhancedBVARParams object.
        
        Args:
            params: EnhancedBVARParams object from MCMC
            
        Returns:
            Tuple of (F, Q, C, H) state space matrices
        """
        
        try:
            # Step 1: Transform EnhancedBVARParams to standardized parameters
            standardized_params = self.adapter.transform_enhanced_bvar_params(params)
            
            # Step 2: Build state space matrices
            return self.builder.build_state_space_matrices(standardized_params)
            
        except Exception as e:
            raise ValueError(
                f"STATE SPACE CONSTRUCTION FAILED: Could not build matrices from EnhancedBVARParams\n"
                f"Error: {e}\n"
                f"Context: Converting EnhancedBVARParams to state space matrices\n"
                f"EnhancedBVARParams structure: {type(params)}\n"
                f"Solution: Check parameter contract and EnhancedBVARParams consistency"
            )
    
    def build_state_space_from_dict(self, param_dict: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build state space matrices from parameter dictionary.
        
        Args:
            param_dict: Dictionary of parameters (MCMC format)
            
        Returns:
            Tuple of (F, Q, C, H) state space matrices
        """
        
        try:
            # Step 1: Transform parameter dictionary to standardized format
            standardized_params = self.adapter.transform_parameter_dict(param_dict)
            
            # Step 2: Build state space matrices
            return self.builder.build_state_space_matrices(standardized_params)
            
        except Exception as e:
            raise ValueError(
                f"STATE SPACE CONSTRUCTION FAILED: Could not build matrices from parameter dictionary\n"
                f"Error: {e}\n"
                f"Context: Converting parameter dictionary to state space matrices\n"
                f"Provided parameters: {sorted(param_dict.keys())}\n"
                f"Solution: Check parameter contract and dictionary consistency"
            )
    
    def get_variable_names(self) -> Dict[str, List[str]]:
        """Get variable names for reporting - compatible with existing code"""
        
        return {
            'trend_variables': self.reduced_model.core_variables,
            'stationary_variables': self.reduced_model.stationary_variables,
            'observed_variables': list(self.reduced_model.reduced_measurement_equations.keys()),
            'parameters': self.reduced_model.parameters
        }
    
    def get_model_summary(self) -> str:
        """Get summary of the reduced model"""
        
        summary = [
            "INTEGRATION COORDINATOR MODEL SUMMARY",
            "=" * 50,
            "",
            f"GPM File: {self.gmp_file_path}",
            f"Core Variables (State): {len(self.reduced_model.core_variables)}",
            f"Stationary Variables: {len(self.reduced_model.stationary_variables)}",
            f"Observed Variables: {len(self.reduced_model.reduced_measurement_equations)}",
            f"State Dimension: {self.state_dim}",
            f"VAR Order: {self.var_order}",
            "",
            "Core Variables:",
        ]
        
        for i, var in enumerate(self.reduced_model.core_variables):
            summary.append(f"  {i+1:2d}. {var}")
        
        summary.extend([
            "",
            "Stationary Variables:",
        ])
        
        for i, var in enumerate(self.reduced_model.stationary_variables):
            summary.append(f"  {i+1:2d}. {var}")
        
        summary.extend([
            "",
            "Observed Variables:",
        ])
        
        for i, var in enumerate(self.reduced_model.reduced_measurement_equations.keys()):
            summary.append(f"  {i+1:2d}. {var}")
        
        return "\n".join(summary)
    
    def validate_mcmc_compatibility(self, mcmc_samples: Dict[str, Any]) -> None:
        """
        Validate that MCMC samples are compatible with parameter contract.
        
        Args:
            mcmc_samples: Dictionary of MCMC samples
            
        Raises:
            ValueError: If MCMC samples don't match contract expectations
        """
        
        # Extract parameter names from first sample
        sample_params = {}
        for key, values in mcmc_samples.items():
            if hasattr(values, '__getitem__') and len(values) > 0:
                sample_params[key] = values[0]
            else:
                sample_params[key] = values
        
        # Validate using contract
        self.contract.validate_mcmc_parameters(sample_params)
        
        print("✓ MCMC samples are compatible with parameter contract")
    
    def test_state_space_construction(self, test_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Test state space construction with test parameters.
        
        Args:
            test_params: Optional test parameters (uses defaults if None)
            
        Returns:
            True if construction successful, False otherwise
        """
        
        if test_params is None:
            test_params = self._create_test_parameters()
        
        try:
            F, Q, C, H = self.build_state_space_from_dict(test_params)
            
            # Basic validation
            assert F.shape == (self.state_dim, self.state_dim), f"F matrix shape mismatch"
            assert Q.shape == (self.state_dim, self.state_dim), f"Q matrix shape mismatch"
            assert C.shape == (self.n_observed, self.state_dim), f"C matrix shape mismatch"
            assert H.shape == (self.n_observed, self.n_observed), f"H matrix shape mismatch"
            
            # Check for finite values
            assert jnp.all(jnp.isfinite(F)), "F matrix contains non-finite values"
            assert jnp.all(jnp.isfinite(Q)), "Q matrix contains non-finite values"
            assert jnp.all(jnp.isfinite(C)), "C matrix contains non-finite values"
            assert jnp.all(jnp.isfinite(H)), "H matrix contains non-finite values"
            
            print("✓ State space construction test passed")
            print(f"  Matrix shapes: F{F.shape}, Q{Q.shape}, C{C.shape}, H{H.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ State space construction test failed: {e}")
            return False
    
    def _create_test_parameters(self) -> Dict[str, Any]:
        """Create test parameters for validation"""
        
        test_params = {}
        
        # Add structural parameters
        for param_name in self.reduced_model.parameters:
            if 'phi' in param_name.lower():
                test_params[f"sigma_{param_name}"] = 1.0  # Use sigma_ prefix for consistency
            else:
                test_params[f"sigma_{param_name}"] = 0.5
        
        # Add trend shock standard deviations
        for shock in getattr(self.reduced_model, 'trend_shocks', []):
            test_params[f"sigma_{shock}"] = 0.1
        
        # Add stationary shock standard deviations
        for shock in getattr(self.reduced_model, 'stationary_shocks', []):
            test_params[f"sigma_{shock}"] = 0.2
        
        # Add VAR matrices (minimal)
        if self.n_stationary > 0:
            test_params['_var_coefficients'] = jnp.zeros((self.var_order, self.n_stationary, self.n_stationary))
            test_params['_var_innovation_cov'] = jnp.eye(self.n_stationary) * 0.1
        
        return test_params


# Factory functions for easy use
def create_integration_coordinator(gmp_file_path: str) -> IntegrationCoordinator:
    """Create integration coordinator for GPM file"""
    return IntegrationCoordinator(gmp_file_path)


def build_state_space_from_mcmc(gmp_file_path: str, mcmc_samples: Dict[str, Any], 
                               sample_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience function to build state space from MCMC sample"""
    coordinator = create_integration_coordinator(gmp_file_path)
    return coordinator.build_state_space_from_mcmc_sample(mcmc_samples, sample_idx)


def build_state_space_from_enhanced_params(gmp_file_path: str, params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience function to build state space from EnhancedBVARParams"""
    coordinator = create_integration_coordinator(gmp_file_path)
    return coordinator.build_state_space_from_enhanced_params(params)


# Compatibility class for existing code
class ReducedGPMIntegration:
    """
    Compatibility wrapper that provides the same interface as the old integration helper.
    
    This allows existing code to work without changes while using the new coordinator internally.
    """
    
    def __init__(self, gmp_file_path: str):
        self.coordinator = create_integration_coordinator(gmp_file_path)
        
        # Expose coordinator attributes for compatibility
        self.reduced_model = self.coordinator.reduced_model
        self.gmp = self.coordinator.gmp
        self.n_trends = self.coordinator.n_trends
        self.n_stationary = self.coordinator.n_stationary
        self.n_observed = self.coordinator.n_observed
        self.var_order = self.coordinator.var_order
        self.state_dim = self.coordinator.state_dim
        self.trend_var_map = self.coordinator.trend_var_map
        self.stat_var_map = self.coordinator.stat_var_map
        self.obs_var_map = self.coordinator.obs_var_map
    
    def build_state_space_matrices(self, params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build state space matrices - compatible interface with existing code.
        
        Args:
            params: Can be EnhancedBVARParams object or dict of parameters
            
        Returns:
            F, Q, C, H matrices for Kalman filter
        """
        
        # Determine parameter type and delegate to coordinator
        if hasattr(params, 'structural_params'):
            # EnhancedBVARParams object
            return self.coordinator.build_state_space_from_enhanced_params(params)
        else:
            # Dictionary
            return self.coordinator.build_state_space_from_dict(params)
    
    def get_variable_names(self) -> Dict[str, List[str]]:
        """Get variable names for reporting - compatible with existing code"""
        return self.coordinator.get_variable_names()


def create_reduced_gmp_model(gmp_file_path: str):
    """
    Factory function that provides compatibility with existing workflow.
    
    Returns the same interface as the old integration helper but uses
    the new coordinator internally.
    """
    
    coordinator = create_integration_coordinator(gmp_file_path)
    integration = ReducedGPMIntegration(gmp_file_path)
    
    # Return same structure as before
    return integration, coordinator.reduced_model, coordinator.builder


if __name__ == "__main__":
    # Test the coordinator
    print("Testing Integration Coordinator...")
    
    try:
        # Create a simple test GPM file
        test_gmp_content = """
parameters var_phi;

estimated_params;
    stderr shk_trend_r_world, inv_gamma_pdf, 2.1, 0.81;
    stderr shk_cycle_y_us, inv_gamma_pdf, 2.1, 0.38;
    var_phi, normal_pdf, 1.0, 0.2;
end;

trends_vars trend_r_world;
stationary_variables cycle_y_us;

trend_shocks;
    shk_trend_r_world
end;

shocks;
    shk_cycle_y_us
end;

trend_model;
    trend_r_world = trend_r_world(-1) + shk_trend_r_world;
end;

varobs y_us;

measurement_equations;
    y_us = trend_r_world + cycle_y_us;
end;

var_prior_setup;
    var_order = 1;
    es = 0.5, 0.3;
    fs = 0.5, 0.5;  
    gs = 2.0, 2.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;
"""
        
        # Save test file
        with open('test_coordinator.gmp', 'w') as f:
            f.write(test_gmp_content)
        
        # Test coordinator
        coordinator = create_integration_coordinator('test_coordinator.gmp')
        
        # Test parameter creation and validation
        success = coordinator.test_state_space_construction()
        
        if success:
            print("✓ Integration Coordinator test passed")
            print(coordinator.get_model_summary())
        else:
            print("✗ Integration Coordinator test failed")
            
    except Exception as e:
        print(f"✗ Integration Coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test file
        import os
        if os.path.exists('test_coordinator.gmp'):
            os.remove('test_coordinator.gmp')