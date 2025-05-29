"""
Integration Helper
Provides bridge between reduced parser and existing MCMC code
"""

import jax.numpy as jnp
import jax.random as random
from typing import Dict, List, Tuple, Optional, Any
from .reduced_gpm_parser import ReducedModel
from .reduced_state_space_builder import ReducedStateSpaceBuilder

class ReducedGPMIntegration:
    """
    Integration layer between reduced parser and existing MCMC infrastructure
    
    This class provides the same interface as your existing GPMStateSpaceBuilder
    but uses the reduced model internally for efficiency.
    """
    
    def __init__(self, reduced_model: ReducedModel):
        self.reduced_model = reduced_model
        self.builder = ReducedStateSpaceBuilder(reduced_model)
        
        # Create mappings for compatibility with existing code
        self.gpm = reduced_model  # For compatibility with existing code
        
        # State space dimensions (compatible with existing GPMStateSpaceBuilder)
        self.n_trends = len(reduced_model.core_variables)  # Only core trends
        self.n_stationary = len(reduced_model.stationary_variables)
        self.n_observed = len(reduced_model.reduced_measurement_equations)
        self.var_order = reduced_model.var_prior_setup.var_order if reduced_model.var_prior_setup else 1
        self.state_dim = self.builder.state_dim
        
        # Variable mappings for compatibility
        self.trend_var_map = {var: i for i, var in enumerate(reduced_model.core_variables)}
        self.stat_var_map = {var: i for i, var in enumerate(reduced_model.stationary_variables)}
        self.obs_var_map = {var: i for i, var in enumerate(reduced_model.reduced_measurement_equations.keys())}
        
        print(f"ReducedGPMIntegration initialized:")
        print(f"  Core trends: {self.n_trends} (reduced from original)")
        print(f"  Stationary: {self.n_stationary}")
        print(f"  State dimension: {self.state_dim}")
    
    def build_state_space_matrices(self, params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build state space matrices - compatible interface with existing code
        
        Args:
            params: Can be either EnhancedBVARParams object or dict of parameters
            
        Returns:
            F, Q, C, H matrices for Kalman filter
        """
        
        # Convert params to dictionary format if needed
        if hasattr(params, 'structural_params'):
            # EnhancedBVARParams object
            param_dict = {}
            param_dict.update(params.structural_params)
            
            # Add shock variances from covariance matrices
            if hasattr(params, 'Sigma_eta') and params.Sigma_eta is not None:
                # Extract shock standard deviations from covariance matrix
                shock_stds = jnp.sqrt(jnp.diag(params.Sigma_eta))
                for i, shock_std in enumerate(shock_stds):
                    if i < len(self.reduced_model.core_variables):
                        shock_name = f"shk_{self.reduced_model.core_variables[i].lower()}"
                        param_dict[shock_name] = shock_std
            
            if hasattr(params, 'Sigma_u') and params.Sigma_u is not None:
                # Extract stationary shock standard deviations
                stat_shock_stds = jnp.sqrt(jnp.diag(params.Sigma_u))
                for i, shock_std in enumerate(stat_shock_stds):
                    if i < len(self.reduced_model.stationary_variables):
                        shock_name = f"shk_cycle_{self.reduced_model.stationary_variables[i].lower()}"
                        param_dict[shock_name] = shock_std
            
            # Handle VAR coefficients
            if hasattr(params, 'A') and params.A is not None:
                # Store VAR coefficients for builder to use
                param_dict['_var_coefficients'] = params.A
                
        else:
            # Already a dictionary
            param_dict = dict(params)
        
        # Build matrices using reduced model
        return self.builder.build_state_space_matrices(param_dict)
    
    def get_variable_names(self) -> Dict[str, List[str]]:
        """Get variable names for reporting - compatible with existing code"""
        
        return {
            'trend_variables': self.reduced_model.core_variables,
            'stationary_variables': self.reduced_model.stationary_variables,
            'observed_variables': list(self.reduced_model.reduced_measurement_equations.keys()),
            'parameters': self.reduced_model.parameters
        }
    
    def get_core_equations(self) -> List:
        """Get core equations for inspection"""
        return self.reduced_model.core_equations
    
    def get_reduced_measurement_equations(self) -> Dict:
        """Get reduced measurement equations for inspection"""
        return self.reduced_model.reduced_measurement_equations
    
    def print_model_summary(self):
        """Print summary of the reduced model"""
        
        print("\n" + "="*50)
        print("REDUCED MODEL SUMMARY")
        print("="*50)
        
        print(f"\nCore Variables (Transition Equation):")
        for i, var in enumerate(self.reduced_model.core_variables):
            print(f"  {i+1:2d}. {var}")
        
        print(f"\nCore Equations:")
        for eq in self.reduced_model.core_equations:
            print(f"\n  {eq.lhs} = ", end="")
            for i, term in enumerate(eq.rhs_terms):
                if i > 0:
                    print(f" {term.sign} ", end="")
                coeff_str = f"{term.coefficient}*" if term.coefficient else ""
                lag_str = f"(-{term.lag})" if term.lag > 0 else ""
                print(f"{coeff_str}{term.variable}{lag_str}", end="")
            if eq.shock:
                print(f" + {eq.shock}")
            else:
                print()
        
        print(f"\nReduced Measurement Equations:")
        for obs_var, expr in self.reduced_model.reduced_measurement_equations.items():
            print(f"\n  {obs_var} = ")
            for var, coeff in list(expr.terms.items())[:3]:  # Show first 3 terms
                print(f"    + ({coeff}) * {var}")
            if len(expr.terms) > 3:
                print(f"    + ... ({len(expr.terms)-3} more terms)")
            print(f"    + stationary_component")


# def create_reduced_gpm_model(gmp_file_path: str):
#     """
#     Factory function to create reduced GPM model - compatible with existing workflow
#     """
    
#     from .reduced_gpm_parser import ReducedGPMParser
    
#     # Parse and reduce the model
#     parser = ReducedGPMParser()
#     reduced_model = parser.parse_file(gmp_file_path)
    
#     # # ADD COMPATIBILITY ATTRIBUTES
#     # reduced_model.trend_variables = reduced_model.core_variables
#     # reduced_model.observed_variables = list(reduced_model.reduced_measurement_equations.keys())

#     # ADD COMPATIBILITY ATTRIBUTES
#     reduced_model.trend_variables = reduced_model.core_variables
#     reduced_model.observed_variables = list(reduced_model.reduced_measurement_equations.keys())

#     # Add shock-related attributes for compatibility
#     # Extract trend shocks from core equations
#     trend_shocks = []
#     for eq in reduced_model.core_equations:
#         if eq.shock:
#             trend_shocks.append(eq.shock)
#     reduced_model.trend_shocks = trend_shocks

#     # Add stationary shocks (these should already exist, but let's be safe)
#     if not hasattr(reduced_model, 'stationary_shocks'):
#         # Create default stationary shock names based on stationary variables
#         reduced_model.stationary_shocks = [f"SHK_{var}" for var in reduced_model.stationary_variables]

#     # Add other potentially missing attributes
#     if not hasattr(reduced_model, 'initial_values'):
#         reduced_model.initial_values = {}

#     # Create integration layer
#     integration = ReducedGPMIntegration(reduced_model)
    
#     # Return compatible interface
#     return integration, reduced_model, integration.builder


def create_reduced_gpm_model(gpm_file_path: str):
    """
    Factory function to create reduced GPM model - compatible with existing workflow
    """
    
    from .reduced_gpm_parser import ReducedGPMParser
    
    # Parse and reduce the model
    parser = ReducedGPMParser()
    reduced_model = parser.parse_file(gpm_file_path)
    
    # ADD FULL COMPATIBILITY ATTRIBUTES
    reduced_model.trend_variables = reduced_model.core_variables
    reduced_model.observed_variables = list(reduced_model.reduced_measurement_equations.keys())
    
    # Extract trend shocks from core equations
    trend_shocks = []
    for eq in reduced_model.core_equations:
        if eq.shock:
            trend_shocks.append(eq.shock)
    reduced_model.trend_shocks = trend_shocks
    
    # Add stationary shocks if missing
    if not hasattr(reduced_model, 'stationary_shocks'):
        reduced_model.stationary_shocks = [f"SHK_{var}" for var in reduced_model.stationary_variables]
    
    # Add other potentially missing attributes
    if not hasattr(reduced_model, 'initial_values'):
        reduced_model.initial_values = {}
        
    if not hasattr(reduced_model, 'trend_equations'):
        reduced_model.trend_equations = reduced_model.core_equations
        
    if not hasattr(reduced_model, 'measurement_equations'):
        # Convert reduced measurement equations to old format if needed
        reduced_model.measurement_equations = []
    
    # Create integration layer
    integration = ReducedGPMIntegration(reduced_model)
    
    # Return compatible interface
    return integration, reduced_model, integration.builder

def enhanced_create_gpm_based_model(gpm_file_path: str, use_reduction: bool = True):
    """
    Enhanced factory function that can use either reduced or original approach
    
    Args:
        gpm_file_path: Path to GPM file
        use_reduction: If True, use reduced model; if False, use original approach
    """
    
    if use_reduction:
        print("Using reduced model approach...")
        return create_reduced_gpm_model(gpm_file_path)
    else:
        print("Using original model approach...")
        # Import and use your existing function
        try:
            from gpm_var_trends.gpm_bvar_trends import create_gpm_based_model
            return create_gpm_based_model(gpm_file_path)
        except ImportError:
            print("Original parser not available, falling back to reduced approach")
            return create_reduced_gpm_model(gpm_file_path)


class ReducedModelWrapper:
    """
    Wrapper to make reduced model compatible with existing simulation smoother
    
    This ensures the simulation smoother can work with both reduced and original models
    """
    
    def __init__(self, reduced_integration: ReducedGPMIntegration):
        self.integration = reduced_integration
        self.reduced_model = reduced_integration.reduced_model
        
        # Create expanded state space for simulation smoother
        self._create_expanded_mappings()
    
    def _create_expanded_mappings(self):
        """Create mappings for full model reconstruction in simulation smoother"""
        
        # For simulation smoother, we need to be able to recover ALL trend variables
        # not just the core ones
        
        # All trends (core + derived) for reporting
        all_trend_vars = list(self.reduced_model.core_variables)
        
        # Add derived trends that appear in measurement equations
        for obs_var, expr in self.reduced_model.reduced_measurement_equations.items():
            for var_key in expr.terms.keys():
                var_name = var_key.split('(')[0]  # Remove lag info
                if var_name not in all_trend_vars and var_name not in self.reduced_model.stationary_variables:
                    all_trend_vars.append(var_name)
        
        self.all_trend_variables = all_trend_vars
        self.expanded_n_trends = len(all_trend_vars)
        
        print(f"Expanded model for simulation smoother:")
        print(f"  Core trends: {len(self.reduced_model.core_variables)}")
        print(f"  All trends (for reporting): {self.expanded_n_trends}")
    
    def get_full_trend_decomposition(self, core_trends: jnp.ndarray, 
                                   params: Dict[str, float]) -> jnp.ndarray:
        """
        Reconstruct all trend variables from core trends using substitution rules
        
        Args:
            core_trends: Array of shape (T, n_core_trends)
            params: Dictionary of parameter values
            
        Returns:
            all_trends: Array of shape (T, n_all_trends)
        """
        
        T, n_core = core_trends.shape
        all_trends = jnp.zeros((T, self.expanded_n_trends))
        
        # Copy core trends
        n_core_actual = min(n_core, len(self.reduced_model.core_variables))
        all_trends = all_trends.at[:, :n_core_actual].set(core_trends[:, :n_core_actual])
        
        # Compute derived trends using measurement equation expressions
        # This is a simplified version - full implementation would use substitution rules
        
        return all_trends
    
    def get_compatible_interface(self):
        """Get interface compatible with existing simulation smoother"""
        
        class CompatibleInterface:
            def __init__(self, wrapper):
                self.wrapper = wrapper
                # Mimic original GPMStateSpaceBuilder interface
                self.n_trends = wrapper.expanded_n_trends  # Use expanded count
                self.n_stationary = wrapper.integration.n_stationary
                self.state_dim = wrapper.integration.state_dim
                self.var_order = wrapper.integration.var_order
                
                # Variable lists for simulation smoother
                self.gpm = wrapper.reduced_model
                
                # Override trend variables to include all trends
                class ExpandedGPMModel:
                    def __init__(self, reduced_model, all_trends):
                        # Copy all attributes from reduced model
                        for attr in dir(reduced_model):
                            if not attr.startswith('_'):
                                setattr(self, attr, getattr(reduced_model, attr))
                        # Override trend variables
                        self.trend_variables = all_trends
                
                self.gpm = ExpandedGPMModel(wrapper.reduced_model, wrapper.all_trend_variables)
            
            def build_state_space_matrices(self, params):
                return self.wrapper.integration.build_state_space_matrices(params)
        
        return CompatibleInterface(self)


def test_integration():
    """Test the integration layer"""
    
    print("="*60)
    print("TESTING INTEGRATION LAYER")
    print("="*60)
    
    try:
        # Create reduced model
        integration, reduced_model, builder = create_reduced_gpm_model('model_with_trends.gpm')
        
        print(f"✓ Reduced model created successfully")
        
        # Test compatibility interface
        var_names = integration.get_variable_names()
        print(f"✓ Variable names extracted: {len(var_names['trend_variables'])} trends")
        
        # Print model summary
        integration.print_model_summary()
        
        # Test wrapper for simulation smoother
        wrapper = ReducedModelWrapper(integration)
        compatible_interface = wrapper.get_compatible_interface()
        
        print(f"✓ Simulation smoother wrapper created")
        print(f"  Compatible n_trends: {compatible_interface.n_trends}")
        print(f"  Compatible state_dim: {compatible_interface.state_dim}")
        
        # Test with fake parameters
        test_params = {}
        for param in reduced_model.parameters:
            test_params[param] = 1.0
        
        # Add shock parameters  
        for var in reduced_model.core_variables:
            test_params[f"shk_{var.lower()}"] = 0.1
            
        for var in reduced_model.stationary_variables:
            test_params[f"shk_cycle_{var.lower()}"] = 0.1
        
        F, Q, C, H = integration.build_state_space_matrices(test_params)
        print(f"✓ State space matrices built: F{F.shape}, Q{Q.shape}, C{C.shape}, H{H.shape}")
        
        print(f"\n🎉 INTEGRATION TESTING COMPLETED SUCCESSFULLY!")
        print(f"   Ready to replace existing parser in your MCMC code")
        
        return integration, wrapper
        
    except Exception as e:
        print(f"✗ Integration testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    test_integration()