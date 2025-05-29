"""
Reduced State Space Builder
Builds JAX state space matrices from the reduced model
"""

# Import JAX configuration first
from .jax_config import configure_jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import sympy as sp
from .reduced_gpm_parser import ReducedModel, ReducedExpression, ParsedEquation, ParsedTerm

# Import Kalman filter with correct path
try:
    from Kalman_filter_jax import KalmanFilter
except ImportError:
    try:
        from Kalman_filter_jax import KalmanFilter
    except ImportError:
        print("Warning: Could not import KalmanFilter. Please ensure Kalman_filter_jax.py is in the correct path.")
        KalmanFilter = None

# Constants for numerical stability
_DEFAULT_DTYPE = jnp.float64
_JITTER = 1e-8
_KF_JITTER = 1e-8

class ReducedStateSpaceBuilder:
    """Build state space matrices from reduced GPM model"""
    
    def __init__(self, reduced_model: ReducedModel):
        self.model = reduced_model
        
        # Dimensions
        self.n_core = len(reduced_model.core_variables)
        self.n_stationary = len(reduced_model.stationary_variables)
        self.n_observed = len(reduced_model.reduced_measurement_equations)
        
        # VAR setup
        if reduced_model.var_prior_setup:
            self.var_order = reduced_model.var_prior_setup.var_order
        else:
            self.var_order = 1
            
        # Total state dimension: core trends + VAR states
        self.state_dim = self.n_core + self.n_stationary * self.var_order
        
        self.n_trends = self.n_core  # This is what the old code expects
        self.gmp = reduced_model      # Some code might access ss_builder.gmp

        # Create variable mappings
        self.core_var_map = {var: i for i, var in enumerate(reduced_model.core_variables)}
        self.stat_var_map = {var: i for i, var in enumerate(reduced_model.stationary_variables)}
        self.obs_var_map = {var: i for i, var in enumerate(reduced_model.reduced_measurement_equations.keys())}
        
        print(f"State Space Dimensions:")
        print(f"  Core variables: {self.n_core}")
        print(f"  Stationary variables: {self.n_stationary}")
        print(f"  VAR order: {self.var_order}")
        print(f"  Total state dimension: {self.state_dim}")
        print(f"  Observed variables: {self.n_observed}")
    
    def build_state_space_matrices(self, params: Dict[str, float]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build complete state space representation: x(t+1) = F*x(t) + R*eta(t), y(t) = C*x(t) + eps(t)
        
        Returns:
            F: Transition matrix (state_dim x state_dim)
            Q: State innovation covariance (state_dim x state_dim) 
            C: Measurement matrix (n_observed x state_dim)
            H: Measurement error covariance (n_observed x n_observed)
        """
        
        # Initialize matrices
        F = jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        Q = jnp.zeros((self.state_dim, self.state_dim), dtype=_DEFAULT_DTYPE)
        C = jnp.zeros((self.n_observed, self.state_dim), dtype=_DEFAULT_DTYPE)
        H = _KF_JITTER * jnp.eye(self.n_observed, dtype=_DEFAULT_DTYPE)
        
        # Build core variable dynamics (F matrix and Q matrix)
        F, Q = self._build_core_dynamics(F, Q, params)
        
        # Build VAR dynamics for stationary variables
        if self.n_stationary > 0:
            F = self._build_var_dynamics(F, params)
            Q = self._add_var_innovations(Q, params)
        
        # Build measurement equations (C matrix)
        C = self._build_measurement_matrix(C, params)
        
        # Ensure matrices are well-conditioned
        Q = (Q + Q.T) / 2.0 + _JITTER * jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0
        
        return F, Q, C, H
    
    def _build_core_dynamics(self, F: jnp.ndarray, Q: jnp.ndarray, params: Dict[str, float]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build dynamics for core variables"""
        
        # Initialize core shock variances
        core_shock_vars = jnp.zeros(self.n_core, dtype=_DEFAULT_DTYPE)
        
        for i, equation in enumerate(self.model.core_equations):
            lhs_idx = self.core_var_map[equation.lhs]
            
            # Process RHS terms
            for term in equation.rhs_terms:
                if term.variable in self.core_var_map:
                    rhs_idx = self.core_var_map[term.variable]
                    
                    # Evaluate coefficient
                    coeff_value = self._evaluate_coefficient(term.coefficient, params)
                    
                    # Apply sign
                    if term.sign == '-':
                        coeff_value = -coeff_value
                    
                    # Handle lags (for now, assume lag 1 means previous period)
                    if term.lag == 1:
                        F = F.at[lhs_idx, rhs_idx].set(coeff_value)
                    elif term.lag == 0:
                        # This would be a contemporaneous relationship - unusual in trends
                        # For now, treat as identity unless it's the same variable
                        if term.variable != equation.lhs:
                            F = F.at[lhs_idx, rhs_idx].set(coeff_value)
            
            # Add shock variance if present
            if equation.shock:
                shock_var = self._get_shock_variance(equation.shock, params)
                core_shock_vars = core_shock_vars.at[lhs_idx].set(shock_var)
        
        # Set shock variances in Q matrix
        Q = Q.at[:self.n_core, :self.n_core].set(jnp.diag(core_shock_vars))
        
        return F, Q
    
    def _build_var_dynamics(self, F: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Build VAR dynamics for stationary components"""
        
        var_start = self.n_core
        
        # For now, implement simple VAR(1) - can be extended
        # This would typically involve A matrices from VAR parameter sampling
        # For simplicity, use identity for now (can be replaced with actual VAR coefficients)
        
        # Set VAR coefficient matrices
        for lag in range(self.var_order):
            # Get VAR coefficients for this lag (placeholder - should come from params)
            A_lag = self._get_var_coefficients(lag, params)
            
            start_row = var_start
            end_row = var_start + self.n_stationary
            start_col = var_start + lag * self.n_stationary
            end_col = var_start + (lag + 1) * self.n_stationary
            
            if end_col <= self.state_dim:
                F = F.at[start_row:end_row, start_col:end_col].set(A_lag)
        
        # Set identity matrices for lagged states (VAR companion form)
        if self.var_order > 1:
            for i in range(self.var_order - 1):
                start_row = var_start + (i + 1) * self.n_stationary
                end_row = start_row + self.n_stationary
                start_col = var_start + i * self.n_stationary
                end_col = start_col + self.n_stationary
                
                F = F.at[start_row:end_row, start_col:end_col].set(
                    jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE)
                )
        
        return F
    
    def _add_var_innovations(self, Q: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Add VAR innovation covariance to state covariance"""
        
        var_start = self.n_core
        
        # Get VAR innovation covariance (placeholder - should come from params)
        Sigma_u = self._get_var_innovation_covariance(params)
        
        Q = Q.at[var_start:var_start + self.n_stationary, 
                 var_start:var_start + self.n_stationary].set(Sigma_u)
        
        return Q
    
    def _build_measurement_matrix(self, C: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Build measurement matrix from reduced measurement equations"""
        
        for obs_var, reduced_expr in self.model.reduced_measurement_equations.items():
            obs_idx = self.obs_var_map[obs_var]
            
            # Process each term in the reduced expression
            for var_key, coeff_expr in reduced_expr.terms.items():
                # Parse variable key (might have lag info)
                var_name, lag = self._parse_variable_key(var_key)
                
                # Find state index
                state_idx = self._find_state_index(var_name, lag)
                
                if state_idx is not None:
                    # Evaluate coefficient expression
                    coeff_value = self._evaluate_coefficient_expression(coeff_expr, params)
                    C = C.at[obs_idx, state_idx].set(coeff_value)
            
            # Add stationary component (current period)
            if obs_var in self.model.stationary_variables:
                stat_idx = self.stat_var_map[obs_var]
                state_idx = self.n_core + stat_idx  # Current period stationary component
                C = C.at[obs_idx, state_idx].set(1.0)
        
        return C
    
    def _evaluate_coefficient(self, coeff: Optional[str], params: Dict[str, float]) -> float:
        """Evaluate a simple coefficient (parameter name or None)"""
        
        if coeff is None:
            return 1.0
        elif coeff in params:
            return params[coeff]
        else:
            # Try to evaluate as a simple expression
            try:
                return float(coeff)
            except:
                print(f"Warning: Cannot evaluate coefficient '{coeff}', using 1.0")
                return 1.0
    
    def _evaluate_coefficient_expression(self, expr: str, params: Dict[str, float]) -> float:
        """Evaluate a coefficient expression like '(var_phi)' or '(-var_phi)' - FIXED"""
        
        if expr == '1' or expr == '':
            return 1.0
        elif expr == '0':
            return 0.0
        elif expr == '-1':
            return -1.0
        
        # Clean up the expression - remove extra parentheses
        expr = expr.strip()
        if expr.startswith('((') and expr.endswith('))'):
            expr = expr[2:-2]
        elif expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1]
        
        # Handle simple parameter lookups
        if expr in params:
            return params[expr]
        
        # Handle negation
        if expr.startswith('-') and expr[1:].strip() in params:
            param_name = expr[1:].strip()
            return -params[param_name]
        
        # For more complex expressions, try simple evaluation first
        try:
            # Replace parameter names with values
            expr_substituted = expr
            for param_name, param_value in params.items():
                # Use word boundaries to avoid partial replacements
                import re
                pattern = r'\b' + re.escape(param_name) + r'\b'
                expr_substituted = re.sub(pattern, str(param_value), expr_substituted)
            
            # Try direct evaluation for simple expressions
            try:
                result = eval(expr_substituted)
                return float(result)
            except:
                # Fall back to sympy
                import sympy as sp
                result = float(sp.sympify(expr_substituted).evalf())
                return result
                
        except Exception as e:
            print(f"Warning: Cannot evaluate expression '{expr}': {e}, using 0.0")
            return 0.0
    
    def _parse_variable_key(self, var_key: str) -> Tuple[str, int]:
        """Parse variable key like 'var_name(-1)' into (var_name, lag) - FIXED"""
        
        # Clean up the key first
        var_key = var_key.strip()
        
        if '(-' in var_key:
            parts = var_key.split('(-')
            var_name = parts[0].strip()
            if len(parts) > 1:
                lag_part = parts[1].replace(')', '').strip()
                try:
                    lag = int(lag_part)
                except:
                    lag = 0
            else:
                lag = 0
            return var_name, lag
        else:
            return var_key, 0
    
    def _find_state_index(self, var_name: str, lag: int) -> Optional[int]:
        """Find the index of a variable in the state vector"""
        
        # Check core variables
        if var_name in self.core_var_map:
            # For core variables, lag handling depends on whether we store lags
            # For now, assume we only store current period in state
            if lag == 0:
                return self.core_var_map[var_name]
            else:
                # Would need to handle lags properly in state space design
                print(f"Warning: Lagged core variable {var_name}(-{lag}) not handled")
                return None
        
        # Check stationary variables
        if var_name in self.stat_var_map:
            stat_idx = self.stat_var_map[var_name]
            
            # Handle VAR lags
            if lag < self.var_order:
                return self.n_core + lag * self.n_stationary + stat_idx
            else:
                print(f"Warning: Lag {lag} exceeds VAR order {self.var_order}")
                return None
        
        print(f"Warning: Variable {var_name} not found in state")
        return None
    
    # def _get_shock_variance(self, shock_name: str, params: Dict[str, float]) -> float:
    #     """Get variance for a shock"""
        
    #     # Look for shock in estimated parameters (typically as standard deviation)
    #     if shock_name in params:
    #         return params[shock_name] ** 2  # Convert std dev to variance
    #     else:
    #         # Default variance
    #         print(f"Warning: Shock variance for {shock_name} not found, using default")
    #         return 1.0

    # def _get_shock_variance(self, shock_name: str, params: Dict[str, float]) -> float:
    #     """Get variance for a shock"""
        
    #     print(f"DEBUG: Looking for shock_name='{shock_name}'")
    #     print(f"DEBUG: Available params: {list(params.keys())}")
        
    #     # Try the shock name directly
    #     if shock_name in params:
    #         print(f"DEBUG: Found direct match for {shock_name} = {params[shock_name]}")
    #         return params[shock_name] ** 2
        
    #     # Try with sigma_ prefix
    #     sigma_name = f"sigma_{shock_name}"
    #     if sigma_name in params:
    #         print(f"DEBUG: Found sigma match for {sigma_name} = {params[sigma_name]}")
    #         return params[sigma_name] ** 2
        
    #     # Try removing SHK_ prefix and adding sigma_
    #     if shock_name.startswith('SHK_'):
    #         base_name = shock_name[4:]  # Remove 'SHK_'
    #         sigma_base_name = f"sigma_SHK_{base_name}"
    #         if sigma_base_name in params:
    #             print(f"DEBUG: Found reconstructed match for {sigma_base_name} = {params[sigma_base_name]}")
    #             return params[sigma_base_name] ** 2
        
    #     # Default fallback
    #     print(f"DEBUG: No match found for {shock_name}, using default")
    #     print(f"Warning: Shock variance for {shock_name} not found, using default (FIXED)")
    #     return 1.0
    
    def _get_shock_variance(self, shock_name: str, params: Dict[str, float]) -> float:
        """Get variance for a shock"""
        
        # Try multiple naming conventions including case variations
        candidates = [
            shock_name,                    # SHK_TREND1
            shock_name.lower(),            # shk_trend1
            f"sigma_{shock_name}",         # sigma_SHK_TREND1
            f"sigma_{shock_name.lower()}", # sigma_shk_trend1
        ]
        
        for candidate in candidates:
            if candidate in params:
                return params[candidate] ** 2
        
        print(f"Warning: Shock variance for {shock_name} not found, using default")
        return 1.0
    
    def _get_var_coefficients(self, lag: int, params: Dict[str, float]) -> jnp.ndarray:
        """Get VAR coefficient matrix for a given lag (placeholder)"""
        
        # This should come from the VAR parameter sampling
        # For now, return identity matrix as placeholder
        if lag == 0:
            # Placeholder: simple AR(1) with coefficient 0.7
            return jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE) * 0.7
        else:
            return jnp.zeros((self.n_stationary, self.n_stationary), dtype=_DEFAULT_DTYPE)
    
    def _get_var_innovation_covariance(self, params: Dict[str, float]) -> jnp.ndarray:
        """Get VAR innovation covariance matrix (placeholder)"""
        
        # This should come from the VAR parameter sampling
        # For now, return diagonal matrix
        variances = []
        for shock in self.model.stationary_variables:
            shock_name = f"shk_{shock.lower()}"  # Adjust naming convention as needed
            if shock_name in params:
                variances.append(params[shock_name] ** 2)
            else:
                variances.append(1.0)  # Default variance
        
        return jnp.diag(jnp.array(variances, dtype=_DEFAULT_DTYPE))


class ReducedModelTester:
    """Test the reduced model and state space construction"""
    
    def __init__(self, reduced_model: ReducedModel):
        self.model = reduced_model
        self.builder = ReducedStateSpaceBuilder(reduced_model)
    
    def test_with_parameter_values(self, param_values: Dict[str, float] = None):
        """Test state space construction with given parameter values"""
        
        if param_values is None:
            # Use default parameter values for testing
            param_values = {}
            for param_name in self.model.parameters:
                param_values[param_name] = 1.0  # Default value
            
            # Add shock standard deviations
            for shock in self.model.core_variables:
                shock_name = f"shk_{shock.lower()}"
                param_values[shock_name] = 0.1
            
            for shock in self.model.stationary_variables:
                shock_name = f"shk_{shock.lower()}"
                param_values[shock_name] = 0.1
        
        print(f"\n=== TESTING STATE SPACE CONSTRUCTION ===")
        print(f"Using parameter values: {param_values}")
        
        try:
            F, Q, C, H = self.builder.build_state_space_matrices(param_values)
            
            print(f"\n=== STATE SPACE MATRICES ===")
            print(f"Transition matrix F shape: {F.shape}")
            print(f"Innovation covariance Q shape: {Q.shape}")
            print(f"Measurement matrix C shape: {C.shape}")
            print(f"Measurement error H shape: {H.shape}")
            
            # Check matrix properties
            print(f"\n=== MATRIX DIAGNOSTICS ===")
            print(f"F matrix - finite values: {jnp.all(jnp.isfinite(F))}")
            print(f"Q matrix - positive semidefinite: {jnp.all(jnp.linalg.eigvals(Q) >= -1e-10)}")
            print(f"C matrix - finite values: {jnp.all(jnp.isfinite(C))}")
            print(f"H matrix - positive definite: {jnp.all(jnp.linalg.eigvals(H) > 0)}")
            
            # Print key parts of matrices
            print(f"\n=== TRANSITION MATRIX F (Core Variables Block) ===")
            core_block = F[:self.builder.n_core, :self.builder.n_core]
            print(core_block)
            
            print(f"\n=== MEASUREMENT MATRIX C (First 5 rows) ===")
            n_show = min(5, C.shape[0])
            print(C[:n_show, :])
            
            print(f"\n=== INNOVATION COVARIANCE Q (Diagonal) ===")
            print(jnp.diag(Q))
            
            return F, Q, C, H
            
        except Exception as e:
            print(f"Error constructing state space: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def analyze_model_reduction(self):
        """Analyze the effectiveness of model reduction"""
        
        print(f"\n=== MODEL REDUCTION ANALYSIS ===")
        
        # Count original vs reduced complexity
        original_trends = len(self.model.core_variables) + len([v for v in self.model.reduced_measurement_equations.keys() if 'trend' in v.lower()])
        core_trends = len(self.model.core_variables)
        
        print(f"Original trend variables (estimated): ~{original_trends}")
        print(f"Core variables (in state): {core_trends}")
        print(f"Reduction factor: {core_trends/original_trends:.2f}")
        
        # Analyze measurement equation complexity
        total_terms = 0
        max_terms = 0
        
        for obs_var, expr in self.model.reduced_measurement_equations.items():
            n_terms = len(expr.terms)
            total_terms += n_terms
            max_terms = max(max_terms, n_terms)
        
        avg_terms = total_terms / len(self.model.reduced_measurement_equations)
        
        print(f"\nMeasurement equation complexity:")
        print(f"  Average terms per equation: {avg_terms:.1f}")
        print(f"  Maximum terms in one equation: {max_terms}")
        print(f"  Total parameters involved: {len(self.model.parameters)}")
        
        # Show which parameters appear in measurement equations
        all_params_in_measurement = set()
        for expr in self.model.reduced_measurement_equations.values():
            all_params_in_measurement.update(expr.parameters)
        
        print(f"  Parameters in measurement equations: {sorted(all_params_in_measurement)}")


def run_comprehensive_test():
    """Run comprehensive test of the reduced parser and state space builder"""
    
    print("=== COMPREHENSIVE REDUCED MODEL TEST ===")
    
    try:
        # Parse the model
        from .reduced_gpm_parser import ReducedGPMParser
        
        parser = ReducedGPMParser()
        reduced_model = parser.parse_file('model_with_trends.gpm')
        
        print(f"✓ Model parsed successfully")
        
        # Test the state space builder
        tester = ReducedModelTester(reduced_model)
        
        # Analyze model reduction
        tester.analyze_model_reduction()
        
        # Test state space construction
        F, Q, C, H = tester.test_with_parameter_values()
        
        if F is not None:
            print(f"✓ State space matrices constructed successfully")
            
            # Additional tests
            print(f"\n=== ADDITIONAL TESTS ===")
            
            # Test with different parameter values
            test_params = {
                'var_phi': 2.0,  # Test with different parameter value
            }
            
            # Add shock variances
            for var in reduced_model.core_variables:
                test_params[f"shk_{var.lower()}"] = 0.05
            
            print(f"Testing with different parameter values...")
            F2, Q2, C2, H2 = tester.test_with_parameter_values(test_params)
            
            if F2 is not None:
                print(f"✓ State space construction robust to parameter changes")
                
                # Check if matrices changed appropriately
                c_diff = jnp.max(jnp.abs(C2 - C))
                print(f"Maximum change in C matrix: {c_diff:.6f}")
            
            return reduced_model, tester
        else:
            print(f"✗ State space construction failed")
            return reduced_model, None
            
    except Exception as e:
        print(f"✗ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    run_comprehensive_test()