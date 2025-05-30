# gpm_validator.py - Comprehensive Validation Module

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from gpm_model_parser import ReducedModel, ParsedEquation
from state_space_builder import StateSpaceBuilder

@dataclass
class ValidationError:
    """Structured validation error"""
    level: str  # "ERROR", "WARNING"
    category: str  # "SYNTAX", "SEMANTIC", "NUMERICAL"
    message: str
    suggestion: Optional[str] = None

class GPMValidator:
    """Comprehensive GPM validation with multiple strictness levels"""
    
    def __init__(self, reduced_model: ReducedModel, ss_builder: StateSpaceBuilder):
        self.model = reduced_model
        self.ss_builder = ss_builder
        self.errors: List[ValidationError] = []
    
    def validate_all(self, strict: bool = True) -> List[ValidationError]:
        """Run all validation checks"""
        self.errors = []
        
        # Core validations
        self._validate_trend_specifications(strict)
        self._validate_var_specifications(strict)
        self._validate_parameter_specifications(strict)
        self._validate_measurement_specifications(strict)
        self._validate_state_space_consistency(strict)
        
        # Advanced validations
        if strict:
            self._validate_economic_constraints(strict)
            self._validate_identification(strict)
        
        return self.errors
    
    def _add_error(self, level: str, category: str, message: str, suggestion: str = None):
        """Add validation error"""
        self.errors.append(ValidationError(level, category, message, suggestion))
    
    def _validate_trend_specifications(self, strict: bool):
        """Validate trend variable specifications"""
        dynamic_trends = [v for v in self.model.core_variables if v not in self.model.stationary_variables]
        trend_equations = {eq.lhs: eq for eq in self.model.core_equations}
        
        # Check all trends have equations
        for trend in dynamic_trends:
            if trend not in trend_equations:
                self._add_error("ERROR", "SEMANTIC", 
                              f"Dynamic trend '{trend}' declared but missing equation in 'trend_model'",
                              f"Add equation: {trend} = {trend}(-1) + SHK_{trend.upper()};")
        
        # Check for deterministic trends
        for trend in dynamic_trends:
            if trend in trend_equations:
                eq = trend_equations[trend]
                if not eq.shock:
                    level = "ERROR" if strict else "WARNING"
                    self._add_error(level, "NUMERICAL",
                                  f"Deterministic trend '{trend}' has no shock - causes Q matrix singularity",
                                  f"Add shock term: {trend} = ... + SHK_{trend.upper()};")
        
        # Check initval specifications
        for trend in dynamic_trends:
            if trend not in self.model.initial_values:
                self._add_error("ERROR", "SEMANTIC",
                              f"Dynamic trend '{trend}' missing 'initval' specification",
                              f"Add: {trend}, normal_pdf, 0, 1; in initval block")
            else:
                spec = self.model.initial_values[trend]
                if spec.init_dist != 'normal_pdf' or len(spec.init_params) < 2:
                    self._add_error("ERROR", "SEMANTIC",
                                  f"Dynamic trend '{trend}' initval must be 'normal_pdf' with mean,std",
                                  f"Use: {trend}, normal_pdf, mean_value, std_value;")
    
    def _validate_var_specifications(self, strict: bool):
        """Validate VAR specifications"""
        has_stationary = bool(self.model.stationary_variables)
        has_var_setup = bool(self.model.var_prior_setup)
        has_stat_shocks = bool(self.model.stationary_shocks)
        
        if has_stationary:
            if not has_var_setup:
                self._add_error("ERROR", "SEMANTIC",
                              "Stationary variables declared but no 'var_prior_setup' block",
                              "Add var_prior_setup block with var_order, es, fs, gs, hs, eta")
            
            if not has_stat_shocks:
                self._add_error("ERROR", "SEMANTIC",
                              "Stationary variables declared but no stationary 'shocks' block",
                              "Add shocks block with var declarations for each stationary variable")
            
            # Check count consistency
            n_stat = len(self.model.stationary_variables)
            n_shocks = len(self.model.stationary_shocks)
            if n_stat != n_shocks:
                self._add_error("ERROR", "SEMANTIC",
                              f"Mismatch: {n_stat} stationary variables but {n_shocks} stationary shocks",
                              "Ensure one shock per stationary variable")
            
            # Check VAR order
            if has_var_setup and self.model.var_prior_setup.var_order < 1:
                self._add_error("ERROR", "SEMANTIC",
                              f"VAR order must be >= 1, got {self.model.var_prior_setup.var_order}",
                              "Set var_order = 1 or higher in var_prior_setup")
            
            # Check for VAR with n_vars < 2 and LKJ
            if n_stat < 2 and has_var_setup:
                level = "WARNING" if not strict else "ERROR"
                self._add_error(level, "NUMERICAL",
                              f"VAR with {n_stat} variable cannot use LKJ correlation prior",
                              "Use univariate model or add more stationary variables")
    
    def _validate_parameter_specifications(self, strict: bool):
        """Validate parameter specifications"""
        
        # Check declared parameters have priors
        for param_name in self.model.parameters:
            if param_name not in self.model.estimated_params:
                self._add_error("ERROR", "SEMANTIC",
                              f"Parameter '{param_name}' declared but no prior in 'estimated_params'",
                              f"Add: {param_name}, normal_pdf, mean, std; to estimated_params")
        
        # Check shock priors
        all_shocks = self.model.trend_shocks + self.model.stationary_shocks
        for shock_name in all_shocks:
            mcmc_name = f"sigma_{shock_name}"
            if shock_name not in self.model.estimated_params and mcmc_name not in self.model.estimated_params:
                self._add_error("ERROR", "SEMANTIC",
                              f"Shock '{shock_name}' declared but no prior in 'estimated_params'",
                              f"Add: stderr {shock_name}, inv_gamma_pdf, shape, scale; to estimated_params")
        
        # Check prior specifications are valid
        for param_name, prior_spec in self.model.estimated_params.items():
            if prior_spec.distribution not in ['normal_pdf', 'inv_gamma_pdf']:
                self._add_error("ERROR", "SEMANTIC",
                              f"Parameter '{param_name}' has unsupported prior '{prior_spec.distribution}'",
                              "Use 'normal_pdf' or 'inv_gamma_pdf'")
            
            if len(prior_spec.params) < 2:
                self._add_error("ERROR", "SEMANTIC",
                              f"Parameter '{param_name}' prior needs at least 2 parameters",
                              "Provide mean,std for normal_pdf or shape,scale for inv_gamma_pdf")
    
    def _validate_measurement_specifications(self, strict: bool):
        """Validate measurement equation specifications"""
        
        # Check all observed variables have measurement equations
        for obs_var in self.model.gpm_observed_variables_original:
            if obs_var not in self.model.reduced_measurement_equations:
                self._add_error("ERROR", "SEMANTIC",
                              f"Observed variable '{obs_var}' missing measurement equation",
                              f"Add: {obs_var} = trend_component + stationary_component; to measurement_equations")
        
        # Check measurement equations only reference valid variables
        all_valid_vars = set(self.model.core_variables) | set(self.model.parameters)
        
        for obs_var, reduced_expr in self.model.reduced_measurement_equations.items():
            for var_key in reduced_expr.terms.keys():
                var_name = var_key.split('(')[0]  # Remove lag notation
                if var_name not in all_valid_vars:
                    level = "WARNING" if not strict else "ERROR"
                    self._add_error(level, "SEMANTIC",
                                  f"Measurement equation for '{obs_var}' references unknown variable '{var_name}'",
                                  f"Ensure '{var_name}' is declared in trends_vars, stationary_variables, or parameters")
    
    def _validate_state_space_consistency(self, strict: bool):
        """Validate state-space construction consistency"""
        
        # Check state dimension calculation
        expected_dim = self.ss_builder.n_dynamic_trends + self.ss_builder.n_stationary * self.ss_builder.var_order
        if self.ss_builder.state_dim != expected_dim:
            self._add_error("ERROR", "NUMERICAL",
                          f"State dimension mismatch: computed {self.ss_builder.state_dim}, expected {expected_dim}",
                          "This is likely a code bug in StateSpaceBuilder")
        
        # Check variable mapping
        for var_name, idx in self.ss_builder.core_var_map.items():
            if idx >= self.ss_builder.state_dim:
                self._add_error("ERROR", "NUMERICAL",
                              f"Variable '{var_name}' mapped to index {idx} >= state_dim {self.ss_builder.state_dim}",
                              "This is likely a code bug in StateSpaceBuilder")
    
    def _validate_identification(self, strict: bool):
        """Validate model identification"""
        
        # Check if enough shocks for identification
        n_obs = len(self.model.gmp_observed_variables_original)
        n_trends = len([v for v in self.model.core_variables if v not in self.model.stationary_variables])
        n_stat = len(self.model.stationary_variables)
        
        total_shocks = len(self.model.trend_shocks) + len(self.model.stationary_shocks)
        min_shocks_needed = n_trends  # At least one shock per trend for identification
        
        if total_shocks < min_shocks_needed:
            self._add_error("WARNING", "IDENTIFICATION",
                          f"May be under-identified: {total_shocks} shocks for {n_trends} trends",
                          f"Consider adding more shocks or reducing trends")
        
        # Check for observability
        if n_obs < n_trends + n_stat:
            self._add_error("WARNING", "IDENTIFICATION",
                          f"More states ({n_trends + n_stat}) than observations ({n_obs})",
                          "Model may be difficult to estimate - consider more observables or fewer states")

    # Corrected Economic Constraints Validation

    def _validate_economic_constraints(self, strict: bool):
        """Validate economic constraints and best practices - CORRECTED"""
        
        # 1. Check for trend persistence patterns
        dynamic_trends = [v for v in self.model.core_variables if v not in self.model.stationary_variables]
        
        for trend_name in dynamic_trends:
            # Find the equation for this trend
            trend_eq = None
            for eq in self.model.core_equations:
                if eq.lhs == trend_name:
                    trend_eq = eq
                    break
            
            if not trend_eq:
                continue  # Already caught in semantic validation
            
            # Check the persistence pattern
            self._check_trend_persistence(trend_name, trend_eq, strict)
        
        # 2. Check VAR stability concerns
        if self.model.var_prior_setup and len(self.model.stationary_variables) > 0:
            self._check_var_stability_priors(strict)
        
        # 3. Check for economic relationships
        self._check_economic_relationships(strict)

    def _check_trend_persistence(self, trend_name: str, trend_eq, strict: bool):
        """Check persistence pattern for a specific trend"""
        
        # Look for lag-1 coefficient of the trend itself
        lag1_coeff = None
        has_lag1_term = False
        
        for term in trend_eq.rhs_terms:
            if term.variable == trend_name and term.lag == 1:
                has_lag1_term = True
                lag1_coeff = term.coefficient
                break
        
        # CORRECTED LOGIC: 
        # Case 1: No lag-1 term at all → Random walk
        if not has_lag1_term:
            self._add_error("INFO", "ECONOMIC",
                        f"Trend '{trend_name}' has no lag-1 term → Random walk (unit root)",
                        "This is typical for trends. Add mean reversion only if theoretically justified")
        
        # Case 2: Has lag-1 term → Check coefficient
        else:
            if lag1_coeff is None or lag1_coeff == "1":
                # Coefficient is 1 → Random walk  
                self._add_error("INFO", "ECONOMIC",
                            f"Trend '{trend_name}' has lag-1 coefficient = 1 → Random walk (unit root)",
                            "This is typical for trends. Change only if mean reversion is intended")
            
            else:
                # Try to evaluate coefficient
                try:
                    coeff_val = float(lag1_coeff)
                    
                    if abs(coeff_val - 1.0) < 0.001:
                        # Very close to 1
                        self._add_error("INFO", "ECONOMIC",
                                    f"Trend '{trend_name}' has lag-1 coefficient ≈ 1 ({coeff_val}) → Near unit root",
                                    "This is typical for trends")
                    
                    elif 0.9 <= coeff_val < 0.99:
                        # Moderate mean reversion
                        self._add_error("INFO", "ECONOMIC",
                                    f"Trend '{trend_name}' has moderate mean reversion (coeff = {coeff_val})",
                                    "Check if this persistence level is economically justified")
                    
                    elif coeff_val < 0.9:
                        # Strong mean reversion
                        level = "WARNING" if strict else "INFO"
                        self._add_error(level, "ECONOMIC",
                                    f"Trend '{trend_name}' has strong mean reversion (coeff = {coeff_val})",
                                    "Strong mean reversion unusual for trends - verify this is intended")
                    
                    elif coeff_val > 1.01:
                        # Explosive
                        self._add_error("WARNING", "ECONOMIC",
                                    f"Trend '{trend_name}' has explosive coefficient ({coeff_val} > 1)",
                                    "Explosive trends cause numerical instability - check specification")
                    
                except (ValueError, TypeError):
                    # Coefficient is a parameter name
                    self._add_error("INFO", "ECONOMIC",
                                f"Trend '{trend_name}' has parametric lag-1 coefficient '{lag1_coeff}'",
                                "Ensure prior keeps this coefficient in stable range [0, 1]")

    def _check_var_stability_priors(self, strict: bool):
        """Check VAR prior specifications for stability"""
        
        prior_setup = self.model.var_prior_setup
        
        # Check diagonal prior mean (persistence)
        if hasattr(prior_setup, 'es') and prior_setup.es:
            diag_mean = prior_setup.es[0] if len(prior_setup.es) > 0 else None
            
            if diag_mean is not None:
                if diag_mean > 0.98:
                    self._add_error("WARNING", "ECONOMIC",
                                f"VAR diagonal prior mean ({diag_mean}) very close to unit root",
                                "Consider using prior mean < 0.95 for better stability")
                
                elif diag_mean > 1.0:
                    self._add_error("ERROR", "ECONOMIC",
                                f"VAR diagonal prior mean ({diag_mean}) > 1 → explosive prior",
                                "Set diagonal prior mean < 1.0 for stability")
                
                elif diag_mean < 0.3:
                    level = "WARNING" if strict else "INFO"
                    self._add_error(level, "ECONOMIC",
                                f"VAR diagonal prior mean ({diag_mean}) quite low → low persistence",
                                "Low persistence unusual for macro variables - verify this is intended")
        
        # Check off-diagonal prior mean
        if hasattr(prior_setup, 'es') and len(prior_setup.es) > 1:
            offdiag_mean = prior_setup.es[1]
            
            if abs(offdiag_mean) > 0.5:
                level = "WARNING" if strict else "INFO"
                self._add_error(level, "ECONOMIC",
                            f"VAR off-diagonal prior mean ({offdiag_mean}) quite large",
                            "Large cross-effects may cause stability issues - check if justified")

    def _check_economic_relationships(self, strict: bool):
        """Check for economically sensible relationships"""
        
        # 1. Check for circular relationships in trends
        trend_dependencies = {}
        
        for eq in self.model.core_equations:
            if eq.lhs not in self.model.stationary_variables:  # It's a trend
                deps = []
                for term in eq.rhs_terms:
                    if term.variable != eq.lhs and term.variable not in self.model.stationary_variables:
                        deps.append(term.variable)
                trend_dependencies[eq.lhs] = deps
        
        # Simple cycle detection
        def has_cycle(var, visited, rec_stack):
            visited.add(var)
            rec_stack.add(var)
            
            for neighbor in trend_dependencies.get(var, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(var)
            return False
        
        visited = set()
        for trend in trend_dependencies:
            if trend not in visited:
                if has_cycle(trend, visited, set()):
                    self._add_error("WARNING", "ECONOMIC",
                                f"Circular dependency detected in trend equations involving '{trend}'",
                                "Circular dependencies can cause identification issues")
                    break
        
        # 2. Check for reasonable parameter ranges in measurement equations
        for obs_var, reduced_expr in self.model.reduced_measurement_equations.items():
            # Check for very large coefficients
            for var_key, coeff_str in reduced_expr.terms.items():
                if coeff_str and coeff_str not in self.model.parameters:
                    try:
                        coeff_val = float(coeff_str)
                        if abs(coeff_val) > 10:
                            self._add_error("WARNING", "ECONOMIC",
                                        f"Large coefficient ({coeff_val}) in measurement equation for '{obs_var}'",
                                        "Very large coefficients may indicate scaling issues")
                    except (ValueError, TypeError):
                        pass  # Coefficient is a parameter or expression

    def _validate_trend_economic_patterns(self, strict: bool):
        """Additional economic pattern validation for trends"""
        
        # Check for common trend patterns
        patterns = {
            'pure_random_walk': [],      # trend = trend(-1) + shock
            'random_walk_drift': [],     # trend = c + trend(-1) + shock  
            'mean_reverting': [],        # trend = c + rho*trend(-1) + shock, rho < 1
            'explosive': [],             # rho > 1
            'deterministic': []          # no shock
        }
        
        for eq in self.model.core_equations:
            if eq.lhs in self.model.stationary_variables:
                continue  # Skip stationary variables
            
            trend_name = eq.lhs
            
            # Analyze equation structure
            has_constant = any(term.is_numeric_variable for term in eq.rhs_terms)
            has_lag1 = any(term.variable == trend_name and term.lag == 1 for term in eq.rhs_terms)
            has_shock = bool(eq.shock)
            
            lag1_coeff = None
            if has_lag1:
                for term in eq.rhs_terms:
                    if term.variable == trend_name and term.lag == 1:
                        lag1_coeff = term.coefficient
                        break
            
            # Classify pattern
            if not has_shock:
                patterns['deterministic'].append(trend_name)
            elif not has_lag1:
                patterns['pure_random_walk'].append(trend_name)
            elif lag1_coeff is None or lag1_coeff == "1":
                if has_constant:
                    patterns['random_walk_drift'].append(trend_name)
                else:
                    patterns['pure_random_walk'].append(trend_name)
            else:
                try:
                    coeff_val = float(lag1_coeff)
                    if coeff_val > 1.0:
                        patterns['explosive'].append(trend_name)
                    else:
                        patterns['mean_reverting'].append(trend_name)
                except (ValueError, TypeError):
                    patterns['mean_reverting'].append(trend_name)  # Parameter-driven
        
        # Report patterns
        if patterns['pure_random_walk']:
            self._add_error("INFO", "ECONOMIC",
                        f"Pure random walk trends: {patterns['pure_random_walk']}",
                        "Standard for many macro trends")
        
        if patterns['random_walk_drift']:
            self._add_error("INFO", "ECONOMIC", 
                        f"Random walk with drift trends: {patterns['random_walk_drift']}",
                        "Consider if deterministic drift is economically justified")
        
        if patterns['mean_reverting']:
            level = "WARNING" if strict else "INFO"
            self._add_error(level, "ECONOMIC",
                        f"Mean-reverting trends: {patterns['mean_reverting']}",
                        "Mean reversion unusual for trends - verify economic justification")
        
        if patterns['explosive']:
            self._add_error("ERROR", "ECONOMIC",
                        f"Explosive trends: {patterns['explosive']}",
                        "Explosive trends cause numerical instability")
        
        if patterns['deterministic']:
            self._add_error("WARNING", "ECONOMIC",
                        f"Deterministic trends: {patterns['deterministic']}",
                        "Deterministic trends can't be updated by data")

def validate_gpm_model(reduced_model: ReducedModel, ss_builder: StateSpaceBuilder, strict: bool = True) -> None:
    """Main validation function"""
    validator = GPMValidator(reduced_model, ss_builder)
    errors = validator.validate_all(strict)
    
    if errors:
        error_lines = []
        warning_lines = []
        
        for error in errors:
            line = f"[{error.category}] {error.message}"
            if error.suggestion:
                line += f"\n    → Suggestion: {error.suggestion}"
            
            if error.level == "ERROR":
                error_lines.append(line)
            else:
                warning_lines.append(line)
        
        if warning_lines:
            print("⚠️  GPM VALIDATION WARNINGS:")
            for line in warning_lines:
                print(f"    {line}")
            print()
        
        if error_lines:
            error_msg = "🚫 GPM VALIDATION ERRORS:\n" + "\n".join(f"    {line}" for line in error_lines)
            if strict:
                raise ValueError(error_msg)
            else:
                print(error_msg)
    else:
        print("✅ GPM validation passed!")


# # Integration into IntegrationOrchestrator
# class IntegrationOrchestrator:
#     def __init__(self, gmp_file_path: str, strict_validation: bool = True):
#         self.gmp_file_path = gmp_file_path
        
#         # Parse model
#         parser = GPMModelParser()
#         self.reduced_model = parser.parse_file(self.gmp_file_path)
#         self.ss_builder = StateSpaceBuilder(self.reduced_model)
        
#         # Validate if requested
#         if strict_validation:
#             validate_gpm_model(self.reduced_model, self.ss_builder, strict=True)
        
#         # ... rest of initialization ...