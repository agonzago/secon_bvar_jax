# clean_gpm_bvar_trends/state_space_builder.py - FIXED VERSION

from typing import Dict, List, Tuple, Optional, Any
import jax.numpy as jnp
import numpy as np
import sympy as sp
import re
import jax 

from .gpm_model_parser import ReducedModel, ParsedTerm, ParsedEquation, ReducedExpression
from .dynamic_parameter_contract import DynamicParameterContract, create_dynamic_parameter_contract  # Updated import
from .common_types import EnhancedBVARParams

# Constants
from .constants import _DEFAULT_DTYPE, _JITTER, _KF_JITTER

class StateSpaceBuilder:
    def __init__(self, reduced_model: ReducedModel, contract: Optional[DynamicParameterContract] = None):
        self.model = reduced_model
        
        # Create dynamic contract from the model if not provided
        self.contract = contract if contract is not None else create_dynamic_parameter_contract(reduced_model)

        self.n_core = len(reduced_model.core_variables)
        self.n_stationary = len(reduced_model.stationary_variables)
        self.n_observed = len(reduced_model.reduced_measurement_equations)
        self.var_order = reduced_model.var_prior_setup.var_order if reduced_model.var_prior_setup and hasattr(reduced_model.var_prior_setup, 'var_order') else 1
        
        # State dimension calculation
        self.n_dynamic_trends = self.n_core - self.n_stationary
        self.state_dim = self.n_dynamic_trends + self.n_stationary * self.var_order

        # For backward compatibility
        self.n_trends = self.n_dynamic_trends  
        self.gpm = reduced_model

        # Update variable mappings
        dynamic_trend_names = [var for var in reduced_model.core_variables if var not in reduced_model.stationary_variables]
        self.core_var_map = {}
        
        # Map dynamic trends to their indices
        for i, var in enumerate(dynamic_trend_names):
            self.core_var_map[var] = i
            
        # Map stationary variables to their current period indices
        for i, var in enumerate(reduced_model.stationary_variables):
            self.core_var_map[var] = self.n_dynamic_trends + i

        self.stat_var_map = {var: i for i, var in enumerate(reduced_model.stationary_variables)}
        self.obs_var_map = {var: i for i, var in enumerate(reduced_model.reduced_measurement_equations.keys())}
        
        print(f"StateSpaceBuilder with Dynamic Contract: State Dim: {self.state_dim}, Dynamic Trends: {self.n_dynamic_trends}, Stationary: {self.n_stationary}")
        print(f"Core var map: {self.core_var_map}")
        # print(f"Dynamic contract summary:")
        # print(self.contract.get_contract_summary())

    # def _get_value_from_mcmc_draw(self, param_name_mcmc: str, all_draws_array: Any, sample_idx: int) -> Any:
    #     if all_draws_array is None:
    #         return jnp.nan
    #     if not hasattr(all_draws_array, 'shape') or not hasattr(all_draws_array, '__getitem__') or len(all_draws_array.shape) == 0:
    #         return all_draws_array
    #     if sample_idx >= all_draws_array.shape[0]:
    #         raise IndexError(f"sample_idx {sample_idx} out of bounds for MCMC param '{param_name_mcmc}' (shape: {all_draws_array.shape}).")
    #     return all_draws_array[sample_idx]

    def _get_value_from_mcmc_draw(self, param_name_mcmc: str, all_draws_array: Any, sample_idx: int) -> Any:
        """FIXED version with better debugging"""
        
        if all_draws_array is None:
            print(f"    {param_name_mcmc}: array is None")
            return jnp.array(jnp.nan, dtype=_DEFAULT_DTYPE)
        
        if not hasattr(all_draws_array, 'shape'):
            print(f"    {param_name_mcmc}: not an array, returning as-is: {all_draws_array}")
            return all_draws_array
        
        if len(all_draws_array.shape) == 0:
            print(f"    {param_name_mcmc}: scalar array: {all_draws_array}")
            return all_draws_array
        
        if sample_idx >= all_draws_array.shape[0]:
            print(f"    {param_name_mcmc}: sample_idx {sample_idx} >= array size {all_draws_array.shape[0]}")
            raise IndexError(f"sample_idx {sample_idx} out of bounds for MCMC param '{param_name_mcmc}' (shape: {all_draws_array.shape})")
        
        extracted_value = all_draws_array[sample_idx]
        
        # Debug the extraction for problematic parameters
        # if param_name_mcmc in ['lambda_pi_US', 'lambda_pi_EA', 'lambda_pi_JP']:
        #     print(f"    DEBUG {param_name_mcmc}: extracted {extracted_value} from index {sample_idx}")
        #     if hasattr(extracted_value, 'shape'):
        #         print(f"      -> shape: {extracted_value.shape}, finite: {jnp.isfinite(extracted_value)}")
        
        return extracted_value
    
    def _extract_params_from_enhanced_bvar(self, bvar_params: EnhancedBVARParams) -> Dict[str, Any]:
        builder_params: Dict[str, Any] = {}
        if bvar_params.A is not None:
            builder_params["_var_coefficients"] = bvar_params.A

        if bvar_params.structural_params:
            for name_in_struct, value in bvar_params.structural_params.items():
                if name_in_struct in self.contract._mcmc_to_builder:
                    builder_name = self.contract.get_builder_name(name_in_struct)
                    builder_params[builder_name] = value
                else:
                    builder_params[name_in_struct] = value
        
        if bvar_params.Sigma_eta is not None:
            builder_params["_trend_innovation_cov_full"] = bvar_params.Sigma_eta
        if bvar_params.Sigma_u is not None:
            builder_params["_var_innovation_cov_full"] = bvar_params.Sigma_u
        if bvar_params.Sigma_eps is not None:
            builder_params["_measurement_error_cov_full"] = bvar_params.Sigma_eps
        
        return builder_params

    def _standardize_direct_params(self, direct_params: Dict[str, Any]) -> Dict[str, Any]:
        builder_params: Dict[str, Any] = {}
        for name, value in direct_params.items():
            if name in self.contract._mcmc_to_builder:
                builder_name = self.contract.get_builder_name(name)
                builder_params[builder_name] = value
            elif name in self.contract._builder_to_mcmc or name.startswith("_"):
                builder_params[name] = value
            else:
                builder_params[name] = value
        return builder_params

    def build_state_space_from_mcmc_sample(self, mcmc_samples_full_dict: Dict[str, jnp.ndarray], sample_idx: int) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._extract_params_from_mcmc_draw(mcmc_samples_full_dict, sample_idx)
        return self._build_matrices_internal(current_builder_params)

    def build_state_space_from_enhanced_bvar(self, bvar_params: EnhancedBVARParams) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._extract_params_from_enhanced_bvar(bvar_params)
        return self._build_matrices_internal(current_builder_params)

    def build_state_space_from_direct_dict(self, direct_params: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._standardize_direct_params(direct_params)
        return self._build_matrices_internal(current_builder_params)

    def _build_matrices_internal(self, params: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        F = jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        Q = jnp.zeros((self.state_dim, self.state_dim), dtype=_DEFAULT_DTYPE)
        C = jnp.zeros((self.n_observed, self.state_dim), dtype=_DEFAULT_DTYPE)
        
        # Measurement error covariance
        H = _KF_JITTER * jnp.eye(self.n_observed, dtype=_DEFAULT_DTYPE)
        if params.get("_measurement_error_cov_full") is not None:
            H_val = params["_measurement_error_cov_full"]
            if H_val is not None and hasattr(H_val, 'shape') and H_val.shape == (self.n_observed, self.n_observed): 
                H = H_val

        F, Q = self._build_core_dynamics(F, Q, params)
        if self.n_stationary > 0:
            F = self._build_var_dynamics(F, params)
            Q = self._add_var_innovations(Q, params)
        C = self._build_measurement_matrix(C, params)

        Q = (Q + Q.T) / 2.0 + _JITTER * jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0

        return F, Q, C, H

    def _build_var_dynamics(self, F_init: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        F = F_init
        var_start = self.n_dynamic_trends  # FIXED: Start after dynamic trends
        A_matrices = params.get("_var_coefficients")

        if A_matrices is None or not isinstance(A_matrices, jnp.ndarray) or A_matrices.ndim != 3 or \
           A_matrices.shape[1:] != (self.n_stationary, self.n_stationary) or \
           A_matrices.shape[0] != self.var_order:
            # Fallback: set first lag block to a default
            if self.n_stationary > 0 and self.var_order > 0:
                F = F.at[var_start : var_start + self.n_stationary,
                         var_start : var_start + self.n_stationary].set(jnp.eye(self.n_stationary)*0.7)
        else:
            for lag_idx in range(self.var_order):
                A_lag = A_matrices[lag_idx]
                row_s, row_e = var_start, var_start + self.n_stationary
                col_s, col_e = var_start + lag_idx * self.n_stationary, var_start + (lag_idx + 1) * self.n_stationary
                if row_e <= self.state_dim and col_e <= self.state_dim:
                    F = F.at[row_s:row_e, col_s:col_e].set(A_lag)

        # Identity blocks for VAR companion form
        if self.var_order > 1:
            for i in range(self.var_order - 1):
                row_s = var_start + (i + 1) * self.n_stationary
                row_e = row_s + self.n_stationary
                col_s = var_start + i * self.n_stationary
                col_e = col_s + self.n_stationary
                if row_e <= self.state_dim and col_e <= self.state_dim:
                     F = F.at[row_s:row_e, col_s:col_e].set(jnp.eye(self.n_stationary, dtype=_DEFAULT_DTYPE))
        return F

    def _add_var_innovations(self, Q_init: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        Q = Q_init
        var_start = self.n_dynamic_trends  # FIXED: Start after dynamic trends
        n_stat = self.n_stationary
        if n_stat == 0: 
            return Q

        Sigma_u_to_set: Optional[jnp.ndarray] = None

        if "_var_innovation_cov_full" in params and params["_var_innovation_cov_full"] is not None:
            Sigma_u_full = params["_var_innovation_cov_full"]
            if Sigma_u_full.shape == (n_stat, n_stat):
                Sigma_u_to_set = Sigma_u_full
        
        if Sigma_u_to_set is None:
            gpm_stat_shocks = getattr(self.model, 'stationary_shocks', [])
            if not isinstance(gpm_stat_shocks, list): 
                gpm_stat_shocks = []

            stat_sigmas_std = jnp.full(n_stat, 0.01, dtype=_DEFAULT_DTYPE) 
            actual_shocks_in_gpm = min(len(gpm_stat_shocks), n_stat)

            for i in range(actual_shocks_in_gpm):
                shock_builder_name = gpm_stat_shocks[i]
                if shock_builder_name in params:
                    val = params[shock_builder_name]
                    stat_sigmas_std = stat_sigmas_std.at[i].set(float(val.item()) if hasattr(val, 'item') else float(val))
            
            Omega_u_chol = params.get("_var_innovation_corr_chol")
            if Omega_u_chol is not None and isinstance(Omega_u_chol, jnp.ndarray) and Omega_u_chol.shape == (n_stat, n_stat):
                Sigma_u_to_set = jnp.diag(stat_sigmas_std) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(stat_sigmas_std)
            else:
                Sigma_u_to_set = jnp.diag(stat_sigmas_std ** 2)

        if Sigma_u_to_set is not None and Sigma_u_to_set.shape == (n_stat, n_stat):
            Q = Q.at[var_start : var_start + n_stat, var_start : var_start + n_stat].set(Sigma_u_to_set)
        
        return Q

    def _evaluate_coefficient(self, coeff_name_or_val: Optional[str], params: Dict[str, Any]) -> jnp.ndarray:
        if coeff_name_or_val is None:
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        if isinstance(coeff_name_or_val, (float, int, np.number)):
            return jnp.array(float(coeff_name_or_val), dtype=_DEFAULT_DTYPE)
        if not isinstance(coeff_name_or_val, str):
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)

        if coeff_name_or_val in params:
            val = params[coeff_name_or_val]
            return val
        try:
            return jnp.array(float(coeff_name_or_val), dtype=_DEFAULT_DTYPE)
        except ValueError:
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)

    def _evaluate_coefficient_expression(self, expr_str: str, params: Dict[str, Any]) -> jnp.ndarray:
        if not expr_str or expr_str.lower() == 'none': 
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        expr_str_orig = expr_str
        expr_str = expr_str.strip()

        if expr_str == '1': return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        if expr_str == '0': return jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        if expr_str == '-1': return jnp.array(-1.0, dtype=_DEFAULT_DTYPE)
        
        while expr_str.startswith('(') and expr_str.endswith(')'):
            try: 
                sp.sympify(expr_str[1:-1])
                expr_str = expr_str[1:-1].strip()
            except (sp.SympifyError, SyntaxError): 
                break

        if expr_str in params: 
            return params[expr_str]
        if expr_str.startswith('-') and expr_str[1:].strip() in params:
            param_name = expr_str[1:].strip()
            return -params[param_name]

        # Use Sympy for evaluation with JAX tracers
        try:
            sympy_symbols_in_expr = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]+", expr_str))
            
            sympy_locals_map = {}
            for s_name in sympy_symbols_in_expr:
                if s_name not in sympy_locals_map:
                    sympy_locals_map[s_name] = sp.symbols(s_name)

            subs_dict_for_sympy = {}
            for sym_obj in sympy_symbols_in_expr:
                s_name = str(sym_obj)
                if s_name in params:
                    subs_dict_for_sympy[sympy_locals_map[s_name]] = params[s_name] 

            s_expr_obj = sp.sympify(expr_str, locals=sympy_locals_map)
            expr_with_tracers = s_expr_obj.subs(subs_dict_for_sympy)
            
            if isinstance(expr_with_tracers, jax.core.Tracer):
                return expr_with_tracers
            if isinstance(expr_with_tracers, (sp.Number, float, int)):
                return jnp.array(float(expr_with_tracers), dtype=_DEFAULT_DTYPE)

            if isinstance(expr_with_tracers, sp.Add):
                result = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
                for arg in expr_with_tracers.args:
                    if isinstance(arg, jax.core.Tracer): 
                        result += arg
                    elif isinstance(arg, (sp.Number, float, int)): 
                        result += float(arg)
                    else:
                        temp_eval = self._evaluate_sympy_expr_with_tracers(arg, params)
                        result += temp_eval
                return result
            elif isinstance(expr_with_tracers, sp.Mul):
                result = jnp.array(1.0, dtype=_DEFAULT_DTYPE)
                for arg in expr_with_tracers.args:
                    if isinstance(arg, jax.core.Tracer): 
                        result *= arg
                    elif isinstance(arg, (sp.Number, float, int)): 
                        result *= float(arg)
                    else: 
                        temp_eval = self._evaluate_sympy_expr_with_tracers(arg, params)
                        result *= temp_eval
                return result

            if isinstance(expr_with_tracers, sp.Symbol) and str(expr_with_tracers) not in params:
                 raise ValueError(f"Coefficient symbol '{expr_with_tracers}' not found in params and not numeric.")

            raise ValueError(f"Cannot convert sympy expression '{expr_with_tracers}' with JAX tracers to a single JAX tracer.")

        except Exception as e:
            return jnp.array(0.0, dtype=_DEFAULT_DTYPE)

    def _evaluate_sympy_expr_with_tracers(self, sympy_expr: Any, params: Dict[str, Any]) -> jnp.ndarray:
        if isinstance(sympy_expr, jax.core.Tracer):
            return sympy_expr
        if isinstance(sympy_expr, (sp.Number, float, int)):
            return jnp.array(float(sympy_expr), dtype=_DEFAULT_DTYPE)
        if isinstance(sympy_expr, sp.Symbol):
            s_name = str(sympy_expr)
            if s_name in params: 
                return params[s_name]
            else: 
                raise ValueError(f"Unrecognized symbol '{s_name}' in sympy expression during JAX evaluation.")
        
        if isinstance(sympy_expr, sp.Add):
            res = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            for arg in sympy_expr.args: 
                res += self._evaluate_sympy_expr_with_tracers(arg, params)
            return res
        elif isinstance(sympy_expr, sp.Mul):
            res = jnp.array(1.0, dtype=_DEFAULT_DTYPE)
            for arg in sympy_expr.args: 
                res *= self._evaluate_sympy_expr_with_tracers(arg, params)
            return res
        elif isinstance(sympy_expr, sp.Pow):
            base = self._evaluate_sympy_expr_with_tracers(sympy_expr.base, params)
            exp = self._evaluate_sympy_expr_with_tracers(sympy_expr.exp, params)
            return base ** exp
        
        raise ValueError(f"Unsupported sympy expression type '{type(sympy_expr)}' for JAX evaluation: {sympy_expr}")

    def _parse_variable_key(self, var_key_str: str) -> Tuple[str, int]:
        var_key_str = var_key_str.strip()
        match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*-\s*(\d+)\s*\))?", var_key_str)
        if match:
            var_name = match.group(1)
            lag_str = match.group(2)
            return var_name, int(lag_str) if lag_str else 0
        
        return var_key_str, 0

    def _find_state_index(self, var_name: str, lag: int) -> Optional[int]:
        if var_name in self.core_var_map:
            if lag == 0: 
                return self.core_var_map[var_name]
            return None
        if var_name in self.stat_var_map:
            if 0 <= lag < self.var_order:
                return self.n_dynamic_trends + (lag * self.n_stationary) + self.stat_var_map[var_name]
            return None
        return None

    def _build_measurement_matrix(self, C_init: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        C = C_init
        for obs_var_name, reduced_expr_obj in self.model.reduced_measurement_equations.items():
            if obs_var_name not in self.obs_var_map:
                continue
            obs_idx = self.obs_var_map[obs_var_name]

            for var_key_in_expr, coeff_str_in_expr in reduced_expr_obj.terms.items():
                var_name_from_key, lag_from_key = self._parse_variable_key(var_key_in_expr)
                state_idx = self._find_state_index(var_name_from_key, lag_from_key)

                if state_idx is not None:
                    coeff_val = self._evaluate_coefficient_expression(coeff_str_in_expr, params)
                    C = C.at[obs_idx, state_idx].add(coeff_val)

        return C
    

    def _get_shock_variance(self, shock_builder_name: str, params: Dict[str, Any]) -> float:
        """Get shock variance with JAX-compatible debugging"""
        # Only print non-JAX values
        # if not hasattr(shock_builder_name, 'shape'):  # Not a JAX tracer
        #     print(f"    _get_shock_variance called with shock='{shock_builder_name}'")
        #     print(f"    Available params: {sorted(list(params.keys()))}")
        
        std_dev = params.get(shock_builder_name)
        
        # if std_dev is None:
        #     if not hasattr(shock_builder_name, 'shape'):
        #         print(f"    ERROR: shock '{shock_builder_name}' not found in params")
        #         print(f"    Available: {list(params.keys())}")
        #     # Return a small default instead of raising error during JAX compilation
        #     return 0.01  # Small default variance (std=0.1)
        
        val = float(std_dev.item()) if hasattr(std_dev, 'item') else float(std_dev)
        if val < 0:
            val = abs(val)
        variance = val ** 2
        
        # if not hasattr(shock_builder_name, 'shape'):
        #     print(f"    SUCCESS: shock='{shock_builder_name}' -> variance={variance:.6f}")
        
        return variance

    def _build_core_dynamics(self, F_init: jnp.ndarray, Q_init: jnp.ndarray, params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        F = F_init
        Q = Q_init
        
        # ... F matrix building code (unchanged) ...
        
        # Handle trend innovation covariance
        if "_trend_innovation_cov_full" in params and params["_trend_innovation_cov_full"] is not None:
            Sigma_eta_full = params["_trend_innovation_cov_full"]
            if Sigma_eta_full.shape == (self.n_dynamic_trends, self.n_dynamic_trends):
                Q = Q.at[:self.n_dynamic_trends, :self.n_dynamic_trends].set(Sigma_eta_full)
                #print("  Used _trend_innovation_cov_full for Q matrix")
        else:
            # print("  Building Q from individual shocks:")
            # print(f"  n_dynamic_trends: {self.n_dynamic_trends}")
            # print(f"  Available params: {sorted(list(params.keys()))}")
            
            trend_shock_vars = jnp.zeros(self.n_dynamic_trends, dtype=_DEFAULT_DTYPE)
            
            #print("  Processing core equations:")
            for equation in self.model.core_equations:
                is_trend_eq = (equation.lhs in self.core_var_map and 
                            equation.shock and 
                            equation.lhs not in self.model.stationary_variables)
                
                #print(f"    Equation: lhs='{equation.lhs}', shock='{equation.shock}', is_trend={is_trend_eq}")
                
                if is_trend_eq:
                    #print(f"  PROCESSING TREND: {equation.lhs} -> shock: {equation.shock}")
                    
                    shock_variance = self._get_shock_variance(equation.shock, params)
                    idx = self.core_var_map[equation.lhs]
                    
                    #print(f"    core_var_map['{equation.lhs}'] = {idx}")
                    
                    if idx < self.n_dynamic_trends:
                        trend_shock_vars = trend_shock_vars.at[idx].set(shock_variance)
                       # print(f"    SET trend_shock_vars[{idx}] = variance")
                    else:
                        print(f"    ERROR: Index {idx} >= n_dynamic_trends {self.n_dynamic_trends}")
            
            #print("  Setting Q matrix diagonal")
            Q = Q.at[:self.n_dynamic_trends, :self.n_dynamic_trends].set(jnp.diag(trend_shock_vars))
            # Don't print Q values during JAX compilation
        
        return F, Q




# Inside StateSpaceBuilder class, _extract_params_from_mcmc_draw method

    def _extract_params_from_mcmc_draw(self, mcmc_samples_full_dict: Dict[str, jnp.ndarray], sample_idx: int) -> Dict[str, Any]:
        """JAX-compatible parameter extraction with dynamic contract"""
        builder_params: Dict[str, Any] = {}
        
        # print(f"  _extract_params_from_mcmc_draw for sample_idx={sample_idx}") # Verbose
        # print(f"  Available MCMC samples: {sorted(list(mcmc_samples_full_dict.keys()))}") # Verbose
        
        for mcmc_name, all_draws_array in mcmc_samples_full_dict.items():
            try:
                param_value = self._get_value_from_mcmc_draw(mcmc_name, all_draws_array, sample_idx)

                try:
                    builder_name = self.contract.get_builder_name(mcmc_name)
                    builder_params[builder_name] = param_value
                        
                except ValueError as contract_error:
                    # Parameter not in contract - handle special cases
                    if mcmc_name in ["A_raw", "A_diag_0", "A_full_0", "Amu_0", "Amu_1", "Aomega_0", "Aomega_1"]:
                        # print(f"    {mcmc_name} -> SKIPPED (VAR intermediate parameter)") # Verbose
                        pass
                    elif mcmc_name == "init_mean_full":
                        # print(f"    {mcmc_name} -> SKIPPED (handled separately)") # Verbose
                        pass
                    # else: # Verbose
                        # print(f"    {mcmc_name} -> ERROR: {contract_error}") 
                        
            except (IndexError, ValueError) as e:
                # print(f"    {mcmc_name} -> ERROR: {e}") # Verbose
                pass
            except Exception as e:
                # print(f"    {mcmc_name} -> UNEXPECTED ERROR: {e}") # Verbose
                pass
        
        # print(f"  Final builder_params keys: {sorted(list(builder_params.keys()))}") # Verbose
        return builder_params