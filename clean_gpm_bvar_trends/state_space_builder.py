# clean_gpm_bvar_trends/state_space_builder.py

from typing import Dict, List, Tuple, Optional, Any
import jax.numpy as jnp
import numpy as np # For type checking or simple array ops if needed before JAX conversion
import sympy as sp
import re
import jax 
# Assuming these are in the same package/directory relative to this file
from gpm_model_parser import ReducedModel, ParsedTerm, ParsedEquation, ReducedExpression # Ensure these are the final names
from parameter_contract import ParameterContract, get_parameter_contract, ParameterType
from common_types import EnhancedBVARParams

# Constants
_DEFAULT_DTYPE = jnp.float64
_JITTER = 1e-8
_KF_JITTER = 1e-8

class StateSpaceBuilder:
    def __init__(self, reduced_model: ReducedModel, contract: Optional[ParameterContract] = None):
        self.model = reduced_model
        self.contract = contract if contract is not None else get_parameter_contract()

        self.n_core = len(reduced_model.core_variables)
        self.n_stationary = len(reduced_model.stationary_variables)
        self.n_observed = len(reduced_model.reduced_measurement_equations)
        self.var_order = reduced_model.var_prior_setup.var_order if reduced_model.var_prior_setup and hasattr(reduced_model.var_prior_setup, 'var_order') else 1
        self.state_dim = self.n_core + self.n_stationary * self.var_order

        self.n_trends = self.n_core # Compatibility
        self.gmp = reduced_model # Compatibility

        self.core_var_map = {var: i for i, var in enumerate(reduced_model.core_variables)}
        self.stat_var_map = {var: i for i, var in enumerate(reduced_model.stationary_variables)}
        self.obs_var_map = {var: i for i, var in enumerate(reduced_model.reduced_measurement_equations.keys())}

        # print(f"StateSpaceBuilder Initialized. State Dim: {self.state_dim}, Observed: {self.n_observed}")

    def _get_value_from_mcmc_draw(self, param_name_mcmc: str, all_draws_array: Any, sample_idx: int) -> Any:
        if all_draws_array is None:
            # print(f"Warning: MCMC sample array for '{param_name_mcmc}' is None. Returning NaN.")
            return jnp.nan # Or raise error
        if not hasattr(all_draws_array, 'shape') or not hasattr(all_draws_array, '__getitem__') or len(all_draws_array.shape) == 0:
            return all_draws_array
        if sample_idx >= all_draws_array.shape[0]:
            raise IndexError(f"sample_idx {sample_idx} out of bounds for MCMC param '{param_name_mcmc}' (shape: {all_draws_array.shape}).")
        return all_draws_array[sample_idx]

    def _extract_params_from_mcmc_draw(self,
                                       mcmc_samples_full_dict: Dict[str, jnp.ndarray],
                                       sample_idx: int) -> Dict[str, Any]:
        builder_params: Dict[str, Any] = {}
        for mcmc_name, all_draws_array in mcmc_samples_full_dict.items():
            try:
                param_value = self._get_value_from_mcmc_draw(mcmc_name, all_draws_array, sample_idx)

                if mcmc_name == "A_transformed":
                    builder_params["_var_coefficients"] = param_value
                elif mcmc_name == "Omega_u_chol":
                    builder_params["_var_innovation_corr_chol"] = param_value
                elif mcmc_name.startswith("sigma_"):
                    builder_shock_name = self.contract.get_builder_name(mcmc_name)
                    builder_params[builder_shock_name] = param_value
                elif mcmc_name in self.contract._mcmc_to_builder:
                    builder_name = self.contract.get_builder_name(mcmc_name)
                    builder_params[builder_name] = param_value
            except (IndexError, ValueError) as e: # ValueError if contract lookup fails or None array
                # print(f"Warning extracting '{mcmc_name}': {e}")
                pass # Failures in _get_value_from_mcmc_draw or contract might lead here.
            except Exception as e:
                # print(f"Unexpected error processing MCMC param {mcmc_name} for builder: {e}")
                pass
        return builder_params

    def _extract_params_from_enhanced_bvar(self,
                                           bvar_params: EnhancedBVARParams) -> Dict[str, Any]:
        builder_params: Dict[str, Any] = {}
        if bvar_params.A is not None:
            builder_params["_var_coefficients"] = bvar_params.A

        if bvar_params.structural_params:
            for name_in_struct, value in bvar_params.structural_params.items():
                if name_in_struct in self.contract._mcmc_to_builder:
                    builder_name = self.contract.get_builder_name(name_in_struct)
                    builder_params[builder_name] = value
                else: # Already a builder name or direct use
                    builder_params[name_in_struct] = value
        
        # Pass full covariance matrices if present
        if bvar_params.Sigma_eta is not None:
            builder_params["_trend_innovation_cov_full"] = bvar_params.Sigma_eta
        if bvar_params.Sigma_u is not None:
            builder_params["_var_innovation_cov_full"] = bvar_params.Sigma_u
        if bvar_params.Sigma_eps is not None:
            builder_params["_measurement_error_cov_full"] = bvar_params.Sigma_eps
        
        return builder_params

    def _standardize_direct_params(self,
                                   direct_params: Dict[str, Any]) -> Dict[str, Any]:
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

    def build_state_space_from_mcmc_sample(self,
                                           mcmc_samples_full_dict: Dict[str, jnp.ndarray],
                                           sample_idx: int) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._extract_params_from_mcmc_draw(mcmc_samples_full_dict, sample_idx)
        return self._build_matrices_internal(current_builder_params)

    def build_state_space_from_enhanced_bvar(self,
                                             bvar_params: EnhancedBVARParams) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._extract_params_from_enhanced_bvar(bvar_params)
        return self._build_matrices_internal(current_builder_params)

    def build_state_space_from_direct_dict(self,
                                           direct_params: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        current_builder_params = self._standardize_direct_params(direct_params)
        return self._build_matrices_internal(current_builder_params)

    def _build_matrices_internal(self, params: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        F = jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        Q = jnp.zeros((self.state_dim, self.state_dim), dtype=_DEFAULT_DTYPE)
        C = jnp.zeros((self.n_observed, self.state_dim), dtype=_DEFAULT_DTYPE)
        
        H_val = params.get("_measurement_error_cov_full")
        # ... (H matrix logic) ...
        H = _KF_JITTER * jnp.eye(self.n_observed, dtype=_DEFAULT_DTYPE) # Default
        if params.get("_measurement_error_cov_full") is not None:
            H_val = params["_measurement_error_cov_full"]
            if H_val is not None and hasattr(H_val, 'shape') and H_val.shape == (self.n_observed, self.n_observed): H = H_val
            # else: print("Warning: _measurement_error_cov_full shape mismatch. Using default H.")


        F, Q = self._build_core_dynamics(F, Q, params)
        if self.n_stationary > 0:
            F = self._build_var_dynamics(F, params)
            Q = self._add_var_innovations(Q, params)
        C = self._build_measurement_matrix(C, params)

        Q = (Q + Q.T) / 2.0 + _JITTER * jnp.eye(self.state_dim, dtype=_DEFAULT_DTYPE)
        H = (H + H.T) / 2.0
        
        # # Final check for NaNs in critical matrices - REMOVE OR COMMENT OUT THIS BLOCK
        # if jnp.any(jnp.isnan(F)) or jnp.any(jnp.isnan(Q)) or \
        #    jnp.any(jnp.isnan(C)) or jnp.any(jnp.isnan(H)):
        #     print("Warning: NaNs detected in generated state-space matrices during _build_matrices_internal.")
        #     # This print will happen for every problematic step if not removed.
        #     # The matrices_ok check in NumPyro model is the JAX-safe way to handle this.

        return F, Q, C, H

    def _build_core_dynamics(self, F_init: jnp.ndarray, Q_init: jnp.ndarray, params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        F = F_init
        Q = Q_init
        
        for i_eq, equation in enumerate(self.model.core_equations):
            if equation.lhs not in self.core_var_map:
                # print(f"Warning: LHS '{equation.lhs}' of core equation not in core_var_map. Skipping eq.")
                continue
            lhs_idx = self.core_var_map[equation.lhs]

            # Set F matrix elements
            for term in equation.rhs_terms:
                if term.variable in self.core_var_map:
                    rhs_idx = self.core_var_map[term.variable]
                    coeff_value = self._evaluate_coefficient(term.coefficient, params)
                    if term.sign == '-': coeff_value = -coeff_value
                    
                    if term.lag == 1:
                        F = F.at[lhs_idx, rhs_idx].set(coeff_value)
                    elif term.lag == 0:
                        if lhs_idx != rhs_idx : F = F.at[lhs_idx, rhs_idx].set(coeff_value)
                        # If lhs_idx == rhs_idx, F is already identity unless overwritten.
                        # If coeff_value is not 1.0, it should overwrite.
                        elif coeff_value != 1.0 or term.coefficient is not None: # if coeff is explicit
                            F = F.at[lhs_idx, rhs_idx].set(coeff_value)


        # Set Q matrix for core shocks
        if "_trend_innovation_cov_full" in params and params["_trend_innovation_cov_full"] is not None:
            Sigma_eta_full = params["_trend_innovation_cov_full"]
            if Sigma_eta_full.shape == (self.n_core, self.n_core):
                Q = Q.at[:self.n_core, :self.n_core].set(Sigma_eta_full)
            else: # Shape mismatch, fall back
                # print(f"Warning: _trend_innovation_cov_full shape mismatch. Falling back to individual shocks for Q.")
                core_shock_vars = jnp.zeros(self.n_core, dtype=_DEFAULT_DTYPE)
                for equation in self.model.core_equations: # Iterate again to get shocks
                    if equation.lhs in self.core_var_map and equation.shock:
                        shock_variance = self._get_shock_variance(equation.shock, params)
                        idx = self.core_var_map[equation.lhs]
                        core_shock_vars = core_shock_vars.at[idx].add(shock_variance) # Use add if multiple shocks affect one var
                Q = Q.at[:self.n_core, :self.n_core].set(jnp.diag(core_shock_vars))
        else: # Construct from individual shock variances
            core_shock_vars = jnp.zeros(self.n_core, dtype=_DEFAULT_DTYPE)
            for equation in self.model.core_equations:
                if equation.lhs in self.core_var_map and equation.shock:
                    shock_variance = self._get_shock_variance(equation.shock, params) # Expects std_dev in params[equation.shock]
                    idx = self.core_var_map[equation.lhs]
                    core_shock_vars = core_shock_vars.at[idx].set(shock_variance) # .set is typical for x=x(-1)+shock
            Q = Q.at[:self.n_core, :self.n_core].set(jnp.diag(core_shock_vars))
            
        return F, Q

    def _build_var_dynamics(self, F_init: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        F = F_init
        var_start = self.n_core
        A_matrices = params.get("_var_coefficients")

        if A_matrices is None or not isinstance(A_matrices, jnp.ndarray) or A_matrices.ndim != 3 or \
           A_matrices.shape[1:] != (self.n_stationary, self.n_stationary) or \
           A_matrices.shape[0] != self.var_order : # Check number of lag matrices
            # print(f"Warning: '_var_coefficients' missing, malformed, or wrong number of lags. VAR F-block may be incorrect.")
            # Fallback: set first lag block to a default (e.g., 0.7 * I) and others to zero
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
        var_start = self.n_core
        n_stat = self.n_stationary
        if n_stat == 0: return Q

        Sigma_u_to_set: Optional[jnp.ndarray] = None

        if "_var_innovation_cov_full" in params and params["_var_innovation_cov_full"] is not None:
            Sigma_u_full = params["_var_innovation_cov_full"]
            if Sigma_u_full.shape == (n_stat, n_stat):
                Sigma_u_to_set = Sigma_u_full
            else:
                # print(f"Warning: _var_innovation_cov_full shape mismatch. Constructing Sigma_u from components.")
                pass # Will fall through to component construction
        
        if Sigma_u_to_set is None: # Construct from components
            # GPM model.stationary_shocks stores builder_names for shocks, e.g., ["shk_cycle_y_us", ...]
            gpm_stat_shocks = getattr(self.model, 'stationary_shocks', [])
            if not isinstance(gpm_stat_shocks, list): gpm_stat_shocks = []

            # Initialize with a small default in case some sigmas are missing
            stat_sigmas_std = jnp.full(n_stat, 0.01, dtype=_DEFAULT_DTYPE) 
            
            actual_shocks_in_gpm = min(len(gpm_stat_shocks), n_stat) # How many shocks are defined for the stat vars

            for i in range(actual_shocks_in_gpm):
                shock_builder_name = gpm_stat_shocks[i]
                if shock_builder_name in params:
                    val = params[shock_builder_name]
                    stat_sigmas_std = stat_sigmas_std.at[i].set(float(val.item()) if hasattr(val, 'item') else float(val))
                # else: print(f"Warning: Std.dev for stat shock '{shock_builder_name}' not in params. Using default for its slot.")
            
            # If fewer shocks defined than n_stat, remaining stat_sigmas_std slots use default.

            Omega_u_chol = params.get("_var_innovation_corr_chol")
            if Omega_u_chol is not None and isinstance(Omega_u_chol, jnp.ndarray) and Omega_u_chol.shape == (n_stat, n_stat):
                Sigma_u_to_set = jnp.diag(stat_sigmas_std) @ Omega_u_chol @ Omega_u_chol.T @ jnp.diag(stat_sigmas_std)
            else:
                if Omega_u_chol is not None: # Malformed
                    # print(f"Warning: _var_innovation_corr_chol malformed. Using diagonal Sigma_u for VAR.")
                    pass
                Sigma_u_to_set = jnp.diag(stat_sigmas_std ** 2)

        if Sigma_u_to_set is not None and Sigma_u_to_set.shape == (n_stat, n_stat):
            Q = Q.at[var_start : var_start + n_stat,
                     var_start : var_start + n_stat].set(Sigma_u_to_set)
        # else: print("Warning: Failed to set Sigma_u for VAR innovations in Q matrix.")
        return Q

    def _get_shock_variance(self, shock_builder_name: str, params: Dict[str, Any]) -> float:
        std_dev = params.get(shock_builder_name) # This key is "SHK_X"
        if std_dev is None:
            # print(f"Warning: Std dev for shock '{shock_builder_name}' not found. Using default 0.1.")
            std_dev = 0.1
        val = float(std_dev.item()) if hasattr(std_dev, 'item') else float(std_dev)
        if val < 0: # Should not happen if MCMC samples std devs correctly
            # print(f"Warning: Negative std_dev {val} for {shock_builder_name}. Using abs value.")
            val = abs(val)
        return val ** 2

    def _evaluate_coefficient(self, coeff_name_or_val: Optional[str], params: Dict[str, Any]) -> jnp.ndarray: # Return JAX array
        """
        Evaluates a coefficient. It can be a direct numeric value, a parameter name, or None (implies 1.0).
        Returns a 0-d JAX array.
        """
        if coeff_name_or_val is None:
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        if isinstance(coeff_name_or_val, (float, int, np.number)): # Check for Python/Numpy numbers
            return jnp.array(float(coeff_name_or_val), dtype=_DEFAULT_DTYPE)
        if not isinstance(coeff_name_or_val, str):
            # print(f"Warning: Coefficient '{coeff_name_or_val}' is not a string or number. Using 1.0.")
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)

        # It's a string
        if coeff_name_or_val in params:
            val = params[coeff_name_or_val] # val is already a JAX tracer (0-d array) from NumPyro sample
            return val # Return the JAX tracer directly
        try:
            # If coeff_name_or_val is a string like "0.5"
            return jnp.array(float(coeff_name_or_val), dtype=_DEFAULT_DTYPE)
        except ValueError:
            # print(f"Warning: Coeff name '{coeff_name_or_val}' not in params & not a float. Using 1.0.")
            return jnp.array(1.0, dtype=_DEFAULT_DTYPE)

    def _evaluate_coefficient_expression(self, expr_str: str, params: Dict[str, Any]) -> jnp.ndarray: # Return JAX array
        """
        Evaluates a coefficient expression string which might involve parameters.
        Returns a 0-d JAX array.
        """
        if not expr_str or expr_str.lower() == 'none': return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        expr_str_orig = expr_str
        expr_str = expr_str.strip()

        if expr_str == '1': return jnp.array(1.0, dtype=_DEFAULT_DTYPE)
        if expr_str == '0': return jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        if expr_str == '-1': return jnp.array(-1.0, dtype=_DEFAULT_DTYPE)
        
        while expr_str.startswith('(') and expr_str.endswith(')'):
            try: sp.sympify(expr_str[1:-1]); expr_str = expr_str[1:-1].strip()
            except (sp.SympifyError, SyntaxError): break

        if expr_str in params: return params[expr_str] # Already a JAX tracer
        if expr_str.startswith('-') and expr_str[1:].strip() in params:
            param_name = expr_str[1:].strip()
            return -params[param_name] # JAX handles negation of tracer

        # Use Sympy for evaluation, but substitute with JAX tracers
        try:
            # Create sympy symbols for all keys in params that are strings (parameter names)
            # And also for any other symbolic entities in expr_str
            sympy_symbols_in_expr = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]+", expr_str))
            
            sympy_locals_map = {} # Maps string name to sp.Symbol object
            for s_name in sympy_symbols_in_expr:
                if s_name not in sympy_locals_map:
                    sympy_locals_map[s_name] = sp.symbols(s_name)

            # Create the substitution dictionary for sympy.
            # Values will be JAX tracers if the symbol name is in `params`.
            subs_dict_for_sympy = {}
            for sym_obj in sympy_symbols_in_expr:
                s_name = str(sym_obj) # Get string name from sympy symbol
                if s_name in params:
                    subs_dict_for_sympy[sympy_locals_map[s_name]] = params[s_name] 
                # If s_name is not in params, it remains a sympy symbol for sympify,
                # which is fine if it's part of a structure that simplifies away or
                # if it's a variable name that should have been handled differently.
                # For coefficient expressions, all symbols should ideally be parameters or numbers.

            s_expr_obj = sp.sympify(expr_str, locals=sympy_locals_map)
            
            # Substitute JAX tracers into the sympy expression
            # This will result in an expression tree containing JAX tracers at the leaves
            expr_with_tracers = s_expr_obj.subs(subs_dict_for_sympy)
            
            # Now, how to "evaluate" this expr_with_tracers to a single JAX tracer?
            # If expr_with_tracers is already a JAX tracer (e.g. if expr_str was just "var_phi")
            if isinstance(expr_with_tracers, jax.core.Tracer):
                return expr_with_tracers
            # If it's a Sympy number after substitution (all params had concrete values - not during tracing)
            if isinstance(expr_with_tracers, (sp.Number, float, int)):
                return jnp.array(float(expr_with_tracers), dtype=_DEFAULT_DTYPE)

            # This is the tricky part: converting a Sympy expression containing JAX tracers
            # back to a JAX computation. Sympy's evalf() won't work with JAX tracers.
            # We need to reconstruct the JAX computation based on the sympy expression tree.
            # For simple arithmetic (+, -, *):
            if isinstance(expr_with_tracers, sp.Add):
                result = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
                for arg in expr_with_tracers.args:
                    # This recursive call needs to handle args correctly.
                    # For simplicity, assuming args are now JAX tracers or Sympy numbers
                    if isinstance(arg, jax.core.Tracer): result += arg
                    elif isinstance(arg, (sp.Number, float, int)): result += float(arg)
                    else: # Complex arg, try to eval it (might fail if it still has unevaluated sympy symbols)
                         # This recursive evaluation path is non-trivial to make fully robust with JAX tracers.
                         # print(f"Warning: Complex sympy arg '{arg}' in expression. Trying to evaluate its string form.")
                         # This might lead back to issues if not careful.
                         # A direct sympy-to-JAX expression converter is ideal but complex.
                         # Let's assume for now coefficients are simple enough that subs() leads to JAX tracers or numbers.
                         # If 'arg' itself is a Mul with tracers, we need to handle that.
                        temp_eval = self._evaluate_sympy_expr_with_tracers(arg, params) # New helper
                        result += temp_eval
                return result
            elif isinstance(expr_with_tracers, sp.Mul):
                result = jnp.array(1.0, dtype=_DEFAULT_DTYPE)
                for arg in expr_with_tracers.args:
                    if isinstance(arg, jax.core.Tracer): result *= arg
                    elif isinstance(arg, (sp.Number, float, int)): result *= float(arg)
                    else: 
                        temp_eval = self._evaluate_sympy_expr_with_tracers(arg, params)
                        result *= temp_eval
                return result
            # Add more cases for sp.Pow etc. if needed

            # If it's a single symbol that should have been a tracer (e.g. if expr_str was "var_phi")
            # and it's still a sympy symbol, it means it wasn't in params.
            if isinstance(expr_with_tracers, sp.Symbol) and str(expr_with_tracers) not in params:
                 raise ValueError(f"Coefficient symbol '{expr_with_tracers}' not found in params and not numeric.")

            # Fallback if it's not easily converted - this path means a JAX-compatible result wasn't formed.
            # print(f"Warning: Sympy expression '{expr_with_tracers}' could not be directly converted to JAX computation. Result might be incorrect for tracing.")
            # Forcing it to string and re-evaluating is risky.
            # It's better to ensure the sympy substitution results in a simple structure of JAX tracers.
            # If expr_str was "0.5*var_phi", expr_with_tracers should be 0.5 * <tracer_for_var_phi>
            # This should be handled by the Mul case above.
            raise ValueError(f"Cannot convert sympy expression '{expr_with_tracers}' with JAX tracers to a single JAX tracer.")

        except Exception as e:
            # print(f"Error evaluating coefficient expression '{expr_str_orig}' with JAX tracers: {e}. Params: {list(params.keys())}. Returning 0.0.")
            return jnp.array(0.0, dtype=_DEFAULT_DTYPE) # Fallback


    def _evaluate_sympy_expr_with_tracers(self, sympy_expr: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Helper to evaluate a sympy expression part that might contain JAX tracers."""
        if isinstance(sympy_expr, jax.core.Tracer):
            return sympy_expr
        if isinstance(sympy_expr, (sp.Number, float, int)):
            return jnp.array(float(sympy_expr), dtype=_DEFAULT_DTYPE)
        if isinstance(sympy_expr, sp.Symbol):
            s_name = str(sympy_expr)
            if s_name in params: return params[s_name] # It's a parameter (JAX tracer)
            else: raise ValueError(f"Unrecognized symbol '{s_name}' in sympy expression during JAX evaluation.")
        
        if isinstance(sympy_expr, sp.Add):
            res = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            for arg in sympy_expr.args: res += self._evaluate_sympy_expr_with_tracers(arg, params)
            return res
        elif isinstance(sympy_expr, sp.Mul):
            res = jnp.array(1.0, dtype=_DEFAULT_DTYPE)
            for arg in sympy_expr.args: res *= self._evaluate_sympy_expr_with_tracers(arg, params)
            return res
        elif isinstance(sympy_expr, sp.Pow):
            base = self._evaluate_sympy_expr_with_tracers(sympy_expr.base, params)
            exp = self._evaluate_sympy_expr_with_tracers(sympy_expr.exp, params)
            return base ** exp # JAX power
        
        raise ValueError(f"Unsupported sympy expression type '{type(sympy_expr)}' for JAX evaluation: {sympy_expr}")


    def _parse_variable_key(self, var_key_str: str) -> Tuple[str, int]:
        var_key_str = var_key_str.strip()
        # Regex to match VARNAME, VARNAME (-LAG), VARNAME(-LAG)
        match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*-\s*(\d+)\s*\))?", var_key_str)
        if match:
            var_name = match.group(1)
            lag_str = match.group(2)
            return var_name, int(lag_str) if lag_str else 0
        
        # print(f"Warning: Could not parse variable key '{var_key_str}'. Returning as is with lag 0.")
        return var_key_str, 0

    def _find_state_index(self, var_name: str, lag: int) -> Optional[int]:
        if var_name in self.core_var_map:
            if lag == 0: return self.core_var_map[var_name]
            # print(f"Info: Lagged core var '{var_name}(-{lag})' requested. Core states are current period.")
            return None
        if var_name in self.stat_var_map:
            if 0 <= lag < self.var_order:
                return self.n_core + (lag * self.n_stationary) + self.stat_var_map[var_name]
            # print(f"Warning: Lag {lag} for stat var '{var_name}' is out of VAR order range [0, {self.var_order-1}] for state vector.")
            return None
        return None

    def _build_measurement_matrix(self, C_init: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        C = C_init
        for obs_var_name, reduced_expr_obj in self.model.reduced_measurement_equations.items():
            if obs_var_name not in self.obs_var_map:
                # print(f"Warning: Observed variable '{obs_var_name}' from GPM not in obs_var_map. Skipping its C matrix row.")
                continue
            obs_idx = self.obs_var_map[obs_var_name]

            for var_key_in_expr, coeff_str_in_expr in reduced_expr_obj.terms.items():
                var_name_from_key, lag_from_key = self._parse_variable_key(var_key_in_expr)
                state_idx = self._find_state_index(var_name_from_key, lag_from_key)

                if state_idx is not None:
                    coeff_val = self._evaluate_coefficient_expression(coeff_str_in_expr, params)
                    C = C.at[obs_idx, state_idx].add(coeff_val)
            
            # This part from your original might be redundant if the reduced_expr_obj.terms correctly capture all links.
            # If an observed variable IS a stationary variable, its link to its own current state
            # (y_obs = 1*y_stat(0) + ...) should ideally be part of reduced_expr_obj.terms.
            # However, if the GPM implies y_obs = TrendPart + y_stat (implicitly y_stat(0)), this ensures the link.
            if obs_var_name in self.stat_var_map:
                current_stat_var_state_idx = self._find_state_index(obs_var_name, 0) # lag 0 for current period
                if current_stat_var_state_idx is not None:
                    # Check if C[obs_idx, current_stat_var_state_idx] is already set by reduced_expr_obj.terms
                    # If not, and a direct link is implied, set to 1.0
                    # This requires knowing if the reduction process makes this explicit.
                    # A simple way to ensure it without double counting if explicit:
                    # C = C.at[obs_idx, current_stat_var_state_idx].set(
                    #     jnp.maximum(C[obs_idx, current_stat_var_state_idx], 1.0)
                    # )
                    # More safely, only set if it's zero, assuming an explicit "1*obs_var_name(0)" would be in terms.
                    # If the convention is that y_obs = Trends + StatVar (where StatVar is obs_var_name),
                    # then a coefficient of 1.0 to its own current state is implied.
                    # Let's assume for now the terms dict is comprehensive. If a problem arises,
                    # one might explicitly add C = C.at[obs_idx, current_stat_var_state_idx].add(1.0)
                    # IF it's not already represented by a term "obs_var_name(0)" in reduced_expr_obj.terms.
                    pass # Relying on reduced_expr_obj.terms to be complete.

        return C

# The ReducedModelTester and __main__ block would typically be in a separate test file.