# clean_gpm_bvar_trends/symbolic_evaluation_utils.py - Simplified

import jax.numpy as jnp
import jax
import sympy as sp
import re
from typing import Dict, Tuple, Optional, Any
import numpy as np


class SymbolicEvaluationUtils:
    """
    Simplified utility class for evaluating symbolic expressions and parsing variable keys.
    Designed to work with JAX arrays/tracers with minimal complexity.
    """

    @staticmethod
    def _is_numeric_string(s: Optional[str]) -> bool:
        """Checks if a string can be parsed as a float."""
        if s is None: 
            return False
        try: 
            float(s)
            return True
        except (ValueError, TypeError): 
            return False

    @staticmethod
    def _parse_variable_key(var_key_str: str) -> Tuple[str, int]:
        """
        Parses a variable key string like 'variable_name(-lag)' into name and lag.
        """
        var_key_str = var_key_str.strip()
        match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*-\s*(\d+)\s*\))?", var_key_str)
        if match:
            var_name = match.group(1)
            lag_str = match.group(2)
            return var_name, int(lag_str) if lag_str else 0
        
        # Fallback
        return var_key_str, 0

    @staticmethod
    def evaluate_coefficient_expression(expr_str: Optional[str], params: Dict[str, Any]) -> jnp.ndarray:
        """
        Simplified evaluation of symbolic expressions.
        Returns a scalar JAX array.
        """
        if expr_str is None or expr_str.strip() == "" or expr_str.lower() == 'none':
            return jnp.array(1.0, dtype=jnp.float64)

        expr_str = expr_str.strip()

        # Handle simple cases
        if expr_str == '1': 
            return jnp.array(1.0, dtype=jnp.float64)
        if expr_str == '0': 
            return jnp.array(0.0, dtype=jnp.float64)
        if expr_str == '-1': 
            return jnp.array(-1.0, dtype=jnp.float64)

        # Strip outer parentheses
        while expr_str.startswith('(') and expr_str.endswith(')'):
            try:
                sp.sympify(expr_str[1:-1])
                expr_str = expr_str[1:-1].strip()
            except:
                break

        # Check if it's a parameter name
        if expr_str in params:
            param_val = params[expr_str]
            return jnp.asarray(param_val, dtype=jnp.float64)

        # Check if it's a negative parameter name
        if expr_str.startswith('-') and expr_str[1:].strip() in params:
            param_name = expr_str[1:].strip()
            param_val = params[param_name]
            return -jnp.asarray(param_val, dtype=jnp.float64)

        # Try to parse as number
        if SymbolicEvaluationUtils._is_numeric_string(expr_str):
            return jnp.array(float(expr_str), dtype=jnp.float64)

        # Use Sympy for more complex expressions
        try:
            # Find symbols in expression
            symbols_in_expr = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", expr_str))
            
            # Create sympy locals only for known parameters
            sympy_locals = {}
            for s_name in symbols_in_expr:
                if s_name in params:
                    sympy_locals[s_name] = sp.symbols(s_name)

            # Parse expression
            s_expr = sp.sympify(expr_str, locals=sympy_locals)

            # Substitute values
            subs_dict = {}
            for s_name in symbols_in_expr:
                if s_name in params:
                    param_val = params[s_name]
                    # Ensure scalar JAX array
                    if isinstance(param_val, (jax.core.Tracer, jnp.ndarray)):
                        if len(param_val.shape) == 0:
                            subs_dict[sympy_locals[s_name]] = param_val
                        else:
                            # Non-scalar parameter
                            return jnp.array(np.nan, dtype=jnp.float64)
                    else:
                        subs_dict[sympy_locals[s_name]] = jnp.asarray(param_val, dtype=jnp.float64)

            # Substitute and evaluate
            expr_with_values = s_expr.subs(subs_dict)
            
            # Simple evaluation for common cases
            if isinstance(expr_with_values, (sp.Number, float, int)):
                return jnp.array(float(expr_with_values), dtype=jnp.float64)
            
            if isinstance(expr_with_values, jax.core.Tracer):
                return expr_with_values
            
            if isinstance(expr_with_values, sp.Add):
                result = jnp.array(0.0, dtype=jnp.float64)
                for arg in expr_with_values.args:
                    if isinstance(arg, jax.core.Tracer):
                        result += arg
                    elif isinstance(arg, (sp.Number, float, int)):
                        result += float(arg)
                    else:
                        # For more complex terms, recursively evaluate
                        term_val = SymbolicEvaluationUtils._evaluate_sympy_term(arg, params)
                        result += term_val
                return result
            
            if isinstance(expr_with_values, sp.Mul):
                result = jnp.array(1.0, dtype=jnp.float64)
                for arg in expr_with_values.args:
                    if isinstance(arg, jax.core.Tracer):
                        result *= arg
                    elif isinstance(arg, (sp.Number, float, int)):
                        result *= float(arg)
                    else:
                        # For more complex terms, recursively evaluate
                        term_val = SymbolicEvaluationUtils._evaluate_sympy_term(arg, params)
                        result *= term_val
                return result

            # For other types, try to convert to float
            try:
                return jnp.array(float(expr_with_values), dtype=jnp.float64)
            except:
                return jnp.array(np.nan, dtype=jnp.float64)

        except Exception as e:
            # If all else fails, return NaN
            return jnp.array(np.nan, dtype=jnp.float64)

    @staticmethod
    def _evaluate_sympy_term(sympy_expr: Any, params: Dict[str, Any]) -> jnp.ndarray:
        """Helper method to evaluate individual sympy terms."""
        if isinstance(sympy_expr, jax.core.Tracer):
            return sympy_expr
        if isinstance(sympy_expr, (sp.Number, float, int)):
            return jnp.array(float(sympy_expr), dtype=jnp.float64)
        if isinstance(sympy_expr, sp.Symbol):
            s_name = str(sympy_expr)
            if s_name in params:
                return jnp.asarray(params[s_name], dtype=jnp.float64)
            else:
                raise ValueError(f"Unknown symbol '{s_name}' in expression")
        
        # For more complex expressions, try simple recursive evaluation
        if isinstance(sympy_expr, sp.Add):
            result = jnp.array(0.0, dtype=jnp.float64)
            for arg in sympy_expr.args:
                result += SymbolicEvaluationUtils._evaluate_sympy_term(arg, params)
            return result
        
        if isinstance(sympy_expr, sp.Mul):
            result = jnp.array(1.0, dtype=jnp.float64)
            for arg in sympy_expr.args:
                result *= SymbolicEvaluationUtils._evaluate_sympy_term(arg, params)
            return result
        
        if isinstance(sympy_expr, sp.Pow):
            base = SymbolicEvaluationUtils._evaluate_sympy_term(sympy_expr.base, params)
            exp = SymbolicEvaluationUtils._evaluate_sympy_term(sympy_expr.exp, params)
            return jnp.power(base, exp)
        
        # Default case
        try:
            return jnp.array(float(sympy_expr), dtype=jnp.float64)
        except:
            return jnp.array(np.nan, dtype=jnp.float64)

    @staticmethod
    def multiply_coefficients(c1_str: Optional[str], c2_str: Optional[str]) -> str:
        """Multiply coefficient strings symbolically."""
        c1 = c1_str if c1_str is not None else "1"
        c2 = c2_str if c2_str is not None else "1"

        if c1 == "0" or c2 == "0":
            return "0"
        if c1 == "1":
            return c2
        if c2 == "1":
            return c1

        # Try sympy for simplification
        try:
            result_expr = sp.sympify(f"({c1}) * ({c2})")
            try:
                num_result = float(sp.N(result_expr))
                if num_result == int(num_result):
                    return str(int(num_result))
                return str(num_result)
            except:
                return str(result_expr)
        except:
            # Fallback to string manipulation
            c1_neg = c1.startswith('-')
            c2_neg = c2.startswith('-')
            c1_abs = c1[1:].strip("()") if c1_neg else c1.strip("()")
            c2_abs = c2[1:].strip("()") if c2_neg else c2.strip("()")
            c1_abs = "1" if not c1_abs else c1_abs
            c2_abs = "1" if not c2_abs else c2_abs

            if c1_abs == "1" and c2_abs == "1":
                prod = "1"
            elif c1_abs == "1":
                prod = c2_abs
            elif c2_abs == "1":
                prod = c1_abs
            else:
                prod = f"({c1_abs})*({c2_abs})"

            return prod if c1_neg == c2_neg else (f"-{prod}" if prod != "0" else "0")

    @staticmethod
    def add_coefficients(c1_str: Optional[str], c2_str: Optional[str]) -> str:
        """Add coefficient strings symbolically."""
        c1 = c1_str if c1_str is not None else "0"
        c2 = c2_str if c2_str is not None else "0"

        if c1 == "0": 
            return c2
        if c2 == "0": 
            return c1

        # Try sympy for simplification
        try:
            result_expr = sp.sympify(f"({c1}) + ({c2})")
            try:
                num_result = float(sp.N(result_expr))
                if num_result == int(num_result):
                    return str(int(num_result))
                return str(num_result)
            except:
                return str(result_expr)
        except:
            # Fallback to string manipulation
            c2_eff = c2.strip("()")
            if c2_eff.startswith('-'):
                return f"{c1} - {c2_eff[1:]}"
            return f"{c1} + {c2_eff}"


# Create a singleton-like instance for backward compatibility
SymbolicEvaluationUtils = SymbolicEvaluationUtils()


# Simplified functions for direct use in other modules
def evaluate_coefficient_expression(expr_str: Optional[str], params: Dict[str, Any]) -> jnp.ndarray:
    """Simplified function for evaluating coefficient expressions."""
    return SymbolicEvaluationUtils.evaluate_coefficient_expression(expr_str, params)


def parse_variable_key(var_key_str: str) -> Tuple[str, int]:
    """Simplified function for parsing variable keys."""
    return SymbolicEvaluationUtils._parse_variable_key(var_key_str)


def is_numeric_string(s: Optional[str]) -> bool:
    """Simplified function for checking if string is numeric."""
    return SymbolicEvaluationUtils._is_numeric_string(s)


if __name__ == "__main__":
    # Simple test
    utils = SymbolicEvaluationUtils()
    
    # Test basic evaluation
    params = {'rho': jnp.array(0.5), 'alpha': jnp.array(2.0)}
    
    test_cases = [
        "rho",
        "0.5 * rho",
        "rho + alpha", 
        "rho * alpha",
        "-rho",
        "1",
        "0"
    ]
    
    print("Testing symbolic evaluation:")
    for expr in test_cases:
        result = utils.evaluate_coefficient_expression(expr, params)
        print(f"  {expr:<15} -> {result}")
    
    # Test variable key parsing
    print("\nTesting variable key parsing:")
    test_keys = ["x", "x(-1)", "trend_var(-2)", "STAT1"]
    for key in test_keys:
        name, lag = utils._parse_variable_key(key)
        print(f"  {key:<15} -> name='{name}', lag={lag}")