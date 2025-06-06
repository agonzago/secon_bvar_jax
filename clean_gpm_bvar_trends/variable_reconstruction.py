# clean_gpm_bvar_trends/variable_reconstruction.py
"""
Centralized variable reconstruction logic that handles nested non-core variable dependencies.
This module is imported by both gpm_bar_smoother.py and simulation_smoothing.py to avoid circular imports.
"""

import jax.numpy as jnp
from typing import Dict, Any, Tuple
from .constants import _DEFAULT_DTYPE


def parse_variable_key(var_key: str) -> Tuple[str, int]:
    """
    Parse a variable key like 'var_name(-1)' into ('var_name', -1).
    For contemporaneous variables like 'var_name', returns ('var_name', 0).
    """
    if '(' in var_key and ')' in var_key:
        # Extract lag from parentheses
        base_name = var_key[:var_key.index('(')]
        lag_str = var_key[var_key.index('(')+1:var_key.index(')')]
        try:
            lag = int(lag_str)
        except ValueError:
            lag = 0
        return base_name.strip(), lag
    else:
        return var_key.strip(), 0


def evaluate_coefficient_expression(expr_str: str, params: Dict[str, Any]) -> float:
    """
    Evaluate a coefficient expression (could be numeric string, parameter name, etc.).
    Falls back to 1.0 if evaluation fails.
    """
    if expr_str is None or expr_str == "":
        return 1.0
    
    # Try direct numeric conversion
    try:
        return float(expr_str)
    except ValueError:
        pass
    
    # Try parameter lookup
    if expr_str in params:
        param_val = params[expr_str]
        try:
            if hasattr(param_val, '__float__'):
                return float(param_val)
            elif hasattr(param_val, 'item'):  # JAX/numpy scalar
                return float(param_val.item())
            else:
                return float(param_val)
        except (ValueError, TypeError):
            pass
    
    # Fallback
    print(f"Warning: Could not evaluate coefficient '{expr_str}', using 1.0")
    return 1.0


# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model,  # ReducedModel type
#     ss_builder,  # StateSpaceBuilder type  
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     FIXED: Robust reconstruction of original GPM variables from core states.
#     Handles nested non-core variable definitions using recursive resolution with memoization.
    
#     Args:
#         core_states_draw: Core state time series array of shape (T_data, state_dim)
#         gpm_model: The parsed GPM model containing variable definitions
#         ss_builder: State space builder with variable mappings
#         current_builder_params_draw: Current parameter values
#         T_data: Time series length
#         state_dim: State vector dimension
        
#     Returns:
#         Tuple of (reconstructed_trends, reconstructed_stationary) arrays
#     """
    
#     print(f"DEBUG: _reconstruct_original_variables called with T_data={T_data}, state_dim={state_dim}")
#     print(f"DEBUG: core_states_draw shape: {core_states_draw.shape}")
    
#     # Cache to store reconstructed variables (prevents infinite recursion)
#     reconstruction_cache: Dict[str, jnp.ndarray] = {}
    
#     # Extract core state values by name using the state space builder's mapping
#     core_var_map = ss_builder.core_var_map
#     current_draw_core_state_values = {}
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             current_draw_core_state_values[var_name] = core_states_draw[:, state_idx]
    
#     print(f"DEBUG: Core variables available: {list(current_draw_core_state_values.keys())}")
#     print(f"DEBUG: Model attributes: {[attr for attr in dir(gpm_model) if 'trend' in attr.lower() or 'stationary' in attr.lower()]}")
#     print(f"DEBUG: gpm_trend_variables_original exists: {hasattr(gpm_model, 'gpm_trend_variables_original')}")
#     if hasattr(gpm_model, 'gpm_trend_variables_original'):
#         print(f"DEBUG: gpm_trend_variables_original content: {gpm_model.gpm_trend_variables_original}")
#         print(f"DEBUG: Length: {len(gpm_model.gpm_trend_variables_original)}")
    
#     # Check if non_core_trend_definitions exists and what's in it
#     if hasattr(gpm_model, 'non_core_trend_definitions'):
#         print(f"DEBUG: non_core_trend_definitions exists with {len(gpm_model.non_core_trend_definitions)} definitions")
#         print(f"DEBUG: Non-core definitions: {list(gpm_model.non_core_trend_definitions.keys())}")
#     else:
#         print(f"DEBUG: non_core_trend_definitions attribute missing!")
    
#     def get_reconstructed_ts(variable_name: str) -> jnp.ndarray:
#         """
#         Recursively reconstructs a variable's time series, resolving all dependencies.
#         Uses memoization to avoid redundant calculations and infinite loops.
        
#         Args:
#             variable_name: Name of variable to reconstruct
            
#         Returns:
#             Time series array of shape (T_data,)
#         """
        
#         # 1. Check cache first (memoization)
#         if variable_name in reconstruction_cache:
#             return reconstruction_cache[variable_name]
        
#         # 2. Check if it's a core state variable (base case)
#         if variable_name in current_draw_core_state_values:
#             reconstruction_cache[variable_name] = current_draw_core_state_values[variable_name]
#             return current_draw_core_state_values[variable_name]
        
#         # 3. Check if it's a non-core variable with definition (recursive case)
#         if hasattr(gpm_model, 'non_core_trend_definitions') and variable_name in gpm_model.non_core_trend_definitions:
#             expr_def = gpm_model.non_core_trend_definitions[variable_name]
            
#             # Start with constant term
#             try:
#                 constant_val = evaluate_coefficient_expression(expr_def.constant_str, current_builder_params_draw)
#             except:
#                 constant_val = 0.0
                
#             reconstructed_ts = jnp.full(T_data, constant_val, dtype=_DEFAULT_DTYPE)
            
#             # Add contributions from each term in the definition
#             for var_key, coeff_str in expr_def.terms.items():
#                 try:
#                     term_var_name, lag = parse_variable_key(var_key)
                    
#                     # Skip lagged terms for now (would need time-series evaluation)
#                     if lag != 0:
#                         print(f"Warning: Lagged term '{var_key}' in definition of '{variable_name}' not supported. Skipping.")
#                         continue
                    
#                     # RECURSIVE CALL: Get the time series for this term's variable
#                     # This automatically resolves nested dependencies!
#                     term_variable_ts = get_reconstructed_ts(term_var_name)
                    
#                     # Evaluate coefficient
#                     try:
#                         coeff_val = evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
#                     except:
#                         coeff_val = 1.0
                    
#                     # Add this term's contribution
#                     reconstructed_ts += coeff_val * term_variable_ts
                    
#                 except Exception as e:
#                     print(f"Warning: Error processing term '{var_key}' in '{variable_name}': {e}. Skipping term.")
#                     continue
            
#             # Cache result and return
#             reconstruction_cache[variable_name] = reconstructed_ts
#             return reconstructed_ts
        
#         # 4. Variable not found - return zeros and warn
#         print(f"ERROR: Variable '{variable_name}' not found as core state or non-core definition. Returning zeros.")
#         print(f"DEBUG: Available core states: {list(current_draw_core_state_values.keys())}")
#         if hasattr(gpm_model, 'non_core_trend_definitions'):
#             print(f"DEBUG: Available non-core definitions: {list(gpm_model.non_core_trend_definitions.keys())}")
#         zeros_ts = jnp.zeros(T_data, dtype=_DEFAULT_DTYPE)
#         reconstruction_cache[variable_name] = zeros_ts
#         return zeros_ts
    
#     # Reconstruct all original trend variables
#     reconstructed_trends_list = []
#     if hasattr(gpm_model, 'gpm_trend_variables_original'):
#         for trend_name in gpm_model.gpm_trend_variables_original:
#             reconstructed_ts = get_reconstructed_ts(trend_name)
#             reconstructed_trends_list.append(reconstructed_ts)
    
#     # Reconstruct all original stationary variables (these should always be core)
#     reconstructed_stationary_list = []
#     if hasattr(gpm_model, 'gpm_stationary_variables_original'):
#         for stat_name in gpm_model.gpm_stationary_variables_original:
#             if stat_name in current_draw_core_state_values:
#                 reconstructed_stationary_list.append(current_draw_core_state_values[stat_name])
#             else:
#                 print(f"Warning: Stationary variable '{stat_name}' not found in core states. Using zeros.")
#                 reconstructed_stationary_list.append(jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
    
#     # Stack into final arrays
#     final_trends = jnp.stack(reconstructed_trends_list, axis=-1) if reconstructed_trends_list else jnp.empty((T_data, 0), dtype=_DEFAULT_DTYPE)
#     final_stationary = jnp.stack(reconstructed_stationary_list, axis=-1) if reconstructed_stationary_list else jnp.empty((T_data, 0), dtype=_DEFAULT_DTYPE)
    
#     return final_trends, final_stationary

# def _reconstruct_original_variables(
#     core_states_draw: jnp.ndarray,
#     gpm_model,
#     ss_builder,
#     current_builder_params_draw: Dict[str, Any],
#     T_data: int,
#     state_dim: int
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     SIMPLE FIXED VERSION: Direct reconstruction without recursion.
#     Handles your specific model structure explicitly.
#     """
    
#     print(f"DEBUG: Simple reconstruction called with T_data={T_data}, state_dim={state_dim}")
    
#     # Get core state values by name
#     core_var_map = ss_builder.core_var_map
#     core_states = {}
#     for var_name, state_idx in core_var_map.items():
#         if state_idx is not None and state_idx < state_dim:
#             core_states[var_name] = core_states_draw[:, state_idx]
    
#     print(f"DEBUG: Core states extracted: {list(core_states.keys())}")
    
#     # Your model structure (from the GPM file and debug output):
#     # Core trends: r_US_idio_trend, r_EA_idio_trend, r_JP_idio_trend, pi_US_idio_trend, pi_EA_idio_trend, pi_JP_idio_trend, y_US_trend, y_EA_trend, y_JP_trend
#     # Non-core trends: rr_US_full_trend, pi_US_full_trend, R_US_short_trend, rr_EA_full_trend, pi_EA_full_trend, R_EA_short_trend, rr_JP_full_trend, pi_JP_full_trend, R_JP_short_trend
    
#     # Expected from your GPM file:
#     # rr_US_full_trend = r_US_idio_trend
#     # pi_US_full_trend = pi_US_idio_trend  
#     # R_US_short_trend = rr_US_full_trend + pi_US_full_trend
    
#     reconstructed_vars = {}
    
#     # Start with core variables
#     reconstructed_vars.update(core_states)
    
#     # Reconstruct non-core variables in the right order
#     try:
#         # Level 1: Direct mappings from core variables
#         reconstructed_vars['rr_US_full_trend'] = core_states.get('r_US_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
#         reconstructed_vars['pi_US_full_trend'] = core_states.get('pi_US_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
#         reconstructed_vars['rr_EA_full_trend'] = core_states.get('r_EA_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
#         reconstructed_vars['pi_EA_full_trend'] = core_states.get('pi_EA_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
#         reconstructed_vars['rr_JP_full_trend'] = core_states.get('r_JP_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
#         reconstructed_vars['pi_JP_full_trend'] = core_states.get('pi_JP_idio_trend', jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
        
#         print(f"DEBUG: Level 1 non-core variables reconstructed")
        
#         # Level 2: Combinations of level 1 variables  
#         reconstructed_vars['R_US_short_trend'] = reconstructed_vars['rr_US_full_trend'] + reconstructed_vars['pi_US_full_trend']
#         reconstructed_vars['R_EA_short_trend'] = reconstructed_vars['rr_EA_full_trend'] + reconstructed_vars['pi_EA_full_trend']
#         reconstructed_vars['R_JP_short_trend'] = reconstructed_vars['rr_JP_full_trend'] + reconstructed_vars['pi_JP_full_trend']
        
#         print(f"DEBUG: Level 2 non-core variables reconstructed")
        
#     except Exception as e:
#         print(f"ERROR in non-core reconstruction: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Build final arrays in the order expected by gpm_trend_variables_original
#     trend_names = gpm_model.gpm_trend_variables_original
#     trends_list = []
    
#     print(f"DEBUG: Building final trend array for {len(trend_names)} variables")
#     for i, name in enumerate(trend_names):
#         if name in reconstructed_vars:
#             trends_list.append(reconstructed_vars[name])
#             print(f"DEBUG: Added {name} to trends (index {i})")
#         else:
#             print(f"ERROR: Missing trend variable {name}, using zeros")
#             trends_list.append(jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
    
#     # Build stationary array
#     stat_names = gpm_model.gpm_stationary_variables_original
#     stat_list = []
    
#     print(f"DEBUG: Building final stationary array for {len(stat_names)} variables")
#     for name in stat_names:
#         if name in reconstructed_vars:
#             stat_list.append(reconstructed_vars[name])
#             print(f"DEBUG: Added {name} to stationary")
#         else:
#             print(f"ERROR: Missing stationary variable {name}, using zeros")
#             stat_list.append(jnp.zeros(T_data, dtype=_DEFAULT_DTYPE))
    
#     # Stack arrays
#     final_trends = jnp.stack(trends_list, axis=-1) if trends_list else jnp.empty((T_data, 0), dtype=_DEFAULT_DTYPE)
#     final_stationary = jnp.stack(stat_list, axis=-1) if stat_list else jnp.empty((T_data, 0), dtype=_DEFAULT_DTYPE)
    
#     print(f"DEBUG: Final shapes - trends: {final_trends.shape}, stationary: {final_stationary.shape}")
    
#     return final_trends, final_stationary

def _reconstruct_original_variables(
    core_states_draw: jnp.ndarray,
    gpm_model,
    ss_builder,
    current_builder_params_draw: Dict[str, Any],
    T_data: int,
    state_dim: int
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    FIXED & ROBUST: Reconstructs all original GPM variables and returns them as dictionaries
    mapping variable names to their time series. This avoids any ambiguity about column ordering.

    Args:
        core_states_draw: Core state time series array of shape (T_data, state_dim)
        gpm_model: The parsed GPM model containing variable definitions
        ss_builder: State space builder with variable mappings
        current_builder_params_draw: Current parameter values
        T_data: Time series length
        state_dim: State vector dimension

    Returns:
        Tuple of (reconstructed_trends_dict, reconstructed_stationary_dict)
    """
    reconstructed_vars: Dict[str, jnp.ndarray] = {}

    # 1. Populate the dictionary with core state variables by name
    core_var_map = ss_builder.core_var_map
    for var_name, state_idx in core_var_map.items():
        if state_idx is not None and state_idx < state_dim:
            reconstructed_vars[var_name] = core_states_draw[:, state_idx]
            print(f"EXTRACTION: {var_name} from state index {state_idx}")
            print(f"  Data sample: {core_states_draw[:3, state_idx]}")
    # 2. Iteratively resolve and compute non-core trend variables
    # This loop handles nested dependencies (e.g., R_short_trend depends on rr_full_trend, which depends on r_idio_trend)
    # We iterate multiple times to ensure all dependencies are resolved.
    non_core_defs = gpm_model.non_core_trend_definitions
    unresolved_vars = set(non_core_defs.keys())
    
    # Using a simple iterative solver for dependencies
    for _ in range(len(unresolved_vars) + 1): # Iterate enough times for resolution
        resolved_this_pass = set()
        for var_name in unresolved_vars:
            expr_def = non_core_defs[var_name]
            
            # Check if all variables in the definition are already reconstructed
            dependencies = {term.split('(')[0].strip() for term in expr_def.terms.keys()}
            if dependencies.issubset(reconstructed_vars.keys()):
                # All dependencies met, we can compute this variable now
                
                # Start with the constant term
                reconstructed_ts = jnp.zeros(T_data, dtype=jnp.float64) # Default to zero
                if expr_def.constant_str and expr_def.constant_str != "0":
                     # Evaluation of constants should use the provided parameters
                     constant_val = ss_builder._evaluate_coefficient_expression(expr_def.constant_str, current_builder_params_draw)
                     reconstructed_ts += constant_val

                # Add contributions from each term in the definition
                for var_key, coeff_str in expr_def.terms.items():
                    term_var_name, lag = parse_variable_key(var_key)
                    
                    if lag != 0:
                        print(f"Warning: Lagged term '{var_key}' in definition of '{var_name}' not supported in reconstruction. Skipping.")
                        continue
                        
                    term_variable_ts = reconstructed_vars[term_var_name]
                    coeff_val = ss_builder._evaluate_coefficient_expression(coeff_str, current_builder_params_draw)
                    reconstructed_ts += coeff_val * term_variable_ts
                
                reconstructed_vars[var_name] = reconstructed_ts
                resolved_this_pass.add(var_name)

        unresolved_vars -= resolved_this_pass
        if not unresolved_vars:
            break # All resolved
            
    if unresolved_vars:
        print(f"Warning: Could not resolve all non-core trend definitions. Unresolved: {unresolved_vars}")

    # 3. Separate the final dictionary into trends and stationary components
    reconstructed_trends_dict = {name: reconstructed_vars[name] for name in gpm_model.gpm_trend_variables_original if name in reconstructed_vars}
    reconstructed_stationary_dict = {name: reconstructed_vars[name] for name in gpm_model.gpm_stationary_variables_original if name in reconstructed_vars}

    return reconstructed_trends_dict, reconstructed_stationary_dict
