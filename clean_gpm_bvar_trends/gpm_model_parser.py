# clean_gpm_bvar_trends/gpm_model_parser.py

import re
from typing import Dict, List, Tuple, Optional, Set, Any as TypingAny, Union # Added Union
from dataclasses import dataclass, field
import sympy as sp
import numpy as np # Added numpy import
import jax.numpy as jnp # Added jax.numpy import
from collections import defaultdict, deque

# --- Data Classes ---
@dataclass
class PriorSpec:
    name: str; distribution: str; params: List[float]
@dataclass
class VariableSpec:
    name: str; init_dist: Optional[str] = None; init_params: Optional[List[float]] = None
@dataclass
class ParsedTerm:
    variable: str; lag: int = 0; coefficient: Optional[str] = None
    sign: str = '+'; is_numeric_variable: bool = False
@dataclass
class ParsedEquation:
    lhs: str; rhs_terms: List[ParsedTerm]; shock: Optional[str] = None
@dataclass
class ReducedExpression:
    terms: Dict[str, str] = field(default_factory=dict)
    constant_str: str = "0"; parameters: Set[str] = field(default_factory=set)
@dataclass
class VarPriorSetup: # Ensure field order is compatible if defaults are mixed
    var_order: int
    es: List[float]
    fs: List[float]
    gs: List[float]
    hs: List[float]
    eta: float
@dataclass
class ReducedModel:
    core_variables: List[str]; core_equations: List[ParsedEquation]
    reduced_measurement_equations: Dict[str, ReducedExpression]
    stationary_variables: List[str]; parameters: List[str]
    estimated_params: Dict[str, PriorSpec]
    var_prior_setup: Optional[VarPriorSetup] = None
    gpm_trend_variables_original: List[str] = field(default_factory=list)
    gpm_stationary_variables_original: List[str] = field(default_factory=list)
    gpm_observed_variables_original: List[str] = field(default_factory=list)
    non_core_trend_definitions: Dict[str, ReducedExpression] = field(default_factory=dict)
    initial_values: Dict[str, VariableSpec] = field(default_factory=dict)
    trend_shocks: List[str] = field(default_factory=list)
    stationary_shocks: List[str] = field(default_factory=list)
    all_original_trend_equations: List[ParsedEquation] = field(default_factory=list)
    all_original_measurement_equations: List[ParsedEquation] = field(default_factory=list)

class SymbolicReducerUtils:
    def _is_numeric_string(self, s: Optional[str]) -> bool:
        if s is None: return False
        try: float(s); return True
        except (ValueError, TypeError): return False

    def parse_expression_to_terms(self, expr_str: str) -> List[ParsedTerm]:
        original_expr_for_warning = expr_str
        expr_str = expr_str.strip()
        if not expr_str: return []
        
        terms_found: List[ParsedTerm] = []

        try:
            potential_gpm_symbols = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*(?:\(\s*-\s*\d+\s*\))?", expr_str)) | \
                                    set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]+", expr_str))
            sympy_locals = {}
            gpm_to_sympy_map = {}; sympy_to_gpm_map = {}
            processed_expr_for_sympy = expr_str

            for i, gpm_sym_name in enumerate(sorted(list(potential_gpm_symbols), key=len, reverse=True)):
                sympy_safe_name = gpm_sym_name; original_name_for_map = gpm_sym_name; lag = 0
                lag_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_.]+)\s*\(\s*-\s*(\d+)\s*\)", gpm_sym_name)
                if lag_match:
                    base_name = lag_match.group(1); lag_val = int(lag_match.group(2))
                    sympy_safe_name = f"{base_name}_L{lag_val}"; original_name_for_map = base_name; lag = lag_val
                if sympy_safe_name not in sympy_locals:
                    sympy_locals[sympy_safe_name] = sp.symbols(sympy_safe_name)
                    gpm_to_sympy_map[gpm_sym_name] = sympy_safe_name
                    sympy_to_gpm_map[sympy_safe_name] = (original_name_for_map, lag)
                processed_expr_for_sympy = re.sub(r'(?<![a-zA-Z0-9_.])' + re.escape(gpm_sym_name) + r'(?![a-zA-Z0-9_.])', sympy_safe_name, processed_expr_for_sympy)


            s_expr = sp.sympify(processed_expr_for_sympy, locals=sympy_locals)
            s_expr_expanded = sp.expand(s_expr)

            expanded_sympy_terms_list = [s_expr_expanded] if not isinstance(s_expr_expanded, sp.Add) else list(s_expr_expanded.args)

            for sympy_term in expanded_sympy_terms_list:
                sign = '+'
                term_body = sympy_term
                num_coeff, rest_factors = sympy_term.as_coeff_Mul() # Separates numeric coefficient
                
                if num_coeff.is_Number and num_coeff < 0:
                    sign = '-'
                    if num_coeff == -1: term_body = rest_factors
                    else: term_body = -num_coeff * rest_factors # Make rest positive
                
                # Now term_body is positive. Deconstruct it into coefficient_str and variable_str
                term_coeff_parts = []
                term_var_name_parts = []

                if isinstance(term_body, sp.Mul):
                    for factor in term_body.args:
                        if factor.is_Number: term_coeff_parts.append(str(factor))
                        # Check if it's a symbol we mapped (parameter or variable base name)
                        elif str(factor) in sympy_locals : term_var_name_parts.append(str(factor)) 
                        else: term_coeff_parts.append(str(factor)) # Treat other sympy objects as part of coeff
                elif term_body.is_Symbol or str(term_body) in sympy_locals:
                    term_var_name_parts.append(str(term_body))
                elif term_body.is_Number:
                    term_coeff_parts.append(str(term_body))
                else: # Fallback for complex unhandled sympy structures
                    term_var_name_parts.append(str(term_body)) # Treat whole thing as variable part

                # Build final coefficient string and variable string for ParsedTerm
                final_coeff_str = "*".join(term_coeff_parts) if term_coeff_parts else None
                
                # Reconstruct GPM variable name (potentially with lag)
                # This part is tricky if var_name_parts has multiple elements (e.g. param*var_L1)
                # For now, assume var_name_parts ideally contains one sympy_safe_name
                parsed_var_name = "1"; parsed_lag = 0; is_numeric = False
                
                if not term_var_name_parts: # Only coefficients, so it's a constant term
                    if final_coeff_str is None: final_coeff_str = "0" # Should not happen if term_body existed
                    parsed_var_name = "1"; is_numeric = True # Placeholder for constant
                elif len(term_var_name_parts) == 1:
                    sym_var_name_part = term_var_name_parts[0]
                    if sym_var_name_part in sympy_to_gpm_map:
                        parsed_var_name, parsed_lag = sympy_to_gpm_map[sym_var_name_part]
                    else: # Not a mapped symbol (e.g. a parameter like 'var_phi')
                         # If no coeff_str yet, this symbol IS the coeff_str, var is "1"
                        if final_coeff_str is None:
                            final_coeff_str = sym_var_name_part
                            parsed_var_name = "1" # Becomes a "constant" term whose value is param
                            is_numeric = True # In the sense that it's not a state variable
                        else: # Already have a coeff, this is another coeff factor
                            final_coeff_str = self._multiply_coefficients(final_coeff_str, sym_var_name_part)
                            parsed_var_name = "1"
                            is_numeric = True

                else: # Multiple variable parts (e.g. X_L1 * Y_L0), treat as complex variable name
                    # This case is unlikely if sympy.expand worked well and GPM is sum-of-products-of-one-var.
                    # For now, join them and map back the first one found.
                    # print(f"Warning: Multiple var parts in term: {term_var_name_parts} from '{original_expr_for_warning}'. Using first.")
                    first_sym_var = term_var_name_parts[0]
                    if first_sym_var in sympy_to_gpm_map:
                         parsed_var_name, parsed_lag = sympy_to_gpm_map[first_sym_var]
                    else: parsed_var_name = first_sym_var # Keep as is
                    # Other var parts become part of coefficient
                    for rest_v_part in term_var_name_parts[1:]:
                        final_coeff_str = self._multiply_coefficients(final_coeff_str, rest_v_part)


                # If no coefficient was found but var is "1", it means the term was a number
                if final_coeff_str is None and parsed_var_name == "1" and is_numeric:
                    pass # This should be handled by constant accumulation later if it was just a number
                elif final_coeff_str is None and parsed_var_name == "1" and not is_numeric : # Should not happen
                    final_coeff_str = "1" # Term was just "1"
                
                if parsed_var_name == "1" and final_coeff_str is not None: is_numeric = True

                terms_found.append(ParsedTerm(variable=parsed_var_name, lag=parsed_lag, 
                                              coefficient=final_coeff_str, sign=sign, 
                                              is_numeric_variable=is_numeric))
            return terms_found
        except Exception as e_sympy:
            # print(f"Sympy processing failed for '{original_expr_for_warning}': {e_sympy}. Using fallback string split.")
            # Fallback using previous string splitting logic
            expr_str_spaced = re.sub(r"(?<=[a-zA-Z0-9_)])\s*([+-])\s*(?=[a-zA-Z0-9_(])", r" \1 ", original_expr_for_warning)
            expr_str_spaced = re.sub(r"\s+", " ", expr_str_spaced).strip(); current_sign_fallback = '+'
            if expr_str_spaced.startswith("+ "): expr_str_spaced = expr_str_spaced[2:].strip()
            elif expr_str_spaced.startswith("- "): current_sign_fallback = '-'; expr_str_spaced = expr_str_spaced[2:].strip()
            components_fb = re.split(r"\s+(?=[+-])", expr_str_spaced) # Split on space before + or -
            if not components_fb[0] and len(components_fb)>0: components_fb = components_fb[1:] if components_fb[0]=='' else components_fb
            
            ops_fb = [current_sign_fallback] + re.findall(r"\s*([+-])\s+", expr_str_spaced)
            if len(ops_fb) < len(components_fb) and not (original_expr_for_warning.strip().startswith(("+", "-"))): ops_fb = ['+'] + ops_fb

            for i, term_content_fb in enumerate(components_fb):
                sign_fb = ops_fb[i] if i < len(ops_fb) else '+'; cleaned_content_fb = term_content_fb.strip()
                if cleaned_content_fb.startswith('+'): cleaned_content_fb = cleaned_content_fb[1:].strip()
                elif cleaned_content_fb.startswith('-'): sign_fb = '-' if sign_fb == '+' else '+'; cleaned_content_fb = cleaned_content_fb[1:].strip()
                if cleaned_content_fb:
                    parsed_fb = self._parse_single_term_component(cleaned_content_fb, sign_fb)
                    if parsed_fb: terms_found.append(parsed_fb)
            return terms_found

    def _parse_single_term_component(self, term_str: str, sign: str) -> Optional[ParsedTerm]:
        # (As before - robust version)
        term_str = term_str.strip(); coeff_str: Optional[str] = None; var_part_str: str = term_str
        is_numeric_const_origin = False
        if not term_str: return None
        multiply_split = term_str.split('*', 1)
        if len(multiply_split) > 1:
            potential_coeff = multiply_split[0].strip()
            potential_var_or_num = multiply_split[1].strip()
            if (re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", potential_coeff) or self._is_numeric_string(potential_coeff)):
                coeff_str = potential_coeff; var_part_str = potential_var_or_num
        var_name_from_part = var_part_str; lag = 0
        is_numeric_var_part = self._is_numeric_string(var_part_str)
        if not is_numeric_var_part:
            lag_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_.]+)\s*\(\s*-\s*(\d+)\s*\)", var_part_str)
            if lag_match: var_name_from_part = lag_match.group(1); lag = int(lag_match.group(2))
            elif not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", var_part_str): pass
        if is_numeric_var_part:
            is_numeric_const_origin = True
            if coeff_str: coeff_str = self._multiply_coefficients(coeff_str, var_part_str)
            else: coeff_str = var_part_str
            var_name_from_part = "1"
        return ParsedTerm(variable=var_name_from_part, lag=lag, coefficient=coeff_str, sign=sign, is_numeric_variable=is_numeric_const_origin)

    def substitute_expression(self, expr: ReducedExpression, substitution_rules: Dict[str, ReducedExpression], all_gpm_parameters: Set[str]) -> ReducedExpression: # As before
        final_terms: Dict[str, str] = defaultdict(lambda: "0"); current_constant_str = expr.constant_str
        final_parameters = set(expr.parameters); queue = deque(list(expr.terms.items()))
        iterations = 0; max_iterations = len(substitution_rules) * 3 + len(expr.terms) + 10 # Increased max_iter slightly
        while queue and iterations < max_iterations:
            iterations += 1; var_key, coeff_expr_str = queue.popleft()
            var_name_for_rule, lag_for_rule = self._parse_var_key_for_rules(var_key)
            if var_name_for_rule in substitution_rules and lag_for_rule == 0 :
                sub_def = substitution_rules[var_name_for_rule]; final_parameters.update(sub_def.parameters)
                for sub_var_key, sub_coeff_expr_str in sub_def.terms.items():
                    combined_coeff_str = self._multiply_coefficients(coeff_expr_str, sub_coeff_expr_str)
                    queue.append((sub_var_key, combined_coeff_str)) 
                if sub_def.constant_str != "0":
                    term_from_const = self._multiply_coefficients(coeff_expr_str, sub_def.constant_str)
                    current_constant_str = self._add_coefficients(current_constant_str, term_from_const)
            else: 
                final_terms[var_key] = self._add_coefficients(final_terms[var_key], coeff_expr_str)
                final_parameters.update(self._extract_params_from_coeff_str(coeff_expr_str, all_gpm_parameters))
        if queue: 
            while queue: var_key, coeff_expr_str = queue.popleft(); final_terms[var_key] = self._add_coefficients(final_terms[var_key], coeff_expr_str)
        cleaned_final_terms = {k: v for k, v in final_terms.items() if v != "0" and v is not None}
        try: simplified_constant = str(sp.N(sp.sympify(current_constant_str)))
        except: simplified_constant = current_constant_str if current_constant_str else "0"
        if simplified_constant != "0": final_parameters.update(self._extract_params_from_coeff_str(simplified_constant, all_gpm_parameters))
        return ReducedExpression(terms=cleaned_final_terms, constant_str=simplified_constant, parameters=final_parameters)

    def _parse_var_key_for_rules(self, var_key: str) -> Tuple[str, int]: # As before
        match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_.]+)\s*(?:\(\s*-\s*(\d+)\s*\))?", var_key)
        if match: var_name = match.group(1); lag_str = match.group(2); return var_name, int(lag_str) if lag_str else 0
        return var_key, 0

    def _multiply_coefficients(self, c1_str: Optional[str], c2_str: Optional[str]) -> str: # As before
        c1 = c1_str if c1_str is not None else "1"; c2 = c2_str if c2_str is not None else "1"
        if c1 == "0" or c2 == "0": return "0"
        if c1 == "1": return c2; 
        if c2 == "1": return c1
        try: return str(sp.N(sp.sympify(f"({c1}) * ({c2})")))
        except: pass
        c1_neg = c1.startswith('-'); c2_neg = c2.startswith('-')
        c1_abs = c1[1:].strip("()") if c1_neg else c1.strip("()")
        c2_abs = c2[1:].strip("()") if c2_neg else c2.strip("()")
        c1_abs = "1" if not c1_abs else c1_abs; c2_abs = "1" if not c2_abs else c2_abs
        if c1_abs == "1" and c2_abs == "1": prod = "1"
        elif c1_abs == "1": prod = c2_abs
        elif c2_abs == "1": prod = c1_abs
        else: prod = f"({c1_abs})*({c2_abs})"
        return prod if c1_neg == c2_neg else (f"-{prod}" if prod != "0" else "0")

    def _add_coefficients(self, c1_str: Optional[str], c2_str: Optional[str]) -> str: # As before
        c1 = c1_str if c1_str is not None else "0"; c2 = c2_str if c2_str is not None else "0"
        if c1 == "0": return c2; 
        if c2 == "0": return c1
        try: return str(sp.N(sp.sympify(f"({c1}) + ({c2})")))
        except: pass
        c2_eff = c2.strip("()")
        if c2_eff.startswith('-'): return f"{c1} - {c2_eff[1:]}"
        return f"{c1} + {c2_eff}"

    def _extract_params_from_coeff_str(self, coeff_str: Optional[str], all_gpm_params: Set[str]) -> Set[str]: # As before
        found = set()
        if coeff_str and coeff_str not in ["0", "1", "-1"] and not self._is_numeric_string(coeff_str):
            for p in all_gpm_params:
                if re.search(r'\b' + re.escape(p) + r'\b', coeff_str): found.add(p)
        return found

    def evaluate_numeric_expression(self, expr_str: Optional[str], params: Dict[str, Union[float, jnp.ndarray]]) -> float:
        if expr_str is None:
            # Or handle as 0.0 if that's more appropriate for coefficients
            raise ValueError("Cannot evaluate None expression.")
        if not isinstance(expr_str, str):
             # It might already be a number if parsing was very thorough
             if isinstance(expr_str, (float, int, np.number)): # np.number used here
                 return float(expr_str)
             raise ValueError(f"Expression must be a string, got {type(expr_str)}")

        # If it's a simple numeric string, convert directly
        if self._is_numeric_string(expr_str):
            return float(expr_str)

        # Prepare substitutions for sympy: sympy expects float values, not jax arrays.
        # And symbols in expr_str must match keys in params.
        subs_dict = {}
        # Identify symbols in the expression string
        # This is a simple way; a more robust way would be to parse symbols from expr_str
        # For now, assume params contains all necessary symbols as keys.
        for p_name, p_val in params.items():
            if hasattr(p_val, 'item'): # Convert JAX array scalar to float
                subs_dict[sp.symbols(p_name)] = float(p_val.item())
            elif isinstance(p_val, (float, int, np.number)): # np.number used here
                subs_dict[sp.symbols(p_name)] = float(p_val)
            else:
                # This case should ideally not happen if params are resolved numbers
                raise ValueError(f"Parameter '{p_name}' has unhandled type '{type(p_val)}' for sympy substitution.")

        try:
            # Sympify the expression string
            sym_expr = sp.sympify(expr_str)

            # Substitute parameter values
            # Filter subs_dict to only include symbols actually present in the expression
            relevant_subs = {s: v for s, v in subs_dict.items() if s in sym_expr.free_symbols}
            numeric_val = sym_expr.subs(relevant_subs)

            # Evaluate to a float
            # Ensure it's fully numerical before evalf
            if not numeric_val.is_Number:
                # This means some symbols were not substituted.
                # Check if they are GPM parameters that were not in `params` dict.
                unresolved_symbols = numeric_val.free_symbols
                # For now, raise error if not fully numeric after substitution
                raise ValueError(f"Expression '{expr_str}' could not be fully resolved to a number. Unresolved: {unresolved_symbols}")

            return float(numeric_val.evalf())
        except (sp.SympifyError, AttributeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}' with params {list(params.keys())}: {e}") from e

class GPMModelParser: # Main Parser Class
    def __init__(self):
        self.utils = SymbolicReducerUtils()
        self.model_data: Dict[str, TypingAny] = {}

    def parse_file(self, filepath: str) -> ReducedModel: # As before
        try:
            with open(filepath, 'r') as file: content = file.read()
            return self.parse_content(content)
        except FileNotFoundError: print(f"Error: GPM file not found: {filepath}"); raise
        except Exception as e: print(f"Error parsing GPM file {filepath}: {e}"); raise


    def _extract_basic_sections(self, content: str): # As before
        content = re.sub(r'//.*$|#.*$', '', content, flags=re.MULTILINE)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        self.model_data = {
            'parameters': [], 'estimated_params': {}, 'trend_variables': [], 
            'stationary_variables': [], 'observed_variables': [], 'trend_shocks': [], 
            'stationary_shocks': [], 'trend_equations': [], 'measurement_equations': [],
            'initial_values': {}, 'var_prior_setup': None,
            'gpm_trend_variables_original': [], 'gpm_stationary_variables_original': [],
            'gpm_observed_variables_original': []
        }
        idx = 0
        while idx < len(lines):
            line = lines[idx]; s_idx = idx; low_line = line.lower(); matched = False
            if not line: idx += 1; continue
            if low_line.startswith('parameters'): idx = self._parse_parameters(lines, s_idx); matched=True
            elif low_line.startswith('estimated_params'): idx = self._parse_estimated_params(lines, s_idx); matched=True
            elif low_line.startswith('trends_vars'): idx = self._parse_variable_list(lines, s_idx, 'trend_variables', 'trends_vars'); self.model_data['gpm_trend_variables_original'] = list(self.model_data['trend_variables']); matched=True
            elif low_line.startswith('stationary_variables'): idx = self._parse_variable_list(lines, s_idx, 'stationary_variables', 'stationary_variables'); self.model_data['gpm_stationary_variables_original'] = list(self.model_data['stationary_variables']); matched=True
            elif low_line.startswith('trend_shocks'): idx = self._parse_shock_list(lines, s_idx, 'trend_shocks'); matched=True
            elif low_line.startswith('shocks'): idx = self._parse_shock_list(lines, s_idx, 'stationary_shocks'); matched=True
            elif low_line.startswith('trend_model'): idx = self._parse_equations(lines, s_idx, 'trend_equations', self._parse_trend_equation_details); matched=True
            elif low_line.startswith('varobs'): idx = self._parse_variable_list(lines, s_idx, 'observed_variables', 'varobs'); self.model_data['gpm_observed_variables_original'] = list(self.model_data['observed_variables']); matched=True
            elif low_line.startswith('measurement_equations'): idx = self._parse_equations(lines, s_idx, 'measurement_equations', self._parse_measurement_equation_details); matched=True
            elif low_line.startswith('initval'): idx = self._parse_initial_values(lines, s_idx); matched=True
            elif low_line.startswith('var_prior_setup'): idx = self._parse_var_prior_setup(lines, s_idx); matched=True
            if not matched: idx += 1 # Advance if no keyword matched
            elif idx == s_idx and matched : idx +=1 # Ensure progress if sub-parser didn't move but claimed match

    def _parse_parameters(self, lines: List[str], start_idx: int) -> int: # As before
        current_line_idx = start_idx; parts_acc = []; first = True; term = False
        while current_line_idx < len(lines):
            content = lines[current_line_idx].split('//')[0].split('#')[0].strip()
            if first: content = re.sub(r'parameters\s*', '', content, flags=re.IGNORECASE); first = False
            if ';' in content: parts_acc.append(content.split(';',1)[0].strip()); term = True; current_line_idx += 1; break
            if content: parts_acc.append(content)
            current_line_idx += 1
        if not term and "".join(parts_acc).strip(): print(f"Warning: 'parameters' at line {start_idx+1} ended prematurely.")
        full_str = " ".join(filter(None, parts_acc))
        if full_str: self.model_data['parameters'].extend(p.strip() for p in full_str.split(',') if p.strip())
        return current_line_idx

    def _parse_estimated_params(self, lines: List[str], start_idx: int) -> int: # As before
        i = start_idx + 1
        while i < len(lines):
            line = lines[i].split('//')[0].split('#')[0].strip()
            if not line: i+=1; continue
            if line.lower().startswith('end;'): i+=1; break
            self._parse_prior_specification(line); i += 1
        return i

    def _parse_prior_specification(self, line: str): # As before
        line = line.rstrip(';'); parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            name_part = parts[0]; name = name_part.split()[-1]; dist = parts[1]
            try:
                d_params = [float(p) for p in parts[2:]]
                self.model_data['estimated_params'][name] = PriorSpec(name, dist, d_params)
            except ValueError: print(f"Warning: Could not parse params for prior: '{line}'")

    def _parse_variable_list(self, lines: List[str], start_idx: int, key: str, keyword: str) -> int: # As before
        i = start_idx; parts_acc = []; first = True; term = False
        while i < len(lines):
            content = lines[i].split('//')[0].split('#')[0].strip()
            if first: content = re.sub(fr'{keyword}\s*', '', content, flags=re.IGNORECASE); first = False
            if ';' in content: parts_acc.append(content.split(';',1)[0].strip()); term = True; i += 1; break
            if content: parts_acc.append(content)
            i += 1
        if not term and "".join(parts_acc).strip(): print(f"Warning: '{keyword}' at line {start_idx+1} ended prematurely.")
        full_str = " ".join(filter(None, parts_acc))
        if full_str:
            temp_vars = []; 
            for c_part in full_str.split(','): temp_vars.extend(s.strip() for s in c_part.split() if s.strip())
            self.model_data[key] = [v for v in temp_vars if v]
        return i

    def _parse_shock_list(self, lines: List[str], start_idx: int, key: str) -> int: # As before
        shocks_found = []; i = start_idx + 1
        while i < len(lines):
            content = lines[i].split('//')[0].split('#')[0].strip()
            if not content: i += 1; continue
            if content.lower() == 'end;': i += 1; break
            clean_line = re.sub(r'^var\s+', '', content, flags=re.IGNORECASE).strip()
            if clean_line:
                p_shocks = re.split(r'[\s,]+', clean_line)
                for s_cand in p_shocks:
                    s = s_cand.strip().rstrip(';').strip()
                    if s: shocks_found.append(s)
            i += 1
        else: print(f"Warning: Shock section '{key}' at line {start_idx+1} did not find 'end;'.")
        self.model_data[key] = shocks_found
        return i
        
    def _parse_equations(self, lines: List[str], start_idx: int, key: str, detail_parser_func) -> int: # As before
        eqs = []; i = start_idx + 1
        while i < len(lines):
            line = lines[i].split('//')[0].split('#')[0].strip()
            if not line: i+=1; continue
            if line.lower().startswith('end;'): i+=1; break
            if '=' in line:
                parsed = detail_parser_func(line)
                if parsed: eqs.append(parsed)
            elif line: print(f"Warning: Skipping unexpected line in '{key}' section at line {i+1}: '{line}'")
            i += 1
        self.model_data[key] = eqs
        return i

    def _parse_initial_values(self, lines: List[str], start_idx: int) -> int: # As before
        i = start_idx + 1
        while i < len(lines):
            line = lines[i].split('//')[0].split('#')[0].strip()
            if not line: i+=1; continue
            if line.lower().startswith('end;'): i+=1; break
            parts = [p.strip() for p in line.rstrip(';').split(',')]
            if len(parts) >= 3:
                var_n, dist_n = parts[0], parts[1]
                try:
                    d_params = [float(p) for p in parts[2:]]
                    self.model_data['initial_values'][var_n] = VariableSpec(var_n, dist_n, d_params)
                except ValueError: print(f"Warning: Could not parse params for initval: '{line}'")
            elif line : print(f"Warning: Malformed initval line: '{line}'")
            i+=1
        return i

    def _parse_var_prior_setup(self, lines: List[str], start_idx: int) -> int: # As before
        i = start_idx + 1; data = {}; current_block_lines = []
        while i < len(lines):
            line_content = lines[i].split('//')[0].split('#')[0].strip()
            if not line_content: i+=1; continue
            if line_content.lower().startswith('end;'): i+=1; break
            current_block_lines.append(line_content); i += 1
        full_block_str = " ".join(current_block_lines)
        parts = [p.strip() for p in full_block_str.split(';') if p.strip()]
        for item in parts:
            if '=' in item:
                key_s, val_s = item.split('=', 1); key = key_s.strip().lower(); val_s = val_s.strip()
                try:
                    if ',' in val_s: data[key] = [float(pv.strip()) for pv in val_s.split(',')]
                    elif key == 'var_order': data[key] = int(val_s)
                    else: data[key] = float(val_s)
                except ValueError: print(f"Warning: Could not parse VAR prior part: '{item}'")
        if data:
             self.model_data['var_prior_setup'] = VarPriorSetup(
                var_order=data.get('var_order', 1), es=data.get('es', [0.0, 0.0]),
                fs=data.get('fs', [1.0, 1.0]), gs=data.get('gs', [1.0, 1.0]),
                hs=data.get('hs', [1.0, 1.0]), eta=data.get('eta', 1.0))
        return i

    # def _identify_core_variables(self) -> List[str]: # As refined
    #     core_vars = set(self.model_data.get('stationary_variables', []))
    #     trend_equations_parsed = self.model_data.get('trend_equations', [])
    #     declared_trend_shocks = set(self.model_data.get('trend_shocks', []))
    #     all_potential_trends = set(self.model_data.get('gpm_trend_variables_original', []))
    #     for eq in trend_equations_parsed: all_potential_trends.add(eq.lhs)
    #     for var_name in all_potential_trends:
    #         if var_name in core_vars: continue 
    #         defining_equation = next((eq for eq in trend_equations_parsed if eq.lhs == var_name), None)
    #         if defining_equation:
    #             if defining_equation.shock and defining_equation.shock in declared_trend_shocks:
    #                 core_vars.add(var_name); continue 
    #             for term in defining_equation.rhs_terms:
    #                 if term.variable == var_name and term.lag > 0:
    #                     core_vars.add(var_name); break 
    #     return sorted(list(core_vars))

    def _identify_core_variables(self) -> List[str]:
        """
        Identifies core variables and returns them in a GPM-defined, stable order.
        FIXED: The returned list is ordered based on original GPM declarations,
        not alphabetically, to ensure system-wide consistency.
        """
        # --- Step 1: Identify the SET of core variables (logic is correct) ---
        core_vars_set = set(self.model_data.get('stationary_variables', []))
        trend_equations_parsed = self.model_data.get('trend_equations', [])
        declared_trend_shocks = set(self.model_data.get('trend_shocks', []))
        
        # Add dynamic trends to the core set
        for eq in trend_equations_parsed:
            is_dynamic_by_shock = eq.shock and eq.shock in declared_trend_shocks
            is_dynamic_by_autoregression = any(term.variable == eq.lhs and term.lag > 0 for term in eq.rhs_terms)
            
            if is_dynamic_by_shock or is_dynamic_by_autoregression:
                core_vars_set.add(eq.lhs)

        # --- Step 2: Build the final ORDERED list of core variables ---
        final_ordered_core_vars = []
        
        # First, add the core trend variables, maintaining the order from the 'trends_vars' block
        gpm_original_trends = self.model_data.get('gpm_trend_variables_original', [])
        for var in gpm_original_trends:
            if var in core_vars_set:
                final_ordered_core_vars.append(var)
        
        # Then, add the stationary variables, maintaining the order from the 'stationary_variables' block
        gpm_original_stationary = self.model_data.get('gpm_stationary_variables_original', [])
        for var in gpm_original_stationary:
            if var in core_vars_set:
                final_ordered_core_vars.append(var)

        # Safeguard: Add any core variables identified that weren't in the original lists.
        # This prevents variables from being dropped but may indicate a GPM file issue.
        if len(final_ordered_core_vars) != len(core_vars_set):
            for var in core_vars_set:
                if var not in final_ordered_core_vars:
                    print(f"Warning (Parser): Core variable '{var}' was identified but not found in original 'trends_vars' or 'stationary_variables' lists. Appending to the end.")
                    final_ordered_core_vars.append(var)
                    
        return final_ordered_core_vars
    
    def _parsed_equation_to_reduced_expression(self, equation: ParsedEquation, all_gpm_params: Set[str]) -> ReducedExpression:
        # (As refined for constant accumulation)
        terms_dict: Dict[str, str] = defaultdict(lambda: "0")
        constant_accumulator_str = "0"; parameters_in_expr: Set[str] = set()
        for term in equation.rhs_terms:
            current_term_coeff_str = term.coefficient if term.coefficient is not None else "1"
            if term.sign == '-': current_term_coeff_str = self.utils._multiply_coefficients("-1", current_term_coeff_str)
            if term.coefficient: parameters_in_expr.update(self.utils._extract_params_from_coeff_str(term.coefficient, all_gpm_params))
            if term.is_numeric_variable and term.variable == "1":
                constant_accumulator_str = self.utils._add_coefficients(constant_accumulator_str, current_term_coeff_str)
            else:
                var_key = term.variable
                if term.lag > 0: var_key = f"{term.variable}(-{term.lag})"
                terms_dict[var_key] = self.utils._add_coefficients(terms_dict[var_key], current_term_coeff_str)
        final_terms = {k: v for k, v in terms_dict.items() if v != "0" and v is not None}
        try: final_constant_str = str(sp.N(sp.sympify(constant_accumulator_str)))
        except: final_constant_str = constant_accumulator_str if constant_accumulator_str else "0"
        if final_constant_str != "0": parameters_in_expr.update(self.utils._extract_params_from_coeff_str(final_constant_str, all_gpm_params))
        return ReducedExpression(terms=final_terms, constant_str=final_constant_str, parameters=parameters_in_expr)

    def _build_definitions_for_non_core_trends(self, core_variables: List[str], all_gpm_params: Set[str]) -> Dict[str, ReducedExpression]:
        # (As before, iterative substitution using self.utils.substitute_expression)
        non_core_defs: Dict[str, ReducedExpression] = {}; core_set = set(core_variables)
        original_trend_eq_dict: Dict[str, ParsedEquation] = {eq.lhs: eq for eq in self.model_data.get('trend_equations', [])}
        candidate_non_core_names = [
            name for name in self.model_data.get('gpm_trend_variables_original', [])
            if name not in core_set and name in original_trend_eq_dict]
        current_rules: Dict[str, ReducedExpression] = {}
        for trend_name in candidate_non_core_names:
            eq = original_trend_eq_dict[trend_name]
            if not eq.shock and not any(term.variable == eq.lhs and term.lag > 0 for term in eq.rhs_terms):
                current_rules[trend_name] = self._parsed_equation_to_reduced_expression(eq, all_gpm_params)
        for _ in range(len(current_rules) + 5): # Iterate enough for multi-level substitutions
            changed = False
            new_pass_rules = {} 
            for name, expr in current_rules.items():
                resolved_expr = self.utils.substitute_expression(expr, current_rules, all_gpm_params)
                new_pass_rules[name] = resolved_expr
                if resolved_expr.terms != expr.terms or resolved_expr.constant_str != expr.constant_str or resolved_expr.parameters != expr.parameters:
                    changed = True
            current_rules = new_pass_rules
            if not changed: break
        for name, expr in current_rules.items(): # Final check for full reduction
            is_fully_reduced = True
            for term_var_key in expr.terms.keys():
                term_var_name, lag = self.utils._parse_var_key_for_rules(term_var_key)
                # A fully reduced term for non-core def should refer to a core var (any lag) or be a param/const
                if not (term_var_name in core_set or \
                        term_var_name in all_gpm_params or \
                        self.utils._is_numeric_string(term_var_name) or \
                        term_var_name == "1"):
                    # print(f"Debug: Non-core def '{name}', term '{term_var_key}' not fully reduced (var '{term_var_name}' not core/param).")
                    is_fully_reduced = False; break
            if is_fully_reduced: non_core_defs[name] = expr
            # else: print(f"Info: Definition for non-core trend '{name}' couldn't be fully reduced to core: {expr.terms}, const: {expr.constant_str}")
        return non_core_defs

    def _reduce_measurement_equations(self, non_core_trend_definitions: Dict[str, ReducedExpression], all_gpm_params: Set[str]) -> Dict[str, ReducedExpression]:
        # (As before, ensuring MEs refer to current core/stationary states)
        reduced_mes: Dict[str, ReducedExpression] = {}
        core_vars_set = set(self.model_data.get('core_variables',[]))
        stat_vars_set = set(self.model_data.get('stationary_variables',[]))

        for eq in self.model_data.get('measurement_equations', []):
            initial_expr = self._parsed_equation_to_reduced_expression(eq, all_gpm_params)
            substituted_expr = self.utils.substitute_expression(initial_expr, non_core_trend_definitions, all_gpm_params)
            final_terms_for_ME: Dict[str,str] = {}; final_constant_for_ME = substituted_expr.constant_str
            final_params_for_ME = set(substituted_expr.parameters)

            for var_key, coeff_str in substituted_expr.terms.items():
                var_name, lag = self.utils._parse_var_key_for_rules(var_key)
                # ME terms must be current period core trends or current period stationary variables
                if lag == 0 and (var_name in core_vars_set or var_name in stat_vars_set):
                    final_terms_for_ME[var_name] = self.utils._add_coefficients(final_terms_for_ME.get(var_name, "0"), coeff_str)
                elif self.utils._is_numeric_string(var_name) and var_name == "1": # Term was a constant
                    final_constant_for_ME = self.utils._add_coefficients(final_constant_for_ME, coeff_str)
                elif var_name in all_gpm_params: # A parameter might appear as a "variable" if not a coefficient
                     final_terms_for_ME[var_name] = self.utils._add_coefficients(final_terms_for_ME.get(var_name, "0"), coeff_str)
                else: 
                    # print(f"Warning: ME for '{eq.lhs}' term '{coeff_str}*{var_key}' not reduced to current core/stat state. Kept as is.")
                    final_terms_for_ME[var_key] = self.utils._add_coefficients(final_terms_for_ME.get(var_key, "0"), coeff_str)
            
            cleaned_final_terms = {k:v for k,v in final_terms_for_ME.items() if v != "0"}
            try: final_constant_for_ME = str(sp.N(sp.sympify(final_constant_for_ME)))
            except: pass
            if final_constant_for_ME != "0": final_params_for_ME.update(self.utils._extract_params_from_coeff_str(final_constant_for_ME, all_gpm_params))
            reduced_mes[eq.lhs] = ReducedExpression(terms=cleaned_final_terms, constant_str=final_constant_for_ME, parameters=final_params_for_ME)
        return reduced_mes

    def _extract_core_equations(self, core_variables: List[str]) -> List[ParsedEquation]: # As before
        core_eqs = []; core_set = set(core_variables)
        for eq in self.model_data.get('trend_equations', []):
            if eq.lhs in core_set: core_eqs.append(eq)
        return core_eqs

    def _parse_trend_equation_details(self, line: str) -> Optional[ParsedEquation]:
        """
        Parse trend equation details. 
        
        Note: Shock identification is deferred to post-processing since
        trend_shocks block might not be parsed yet when this is called.
        """
        line = line.rstrip(';')
        lhs_s, rhs_s = line.split('=', 1)
        lhs = lhs_s.strip()
        terms = self.utils.parse_expression_to_terms(rhs_s)
        
        # For now, treat all terms as regular terms
        # We'll identify shocks in post-processing after all sections are parsed
        return ParsedEquation(lhs=lhs, rhs_terms=terms, shock=None)

    def _parse_measurement_equation_details(self, line: str) -> Optional[ParsedEquation]:
        """
        Parse measurement equation details.
        
        Note: For measurement equations, we don't expect shocks, but we should
        still be consistent with our approach of not guessing by naming conventions.
        """
        line = line.rstrip(';')
        lhs_s, rhs_s = line.split('=', 1)
        lhs = lhs_s.strip()
        terms = self.utils.parse_expression_to_terms(rhs_s)
        
        # Check for any stationary variable coefficient constraints
        stat_vars = set(self.model_data.get('stationary_variables', []))
        final_terms = []
        
        for term in terms:
            if term.variable in stat_vars:
                # Enforce that stationary variables in MEs have coeff of +/-1
                is_simple_coeff = term.coefficient is None or \
                                self.utils._is_numeric_string(term.coefficient) and \
                                abs(float(term.coefficient)) == 1.0
                if not is_simple_coeff:
                    print(f"Warning: Measurement eq for '{lhs}', term '{term.sign}{term.coefficient or ''}*{term.variable}' has non-unitary coefficient for stationary var. Forcing to +/-1.")
                    term.coefficient = "1"  # Sign is handled by term.sign
            final_terms.append(term)
        
        return ParsedEquation(lhs=lhs, rhs_terms=final_terms, shock=None)

    def _post_process_trend_equations_for_shocks(self):
        """
        Post-process trend equations to identify shocks based on declared trend_shocks.
        
        This must be called after all sections are parsed.
        """
        declared_trend_shocks = set(self.model_data.get('trend_shocks', []))
        
        processed_equations = []
        
        for eq in self.model_data.get('trend_equations', []):
            reg_terms = []
            shock = None
            
            for term in eq.rhs_terms:
                if term.variable in declared_trend_shocks:
                    if shock:
                        print(f"Warning: Multiple shocks for {eq.lhs}. Using last: {term.variable}")
                    shock = term.variable
                else:
                    reg_terms.append(term)
            
            # Create new equation with properly identified shock
            processed_eq = ParsedEquation(lhs=eq.lhs, rhs_terms=reg_terms, shock=shock)
            processed_equations.append(processed_eq)
        
        # Update the model data with processed equations
        self.model_data['trend_equations'] = processed_equations

    def _validate_shock_usage(self):
        """
        Validate that all declared shocks are used appropriately and consistently.
        """
        declared_trend_shocks = set(self.model_data.get('trend_shocks', []))
        declared_stationary_shocks = set(self.model_data.get('stationary_shocks', []))
        all_declared_shocks = declared_trend_shocks | declared_stationary_shocks
        
        # Check trend equations use only declared trend shocks
        used_trend_shocks = set()
        for eq in self.model_data.get('trend_equations', []):
            if eq.shock:
                used_trend_shocks.add(eq.shock)
                if eq.shock not in declared_trend_shocks:
                    print(f"Warning: Trend equation for '{eq.lhs}' uses shock '{eq.shock}' not declared in trend_shocks block")
        
        # Check for unused declared trend shocks
        unused_trend_shocks = declared_trend_shocks - used_trend_shocks
        if unused_trend_shocks:
            print(f"Warning: Declared trend shocks not used in any equations: {unused_trend_shocks}")
        
        # Validate that stationary shocks are not used in trend equations
        trend_eq_terms_all = []
        for eq in self.model_data.get('trend_equations', []):
            trend_eq_terms_all.extend([term.variable for term in eq.rhs_terms])
        
        stationary_shocks_in_trends = declared_stationary_shocks.intersection(set(trend_eq_terms_all))
        if stationary_shocks_in_trends:
            print(f"Warning: Stationary shocks used in trend equations: {stationary_shocks_in_trends}")

    def parse_content(self, content: str) -> ReducedModel:
        """Parse GPM content and return ReducedModel with proper shock identification."""
        self._extract_basic_sections(content)
        
        # POST-PROCESS: Identify shocks in trend equations now that all sections are parsed
        self._post_process_trend_equations_for_shocks()
        
        # VALIDATE: Check shock usage consistency
        self._validate_shock_usage()
        
        core_vars_list = self._identify_core_variables()
        all_gpm_params_set = set(self.model_data.get('parameters', []))
        non_core_trend_defs = self._build_definitions_for_non_core_trends(core_vars_list, all_gpm_params_set)
        reduced_meas_eqs = self._reduce_measurement_equations(non_core_trend_defs, all_gpm_params_set)
        core_eqs_list = self._extract_core_equations(core_vars_list)
        
        return ReducedModel(
            core_variables=core_vars_list, 
            core_equations=core_eqs_list,
            reduced_measurement_equations=reduced_meas_eqs,
            stationary_variables=self.model_data.get('stationary_variables', []),
            parameters=self.model_data.get('parameters', []),
            estimated_params=self.model_data.get('estimated_params', {}),
            var_prior_setup=self.model_data.get('var_prior_setup'),
            gpm_trend_variables_original=self.model_data.get('gpm_trend_variables_original', []),
            gpm_stationary_variables_original=self.model_data.get('gpm_stationary_variables_original', []),
            gpm_observed_variables_original=self.model_data.get('gpm_observed_variables_original', []),
            non_core_trend_definitions=non_core_trend_defs,
            initial_values=self.model_data.get('initial_values', {}),
            trend_shocks=self.model_data.get('trend_shocks', []),
            stationary_shocks=self.model_data.get('stationary_shocks', []),
            all_original_trend_equations=self.model_data.get('trend_equations', []),
            all_original_measurement_equations=self.model_data.get('measurement_equations', [])
        )

# --- Test Function ---
def _test_gpm_model_parser(gpm_file_content: str, file_name_for_test:str = "test_parser_model.gpm"):
    # (As before)
    with open(file_name_for_test, "w") as f: f.write(gpm_file_content)
    parser = GPMModelParser()
    try:
        print(f"\n--- Attempting to parse GPM content from: {file_name_for_test} ---")
        model = parser.parse_file(file_name_for_test)
        print(f"\n--- Parsed ReducedModel for {file_name_for_test} ---")
        print(f"Parameters Declared: {model.parameters}")
        print(f"GPM Original Trend Vars: {model.gpm_trend_variables_original}")
        print(f"GPM Original Stationary Vars: {model.gpm_stationary_variables_original}")
        print(f"GPM Original Observed Vars: {model.gpm_observed_variables_original}")
        print(f"Trend Shocks Declared: {model.trend_shocks}")
        print(f"Stationary Shocks Declared: {model.stationary_shocks}")
        if model.var_prior_setup: print(f"VAR Prior: Order={model.var_prior_setup.var_order}, es={model.var_prior_setup.es}, eta={model.var_prior_setup.eta}")
        else: print("VAR Prior: Not specified.")
        # print(f"Initial Values: {model.initial_values}") # Can be verbose
        print(f"Core Variables Identified: {model.core_variables} (Count: {len(model.core_variables)})")
        print(f"\nCore Equations ({len(model.core_equations)}):")
        for eq in model.core_equations:
            rhs_parts = []
            for t_idx, t in enumerate(eq.rhs_terms):
                c = t.coefficient if t.coefficient and t.coefficient!="1" else ""
                c = c+"*" if c and t.variable!="1" else c
                l = f"(-{t.lag})" if t.lag > 0 else ""
                s = t.sign if t_idx > 0 or t.sign=='-' else "" 
                if t_idx == 0 and t.sign=='+': s = ""
                rhs_parts.append(f"{s}{c}{t.variable}{l}")
            shk = f" + {eq.shock}" if eq.shock else ""
            print(f"  {eq.lhs} = {' '.join(rhs_parts).replace(' + -',' - ')}{shk}")
        print("\nNon-core Trend Definitions:")
        if model.non_core_trend_definitions:
            for name, expr in model.non_core_trend_definitions.items(): print(f"  {name} = Terms: {expr.terms}, Const: '{expr.constant_str}', Params: {expr.parameters}")
        else: print("  (None)")
        print("\nReduced Measurement Equations:")
        if model.reduced_measurement_equations:
            for name, expr in model.reduced_measurement_equations.items(): print(f"  {name} = Terms: {expr.terms}, Const: '{expr.constant_str}', Params: {expr.parameters}")
        else: print("  (None)")
    except Exception as e:
        import traceback; print(f"Error testing parser: {e}"); traceback.print_exc()
    finally:
        import os; 
        if os.path.exists(file_name_for_test):
            try: os.remove(file_name_for_test)
            except Exception as e_rem: print(f"Warning: Could not remove temp test file: {e_rem}")

if __name__ == "__main__":
    full_gpm_example_content_for_test = """
parameters 
var_phi  // Coefficient of relative risk aversion
;            



estimated_params;
    // ... standard errors variance...
    stderr shk_trend_r_world, inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_pi_world,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_sp_trend_world,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_theta_world,inv_gamma_pdf, 0.01, 0.005; 
    
    // Country-specific trend shocks
    stderr shk_trend_r_us,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_pi_us,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_r_ea,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_pi_ea,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_r_jp,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_trend_pi_jp,inv_gamma_pdf, 0.01, 0.005; 
    
    // Country-specific risk premium shocks
    stderr shk_sp_trend_us,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_sp_trend_ea,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_sp_trend_jp,inv_gamma_pdf, 0.01, 0.005; 
    
    // Country-specific productivity/preference shocks
    stderr shk_theta_us,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_theta_ea,inv_gamma_pdf, 0.01, 0.005; 
    stderr shk_theta_jpinv_gamma_pdf, 0.01, 0.005; 

    stderr SHK_L_GDP_TREND, inv_gamma_pdf, 0.01, 0.005; // Example: Inverse Gamma prior with alpha 0.01 and beta 0.005
    stderr SHK_G_TREND, inv_gamma_pdf, 0.01, 0.005; // Example: Inverse Gamma prior with alpha 0.01 and beta 0.005
    stderr SHK_PI_TREND, inv_gamma_pdf, 0.01, 0.005; // Example: Inverse Gamma prior with alpha 0.01 and beta 0.005
    stderr SHK_RR_TREND, inv_gamma_pdf, 0.01, 0.005; // Example: Inverse Gamma prior with alpha 0.01 and beta 0.005
    stderr SHK_L_GDP_GAP, inv_gamma_pdf, 0.01, 0.005;
    stderr SHK_DLA_CPI, inv_gamma_pdf, 0.01, 0.005;
    stderr SHK_RS, inv_gamma_pdf, 0.01, 0.005;

    //Shocks to stationary_variables

    stderr shk_cycle_y_us, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_y_ea, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_y_jp, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_r_us, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_r_ea, inv_gamma_pdf, 0.01, 0.005;   
    stderr shk_cycle_r_jp, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_pi_jp, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_pi_us, inv_gamma_pdf, 0.01, 0.005;
    stderr shk_cycle_pi_ea, inv_gamma_pdf, 0.01, 0.005;


    // ... other parameters ...
    var_phi, normal_pdf, 1, 0.2; //mean and variace normal distribution

end;

// ----------------------------------------------------------------------------
// TREND VARIABLES
// ----------------------------------------------------------------------------
trends_vars
    // World level trends
    trend_r_world,
    trend_y_world,
    trend_pi_world,
    sp_trend_world,
    rr_trend_world,
    rs_world_trend,
    theta_world,
    
    // Country-specific trend components
    trend_r_us,
    trend_pi_us,
    trend_r_ea,
    trend_pi_ea,
    trend_r_jp,
    trend_pi_jp,
    
    // Country-specific risk premia
    sp_trend_us,
    sp_trend_ea,
    sp_trend_jp,
    
    // Country real and nominal rates
    rr_trend_us,
    rr_trend_ea,
    rr_trend_jp,
    rs_us_trend,
    rs_ea_trend,
    rs_jp_trend,
    
    // Country growth deviations and total growth
    dev_y_us,
    dev_y_ea,
    dev_y_jp,
    trend_y_us,
    trend_y_ea,
    trend_y_jp,
    
    // Country-specific productivity/preference factors
    theta_us,
    theta_ea,
    theta_jp,
    
    // Country inflation trends (aggregated)
    pi_us_trend,
    pi_ea_trend,
    pi_jp_trend,
    

;


trend_shocks;
    // World level shocks
    shk_trend_r_world,
    shk_trend_pi_world,
    shk_sp_trend_world,
    shk_theta_world,
    
    // Country-specific trend shocks
    shk_trend_r_us,
    shk_trend_pi_us,
    shk_trend_r_ea,
    shk_trend_pi_ea,
    shk_trend_r_jp,
    shk_trend_pi_jp,
    
    // Country-specific risk premium shocks
    shk_sp_trend_us,
    shk_sp_trend_ea,
    shk_sp_trend_jp,
    
    // Country-specific productivity/preference shocks
    shk_theta_us,
    shk_theta_ea,
    shk_theta_jp
;
end;

stationary_variables
    cycle_y_us,
    cycle_y_ea,
    cycle_y_jp,
    cycle_r_us,
    cycle_r_ea,
    cycle_r_jp,
    cycle_pi_us,
    cycle_pi_ea,
    cycle_pi_jp
;

// shocks to Stationary variables  
shocks;
    shk_cycle_y_us,
    shk_cycle_y_ea,
    shk_cycle_y_jp,
    shk_cycle_r_us,
    shk_cycle_r_ea,    
    shk_cycle_r_jp,
    shk_cycle_pi_jp,
    shk_cycle_pi_us,
    shk_cycle_pi_ea
end;

trend_model;
    
// ============================================================================
// COMPLETE MULTI-COUNTRY RBC TREND MODEL WITH GROWTH-INTEREST RATE LINKAGES
// ============================================================================

// ----------------------------------------------------------------------------
// 1. WORLD LEVEL TRENDS
// ----------------------------------------------------------------------------

// World real interest rate and inflation trends (exogenous processes)
trend_r_world = trend_r_world(-1) + shk_trend_r_world;
trend_pi_world = trend_pi_world(-1) + shk_trend_pi_world;

// World risk premium
sp_trend_world = sp_trend_world(-1) + shk_sp_trend_world;

// World real rate
rr_trend_world = trend_r_world + sp_trend_world;

// World nominal rate
rs_world_trend = rr_trend_world + trend_pi_world;

// World consumption growth follows world real rate (Euler equation)
trend_y_world = (var_phi) * rr_trend_world + theta_world;
theta_world = theta_world(-1) + shk_theta_world;

// ----------------------------------------------------------------------------
// 2. COUNTRY-SPECIFIC TRENDS
// ----------------------------------------------------------------------------

// Country-specific interest rate and inflation trends
trend_r_us = trend_r_us(-1) + shk_trend_r_us;
trend_pi_us = trend_pi_us(-1) + shk_trend_pi_us;

trend_r_ea = trend_r_ea(-1) + shk_trend_r_ea;
trend_pi_ea = trend_pi_ea(-1) + shk_trend_pi_ea;

trend_r_jp = trend_r_jp(-1) + shk_trend_r_jp;
trend_pi_jp = trend_pi_jp(-1) + shk_trend_pi_jp;

// Country-specific risk premia
sp_trend_us = sp_trend_us(-1) + shk_sp_trend_us;
sp_trend_ea = sp_trend_ea(-1) + shk_sp_trend_ea;
sp_trend_jp = sp_trend_jp(-1) + shk_sp_trend_jp;

// ----------------------------------------------------------------------------
// 3. NO-ARBITRAGE CONDITIONS WITH RISK PREMIA
// ----------------------------------------------------------------------------

// Country real rates = world rate + country premium + risk premium
rr_trend_us = rr_trend_world + trend_r_us + sp_trend_us;
rr_trend_ea = rr_trend_world + trend_r_ea + sp_trend_ea;
rr_trend_jp = rr_trend_world + trend_r_jp + sp_trend_jp;

// Country nominal rates
rs_us_trend = rr_trend_us + trend_pi_us;
rs_ea_trend = rr_trend_ea + trend_pi_ea;
rs_jp_trend = rr_trend_jp + trend_pi_jp;

// ----------------------------------------------------------------------------
// 4. GROWTH-INTEREST RATE LINKAGES (Euler Equations)
// ----------------------------------------------------------------------------

// Country deviations from world growth (driven by interest rate differentials)
dev_y_us = var_phi * (rr_trend_us - rr_trend_world) + theta_us;
dev_y_ea = var_phi * (rr_trend_ea - rr_trend_world) + theta_ea;
dev_y_jp = var_phi * (rr_trend_jp - rr_trend_world) + theta_jp;

// Total country growth rates
trend_y_us = trend_y_world + dev_y_us;
trend_y_ea = trend_y_world + dev_y_ea;
trend_y_jp = trend_y_world + dev_y_jp;

// Country-specific productivity/preference shocks
theta_us = theta_us(-1) + shk_theta_us;
theta_ea = theta_ea(-1) + shk_theta_ea;
theta_jp = theta_jp(-1) + shk_theta_jp;

// ----------------------------------------------------------------------------
// 5. INFLATION AGGREGATION
// ----------------------------------------------------------------------------

// Country inflation = world inflation + country-specific component
pi_us_trend = trend_pi_world + trend_pi_us;
pi_ea_trend = trend_pi_world + trend_pi_ea;
pi_jp_trend = trend_pi_world + trend_pi_jp;


    
end;

varobs 
    y_us,
    y_ea,
    y_jp,
    r_us,
    r_ea,
    r_jp,
    pi_us,
    pi_ea,
    pi_jp
;

measurement_equations;
    y_us = trend_y_us + cycle_y_us;
    y_ea = trend_y_ea + cycle_y_ea;
    y_jp = trend_y_jp + cycle_y_jp;

    r_ea = rs_ea_trend + cycle_r_ea;
    r_jp = rs_jp_trend + cycle_r_jp;
    r_us = rs_us_trend + cycle_r_us;
    
    pi_ea = pi_ea_trend + cycle_pi_ea;
    pi_us = pi_us_trend + cycle_pi_us;    
    pi_jp = pi_jp_trend + cycle_pi_jp;

end;


initval;
    trend_r_world, normal_pdf, 0, 0.1;
    trend_y_world, normal_pdf, 0, 0.1;
    trend_pi_world, normal_pdf, 0, 0.1;
    trend_y_us, normal_pdf, 0, 10;
    trend_pi_us, normal_pdf, 0, 10;
    trend_r_us, normal_pdf, 0, 10;
    trend_pi_ea, normal_pdf, 0, 10;
    trend_y_ea, normal_pdf, 0, 10;
    trend_r_ea, normal_pdf, 0, 10;
    trend_pi_jp, normal_pdf, 0, 10;
    trend_y_jp, normal_pdf, 0, 10;
    trend_r_jp, normal_pdf, 0, 10;
    r_us_trend, normal_pdf, 0, 10;
    y_us_trend, normal_pdf, 0, 10;
    pi_us_trend, normal_pdf, 0, 10;
    y_ea_trend, normal_pdf, 0, 10;
    r_ea_trend, normal_pdf, 0, 10;
    pi_ea_trend, normal_pdf, 0, 10;
    y_jp_trend, normal_pdf, 0, 10;
    r_jp_trend, normal_pdf, 0, 10;
    pi_jp_trend, normal_pdf, 0, 10;

    sp_trend_world, normal_pdf, 0, 0.1;
    sp_trend_us, normal_pdf, 0, 0.05;
    sp_trend_ea, normal_pdf, 0, 0.05;
    sp_trend_jp, normal_pdf, 0, 0.05;
    rr_trend_world, normal_pdf, 0, 0.1;
    rr_trend_us, normal_pdf, 0, 0.1;
    rr_trend_ea, normal_pdf, 0, 0.1;
    rr_trend_jp, normal_pdf, 0, 0.1;
    theta_world, normal_pdf, 0, 0.1;
    theta_us, normal_pdf, 0, 0.05;
    theta_ea, normal_pdf, 0, 0.05;
    theta_jp, normal_pdf, 0, 0.05;
    dev_y_us, normal_pdf, 0, 0.05;
    dev_y_ea, normal_pdf, 0, 0.05;
    dev_y_jp, normal_pdf, 0, 0.05;
    rs_world_trend, normal_pdf, 0, 0.1;
    rs_us_trend, normal_pdf, 0, 0.1;
    rs_ea_trend, normal_pdf, 0, 0.1;
    rs_jp_trend, normal_pdf, 0, 0.1;
    y_world_trend, normal_pdf, 0, 0.1;
end;

var_prior_setup;
    var_order    = 1;  //VAR(p) for stationary components
    es = 0.6, 0.15;  //# Mean for diagonal A, Mean for off-diagonal A
    fs = 0.15, 0.15; //# Std Dev for diagonal A, Std Dev for off-diagonal A (Tighter)
    gs = 3.0 , 3.0;  //# Gamma shape parameters for precision (alpha in IG is gs+1) 
    hs = 1.0 , 1.0 ; //# Gamma scale parameters for precision (beta in IG is hs)
    eta = 2.0 ; //LKJ concentration parameter for the CORRELATION matrix of cycles
end;

    """
    _test_gpm_model_parser(full_gpm_example_content_for_test, file_name_for_test="my_full_gpm_test.gpm")