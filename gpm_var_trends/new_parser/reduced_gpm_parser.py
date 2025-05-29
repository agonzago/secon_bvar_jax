"""
Reduced GPM Parser - Model Reduction Approach
Implements the core-variable identification and model reduction logic
"""

import re
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from copy import deepcopy
import sympy as sp
from collections import defaultdict, deque

@dataclass
class PriorSpec:
    """Specification for a parameter prior"""
    name: str
    distribution: str
    params: List[float]

@dataclass
class VariableSpec:
    """Specification for a variable"""
    name: str
    init_dist: Optional[str] = None
    init_params: Optional[List[float]] = None

@dataclass
class ParsedTerm:
    """A single term in an equation: coefficient * variable^lag"""
    variable: str
    lag: int = 0
    coefficient: Optional[str] = None  # Parameter name or None for 1
    sign: str = '+'

@dataclass
class ParsedEquation:
    """A parsed equation with structured terms"""
    lhs: str
    rhs_terms: List[ParsedTerm]
    shock: Optional[str] = None
    equation_type: str = 'unknown'  # 'core', 'static', 'derived'

@dataclass
class ReducedExpression:
    """Expression in terms of core variables only"""
    terms: Dict[str, str]  # core_variable -> coefficient_expression
    constant: str = '0'
    parameters: Set[str] = field(default_factory=set)

@dataclass
class ReducedModel:
    """The reduced model with core variables and expanded measurement equations"""
    core_variables: List[str]
    core_equations: List[ParsedEquation]
    reduced_measurement_equations: Dict[str, ReducedExpression]
    stationary_variables: List[str]
    parameters: List[str]
    estimated_params: Dict[str, PriorSpec]
    var_prior_setup: Optional[object] = None

class SymbolicReducer:
    """Handles symbolic manipulation and model reduction"""
    
    def __init__(self):
        self.substitution_rules = {}
        self.parameters = set()
        
    def parse_expression_to_terms(self, expr_str: str) -> List[ParsedTerm]:
        """Parse an expression string into structured terms"""
        
        # Clean the expression
        expr_str = expr_str.strip()
        if not expr_str:
            return []
            
        # Split by + and - while preserving signs
        terms = []
        
        # Use regex to split by operators while capturing them
        parts = re.split(r'\s*([+-])\s*', expr_str)
        
        current_sign = '+'
        for i, part in enumerate(parts):
            if part in ['+', '-']:
                current_sign = part
            elif part.strip():
                term = self._parse_single_term(part.strip(), current_sign)
                if term:
                    terms.append(term)
                current_sign = '+'
                
        return terms
    
    def _parse_single_term(self, term_str: str, sign: str) -> Optional[ParsedTerm]:
        """Parse a single term like 'var_phi * rr_trend_us(-1)'"""
        
        if not term_str:
            return None
            
        # Handle multiplication
        if '*' in term_str:
            parts = [p.strip() for p in term_str.split('*')]
            # Last part should be the variable, others are coefficients
            var_part = parts[-1]
            coeff_parts = parts[:-1]
            
            # Combine coefficient parts
            coefficient = ' * '.join(coeff_parts) if coeff_parts else None
        else:
            var_part = term_str
            coefficient = None
            
        # Extract variable name and lag
        if '(-' in var_part and ')' in var_part:
            var_name = var_part.split('(')[0].strip()
            lag_match = re.search(r'\(-(\d+)\)', var_part)
            lag = int(lag_match.group(1)) if lag_match else 0
        else:
            var_name = var_part.strip()
            lag = 0
            
        return ParsedTerm(
            variable=var_name,
            lag=lag,
            coefficient=coefficient,
            sign=sign
        )
    
    def substitute_expression(self, expr: ReducedExpression, 
                            substitution_rules: Dict[str, ReducedExpression]) -> ReducedExpression:
        """Substitute variables in an expression using substitution rules - FIXED"""
        
        result_terms = {}
        result_parameters = set(expr.parameters)
        
        for var, coeff_expr in expr.terms.items():
            # Clean variable name (remove extra parentheses)
            clean_var = var.strip('()')
            
            if clean_var in substitution_rules:
                # Substitute this variable
                sub_expr = substitution_rules[clean_var]
                
                # Multiply each term in substitution by coefficient
                for sub_var, sub_coeff in sub_expr.terms.items():
                    combined_coeff = self._multiply_coefficients(coeff_expr, sub_coeff)
                    
                    if sub_var in result_terms:
                        # Add to existing term
                        result_terms[sub_var] = self._add_coefficients(result_terms[sub_var], combined_coeff)
                    else:
                        result_terms[sub_var] = combined_coeff
                
                # Add parameters from substitution
                result_parameters.update(sub_expr.parameters)
            else:
                # Keep this variable as is
                if var in result_terms:
                    result_terms[var] = self._add_coefficients(result_terms[var], coeff_expr)
                else:
                    result_terms[var] = coeff_expr
        
        return ReducedExpression(
            terms=result_terms,
            constant=expr.constant,
            parameters=result_parameters
        )
    
    def _multiply_coefficients(self, coeff1: str, coeff2: str) -> str:
        """Multiply two coefficient expressions"""
        if coeff1 == '1' and coeff2 == '1':
            return '1'
        elif coeff1 == '1':
            return coeff2
        elif coeff2 == '1':
            return coeff1
        else:
            return f"({coeff1}) * ({coeff2})"
    
    def _add_coefficients(self, coeff1: str, coeff2: str) -> str:
        """Add two coefficient expressions"""
        if coeff1 == '0':
            return coeff2
        elif coeff2 == '0':
            return coeff1
        else:
            return f"({coeff1}) + ({coeff2})"

class ReducedGPMParser:
    """Main parser that implements model reduction"""
    
    def __init__(self):
        self.reducer = SymbolicReducer()
        self.model_data = {}
        
    def parse_file(self, filepath: str) -> ReducedModel:
        """Parse GPM file and return reduced model"""
        
        with open(filepath, 'r') as file:
            content = file.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> ReducedModel:
        """Parse GPM content and return reduced model"""
        
        # First pass: extract all sections using existing parser logic
        self._extract_basic_sections(content)
        
        # Identify core variables (those with shocks or lags)
        core_variables = self._identify_core_variables()
        
        # Build substitution rules for non-core variables
        substitution_rules = self._build_substitution_rules(core_variables)
        
        # Apply model reduction
        reduced_measurement_eqs = self._reduce_measurement_equations(substitution_rules)
        
        # Extract core equations (only those for core variables)
        core_equations = self._extract_core_equations(core_variables)
        
        return ReducedModel(
            core_variables=core_variables,
            core_equations=core_equations,
            reduced_measurement_equations=reduced_measurement_eqs,
            stationary_variables=self.model_data.get('stationary_variables', []),
            parameters=self.model_data.get('parameters', []),
            estimated_params=self.model_data.get('estimated_params', {}),
            var_prior_setup=self.model_data.get('var_prior_setup')
        )
    
    def _extract_basic_sections(self, content: str):
        """Extract basic sections from GPM file (similar to original parser)"""
        
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        self.model_data = {
            'parameters': [],
            'estimated_params': {},
            'trend_variables': [],
            'stationary_variables': [],
            'trend_shocks': [],
            'stationary_shocks': [],
            'trend_equations': [],
            'measurement_equations': [],
            'observed_variables': [],
            'initial_values': {},
            'var_prior_setup': None
        }
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('parameters'):
                i = self._parse_parameters(lines, i)
            elif line.startswith('estimated_params'):
                i = self._parse_estimated_params(lines, i)
            elif line.startswith('trends_vars'):
                i = self._parse_variable_list(lines, i, 'trend_variables')
            elif line.startswith('stationary_variables'):
                i = self._parse_variable_list(lines, i, 'stationary_variables')
            elif line.startswith('trend_shocks'):
                i = self._parse_shock_list(lines, i, 'trend_shocks')
            elif line.startswith('shocks'):
                i = self._parse_shock_list(lines, i, 'stationary_shocks')
            elif line.startswith('trend_model'):
                i = self._parse_trend_model(lines, i)
            elif line.startswith('varobs'):
                i = self._parse_variable_list(lines, i, 'observed_variables')
            elif line.startswith('measurement_equations'):
                i = self._parse_measurement_equations(lines, i)
            elif line.startswith('initval'):
                i = self._parse_initial_values(lines, i)
            elif line.startswith('var_prior_setup'):
                i = self._parse_var_prior_setup(lines, i)
            else:
                i += 1
    
    # def _parse_parameters(self, lines: List[str], start_idx: int) -> int:
    #     """Parse parameters section - FIXED"""
    #     line = lines[start_idx]
        
    #     # Handle empty parameters section
    #     if 'parameters' in line and ';' in line and line.count(',') == 0:
    #         # Check if there are actual parameter names
    #         match = re.search(r'parameters\s+([^;]+);', line)
    #         if match:
    #             param_str = match.group(1).strip()
    #             if param_str:  # Not empty
    #                 self.model_data['parameters'] = [p.strip() for p in param_str.split(',') if p.strip()]
    #             else:
    #                 self.model_data['parameters'] = []
    #         else:
    #             self.model_data['parameters'] = []
    #     else:
    #         # Multi-line parameters (shouldn't happen but handle it)
    #         match = re.search(r'parameters\s+([^;]+);', line)
    #         if match:
    #             param_str = match.group(1)
    #             self.model_data['parameters'] = [p.strip() for p in param_str.split(',') if p.strip()]
        
    #     print(f"Parsed parameters: {self.model_data['parameters']}")  # Debug output
    #     return start_idx + 1
    def _parse_parameters(self, lines: List[str], start_idx: int) -> int:
        """
        Parse a 'parameters' declaration which might span multiple lines
        and handle comments robustly. Appends found parameters to
        self.model_data['parameters'].
        Correctly handles cases like:
        parameters param1, param2;
        parameters param1,
                   param2; // EOL comment
        parameters param1; // Comment
        parameters param2; // Comment
        parameters ; // Empty declaration
        parameters param1, param2; // Another EOL comment
        """
        current_line_idx = start_idx
        parameter_declaration_parts = []
        first_line_of_this_statement = True
        statement_terminated_by_semicolon = False

        # Loop to collect all parts of a single 'parameters ... ;' statement
        while current_line_idx < len(lines):
            line_text = lines[current_line_idx]
            
            # Content relevant for parameter parsing (before any EOL comment)
            content_to_parse = line_text.split('//', 1)[0].strip()

            if not content_to_parse and not line_text.strip().startswith('//'):
                # Line is effectively empty or became empty after stripping comment,
                # but wasn't a full comment line itself.
                # If it's truly an empty line, we can just skip.
                if not line_text.strip():
                    current_line_idx +=1
                    continue
                # If it was like "   // comment", content_to_parse is empty.
                # If it was just "   ", content_to_parse is empty.
                # We only break if it's a significant structural change or end of file.
                # For now, let's assume empty lines don't break a multi-line parameter list.
            
            # Remove "parameters" keyword from the first line of the statement
            if first_line_of_this_statement:
                # Case-insensitive removal of "parameters"
                if content_to_parse.lower().startswith('parameters'):
                    content_to_parse = content_to_parse[len('parameters'):].strip()
                first_line_of_this_statement = False # Processed the keyword part

            # Check for semicolon to terminate the current statement
            if ';' in content_to_parse:
                # Take content up to the first semicolon
                part_before_semicolon = content_to_parse.split(';', 1)[0]
                parameter_declaration_parts.append(part_before_semicolon.strip())
                statement_terminated_by_semicolon = True
                current_line_idx += 1 # Consume this line
                break # End of this 'parameters' statement
            else:
                # No semicolon on this line (or it was in a comment)
                # Add the content if it's not empty
                if content_to_parse:
                    parameter_declaration_parts.append(content_to_parse)
                current_line_idx += 1
        
        if not statement_terminated_by_semicolon and "".join(parameter_declaration_parts).strip():
            # Reached end of input lines, but the statement wasn't properly terminated.
            # This might be a syntax error in the GPM file.
            print(f"Warning: 'parameters' declaration starting near line {start_idx + 1} "
                  f"did not have a terminating ';' before end of file or next section. "
                  f"Attempting to parse collected content: '{' '.join(parameter_declaration_parts)}'")

        # Join collected parts (e.g., from multi-line) into a single string
        # Using a space as a separator handles cases where a parameter might be split
        # across lines, though typically commas would still be present.
        full_parameter_string = " ".join(filter(None, parameter_declaration_parts))

        # Split by comma, then strip whitespace from each part, and filter out empty strings
        if full_parameter_string: # Ensure not empty (e.g. from "parameters ;")
            parsed_params = [p.strip() for p in full_parameter_string.split(',') if p.strip()]
            if parsed_params: # If after splitting and stripping, we have parameters
                self.model_data['parameters'].extend(parsed_params)
                print(f"Parsed parameters: {parsed_params} (cumulative: {self.model_data['parameters']})") # Debug

        return current_line_idx # Return the index of the next line to process
        
    def _parse_estimated_params(self, lines: List[str], start_idx: int) -> int:
        """Parse estimated_params section"""
        i = start_idx + 1
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//'):
                self._parse_prior_specification(line)
            i += 1
        return i + 1
    
    def _parse_prior_specification(self, line: str):
        """Parse a single prior specification line"""
        line = line.rstrip(';')
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 3:
            param_name = parts[0].split()[-1]
            distribution = parts[1]
            params = [float(p) for p in parts[2:]]
            
            self.model_data['estimated_params'][param_name] = PriorSpec(
                name=param_name,
                distribution=distribution,
                params=params
            )
    
    def _parse_variable_list(self, lines: List[str], start_idx: int, attr_name: str) -> int:
        """Parse a variable list section - FIXED"""
        variables = []
        i = start_idx + 1
        
        # Collect all lines until we hit semicolon
        collected_text = ""
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('//'):
                i += 1
                continue
                
            collected_text += " " + line
            if line.endswith(';'):
                break
            i += 1
        
        # Clean up the collected text and extract variables
        if collected_text:
            # Remove section name if present
            for section_name in ['trends_vars', 'stationary_variables', 'varobs']:
                collected_text = collected_text.replace(section_name, '')
            
            # Remove semicolon and split by comma
            var_text = collected_text.replace(';', '').strip()
            if var_text:
                variables = [v.strip() for v in var_text.split(',') if v.strip()]
        
        self.model_data[attr_name] = variables
        print(f"Parsed {attr_name}: {variables}")  # Debug output
        return i + 1
    
    def _parse_shock_list(self, lines: List[str], start_idx: int, attr_name: str) -> int:
        """Parse shock list section"""
        shocks = []
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line.startswith('var '):
                shock_name = line.replace('var ', '').strip().rstrip(',')
                shocks.append(shock_name)
            i += 1
        
        self.model_data[attr_name] = shocks
        return i + 1
    
    def _parse_trend_model(self, lines: List[str], start_idx: int) -> int:
        """Parse trend_model section"""
        equations = []
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//') and '=' in line:
                equation = self._parse_trend_equation(line)
                if equation:
                    equations.append(equation)
            i += 1
        
        self.model_data['trend_equations'] = equations
        return i + 1
    
    def _parse_trend_equation(self, line: str) -> Optional[ParsedEquation]:
        """Parse a single trend equation"""
        line = line.rstrip(';')
        
        if '=' not in line:
            return None
        
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Parse RHS into terms
        terms = self.reducer.parse_expression_to_terms(rhs)
        
        # Separate shock from regular terms
        shock = None
        regular_terms = []
        
        for term in terms:
            if term.variable.startswith('shk_') or term.variable.startswith('SHK_'):
                shock = term.variable
            else:
                regular_terms.append(term)
        
        return ParsedEquation(
            lhs=lhs,
            rhs_terms=regular_terms,
            shock=shock
        )
    
    def _parse_measurement_equations(self, lines: List[str], start_idx: int) -> int:
        """Parse measurement_equations section"""
        equations = []
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//') and '=' in line:
                equation = self._parse_measurement_equation(line)
                if equation:
                    equations.append(equation)
            i += 1
        
        self.model_data['measurement_equations'] = equations
        return i + 1
    
    def _parse_measurement_equation(self, line: str) -> Optional[ParsedEquation]:
        """Parse a single measurement equation"""
        line = line.rstrip(';')
        
        if '=' not in line:
            return None
        
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        terms = self.reducer.parse_expression_to_terms(rhs)
        
        return ParsedEquation(
            lhs=lhs,
            rhs_terms=terms,
            shock=None
        )
    
    def _parse_initial_values(self, lines: List[str], start_idx: int) -> int:
        """Parse initval section (simplified)"""
        # Implementation similar to original parser
        return start_idx + 1
    
    def _parse_var_prior_setup(self, lines: List[str], start_idx: int) -> int:
        """Parse var_prior_setup section (simplified)"""
        # Implementation similar to original parser  
        return start_idx + 1
    
    def _identify_core_variables(self) -> List[str]:
        """Identify core variables that must be in the transition equation"""
        
        core_vars = set()
        
        # Variables with shocks are always core
        for eq in self.model_data['trend_equations']:
            if eq.shock is not None:
                core_vars.add(eq.lhs)
        
        # Variables with lags of themselves are core
        for eq in self.model_data['trend_equations']:
            for term in eq.rhs_terms:
                if term.variable == eq.lhs and term.lag > 0:
                    core_vars.add(eq.lhs)
        
        # Variables that appear in measurement equations with parameters should be core
        # (This catches cases like dev_y_us that have parameters but no direct shocks)
        for eq in self.model_data['trend_equations']:
            has_parameters = any(term.coefficient in self.model_data['parameters'] 
                               for term in eq.rhs_terms 
                               if term.coefficient is not None)
            
            if has_parameters:
                # Check if this variable or its dependencies appear in measurement equations
                if self._affects_measurement_equations(eq.lhs):
                    core_vars.add(eq.lhs)
        
        return sorted(list(core_vars))
    
    def _affects_measurement_equations(self, var_name: str) -> bool:
        """Check if a variable affects measurement equations (directly or indirectly)"""
        
        # Direct appearance
        for eq in self.model_data['measurement_equations']:
            if any(term.variable == var_name for term in eq.rhs_terms):
                return True
        
        # Indirect appearance through other variables (simplified check)
        return True  # Conservative: assume it affects measurement
    
    def _build_substitution_rules(self, core_variables: List[str]) -> Dict[str, ReducedExpression]:
        """Build substitution rules for non-core variables"""
        
        substitution_rules = {}
        core_set = set(core_variables)
        
        # Find equations for non-core variables
        for eq in self.model_data['trend_equations']:
            if eq.lhs not in core_set and eq.shock is None:
                # This is a definition equation for a non-core variable
                expr = self._equation_to_reduced_expression(eq)
                substitution_rules[eq.lhs] = expr
        
        # Iteratively substitute until convergence
        max_iterations = 10
        for iteration in range(max_iterations):
            changed = False
            for var_name, expr in substitution_rules.items():
                new_expr = self.reducer.substitute_expression(expr, substitution_rules)
                if new_expr.terms != expr.terms:
                    substitution_rules[var_name] = new_expr
                    changed = True
            
            if not changed:
                break
        
        return substitution_rules
    
    def _equation_to_reduced_expression(self, equation: ParsedEquation) -> ReducedExpression:
        """Convert a parsed equation to a reduced expression"""
        
        terms = {}
        parameters = set()
        
        for term in equation.rhs_terms:
            var_key = f"{term.variable}"
            if term.lag > 0:
                var_key += f"(-{term.lag})"
            
            coeff = term.coefficient if term.coefficient else '1'
            
            # Apply sign
            if term.sign == '-':
                coeff = f"-({coeff})" if coeff != '1' else '-1'
            
            if var_key in terms:
                terms[var_key] = self.reducer._add_coefficients(terms[var_key], coeff)
            else:
                terms[var_key] = coeff
            
            # Track parameters
            if term.coefficient and term.coefficient in self.model_data['parameters']:
                parameters.add(term.coefficient)
        
        return ReducedExpression(terms=terms, parameters=parameters)
    
    def _reduce_measurement_equations(self, substitution_rules: Dict[str, ReducedExpression]) -> Dict[str, ReducedExpression]:
        """Reduce measurement equations by substituting out non-core variables"""
        
        reduced_equations = {}
        
        for eq in self.model_data['measurement_equations']:
            # Convert measurement equation to reduced expression
            expr = self._equation_to_reduced_expression(eq)
            
            # Apply substitutions
            reduced_expr = self.reducer.substitute_expression(expr, substitution_rules)
            
            reduced_equations[eq.lhs] = reduced_expr
        
        return reduced_equations
    
    def _extract_core_equations(self, core_variables: List[str]) -> List[ParsedEquation]:
        """Extract equations for core variables only"""
        
        core_equations = []
        core_set = set(core_variables)
        
        for eq in self.model_data['trend_equations']:
            if eq.lhs in core_set:
                core_equations.append(eq)
        
        return core_equations


def test_reduced_parser(gpm_file='model_with_trends.gpm'):
    """Test the reduced parser with your GPM file - FIXED"""
    
    try:
        parser = ReducedGPMParser()
        reduced_model = parser.parse_file(gpm_file)
        
        print("=== REDUCED MODEL ANALYSIS ===")
        print(f"\nOriginal trend variables: {len(parser.model_data['trend_variables'])}")
        print(f"Core variables (transition eq): {len(reduced_model.core_variables)}")
        
        # FIXED: Avoid division by zero
        original_trends = len(parser.model_data['trend_variables'])
        if original_trends > 0:
            print(f"Reduction ratio: {len(reduced_model.core_variables) / original_trends:.2f}")
        else:
            print(f"Reduction ratio: N/A (no original trend variables parsed)")
            print(f"Note: All {len(reduced_model.core_variables)} variables identified as core")
        
        print(f"\n=== CORE VARIABLES (Transition Equation) ===")
        for var in reduced_model.core_variables:
            print(f"  {var}")
        
        print(f"\n=== CORE EQUATIONS ===")
        for eq in reduced_model.core_equations:
            print(f"\n{eq.lhs} = ", end="")
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
        
        print(f"\n=== REDUCED MEASUREMENT EQUATIONS ===")
        for obs_var, expr in reduced_model.reduced_measurement_equations.items():
            print(f"\n{obs_var} = ")
            for var, coeff in expr.terms.items():
                print(f"  + ({coeff}) * {var}")
            print(f"  + stationary_component")
            
            if expr.parameters:
                print(f"  Parameters involved: {expr.parameters}")
        
        # Additional debugging
        print(f"\n=== PARSING DEBUG INFO ===")
        print(f"Parameters found: {parser.model_data['parameters']}")
        print(f"Stationary variables: {parser.model_data['stationary_variables']}")
        print(f"Trend equations parsed: {len(parser.model_data['trend_equations'])}")
        print(f"Measurement equations parsed: {len(parser.model_data['measurement_equations'])}")
        
        return reduced_model
        
    except Exception as e:
        print(f"Error testing reduced parser: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_reduced_parser()