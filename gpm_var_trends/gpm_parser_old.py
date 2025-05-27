import re
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import numpy as np


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
class TrendEquation:
    """Specification for a trend equation"""
    lhs: str  # Left-hand side variable
    rhs_terms: List[Tuple[str, int, Optional[str]]]  # (variable, lag, coefficient_name)
    shock: str


@dataclass
class MeasurementEquation:
    """Specification for a measurement equation"""
    lhs: str  # Observed variable
    rhs_terms: List[Tuple[str, Optional[str]]]  # (variable, coefficient_name)


@dataclass
class VARPriorSetup:
    """VAR prior hyperparameters"""
    var_order: int
    es: List[float]  # Mean for diagonal and off-diagonal A
    fs: List[float]  # Std dev for diagonal and off-diagonal A
    gs: List[float]  # Gamma shape parameters for precision
    hs: List[float]  # Gamma scale parameters for precision
    eta: float  # LKJ concentration parameter


class GPMModel:
    """Parsed GPM model specification"""
    
    def __init__(self):
        self.parameters: List[str] = []
        self.estimated_params: Dict[str, PriorSpec] = {}
        self.trend_variables: List[str] = []
        self.stationary_variables: List[str] = []
        self.trend_shocks: List[str] = []
        self.stationary_shocks: List[str] = []
        self.trend_equations: List[TrendEquation] = []
        self.measurement_equations: List[MeasurementEquation] = []
        self.observed_variables: List[str] = []
        self.initial_values: Dict[str, VariableSpec] = {}
        self.var_prior_setup: Optional[VARPriorSetup] = None


class GPMParser:
    """Parser for .gpm files"""
    
    def __init__(self):
        self.current_section = None
        self.model = GPMModel()
    
    def parse_file(self, filepath: str) -> GPMModel:
        """Parse a .gpm file and return the model specification"""
        with open(filepath, 'r') as file:
            content = file.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> GPMModel:
        """Parse .gpm content string"""
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # Split into lines and clean
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for section headers
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
        
        return self.model
    
    def _parse_parameters(self, lines: List[str], start_idx: int) -> int:
        """Parse parameters section"""
        line = lines[start_idx]
        # Extract parameters from the line: "parameters b1, b2;"
        match = re.search(r'parameters\s+([^;]+);', line)
        if match:
            param_str = match.group(1)
            self.model.parameters = [p.strip() for p in param_str.split(',')]
        return start_idx + 1
    
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
        # Format: "stderr SHK_L_GDP_TREND, inv_gamma_pdf, 0.01, 0.005;"
        # or: "b1, normal_pdf, 0.1, 0.2;"
        line = line.rstrip(';')
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 3:
            param_type = parts[0].split()[0]  # 'stderr' or parameter name
            param_name = parts[0].split()[-1] if len(parts[0].split()) > 1 else parts[0]
            distribution = parts[1]
            params = [float(p) for p in parts[2:]]
            
            self.model.estimated_params[param_name] = PriorSpec(
                name=param_name,
                distribution=distribution,
                params=params
            )
    
    def _parse_variable_list(self, lines: List[str], start_idx: int, attr_name: str) -> int:
        """Parse a variable list section"""
        variables = []
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].endswith(';'):
            line = lines[i].rstrip(',')
            if line and not line.startswith('//'):
                variables.append(line)
            i += 1
        
        # Handle the last line with semicolon
        if i < len(lines):
            last_line = lines[i].rstrip(';').rstrip(',')
            if last_line and not last_line.startswith('//'):
                variables.append(last_line)
        
        setattr(self.model, attr_name, variables)
        return i + 1
    
    def _parse_shock_list(self, lines: List[str], start_idx: int, attr_name: str) -> int:
        """Parse shock list section"""
        shocks = []
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line.startswith('var '):
                shock_name = line.replace('var ', '').strip()
                shocks.append(shock_name)
            i += 1
        
        setattr(self.model, attr_name, shocks)
        return i + 1
    
    def _parse_trend_model(self, lines: List[str], start_idx: int) -> int:
        """Parse trend_model section"""
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//') and '=' in line:
                equation = self._parse_trend_equation(line)
                if equation:
                    self.model.trend_equations.append(equation)
            i += 1
        
        return i + 1
    
    def _parse_trend_equation(self, line: str) -> Optional[TrendEquation]:
        """Parse a single trend equation"""
        # Format: "L_GDP_TREND = L_GDP_TREND(-1) + b1*G_TREND(-1) + SHK_L_GDP_TREND;"
        line = line.rstrip(';')
        
        if '=' not in line:
            return None
        
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Parse RHS terms
        terms = []
        shock = None
        
        # Split by + and - while preserving the signs
        rhs_parts = re.split(r'(\+|\-)', rhs)
        current_sign = '+'
        
        for part in rhs_parts:
            part = part.strip()
            if part in ['+', '-']:
                current_sign = part
                continue
            
            if not part:
                continue
            
            # Check if this is a shock (typically starts with SHK_)
            if part.startswith('SHK_'):
                shock = part
            else:
                # Parse variable terms like "b1*G_TREND(-1)" or "L_GDP_TREND(-1)"
                term = self._parse_term(part, current_sign)
                if term:
                    terms.append(term)
        
        return TrendEquation(lhs=lhs, rhs_terms=terms, shock=shock)
    
    def _parse_term(self, term: str, sign: str) -> Optional[Tuple[str, int, Optional[str]]]:
        """Parse a single term in an equation"""
        # Handle coefficient * variable cases
        if '*' in term:
            coeff, var_part = term.split('*', 1)
            coeff = coeff.strip()
        else:
            coeff = None
            var_part = term
        
        var_part = var_part.strip()
        
        # Extract variable name and lag
        if '(-' in var_part and ')' in var_part:
            # Variable with lag: "G_TREND(-1)"
            var_name = var_part.split('(')[0]
            lag_str = var_part.split('(-')[1].split(')')[0]
            lag = int(lag_str)
        else:
            # Variable without lag (current period)
            var_name = var_part
            lag = 0
        
        # Apply sign to coefficient if present
        if coeff and sign == '-':
            coeff = '-' + coeff
        
        return (var_name, lag, coeff)
    
    def _parse_measurement_equations(self, lines: List[str], start_idx: int) -> int:
        """Parse measurement_equations section"""
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//') and '=' in line:
                equation = self._parse_measurement_equation(line)
                if equation:
                    self.model.measurement_equations.append(equation)
            i += 1
        
        return i + 1
    
    def _parse_measurement_equation(self, line: str) -> Optional[MeasurementEquation]:
        """Parse a single measurement equation"""
        # Format: "L_GDP_OBS = L_GDP_TREND + L_GDP_GAP;"
        line = line.rstrip(';')
        
        if '=' not in line:
            return None
        
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Parse RHS terms (simpler than trend equations, no lags typically)
        terms = []
        rhs_parts = re.split(r'(\+|\-)', rhs)
        current_sign = '+'
        
        for part in rhs_parts:
            part = part.strip()
            if part in ['+', '-']:
                current_sign = part
                continue
            
            if not part:
                continue
            
            # Handle coefficient * variable cases
            if '*' in part:
                coeff, var_name = part.split('*', 1)
                coeff = coeff.strip()
                var_name = var_name.strip()
            else:
                coeff = None
                var_name = part
            
            # Apply sign to coefficient if present
            if coeff and current_sign == '-':
                coeff = '-' + coeff
            
            terms.append((var_name, coeff))
        
        return MeasurementEquation(lhs=lhs, rhs_terms=terms)
    
    def _parse_initial_values(self, lines: List[str], start_idx: int) -> int:
        """Parse initval section"""
        i = start_idx + 1
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//'):
                self._parse_initial_value(line)
            i += 1
        
        return i + 1
    
    def _parse_initial_value(self, line: str):
        """Parse a single initial value specification"""
        # Format: "L_GDP_GAP, normal_pdf, 0, 0.1;"
        line = line.rstrip(';')
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 3:
            var_name = parts[0]
            distribution = parts[1]
            params = [float(p) for p in parts[2:]]
            
            self.model.initial_values[var_name] = VariableSpec(
                name=var_name,
                init_dist=distribution,
                init_params=params
            )
    
    def _parse_var_prior_setup(self, lines: List[str], start_idx: int) -> int:
        """Parse var_prior_setup section"""
        i = start_idx + 1
        setup_dict = {}
        
        while i < len(lines) and not lines[i].startswith('end;'):
            line = lines[i]
            if line and not line.startswith('//') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.rstrip(';').strip()
                
                if key == 'var_order':
                    setup_dict[key] = int(value)
                elif key in ['es', 'fs', 'gs', 'hs']:
                    # Parse comma-separated values
                    values = [float(v.strip()) for v in value.split(',')]
                    setup_dict[key] = values
                elif key == 'eta':
                    setup_dict[key] = float(value)
            i += 1
        
        if setup_dict:
            self.model.var_prior_setup = VARPriorSetup(**setup_dict)
        
        return i + 1


class GPMModelBuilder:
    """Builds Numpyro models from GPM specifications"""
    
    def __init__(self, gpm_model: GPMModel):
        self.gpm = gpm_model
        self.param_map = {}  # Maps parameter names to their specifications
        
    def build_numpyro_model(self):
        """Build a Numpyro model function from the GPM specification"""
        
        def model(y: jnp.ndarray):
            # Sample structural parameters
            structural_params = self._sample_structural_parameters()
            
            # Build state space matrices based on the GPM specification
            F, R, C, H, init_mean, init_cov = self._build_state_space_from_gpm(structural_params, y.shape)
            
            # Set up and run Kalman filter
            from Kalman_filter_jax import KalmanFilter
            
            kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
            
            # Compute likelihood
            n_vars = len(self.gmp.observed_variables)
            valid_obs_idx = jnp.arange(n_vars, dtype=int)
            I_obs = jnp.eye(n_vars)
            
            loglik = kf.log_likelihood(y, valid_obs_idx, n_vars, C, H, I_obs)
            numpyro.factor("loglik", loglik)
        
        return model
    
    def _sample_structural_parameters(self) -> Dict[str, jnp.ndarray]:
        """Sample all structural parameters based on GPM priors"""
        params = {}
        
        # Sample trend shock standard deviations
        for shock in self.gpm.trend_shocks:
            if shock in self.gpm.estimated_params:
                prior_spec = self.gpm.estimated_params[shock]
                params[shock] = self._sample_parameter(shock, prior_spec)
        
        # Sample stationary shock standard deviations
        for shock in self.gpm.stationary_shocks:
            if shock in self.gpm.estimated_params:
                prior_spec = self.gpm.estimated_params[shock]
                params[shock] = self._sample_parameter(shock, prior_spec)
        
        # Sample structural coefficients
        for param_name in self.gpm.parameters:
            if param_name in self.gpm.estimated_params:
                prior_spec = self.gpm.estimated_params[param_name]
                params[param_name] = self._sample_parameter(param_name, prior_spec)
        
        # Sample VAR parameters if specified
        if self.gpm.var_prior_setup and self.gpm.stationary_variables:
            var_params = self._sample_var_parameters()
            params.update(var_params)
        
        return params
    
    def _sample_parameter(self, name: str, prior_spec: PriorSpec) -> jnp.ndarray:
        """Sample a single parameter based on its prior specification"""
        if prior_spec.distribution == 'normal_pdf':
            mean, std = prior_spec.params
            return numpyro.sample(name, dist.Normal(mean, std))
        elif prior_spec.distribution == 'inv_gamma_pdf':
            alpha, beta = prior_spec.params
            return numpyro.sample(name, dist.InverseGamma(alpha, beta))
        else:
            raise ValueError(f"Unknown distribution: {prior_spec.distribution}")
    
    def _sample_var_parameters(self) -> Dict[str, jnp.ndarray]:
        """Sample VAR parameters using hierarchical prior if specified"""
        setup = self.gpm.var_prior_setup
        n_vars = len(self.gpm.stationary_variables)
        n_lags = setup.var_order
        
        # Sample hierarchical hyperparameters
        Amu = [numpyro.sample(f"Amu_{i}", dist.Normal(setup.es[i], setup.fs[i])) 
               for i in range(2)]
        Aomega = [numpyro.sample(f"Aomega_{i}", dist.Gamma(setup.gs[i], setup.hs[i])) 
                  for i in range(2)]
        
        # Sample VAR coefficient matrices
        A_matrices = []
        for lag in range(n_lags):
            # Sample full matrix with off-diagonal prior
            A_full = numpyro.sample(f"A_full_{lag}", 
                                   dist.Normal(Amu[1], 1/jnp.sqrt(Aomega[1])).expand([n_vars, n_vars]))
            
            # Sample diagonal elements with diagonal prior
            A_diag = numpyro.sample(f"A_diag_{lag}", 
                                   dist.Normal(Amu[0], 1/jnp.sqrt(Aomega[0])).expand([n_vars]))
            
            # Combine
            A_lag = A_full.at[jnp.arange(n_vars), jnp.arange(n_vars)].set(A_diag)
            A_matrices.append(A_lag)
        
        # Sample correlation matrix
        Omega_chol = numpyro.sample("Omega_stat_chol", 
                                   dist.LKJCholesky(n_vars, concentration=setup.eta))
        
        # Sample shock standard deviations
        sigma_stat = numpyro.sample("sigma_stat", 
                                   dist.InverseGamma(2.0, 1.0).expand([n_vars]))
        
        return {
            'A_matrices': jnp.stack(A_matrices),
            'Omega_stat_chol': Omega_chol,
            'sigma_stat': sigma_stat
        }
    
    def _build_state_space_from_gpm(self, params: Dict, data_shape: Tuple[int, int]):
        """Build state space matrices from GPM specification and parameters"""
        T, n_obs = data_shape
        
        # This would need to be implemented based on the specific GPM structure
        # For now, return placeholder matrices
        n_trend = len(self.gpm.trend_variables)
        n_stat = len(self.gpm.stationary_variables)
        n_lags = self.gpm.var_prior_setup.var_order if self.gpm.var_prior_setup else 1
        
        state_dim = n_trend + n_stat * n_lags
        
        # Build F matrix based on trend equations and VAR structure
        F = jnp.eye(state_dim)  # Placeholder
        
        # Build other matrices...
        R = jnp.eye(state_dim)  # Placeholder
        C = jnp.eye(n_obs, state_dim)  # Placeholder
        H = jnp.eye(n_obs) * 1e-6  # Placeholder
        
        init_mean = jnp.zeros(state_dim)
        init_cov = jnp.eye(state_dim) * 1e6
        
        return F, R, C, H, init_mean, init_cov


# # Example usage
# def example_gpm_usage():
#     """Example of parsing and using a GPM file"""
    
#     # Parse the GPM file
#     parser = GPMParser()
#     gpm_model = parser.parse_file("model_with_trends.gpm")
    
#     print("Parsed GPM Model:")
#     print(f"Parameters: {gpm_model.parameters}")
#     print(f"Trend variables: {gpm_model.trend_variables}")
#     print(f"Stationary variables: {gpm_model.stationary_variables}")
#     print(f"Observed variables: {gpm_model.observed_variables}")
    
#     if gpm_model.var_prior_setup:
#         print(f"VAR order: {gpm_model.var_prior_setup.var_order}")
#         print(f"VAR priors: es={gpm_model.var_prior_setup.es}, fs={gpm_model.var_prior_setup.fs}")
    
#     # Build and use the model
#     builder = GPMModelBuilder(gpm_model)
#     model_fn = builder.build_numpyro_model()
    
#     return gpm_model, model_fn


# if __name__ == "__main__":
    example_gpm_usage()