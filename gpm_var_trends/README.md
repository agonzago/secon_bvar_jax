# GPM Framework for BVAR with Trends

## Overview

The GPM (Generalized Parametric Model) framework provides a declarative way to specify and estimate different variants of BVAR with trends models. Similar to Dynare's approach, you can define your model structure in a `.gpm` file and the framework automatically builds the appropriate state space representation and sets up the Bayesian estimation.

## Key Features

- **Declarative Model Specification**: Define models using human-readable `.gpm` files
- **Flexible Trend Specifications**: Support for various trend dynamics and structural relationships
- **Hierarchical VAR Priors**: Configurable shrinkage priors for stationary components
- **Automatic State Space Construction**: Framework builds state space matrices from model specification
- **Integration with JAX/Numpyro**: Efficient Bayesian estimation with modern computational tools

## GPM File Structure

A `.gpm` file consists of several sections:

### 1. Parameters
```
parameters b1, b2, alpha;
```
Lists structural parameters to be estimated.

### 2. Estimated Parameters
```
estimated_params;
    stderr SHK_GDP_TREND, inv_gamma_pdf, 2.0, 1.0;
    b1, normal_pdf, 0.1, 0.2;
end;
```
Specifies priors for parameters and shock standard deviations.

### 3. Variable Classifications
```
trends_vars
    GDP_TREND,
    INFLATION_TREND
;

stationary_variables
    GDP_GAP,
    INFLATION_GAP
;
```

### 4. Shock Specifications
```
trend_shocks;
    var SHK_GDP_TREND
    var SHK_INFLATION_TREND
end;

shocks;
    var SHK_GDP_GAP
    var SHK_INFLATION_GAP
end;
```

### 5. Trend Model
```
trend_model;
    GDP_TREND = GDP_TREND(-1) + b1*INFLATION_TREND(-1) + SHK_GDP_TREND;
    INFLATION_TREND = INFLATION_TREND(-1) + SHK_INFLATION_TREND;
end;
```
Defines the dynamics of trend components.

### 6. Observation and Measurement
```
varobs 
    GDP_OBS,
    INFLATION_OBS
;

measurement_equations;
    GDP_OBS = GDP_TREND + GDP_GAP;
    INFLATION_OBS = INFLATION_TREND + INFLATION_GAP;
end;
```

### 7. VAR Prior Setup
```
var_prior_setup;
    var_order = 2;
    es = 0.6, 0.15;  // Diagonal, off-diagonal means
    fs = 0.15, 0.15; // Diagonal, off-diagonal std devs
    gs = 3.0, 3.0;   // Gamma shape parameters
    hs = 1.0, 1.0;   // Gamma rate parameters
    eta = 2.0;       // LKJ concentration
end;
```

### 8. Initial Values
```
initval;
    GDP_GAP, normal_pdf, 0, 0.1;
    GDP_TREND, normal_pdf, 0, 10;
end;
```

## Usage Examples

### Basic Usage

```python
from enhanced_bvar_with_gpm import fit_gpm_model
import jax.numpy as jnp

# Load your data
y = jnp.array(your_data)  # Shape: (T, n_variables)

# Fit model specified in GPM file
mcmc, gpm_model, ss_builder = fit_gpm_model(
    'your_model.gpm', 
    y, 
    num_warmup=1000, 
    num_samples=2000, 
    num_chains=4
)

# Print results
mcmc.print_summary()
```

### Parsing Only

```python
from gpm_parser import GPMParser

parser = GPMParser()
gpm_model = parser.parse_file('your_model.gpm')

print(f"Trend variables: {gpm_model.trend_variables}")
print(f"VAR order: {gpm_model.var_prior_setup.var_order}")
```

### Model Comparison

```python
from gpm_usage_example import compare_model_variants

# Compare different model specifications
results = compare_model_variants()

for model_name, result in results.items():
    if result is not None:
        print(f"{model_name}: successful")
        # Access MCMC results: result['mcmc']
```

## Model Variants

The framework supports various model types:

### 1. Simple Random Walk Trends
- Independent random walk processes for each trend
- Minimal structural relationships

### 2. Cointegrated Trends
- Common stochastic trends with loading coefficients
- Structural relationships between trends

### 3. Macroeconomic Models
- GDP trends with productivity relationships
- Inflation targeting frameworks
- Interest rate rules

### 4. Custom Specifications
- Arbitrary trend dynamics
- Complex measurement equations
- Structural coefficients

## Advanced Features

### Custom Prior Distributions

The framework supports:
- `normal_pdf`: Normal distribution
- `inv_gamma_pdf`: Inverse Gamma distribution
- Easily extensible to other distributions

### Hierarchical VAR Priors

Configure shrinkage for stationary components:
- `es`: Prior means for diagonal/off-diagonal coefficients
- `fs`: Prior standard deviations
- `gs`, `hs`: Gamma parameters for precision priors
- `eta`: LKJ concentration for correlation matrix

### State Space Integration

The framework automatically:
- Builds transition matrices from trend equations
- Constructs observation matrices from measurement equations
- Handles VAR dynamics for stationary components
- Ensures numerical stability

## Integration with Existing Code

The GPM framework integrates seamlessly with your existing BVAR infrastructure:

- Uses your existing `stationary_prior_jax_simplified.py` for stationarity enforcement
- Leverages your `Kalman_filter_jax.py` for efficient likelihood computation
- Compatible with your Durbin & Koopman simulation smoother
- Works with your plotting and diagnostic functions

## File Structure

```
├── gpm_parser.py              # Core GPM parsing functionality
├── enhanced_bvar_with_gmp.py  # Enhanced BVAR with GPM integration
├── gmp_usage_example.py       # Usage examples and demonstrations
├── model_with_trends.gpm      # Your original model specification
└── README.md                  # This file
```

## Error Handling

The framework includes robust error handling:
- Validates GPM file syntax
- Checks for missing required sections
- Handles numerical instabilities in state space construction
- Provides informative error messages

## Performance Considerations

- Uses JAX for efficient automatic differentiation
- Supports multi-chain parallel MCMC
- Optimized state space operations
- Memory-efficient matrix operations

## Extending the Framework

### Adding New Prior Distributions

```python
def _sample_parameter(name: str, prior_spec: PriorSpec):
    if prior_spec.distribution == 'your_new_distribution':
        # Implementation here
        return numpyro.sample(name, your_distribution(*prior_spec.params))
```

### Custom Trend Dynamics

Simply specify in the `trend_model` section of your `.gpm` file:
```
trend_model;
    YOUR_TREND = custom_function(OTHER_VARS) + SHOCK;
end;
```

### Additional Measurement Structures

The measurement equations support arbitrary linear combinations:
```
measurement_equations;
    OBS = coeff1*TREND1 + coeff2*TREND2 + STATIONARY;
end;
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure `.gpm` file is in the correct directory
2. **Parsing Errors**: Check syntax, especially semicolons and section markers
3. **Numerical Issues**: Verify parameter priors are reasonable
4. **Convergence Problems**: Increase warmup samples or adjust priors

### Debugging Tips

- Use the parser directly to check file parsing
- Start with simple model specifications
- Check data preprocessing and scaling
- Verify variable name consistency across sections

## Examples

See `gmp_usage_example.py` for comprehensive examples including:
- Parsing your existing model file
- Generating synthetic data
- Fitting different model variants
- Comparing model performance

## Future Extensions

Planned enhancements:
- Support for time-varying parameters
- Non-linear trend dynamics
- Mixed-frequency data handling
- Model averaging capabilities
- Additional prior specifications