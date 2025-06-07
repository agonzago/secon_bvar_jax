# Clean GPM BVAR Trends Package

This package provides tools for GPM (Generalized Potentially Misspecified) BVAR (Bayesian Vector Autoregression) trend analysis. It is designed to allow users to define, fit, and analyze BVAR models with various trend specifications.

## Features

- Define GPM models using `.gpm` files.
- Fit models using MCMC methods via NumPyro.
- Analyze results, including trend decomposition and plotting.
- Calibrate priors using provided utility scripts.

## GPM Processing Pipeline

A complete analysis follows a three stage pipeline:

1. **Parsing** – The parser reads the `.gpm` file and produces a canonical
   ordering of all variables as well as the relationships between them.
2. **Reduction** – Using this canonical order, the `StateSpaceBuilder` creates
   a minimal state space representation that is ready for Kalman filtering.
3. **Reconstruction** – After smoothing, the state vector is converted back
   into economic variables using the exact same ordering established by the
   parser.

Keeping the ordering consistent across these stages is critical.  The parser is
the single source of truth and the reduction and reconstruction stages use the
provided lists (e.g. `core_variables`, `gpm_trend_variables_original`) to map
states to variable names reliably.

## Installation

To install the package, clone the repository and install it using pip:

```bash
git clone https://github.com/yourusername/yourrepository # Replace with actual URL
cd yourrepository
pip install .
```

## Usage

The primary way to use the package is by importing functions from `clean_gpm_bvar_trends`. The `main_script_test.py` in the `examples` directory (once moved) demonstrates basic usage.

```python
# Example from examples/main_script_test.py (path may change)
from clean_gpm_bvar_trends import run_complete_gpm_analysis, run_quick_example

# Run a quick example
results_quick = run_quick_example(analysis_type="mcmc", save_results=True)

# Run with your own data and GPM model
# Ensure your GPM model file (e.g., my_model.gpm) is accessible,
# for example, by placing it in clean_gpm_bvar_trends/models/
# and data (e.g., my_data.csv) is in the working directory or provide a full path.

    # Ensure 'my_data.csv' is in your current working directory or provide a full path.
    # Ensure 'my_model.gpm' is accessible. If using models from the package,
    # you might need to locate them within the installed package's 'models' directory.
    # For a direct path:
    my_gpm_file = "path/to/your/my_model.gpm"
    # If you copied one from the package's models folder for example:
    # my_gpm_file = "./my_copied_model.gpm"

    results_full = run_complete_gpm_analysis(
        data="my_data.csv",  # Path to your data file
        gpm_file=my_gpm_file, # Path to your GPM file
        analysis_type="mcmc"  # or "fixed_params"
    )

print(results_full.keys())
```

## Models

The package uses `.gpm` files to define models. Example models are provided in the `clean_gpm_bvar_trends/models` directory.

## Testing

The package requires `yax` and `numpyro` for its core functionality and testing. Ensure these are installed (they are included in `install_requires`). JAX is configured to use CPU by default.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This package is licensed under the MIT License. See the `LICENSE` file for details.
