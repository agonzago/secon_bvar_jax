# --- START OF FILE example_single_parameter_run.py ---

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os # Needed to check if files exist
from typing import Dict, List, Optional

# Import the main test function from the test utilities file
from gpm_test_utils import test_gpm_model_with_parameters

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 # Ensure this is defined

# --- Configuration ---
DATA_FILE_PATH = 'sim_data.csv'      # <-- SET THIS to your CSV data file
GPM_FILE_PATH = 'test_model.gpm'     # <-- SET THIS to your GPM model file
# Example: Assuming your data has columns like 'GDP', 'Inflation', 'Interest Rate'
OBSERVED_VARIABLE_NAMES = ['OBS1', 'OBS2', 'OBS3'] # <-- SET THIS to the names of the *observed* variables in your GPM file (matching CSV column names is common)

# Example fixed parameter values - replace with values from your MCMC results
# The keys here MUST match the parameter names in your GPM file's 'estimated_params' section
FIXED_PARAMETER_VALUES: Dict[str, float] = {
    # Example:
    'sigma_SHK_TREND1': 0.15,
    'sigma_SHK_TREND2': 0.20,
    'sigma_SHK_TREND3': 0.10,
    'sigma_SHK_STAT1': 0.30,
    'sigma_SHK_STAT2': 0.40,
    'sigma_SHK_STAT3': 0.20,
    # Add any structural coefficients if they are in your GPM estimated_params
    # e.g., 'my_beta_coeff': 0.5,
}
# --- End Configuration ---


def run_my_single_parameter_test():
    """
    Loads data and GPM, runs the single parameter test, and plots results.
    """
    print(f"--- Running Single Parameter Test Workflow ---")
    print(f"Data file: {DATA_FILE_PATH}")
    print(f"GPM file: {GPM_FILE_PATH}")

    # --- 1. Load Data ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please update DATA_FILE_PATH in this script.")
        return None

    try:
        # Read the CSV file using pandas
        dta = pd.read_csv(DATA_FILE_PATH)
        # Convert to NumPy array
        y_np = dta.values
        # Convert to JAX NumPy array with the desired dtype
        y_jax = jnp.asarray(y_np, dtype=_DEFAULT_DTYPE)
        T, n_obs = y_jax.shape
        print(f"Successfully loaded data with shape: {y_jax.shape}")

    except Exception as e:
        print(f"Error loading or processing data file: {e}")
        return None

    # --- 2. Check GPM File ---
    if not os.path.exists(GPM_FILE_PATH):
        print(f"Error: GPM file not found at {GPM_FILE_PATH}")
        print("Please update GPM_FILE_PATH in this script.")
        # Optional: You could add a function here to create a *placeholder* GPM file
        # if it doesn't exist, similar to the one in single_parameter_run.py
        return None

    # --- 3. Run the Test Function ---
    print("\nCalling test_gpm_model_with_parameters...")
    try:
        results = test_gpm_model_with_parameters(
            gpm_file_path=GPM_FILE_PATH,
            y=y_jax,
            param_values=FIXED_PARAMETER_VALUES,
            num_sim_draws=100,          # Set number of simulation smoother draws
            rng_key=jax.random.PRNGKey(123), # Set random key for reproducibility
            plot_results=True,          # Set to True to show plots
            variable_names=OBSERVED_VARIABLE_NAMES, # Pass observed variable names for plots
            use_gamma_init_for_test=True, # Set to True to use gamma-based init cov, False for 1e6/1e-6
            gamma_init_scaling=1.0      # Scaling factor for gamma init cov (try < 1 if needed)
        )

        print("\n--- Test Run Complete ---")
        if 'loglik' in results:
             print(f"Final Log-likelihood: {results['loglik']:.2f}")
        if 'sim_draws' in results:
             print(f"Total simulation draws obtained: {results['sim_draws'].shape[0]}")
        if 'error' in results:
            print(f"Test encountered an error: {results['error']}")

        return results

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Ensure JAX is configured before running
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu") # You might want to use GPU if available

    test_results = run_my_single_parameter_test()

    if test_results:
        print("\nWorkflow finished.")
    else:
        print("\nWorkflow failed.")

# --- END OF FILE example_single_parameter_run.py ---