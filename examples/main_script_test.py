# examples/main_script_test.py

import sys
import os
# Add the repository root to sys.path to locate the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clean_gpm_bvar_trends.complete_gpm_workflow import run_complete_gpm_analysis, run_quick_example

def main():
    print("--- Running Quick Example (MCMC) ---")
    # This example uses a default, simple GPM model created internally by run_quick_example
    # and internally generated data. It's good for a basic test.
    # Output will be in examples/example_output/quick_mcmc relative to where script is run.
    # If script is run from examples/, then it's examples/example_output/quick_mcmc
    results_quick = run_quick_example(analysis_type="mcmc", save_results=True, output_dir="example_output/quick_mcmc")
    if results_quick:
        print("Quick MCMC example completed. Results (like plots) in 'example_output/quick_mcmc'.")
    else:
        print("Quick MCMC example failed.")

    print("\n--- Running Full Analysis Example (MCMC) ---")
    # This script (main_script_test.py) is in the 'examples' directory.
    # data_m5.csv is also in the 'examples' directory.
    # The GPM model is in 'clean_gpm_bvar_trends/models/' relative to the repository root.

    # Path for data file relative to the repository root
    data_for_full_run = "examples/data_m5.csv"

    # Path for GPM file relative to the repository root
    gpm_file_for_full_run = "clean_gpm_bvar_trends/models/gpm_factor_y_pi_rshort.gpm"

    print(f"Attempting to use GPM file: {os.path.abspath(gpm_file_for_full_run)}")
    print(f"Attempting to use data file: {os.path.abspath(data_for_full_run)}")

    results_full = run_complete_gpm_analysis(
        data=data_for_full_run,         # Note: run_complete_gpm_analysis uses 'data'
        gpm_file=gpm_file_for_full_run, # Note: run_complete_gpm_analysis uses 'gpm_file'
        analysis_type="mcmc",
        num_warmup=50, # Keep low for example
        num_samples=50, # Keep low for example
        num_chains=1,
        save_plots=True,
        plot_save_path="example_output/full_mcmc_analysis" # Subdirectory for these plots
    )

    if results_full:
        print("Full MCMC analysis example completed. Results in 'example_output/full_mcmc_analysis'.")
    else:
        print("Full MCMC analysis example failed.")
        print("Please ensure:")
        print(f"  1. GPM file exists at: {os.path.abspath(gpm_file_for_full_run)}")
        print(f"  2. Data file exists at: {os.path.abspath(data_for_full_run)}")
        print(f"  3. The GPM file is compatible with the data.")

if __name__ == "__main__":
    main()