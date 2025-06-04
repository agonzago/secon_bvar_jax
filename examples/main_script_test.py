# examples/main_script_test.py - SIMPLE FIXES

import sys
import os

# FIX 1: Add this line to use non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Add the repository root to sys.path to locate the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clean_gpm_bvar_trends.complete_gpm_workflow import run_complete_gpm_analysis, run_quick_example

def main():
    print("\n--- Running Full Analysis Example (MCMC) ---")
    
    # FIX 2: Create the output directory explicitly
    output_dir = os.path.join(os.getcwd(), "example_output", "full_mcmc_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Path for data file relative to the repository root
    data_for_full_run = "/home/andres/secon_bvar_jax/Application/data_m5.csv"

    # Path for GPM file relative to the repository root
    gpm_file_for_full_run = "/home/andres/secon_bvar_jax/clean_gpm_bvar_trends/models/gpm_factor_y_pi_rshort.gpm"

    print(f"Attempting to use GPM file: {os.path.abspath(gpm_file_for_full_run)}")
    print(f"Attempting to use data file: {os.path.abspath(data_for_full_run)}")

    results_full = run_complete_gpm_analysis(
        data=data_for_full_run,         
        gpm_file=gpm_file_for_full_run, 
        analysis_type="mcmc",
        num_warmup=200,
        num_samples=500,
        num_chains=4,
        
        # Add these plot parameters
        generate_plots=True,
        save_plots=True,
        plot_save_path=output_dir
    )

    if results_full:
        print("Full MCMC analysis example completed.")
        
        # FIX 5: Check what files were actually created
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            print(f"Generated {len(files)} plot files:")
            for f in files:
                print(f"  - {f}")
        else:
            print("Output directory not found")
    else:
        print("Full MCMC analysis example failed.")

if __name__ == "__main__":
    main()