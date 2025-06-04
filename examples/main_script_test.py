# examples/main_script_test.py - SIMPLE FIXES

import sys
import os
#Hola
# FIX 1: Add this line to use non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Add the repository root to sys.path to locate the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clean_gpm_bvar_trends.complete_gpm_workflow import run_complete_gpm_analysis, run_quick_example, create_example_gpm_model, generate_example_data
import pandas as pd
import numpy as np # Ensure numpy is imported as np
import jax.numpy as jnp # For creating example data if needed directly

def main():
    # --- Original Full Analysis Example (Optional, can be kept or commented out) ---
    # print("\n--- Running Full Analysis Example (MCMC) ---")
    # output_dir_full = os.path.join(os.getcwd(), "example_output", "full_mcmc_analysis")
    # os.makedirs(output_dir_full, exist_ok=True)
    # print(f"Output directory for full analysis: {output_dir_full}")
    # data_for_full_run = "/home/andres/secon_bvar_jax/Application/data_m5.csv" # This path is specific, might not exist elsewhere
    # gpm_file_for_full_run = "/home/andres/secon_bvar_jax/clean_gpm_bvar_trends/models/gpm_factor_y_pi_rshort.gpm" # Also specific
    # print(f"Attempting to use GPM file: {os.path.abspath(gpm_file_for_full_run)}")
    # print(f"Attempting to use data file: {os.path.abspath(data_for_full_run)}")
    # results_full = run_complete_gpm_analysis(
    #     data=data_for_full_run,
    #     gpm_file=gpm_file_for_full_run,
    #     analysis_type="mcmc",
    #     num_warmup=200, # Reduced for quicker test if uncommented
    #     num_samples=500, # Reduced
    #     num_chains=2, # Reduced
    #     generate_plots=True,
    #     save_plots=True,
    #     plot_save_path=output_dir_full
    # )
    # if results_full:
    #     print("Full MCMC analysis example completed.")
    # else:
    #     print("Full MCMC analysis example failed.")

    print("\n\n--- Running P0 Override Examples (MCMC) ---")
    
    # Setup for P0 override examples
    example_gpm_filename = "test_p0_override_model.gpm"
    # Create a model with 2 trend variables (TREND1, TREND2) and 2 stationary variables (STAT1, STAT2)
    # OBS1 = TREND1 + STAT1, OBS2 = TREND2 + STAT2
    create_example_gpm_model(filename=example_gpm_filename, num_obs_vars=2, num_stat_vars=2, include_var_setup=True)

    # Generate some synthetic data for this model
    # For simplicity, using random data. For more realistic test, use generate_example_data from workflow
    num_periods = 100
    example_data_np = np.random.randn(num_periods, 2) * 5 + np.arange(num_periods)[:, None] * 0.1 # Trend + noise
    example_data_df = pd.DataFrame(example_data_np, columns=['OBS1', 'OBS2'],
                                   index=pd.date_range(start='2000-01-01', periods=num_periods, freq='QE'))

    p0_output_base_dir = os.path.join(os.getcwd(), "example_output", "p0_override_tests")

    common_mcmc_params = {
        "num_warmup": 50, # Quick run
        "num_samples": 100, # Quick run
        "num_chains": 1,
        "generate_plots": True,
        "save_plots": True,
        "use_gamma_init": False, # Using standard P0 for simpler override demonstration unless gamma is specifically tested
    }

    # Scenario 1: mcmc_trend_P0_scales as a dictionary
    print("\n--- Scenario 1: MCMC Trend P0 Scales (Dict) ---")
    output_dir_s1 = os.path.join(p0_output_base_dir, "scenario1_trend_dict")
    os.makedirs(output_dir_s1, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        mcmc_trend_P0_scales={"TREND1": 1e7, "TREND2": 8e5}, # TREND1, TREND2 are default names from create_example_gpm_model
        plot_save_path=output_dir_s1, **common_mcmc_params
    )

    # Scenario 2: mcmc_trend_P0_scales as a float
    print("\n--- Scenario 2: MCMC Trend P0 Scales (Float) ---")
    output_dir_s2 = os.path.join(p0_output_base_dir, "scenario2_trend_float")
    os.makedirs(output_dir_s2, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        mcmc_trend_P0_scales=2e6,
        plot_save_path=output_dir_s2, **common_mcmc_params
    )

    # Scenario 3: mcmc_stationary_P0_scale as a float
    print("\n--- Scenario 3: MCMC Stationary P0 Scale (Float) ---")
    output_dir_s3 = os.path.join(p0_output_base_dir, "scenario3_stat_float")
    os.makedirs(output_dir_s3, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        mcmc_stationary_P0_scale=12.0,
        plot_save_path=output_dir_s3, **common_mcmc_params
    )

    # Scenario 4: Combination of dict trend scales and stationary scale
    print("\n--- Scenario 4: MCMC Trend (Dict) & Stationary (Float) P0 Scales ---")
    output_dir_s4 = os.path.join(p0_output_base_dir, "scenario4_trend_dict_stat_float")
    os.makedirs(output_dir_s4, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        mcmc_trend_P0_scales={"TREND1": 1.5e7, "TREND2": 7e5},
        mcmc_stationary_P0_scale=15.0,
        plot_save_path=output_dir_s4, **common_mcmc_params
    )

    # Scenario 5: Combination of float trend scale and stationary scale (with gamma init for MCMC)
    print("\n--- Scenario 5: MCMC Trend (Float) & Stationary (Float) P0 Scales with Gamma P0 for MCMC ---")
    output_dir_s5 = os.path.join(p0_output_base_dir, "scenario5_trend_float_stat_float_gamma")
    os.makedirs(output_dir_s5, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        mcmc_trend_P0_scales=3e6,
        mcmc_stationary_P0_scale=8.0,
        use_gamma_init=True, # Test interaction with gamma P0 for MCMC
        gamma_scale_factor=1.5,
        plot_save_path=output_dir_s5, **common_mcmc_params
    )

    # Scenario 6: Smoother P0 overrides (dictionary for trends, float for stationary)
    # MCMC will use its defaults (or could be combined with MCMC overrides too)
    print("\n--- Scenario 6: Smoother P0 Scales (Trend Dict, Stat Float) ---")
    output_dir_s6 = os.path.join(p0_output_base_dir, "scenario6_smoother_trend_dict_stat_float")
    os.makedirs(output_dir_s6, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        # MCMC P0s can be default or specified:
        # mcmc_trend_P0_scales=1e6,
        smoother_trend_P0_scales={"TREND1": 2e7, "TREND2": 1e6}, # Different scales for smoother
        smoother_stationary_P0_scale=20.0,
        plot_save_path=output_dir_s6, **common_mcmc_params
    )

    # Scenario 7: Smoother P0 overrides (float for trends, float for stationary)
    print("\n--- Scenario 7: Smoother P0 Scales (Trend Float, Stat Float) ---")
    output_dir_s7 = os.path.join(p0_output_base_dir, "scenario7_smoother_trend_float_stat_float")
    os.makedirs(output_dir_s7, exist_ok=True)
    run_complete_gpm_analysis(
        data=example_data_df.copy(), gpm_file=example_gpm_filename, analysis_type="mcmc",
        smoother_trend_P0_scales=2.5e6,
        smoother_stationary_P0_scale=25.0,
        plot_save_path=output_dir_s7, **common_mcmc_params
    )

    print("\nAll P0 override examples completed.")

    # Clean up the example GPM file
    if os.path.exists(example_gpm_filename):
        os.remove(example_gpm_filename)
        print(f"Cleaned up {example_gpm_filename}")

    # --- Original Quick Example (can be kept) ---
    print("\n--- Running Quick Example (MCMC) ---")
    # output_dir_quick = os.path.join(os.getcwd(), "example_output", "quick_mcmc") # FIX 3 & 4
    # os.makedirs(output_dir_quick, exist_ok=True)
    # quick_results = run_quick_example(analysis_type="mcmc", save_results=True, output_dir=output_dir_quick)
    # if quick_results:
    #     print("Quick MCMC example completed.")
    # else:
    #     print("Quick MCMC example failed.")

    # FIX 6: Call main()
if __name__ == "__main__":
    main()
    # Also run the quick example as a final check if desired
    print("\n--- Running Quick Example (MCMC) as final check ---")
    output_dir_quick_final = os.path.join(os.getcwd(), "example_output", "quick_mcmc_final")
    os.makedirs(output_dir_quick_final, exist_ok=True)
    run_quick_example(analysis_type="mcmc", save_results=True, output_dir=output_dir_quick_final)