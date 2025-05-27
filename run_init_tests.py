import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import os

# Attempt to import from the GPM core package
try:
    from gpmcore.gpm_bvar_trends import fit_gpm_model
    from gpmcore.gpm_parser import GPMParser # For creating GPM model object if needed by fit_gpm_model directly
except ImportError as e:
    print(f"Failed to import from gpmcore: {e}")
    print("Ensure gpmcore is correctly structured and __init__.py is set up.")
    exit()

# Basic GPM model content with initial values
gpm_content_for_test = '''
parameters ; 

estimated_params;
    stderr SHK_TREND1, inv_gamma_pdf, 2.0, 1.0;
    stderr SHK_TREND2, inv_gamma_pdf, 2.0, 1.0;
    stderr SHK_STAT1, inv_gamma_pdf, 2.0, 1.0;
    stderr SHK_STAT2, inv_gamma_pdf, 2.0, 1.0;
end;

trends_vars
    TREND1,
    TREND2
;

stationary_variables
    STAT1,
    STAT2
;

trend_shocks;
    var SHK_TREND1
    var SHK_TREND2
end;

shocks;
    var SHK_STAT1
    var SHK_STAT2
end;

trend_model;
    TREND1 = TREND1(-1) + SHK_TREND1;
    TREND2 = TREND2(-1) + SHK_TREND2;
end;

varobs 
    OBS1,
    OBS2
;

measurement_equations;
    OBS1 = TREND1 + STAT1;
    OBS2 = TREND2 + STAT2;
end;

var_prior_setup;
    var_order = 1; // Simplified VAR order
    es = 0.6, 0.15;
    fs = 0.15, 0.15;
    gs = 3.0, 3.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;

initial_values; // Crucial for MCMC conditional prior test
    TREND1, normal_pdf, 0.0, 5.0; 
end;
'''

# Save to a temporary .gpm file
test_gpm_file_path = 'temp_test_model.gpm'
with open(test_gpm_file_path, 'w') as f:
    f.write(gpm_content_for_test)

# Generate or load simple data
T = 50
n_obs_vars = 2
np.random.seed(42)
y_data_np = np.random.randn(T, n_obs_vars)
y_data_jax = jnp.array(y_data_np)

# MCMC settings for quick tests
mcmc_params_test = {
    'num_warmup': 10, # Minimal
    'num_samples': 20, # Minimal
    'num_chains': 1
}

rng_key_test = random.PRNGKey(123)

initialization_methods_to_test = {
    "default": {"use_conditional_init": False, "use_mcmc_conditional_init": False, "use_hierarchical_init": False},
    "conditional_init": {"use_conditional_init": True, "use_mcmc_conditional_init": False, "use_hierarchical_init": False},
    "mcmc_conditional_init": {"use_conditional_init": False, "use_mcmc_conditional_init": True, "use_hierarchical_init": False},
    "hierarchical_init": {"use_conditional_init": False, "use_mcmc_conditional_init": False, "use_hierarchical_init": True},
}

print("Starting tests for different initial condition methods...")

for method_name, flags in initialization_methods_to_test.items():
    print(f"\n--- Testing method: {method_name} ---")
    current_rng_key, rng_key_test = random.split(rng_key_test) # Ensure fresh key for each run
    try:
        mcmc_result, _, _ = fit_gpm_model(
            gpm_file_path=test_gpm_file_path,
            y=y_data_jax,
            num_warmup=mcmc_params_test['num_warmup'],
            num_samples=mcmc_params_test['num_samples'],
            num_chains=mcmc_params_test['num_chains'],
            rng_key=current_rng_key,
            use_conditional_init=flags["use_conditional_init"],
            use_mcmc_conditional_init=flags["use_mcmc_conditional_init"],
            use_hierarchical_init=flags["use_hierarchical_init"]
            # conditioning_strength will use its default in fit_gpm_model
        )
        print(f"Successfully ran MCMC for {method_name}.")
        if mcmc_result:
            mcmc_result.print_summary(exclude_deterministic=False)
        else:
            print("MCMC result was None.")
        print(f"--- Test for {method_name} PASSED (completed run) ---")
    except Exception as e:
        print(f"!!! Test for {method_name} FAILED: {e} !!!")
        import traceback
        traceback.print_exc()

# Clean up the temporary GPM file
if os.path.exists(test_gpm_file_path):
    os.remove(test_gpm_file_path)
print("\nAll tests concluded.")
