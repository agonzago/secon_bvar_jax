# applications/main.py
import sys
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import time

import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'clean_gpm_bvar_trends'))
from gpm_bar_smoother import complete_gpm_workflow_with_smoother
from gpm_prior_calibration_example import run_sensitivity_analysis_workflow
from constants import _DEFAULT_DTYPE
from gpm_numpyro_models import fit_gpm_numpyro_model


import jax
import jax.numpy as jnp
# Configure JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpyro
numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)  # Use 4 CPU cores for parallel chains


# Four lines to use the GPM framework:
dta_path=os.path.join(os.path.dirname(__file__),"data_m5.csv")
data_source = dta_path  # Load your data file

# 1. Load or Generate Data

dta = pd.read_csv(data_source)
dta['Date'] = pd.to_datetime(dta['Date'])  # Convert '3/31/1970' to datetime
dta.set_index('Date', inplace=True)
dta = dta.asfreq('QE')  
# Subsample data 
data_sub = dta[['y_us', 'y_ea', 'y_jp']]


import matplotlib.pyplot as plt

# Plot all three series
data_sub.plot(figsize=(12, 8))
plt.title('US, EA, JP Growth Rates')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


#data for mcmc estimation 
y_numpy = data_sub.values.astype(_DEFAULT_DTYPE)
# y_jax = jnp.asarray(y_numpy)
# T_actual, N_actual_obs = y_jax.shape
# print(f"   ✓ Data shape: ({T_actual}, {N_actual_obs})")


print(f"\n3. Fitting gpm Model...")

gpm_file = 'gdps_1.gpm'
gpm_file=os.path.join(os.path.dirname(__file__),'gdps_1.gpm')
# start_time = time.time()
# mcmc_results, parsed_gpm_model, state_space_builder = fit_gpm_numpyro_model(
#     gpm_file_path=gpm_file,
#     y_data=y_jax,
#     num_warmup=100,
#     num_samples=100,
#     num_chains=2,
#     rng_key_seed=47,
#     use_gamma_init_for_P0=True,
#     gamma_init_scaling_for_P0=1.0,
#     target_accept_prob=0.9
# )
# fit_time = time.time() - start_time
# print(f"   ✓ MCMC completed in {fit_time:.1f}s")

# mcmc_results.print_summary()
# if mcmc_results is None:
#     raise RuntimeError("MCMC fitting returned None")


results = complete_gpm_workflow_with_smoother(
    data=data_sub,
    gpm_file = gpm_file, 
    num_warmup  = 100,
    num_samples = 10,
    num_chains  = 2,
    target_accept_prob = 0.85,
    # P0 initialization settings
    use_gamma_init = False,
    gamma_scale_factor = 1.0,
    # Smoother settings
    num_extract_draws = 5,
    # Plotting
    generate_plots = True,
    hdi_prob_plot  = 0.9)  


# sensitivity = run_sensitivity_analysis_workflow(config=None, param_to_study="rho")  # Try alternative priors  
# print(results.keys())  # Results contain: mcmc_object, reconstructed_trend_draws, reconstructed_stationary_draws, etc.
