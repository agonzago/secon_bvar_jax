
"""
JAX Configuration Module
Centralized JAX configuration for all modules
"""
# In clean_gpm_bvar_trends/jax_config.py
import jax
import os

def configure_jax():
    # It's often better to set XLA_FLAGS as an environment variable *before*
    # Python starts, but setting it here can work in some cases.
    if "XLA_FLAGS" not in os.environ:
        # Example: Force JAX to use a specific number of CPU devices.
        # Adjust if you have multiple CPUs and want to control parallelism for JAX itself.
        # For NumPyro MCMC, num_chains often controls CPU usage more directly.
        # num_cpus = os.cpu_count() or 1 # Get number of CPUs
        # os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cpus}"
        pass # Often, you don't need to force this if JAX detects CPUs correctly.

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu") # Or "gpu"/"tpu" if available and desired

    print("✓ JAX configured from jax_config.py: Target platform CPU, float64 enabled.")
    # Test configuration (optional, but good for verification)
    # import jax.numpy as jnp
    # test_array = jnp.array([1.0])
    # print(f"✓ JAX test: dtype={test_array.dtype}, platform={jax.devices()[0].platform}")

if __name__ == 'clean_gpm_bvar_trends.jax_config': # Ensures it runs when imported
    configure_jax()