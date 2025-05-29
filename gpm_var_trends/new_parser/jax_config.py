
"""
JAX Configuration Module
Centralized JAX configuration for all modules
"""

import jax
import os

def configure_jax():
    """Configure JAX for consistent behavior across all modules"""
    
    # Configure JAX for float64 precision and CPU execution
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    
    # Set environment variables for consistency
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    print("✓ JAX configured: CPU platform, float64 precision enabled")
    
    # Test configuration
    import jax.numpy as jnp
    test_array = jnp.array([1.0])
    print(f"✓ JAX test: dtype={test_array.dtype}, platform={jax.devices()[0].platform}")

# Configure JAX as soon as this module is imported
configure_jax()