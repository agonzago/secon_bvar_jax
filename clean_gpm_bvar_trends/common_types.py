# common_types.py
from typing import NamedTuple, Dict, Optional
import jax.numpy as jnp # If jnp is used in type hints


# Keep existing EnhancedBVARParams structure unchanged
class EnhancedBVARParams(NamedTuple):
    """Enhanced parameters for BVAR with trends supporting GPM specifications"""
    # Core VAR parameters
    A: jnp.ndarray                    # VAR coefficient matrices
    Sigma_u: jnp.ndarray             # Stationary innovation covariance
    Sigma_eta: jnp.ndarray           # Trend innovation covariance
    
    # Structural parameters from GPM
    structural_params: Dict[str, jnp.ndarray] = {}
    
    # Measurement error (optional)
    Sigma_eps: Optional[jnp.ndarray] = None


