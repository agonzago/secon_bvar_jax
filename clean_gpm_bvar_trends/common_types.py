# clean_gpm_bvar_trends/common_types.py
from typing import NamedTuple, Dict, Optional, List, Any # Added List, Any
import jax.numpy as jnp
from dataclasses import dataclass # Added dataclass
import numpy as np # Added numpy for array type hint

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


@dataclass
class SmootherResults:
    """Standardized container for simulation smoother results."""
    observed_data: np.ndarray            # (T, N_obs)
    observed_variable_names: List[str]   # (N_obs,)
    time_index: Optional[Any]            # Time index (e.g., pd.Index)

    trend_draws: np.ndarray              # (N_draws, T, N_trends)
    trend_names: List[str]               # (N_trends,)
    trend_stats: Dict[str, np.ndarray]   # Summary stats (mean, median, etc.), each (T, N_trends)
    trend_hdi_lower: Optional[np.ndarray]# HDI lower bounds (T, N_trends)
    trend_hdi_upper: Optional[np.ndarray]# HDI upper bounds (T, N_trends)

    stationary_draws: np.ndarray         # (N_draws, T, N_stat)
    stationary_names: List[str]          # (N_stat,)
    stationary_stats: Dict[str, np.ndarray]# Summary stats (mean, median, etc.), each (T, N_stat)
    stationary_hdi_lower: Optional[np.ndarray]# HDI lower bounds (T, N_stat)
    stationary_hdi_upper: Optional[np.ndarray]# HDI upper bounds (T, N_stat)

    # Information derived from the GPM model used
    reduced_measurement_equations: Optional[Dict[str, Any]] = None # Parsed ME for plotting fitted/components
    gpm_model: Optional[Any] = None # Keep reference to the full parsed model if needed

    # Optional: Information about the parameters used for evaluation (e.g., for fixed eval)
    parameters_used: Optional[Dict[str, Any]] = None

    # Optional: Log-likelihood if available (e.g., from fixed evaluation)
    log_likelihood: Optional[float] = None

    # Store some run details
    n_draws: int = 0
    hdi_prob: float = 0.9
