# Configure JAX as soon as the package is imported
from . import jax_config
jax_config.configure_jax()

# clean_gpm_bvar_trends/__init__.py

"""
Clean GPM BVAR Trends Package
A comprehensive framework for GPM-based Bayesian VAR with trends analysis.
"""

# Import main workflow functions
from .complete_gpm_workflow import (
    run_complete_gpm_analysis,
    run_quick_example,
    create_example_gpm_model,
    generate_example_data
)

# Import calibration utilities
from .calibration_utils import (
    PriorCalibrationConfig,
    validate_calibration_config,
    load_data_for_calibration,
    run_mcmc_workflow,
    run_fixed_parameter_evaluation
)

# Import common types
from .common_types import SmootherResults, EnhancedBVARParams

# Import core evaluation function
from .gpm_prior_evaluator import evaluate_gpm_at_parameters

# Import main smoother workflow
from .gpm_bar_smoother import complete_gpm_workflow_with_smoother_fixed

__version__ = "0.1.0"
__author__ = "GPM BVAR Development Team"

__all__ = [
    'run_complete_gpm_analysis',
    'run_quick_example', 
    'create_example_gpm_model',
    'generate_example_data',
    'PriorCalibrationConfig',
    'validate_calibration_config',
    'load_data_for_calibration',
    'run_mcmc_workflow',
    'run_fixed_parameter_evaluation',
    'SmootherResults',
    'EnhancedBVARParams',
    'evaluate_gpm_at_parameters',
    'complete_gpm_workflow_with_smoother_fixed'
]