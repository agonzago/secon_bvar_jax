# clean_gpm_bvar_trends/complete_gpm_workflow.py - Main Integration

"""
Simplified main entry point for GPM BVAR workflows.
This module provides a clean interface to the core functionality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Core imports
from .gpm_bar_smoother import (
    complete_gpm_workflow_with_smoother_fixed,
    create_default_gpm_file_if_needed,
    generate_synthetic_data_for_gpm,
    debug_smoother_draws
)
from .gpm_prior_evaluator import evaluate_gpm_at_parameters
from .common_types import SmootherResults
from .constants import _DEFAULT_DTYPE

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def run_complete_gpm_analysis(
    data: Union[jnp.ndarray, pd.DataFrame, str],
    gpm_file: str,
    analysis_type: str = "mcmc",
    **kwargs
) -> Optional[SmootherResults]:
    """
    Main entry point for running complete GPM analysis.
    
    Args:
        data: Input data (array, DataFrame, or path to CSV)
        gpm_file: Path to GPM model file
        analysis_type: Type of analysis ("mcmc" or "fixed_params")
        **kwargs: Additional arguments for the specific analysis type
        
    Returns:
        SmootherResults object or None if failed
    """
    print(f"\n{'='*60}")
    print(f"  COMPLETE GPM ANALYSIS - {analysis_type.upper()}")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPM File: {gpm_file}")
    
    # Load data if it's a file path
    if isinstance(data, str):
        if not os.path.exists(data):
            print(f"Error: Data file not found: {data}")
            return None
        
        try:
            if data.endswith('.csv'):
                data = pd.read_csv(data, index_col=0, parse_dates=True)
            else:
                print(f"Error: Unsupported data file format: {data}")
                return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Validate GPM file
    if not os.path.exists(gpm_file):
        print(f"Error: GPM file not found: {gmp_file}")
        return None
    
    # Route to appropriate analysis
    try:
        if analysis_type.lower() == "mcmc":
            return _run_mcmc_analysis(data, gpm_file, **kwargs)
        elif analysis_type.lower() == "fixed_params":
            return _run_fixed_params_analysis(data, gpm_file, **kwargs)
        else:
            print(f"Error: Unknown analysis type: {analysis_type}")
            return None
            
    except Exception as e:
        import traceback
        print(f"Error during {analysis_type} analysis: {e}")
        traceback.print_exc()
        return None
    
    finally:
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


def _run_mcmc_analysis(
    data: Union[jnp.ndarray, pd.DataFrame],
    gpm_file: str,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 2,
    use_gamma_init: bool = False,
    gamma_scale_factor: float = 1.0,
    num_extract_draws: int = 100,
    generate_plots: bool = True,
    plot_default_observed_vs_fitted: bool = True,
    hdi_prob_plot: float = 0.9,
    show_plot_info_boxes: bool = False,
    plot_save_path: Optional[str] = None,
    save_plots: bool = False,
    custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
    variable_names_override: Optional[List[str]] = None,
    target_accept_prob: float = 0.85,
    **kwargs
) -> Optional[SmootherResults]:
    """
    Run MCMC-based analysis using the complete GPM workflow.
    
    Args:
        data: Input data
        gmp_file: Path to GPM model file
        num_warmup: Number of MCMC warmup steps
        num_samples: Number of MCMC sampling steps
        num_chains: Number of MCMC chains
        use_gamma_init: Whether to use gamma-based P0 initialization
        gamma_scale_factor: Scaling factor for gamma P0
        num_extract_draws: Number of draws to extract for smoother
        generate_plots: Whether to generate plots
        plot_default_observed_vs_fitted: Whether to plot observed vs fitted
        hdi_prob_plot: HDI probability for plots
        show_plot_info_boxes: Whether to show info boxes on plots
        plot_save_path: Path to save plots
        save_plots: Whether to save plots
        custom_plot_specs: Custom plot specifications
        variable_names_override: Override for variable names
        target_accept_prob: Target acceptance probability for MCMC
        **kwargs: Additional arguments
        
    Returns:
        SmootherResults object or None if failed
    """
    print(f"\n--- Running MCMC Analysis ---")
    print(f"  MCMC Settings: {num_warmup} warmup, {num_samples} samples, {num_chains} chains")
    print(f"  Target accept prob: {target_accept_prob}")
    print(f"  Use gamma P0: {use_gamma_init}")
    print(f"  Smoother draws: {num_extract_draws}")
    
    try:
        results = complete_gmp_workflow_with_smoother_fixed(
            data=data,
            gpm_file=gpm_file,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            use_gamma_init=use_gamma_init,
            gamma_scale_factor=gamma_scale_factor,
            num_extract_draws=num_extract_draws,
            generate_plots=generate_plots,
            plot_default_observed_vs_fitted=plot_default_observed_vs_fitted,
            hdi_prob_plot=hdi_prob_plot,
            show_plot_info_boxes=show_plot_info_boxes,
            plot_save_path=plot_save_path,
            save_plots=save_plots,
            custom_plot_specs=custom_plot_specs,
            variable_names_override=variable_names_override,
            data_file_source_for_summary=getattr(data, 'name', 'In-memory data'),
            target_accept_prob=target_accept_prob
        )
        
        if results is not None:
            print(f"✓ MCMC analysis completed successfully")
            print(f"  Generated {results.n_draws} smoother draws")
            return results
        else:
            print("✗ MCMC analysis failed")
            return None
            
    except Exception as e:
        import traceback
        print(f"✗ MCMC analysis failed with error: {e}")
        traceback.print_exc()
        return None


def _run_fixed_params_analysis(
    data: Union[jnp.ndarray, pd.DataFrame],
    gpm_file: str,
    param_values: Dict[str, Any],
    initial_state_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    num_sim_draws: int = 50,
    plot_results: bool = True,
    plot_default_observed_vs_fitted: bool = True,
    plot_default_observed_vs_trend_components: bool = True,
    custom_plot_specs: Optional[List[Dict[str, Any]]] = None,
    variable_names: Optional[List[str]] = None,
    use_gamma_init_for_test: bool = True,
    gamma_init_scaling: float = 1.0,
    hdi_prob: float = 0.9,
    trend_P0_var_scale: float = 1e4,
    var_P0_var_scale: float = 1.0,
    save_plots_path_prefix: Optional[str] = None,
    show_plot_info_boxes: bool = False,
    **kwargs
) -> Optional[SmootherResults]:
    """
    Run fixed parameter analysis using the GPM prior evaluator.
    
    Args:
        data: Input data
        gpm_file: Path to GPM model file
        param_values: Fixed parameter values to use
        initial_state_prior_overrides: Override for initial state priors
        num_sim_draws: Number of simulation draws
        plot_results: Whether to generate plots
        plot_default_observed_vs_fitted: Whether to plot observed vs fitted
        plot_default_observed_vs_trend_components: Whether to plot obs vs trend components
        custom_plot_specs: Custom plot specifications
        variable_names: Variable names for the data
        use_gamma_init_for_test: Whether to use gamma-based P0
        gamma_init_scaling: Scaling for gamma P0
        hdi_prob: HDI probability
        trend_P0_var_scale: Scale for trend P0 variance
        var_P0_var_scale: Scale for VAR P0 variance
        save_plots_path_prefix: Prefix for saving plots
        show_plot_info_boxes: Whether to show info boxes
        **kwargs: Additional arguments
        
    Returns:
        SmootherResults object or None if failed
    """
    print(f"\n--- Running Fixed Parameters Analysis ---")
    print(f"  Parameter values: {param_values}")
    print(f"  Simulation draws: {num_sim_draws}")
    print(f"  Use gamma P0: {use_gamma_init_for_test}")
    
    try:
        results = evaluate_gpm_at_parameters(
            gpm_file_path=gpm_file,
            y=data,
            param_values=param_values,
            initial_state_prior_overrides=initial_state_prior_overrides,
            num_sim_draws=num_sim_draws,
            plot_results=plot_results,
            plot_default_observed_vs_fitted=plot_default_observed_vs_fitted,
            plot_default_observed_vs_trend_components=plot_default_observed_vs_trend_components,
            custom_plot_specs=custom_plot_specs,
            variable_names=variable_names,
            use_gamma_init_for_test=use_gamma_init_for_test,
            gamma_init_scaling=gamma_init_scaling,
            hdi_prob=hdi_prob,
            trend_P0_var_scale=trend_P0_var_scale,
            var_P0_var_scale=var_P0_var_scale,
            save_plots_path_prefix=save_plots_path_prefix,
            show_plot_info_boxes=show_plot_info_boxes
        )
        
        if results is not None:
            print(f"✓ Fixed parameter analysis completed successfully")
            print(f"  Log-likelihood: {results.log_likelihood:.4f}")
            print(f"  Generated {results.n_draws} simulation draws")
            return results
        else:
            print("✗ Fixed parameter analysis failed")
            return None
            
    except Exception as e:
        import traceback
        print(f"✗ Fixed parameter analysis failed with error: {e}")
        traceback.print_exc()
        return None


def create_example_gpm_model(
    filename: str = "example_model.gpm",
    num_obs_vars: int = 2,
    num_stat_vars: int = 2,
    include_var_setup: bool = True
) -> str:
    """
    Create an example GPM model file for testing and demonstration.
    
    Args:
        filename: Name of the GPM file to create
        num_obs_vars: Number of observed variables
        num_stat_vars: Number of stationary variables
        include_var_setup: Whether to include VAR prior setup
        
    Returns:
        Path to the created GPM file
    """
    print(f"Creating example GPM model: {filename}")
    
    create_default_gpm_file_if_needed(
        filename=filename,
        num_obs_vars=num_obs_vars,
        num_stat_vars=num_stat_vars
    )
    
    # If VAR setup wasn't included by default, add it
    if include_var_setup and num_stat_vars > 0:
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            if 'var_prior_setup' not in content.lower():
                # Add VAR prior setup
                var_setup = """
var_prior_setup;
    var_order = 1;
    es = 0.6, 0.15;
    fs = 0.15, 0.15;
    gs = 3.0, 3.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;
"""
                # Insert before the last initval section
                if 'initval;' in content:
                    content = content.replace('initval;', var_setup + '\ninitval;')
                else:
                    content = content + var_setup
                
                with open(filename, 'w') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"Warning: Could not add VAR setup to {filename}: {e}")
    
    print(f"✓ Created example GPM model: {filename}")
    return filename


def generate_example_data(
    gpm_file: str,
    true_params: Optional[Dict[str, Any]] = None,
    num_periods: int = 150,
    random_seed: int = 42
) -> Optional[jnp.ndarray]:
    """
    Generate synthetic data for a GPM model.
    
    Args:
        gpm_file: Path to GPM model file
        true_params: True parameter values for simulation
        num_periods: Number of time periods
        random_seed: Random seed for reproducibility
        
    Returns:
        Simulated data array or None if failed
    """
    print(f"Generating example data for: {gpm_file}")
    
    if true_params is None:
        # Use default parameters
        true_params = {
            "rho": 0.5,
            "SHK_TREND1": 0.1,
            "SHK_TREND2": 0.15,
            "SHK_STAT1": 0.2,
            "SHK_STAT2": 0.25,
            "_var_coefficients": jnp.array([[[0.7, 0.1], [0.0, 0.6]]], dtype=_DEFAULT_DTYPE),
            "_var_innovation_corr_chol": jnp.array([[1.0, 0.0], [0.3, jnp.sqrt(1-0.3**2)]], dtype=_DEFAULT_DTYPE)
        }
    
    try:
        sim_data = generate_synthetic_data_for_gpm(
            gpm_file_path=gpm_file,
            true_params=true_params,
            num_steps=num_periods,
            rng_key_seed=random_seed
        )
        
        if sim_data is not None:
            print(f"✓ Generated synthetic data with shape: {sim_data.shape}")
            return sim_data
        else:
            print("✗ Failed to generate synthetic data")
            return None
            
    except Exception as e:
        print(f"✗ Error generating synthetic data: {e}")
        return None


def run_quick_example(
    analysis_type: str = "mcmc",
    save_results: bool = False,
    output_dir: str = "example_output"
) -> Optional[SmootherResults]:
    """
    Run a quick example analysis with generated data and model.
    
    Args:
        analysis_type: Type of analysis ("mcmc" or "fixed_params")
        save_results: Whether to save results and plots
        output_dir: Directory for output files
        
    Returns:
        SmootherResults object or None if failed
    """
    print(f"\n{'='*60}")
    print(f"  RUNNING QUICK EXAMPLE - {analysis_type.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create example model
        gpm_file = create_example_gmp_model("quick_example.gpm", num_obs_vars=2, num_stat_vars=2)
        
        # Generate example data
        sim_data = generate_example_data(gpm_file, num_periods=100)
        
        if sim_data is None:
            print("Failed to generate example data")
            return None
        
        # Create DataFrame for better handling
        data_df = pd.DataFrame(
            sim_data, 
            columns=['OBS1', 'OBS2'],
            index=pd.date_range(start='2000-01-01', periods=sim_data.shape[0], freq='QE')
        )
        
        # Prepare output directory if saving
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            plot_save_path = os.path.join(output_dir, "plots")
        else:
            plot_save_path = None
        
        # Run analysis
        if analysis_type.lower() == "mcmc":
            results = run_complete_gpm_analysis(
                data=data_df,
                gpm_file=gmp_file,
                analysis_type="mcmc",
                num_warmup=50,
                num_samples=100,
                num_chains=1,
                num_extract_draws=25,
                generate_plots=True,
                save_plots=save_results,
                plot_save_path=plot_save_path,
                use_gamma_init=True,
                gamma_scale_factor=1.0
            )
        else:  # fixed_params
            # Define fixed parameter values
            fixed_params = {
                'rho': 0.7,
                'SHK_TREND1': 0.12,
                'SHK_TREND2': 0.18,
                'SHK_STAT1': 0.22,
                'SHK_STAT2': 0.28,
                '_var_coefficients': jnp.array([[[0.75, 0.05], [0.02, 0.65]]], dtype=_DEFAULT_DTYPE),
                '_var_innovation_corr_chol': jnp.array([[1.0, 0.0], [0.25, jnp.sqrt(1-0.25**2)]], dtype=_DEFAULT_DTYPE)
            }
            
            results = run_complete_gpm_analysis(
                data=data_df,
                gpm_file=gpm_file,
                analysis_type="fixed_params",
                param_values=fixed_params,
                num_sim_draws=50,
                plot_results=True,
                save_plots_path_prefix=os.path.join(plot_save_path, "fixed_eval_plot") if plot_save_path else None,
                use_gamma_init_for_test=True
            )
        
        # Print summary
        if results is not None:
            print(f"\n✓ Quick example completed successfully!")
            print(f"  Analysis type: {analysis_type}")
            print(f"  Number of draws: {results.n_draws}")
            if hasattr(results, 'log_likelihood') and results.log_likelihood is not None:
                print(f"  Log-likelihood: {results.log_likelihood:.4f}")
            if save_results:
                print(f"  Results saved to: {output_dir}")
            
            # Debug draws if available
            if results.n_draws > 0:
                debug_smoother_draws(results)
                
        return results
        
    except Exception as e:
        import traceback
        print(f"✗ Quick example failed: {e}")
        traceback.print_exc()
        return None
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists("quick_example.gpm"):
                os.remove("quick_example.gpm")
        except:
            pass


if __name__ == "__main__":
    # Example usage
    print("Complete GPM Workflow - Example Usage")
    print("="*50)
    
    # Run quick MCMC example
    print("\n1. Running quick MCMC example...")
    mcmc_results = run_quick_example(analysis_type="mcmc", save_results=True)
    
    # Run quick fixed params example
    print("\n2. Running quick fixed parameters example...")
    fixed_results = run_quick_example(analysis_type="fixed_params", save_results=True)
    
    print("\nExamples completed. Check 'example_output' directory for results.")