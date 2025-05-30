# reporting_plots.py - Enhanced plotting with better HDI visibility

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, List
from simulation_smoothing import _compute_and_format_hdi_az

def plot_observed_and_fitted_enhanced(
    y_np: np.ndarray,
    trend_draws: jnp.ndarray,
    stationary_draws: jnp.ndarray,
    hdi_prob: float = 0.9,
    variable_names: Optional[List[str]] = None,
    alpha_bands: float = 0.4,
    show_multiple_percentiles: bool = False
):
    """
    Enhanced version with better confidence band visibility
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Check for sufficient draws
    if trend_draws.shape[0] < 2 or stationary_draws.shape[0] < 2:
        print("Warning: Not enough draws for confidence intervals")
        return

    # Calculate fitted draws
    fitted_draws = trend_draws + stationary_draws
    fitted_draws_np = np.asarray(fitted_draws)

    # Calculate statistics
    fitted_median_np = np.percentile(fitted_draws_np, 50, axis=0)
    
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    elif len(variable_names) != n_vars:
        print(f"Warning: Variable names count mismatch. Using defaults.")
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]

    # Create subplots
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), squeeze=False)

    for i in range(n_vars):
        ax = axes[i, 0]
        
        # Plot observed data
        ax.plot(time_index, y_np[:, i], 'k-', label='Observed Data', 
                alpha=0.8, linewidth=2, zorder=3)

        # Plot fitted median
        ax.plot(time_index, fitted_median_np[:, i], 'g-', 
                label='Estimated Fitted (median)', linewidth=2, zorder=4)

        if show_multiple_percentiles:
            # Show multiple percentile bands for better visualization
            percentiles = [5, 25, 75, 95]
            colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightblue']
            alphas = [0.3, 0.4, 0.4, 0.3]
            
            fitted_percentiles = np.percentile(fitted_draws_np, percentiles, axis=0)
            
            # Plot nested bands
            ax.fill_between(time_index, 
                           fitted_percentiles[0, :, i], fitted_percentiles[3, :, i],
                           color=colors[0], alpha=alphas[0], 
                           label='5-95% range', zorder=1)
            
            ax.fill_between(time_index, 
                           fitted_percentiles[1, :, i], fitted_percentiles[2, :, i],
                           color=colors[1], alpha=alphas[1], 
                           label='25-75% range', zorder=2)
        else:
            # Standard HDI approach with enhancements
            fitted_hdi = _compute_and_format_hdi_az(fitted_draws, hdi_prob=hdi_prob)
            
            if fitted_hdi and 'low' in fitted_hdi and 'high' in fitted_hdi:
                # Check HDI width for this variable
                hdi_width = fitted_hdi['high'][:, i] - fitted_hdi['low'][:, i]
                max_width = np.max(hdi_width)
                mean_width = np.mean(hdi_width)
                
                print(f"Variable {i+1}: HDI width stats - mean: {mean_width:.6f}, max: {max_width:.6f}")
                
                # Adaptive alpha based on HDI width
                adaptive_alpha = min(0.8, max(alpha_bands, alpha_bands + 0.3))
                
                # Main HDI band
                ax.fill_between(
                    time_index,
                    fitted_hdi['low'][:, i],
                    fitted_hdi['high'][:, i],
                    color='green',
                    alpha=adaptive_alpha,
                    label=f'Fitted ({int(hdi_prob*100)}% HDI)',
                    zorder=2,
                    edgecolor='darkgreen',
                    linewidth=0.5
                )
                
                # If HDI is very narrow, add visible edge lines
                if mean_width / (np.max(y_np[:, i]) - np.min(y_np[:, i])) < 0.02:
                    ax.plot(time_index, fitted_hdi['low'][:, i], 
                           'g--', alpha=0.8, linewidth=1, label='HDI bounds')
                    ax.plot(time_index, fitted_hdi['high'][:, i], 
                           'g--', alpha=0.8, linewidth=1)
            else:
                print(f"Warning: HDI calculation failed for variable {i+1}")

        ax.set_title(f'{variable_names[i]} - Observed Data and Fitted', fontsize=14)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_estimated_components_enhanced(
    trend_draws: jnp.ndarray,
    stationary_draws: jnp.ndarray,
    hdi_prob: float = 0.9,
    trend_variable_names: Optional[List[str]] = None,
    stationary_variable_names: Optional[List[str]] = None,
    alpha_bands: float = 0.4
):
    """
    Enhanced component plotting with better visibility
    """
    num_draws_t, T, n_trends = trend_draws.shape
    num_draws_s, T_s, n_stationary = stationary_draws.shape

    if T != T_s:
        print(f"Error: Time dimensions mismatch ({T} vs {T_s})")
        return

    time_index = np.arange(T)

    # Set default names
    if trend_variable_names is None:
        trend_variable_names = [f'Trend {i+1}' for i in range(n_trends)]
    if stationary_variable_names is None:
        stationary_variable_names = [f'Stationary {i+1}' for i in range(n_stationary)]

    # Check for sufficient draws
    has_trend_draws = num_draws_t > 1
    has_stationary_draws = num_draws_s > 1

    # Calculate statistics
    if has_trend_draws:
        trend_draws_np = np.asarray(trend_draws)
        trend_median_np = np.percentile(trend_draws_np, 50, axis=0)
        trend_hdi = _compute_and_format_hdi_az(trend_draws, hdi_prob=hdi_prob)
    
    if has_stationary_draws:
        stationary_draws_np = np.asarray(stationary_draws)
        stationary_median_np = np.percentile(stationary_draws_np, 50, axis=0)
        stationary_hdi = _compute_and_format_hdi_az(stationary_draws, hdi_prob=hdi_prob)

    # Create plots
    num_plots = n_trends + n_stationary
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), squeeze=False)

    plot_idx = 0

    # Plot Trend Components
    for i in range(n_trends):
        ax = axes[plot_idx, 0]
        
        if has_trend_draws:
            # Plot median
            ax.plot(time_index, trend_median_np[:, i], 'r-', 
                   label='Estimated Trend (median)', linewidth=2)
            
            # Plot HDI
            if trend_hdi and 'low' in trend_hdi and 'high' in trend_hdi:
                hdi_width = trend_hdi['high'][:, i] - trend_hdi['low'][:, i]
                mean_width = np.mean(hdi_width)
                
                # Enhanced HDI visibility
                ax.fill_between(
                    time_index,
                    trend_hdi['low'][:, i],
                    trend_hdi['high'][:, i],
                    color='red',
                    alpha=alpha_bands + 0.2,
                    label=f'Trend ({int(hdi_prob*100)}% HDI)',
                    edgecolor='darkred',
                    linewidth=0.5
                )
                
                # Add percentile lines for very narrow HDI
                if mean_width > 0:
                    range_ratio = mean_width / (np.max(trend_median_np[:, i]) - np.min(trend_median_np[:, i]))
                    if range_ratio < 0.05:
                        # Add 25th and 75th percentiles as dashed lines
                        trend_25 = np.percentile(trend_draws_np, 25, axis=0)[:, i]
                        trend_75 = np.percentile(trend_draws_np, 75, axis=0)[:, i]
                        ax.plot(time_index, trend_25, 'r--', alpha=0.7, linewidth=1)
                        ax.plot(time_index, trend_75, 'r--', alpha=0.7, linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No trend draws available', transform=ax.transAxes, 
                   ha='center', va='center')

        ax.set_title(f'{trend_variable_names[i]} - Estimated Trend Component', fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1

    # Plot Stationary Components  
    for i in range(n_stationary):
        ax = axes[plot_idx, 0]

        # Add zero reference line
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        if has_stationary_draws:
            # Plot median
            ax.plot(time_index, stationary_median_np[:, i], 'b-', 
                   label='Estimated Stationary (median)', linewidth=2)
            
            # Plot HDI
            if stationary_hdi and 'low' in stationary_hdi and 'high' in stationary_hdi:
                # Enhanced HDI visibility for stationary components
                ax.fill_between(
                    time_index,
                    stationary_hdi['low'][:, i],
                    stationary_hdi['high'][:, i],
                    color='blue',
                    alpha=alpha_bands + 0.2,
                    label=f'Stationary ({int(hdi_prob*100)}% HDI)',
                    edgecolor='darkblue',
                    linewidth=0.5
                )
                
                # Check if we need to enhance visibility further
                hdi_width = stationary_hdi['high'][:, i] - stationary_hdi['low'][:, i]
                mean_width = np.mean(hdi_width)
                max_abs_val = max(abs(np.min(stationary_median_np[:, i])), 
                                abs(np.max(stationary_median_np[:, i])))
                
                if max_abs_val > 0:
                    range_ratio = mean_width / (2 * max_abs_val)
                    if range_ratio < 0.05:
                        # Add additional visual cues
                        stat_25 = np.percentile(stationary_draws_np, 25, axis=0)[:, i]
                        stat_75 = np.percentile(stationary_draws_np, 75, axis=0)[:, i]
                        ax.plot(time_index, stat_25, 'b:', alpha=0.8, linewidth=1.5)
                        ax.plot(time_index, stat_75, 'b:', alpha=0.8, linewidth=1.5)
        else:
            ax.text(0.5, 0.5, 'No stationary draws available', transform=ax.transAxes, 
                   ha='center', va='center')

        ax.set_title(f'{stationary_variable_names[i]} - Estimated Stationary Component', fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1

    plt.tight_layout()
    plt.show()


def plot_with_error_bars_alternative(
    y_np: np.ndarray,
    trend_draws: jnp.ndarray,
    stationary_draws: jnp.ndarray,
    variable_names: Optional[List[str]] = None,
    error_every: int = 5  # Show error bars every N points
):
    """
    Alternative plotting using error bars instead of fill_between
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)
    
    if trend_draws.shape[0] < 2 or stationary_draws.shape[0] < 2:
        print("Not enough draws for error bars")
        return
    
    # Calculate fitted draws and statistics
    fitted_draws = trend_draws + stationary_draws
    fitted_draws_np = np.asarray(fitted_draws)
    fitted_median = np.percentile(fitted_draws_np, 50, axis=0)
    fitted_25 = np.percentile(fitted_draws_np, 25, axis=0)
    fitted_75 = np.percentile(fitted_draws_np, 75, axis=0)
    
    # Error bar data (show every N-th point to avoid clutter)
    time_errorbar = time_index[::error_every]
    fitted_median_eb = fitted_median[::error_every]
    yerr_lower = fitted_median_eb - fitted_25[::error_every]
    yerr_upper = fitted_75[::error_every] - fitted_median_eb
    
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), squeeze=False)
    
    for i in range(n_vars):
        ax = axes[i, 0]
        
        # Plot observed data
        ax.plot(time_index, y_np[:, i], 'k-', label='Observed Data', 
                linewidth=2, alpha=0.8)
        
        # Plot fitted line
        ax.plot(time_index, fitted_median[:, i], 'g-', 
                label='Fitted (median)', linewidth=2)
        
        # Plot error bars
        ax.errorbar(time_errorbar, fitted_median_eb[:, i],
                   yerr=[yerr_lower[:, i], yerr_upper[:, i]],
                   fmt='o', color='green', alpha=0.7, markersize=4,
                   elinewidth=2, capsize=3, capthick=2,
                   label='25-75% range')
        
        ax.set_title(f'{variable_names[i]} - Observed and Fitted (Error Bar Style)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_observed_and_trend(y_np, trend_draws, hdi_prob=0.9, variable_names=None):
    """Plot observed data vs trend component"""
    import matplotlib.pyplot as plt
    import numpy as np
    from simulation_smoothing import _compute_and_format_hdi_az
    
    T, n_vars = y_np.shape
    time_index = np.arange(T)
    trend_draws_np = np.asarray(trend_draws)
    
    if trend_draws_np.shape[0] > 1:
        trend_median_np = np.percentile(trend_draws_np, 50, axis=0)
        trend_hdi = _compute_and_format_hdi_az(trend_draws, hdi_prob=hdi_prob)
    else:
        print("Not enough draws for confidence intervals")
        return
    
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), squeeze=False)
    
    for i in range(min(n_vars, trend_draws_np.shape[2])):
        ax = axes[i, 0] if n_vars > 1 else axes
        
        # Observed data
        ax.plot(time_index, y_np[:, i], 'k-', label='Observed Data', linewidth=2)
        
        # Trend median
        ax.plot(time_index, trend_median_np[:, i], 'r-', 
               label='Estimated Trend (median)', linewidth=2)
        
        # HDI if available and meaningful
        if trend_hdi and 'low' in trend_hdi and 'high' in trend_hdi:
            hdi_width = trend_hdi['high'][:, i] - trend_hdi['low'][:, i]
            if not np.isnan(np.mean(hdi_width)) and np.mean(hdi_width) > 1e-10:
                ax.fill_between(time_index, trend_hdi['low'][:, i], trend_hdi['high'][:, i],
                               color='red', alpha=0.3, label=f'Trend ({int(hdi_prob*100)}% HDI)')
        
        ax.set_title(f'{variable_names[i]} - Observed Data and Trend Component')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Drop-in replacement functions with enhanced visibility
def plot_observed_and_fitted(y_np, trend_draws, stationary_draws, hdi_prob=0.9, variable_names=None):
    """Drop-in replacement with enhanced visibility"""
    plot_observed_and_fitted_enhanced(y_np, trend_draws, stationary_draws, 
                                     hdi_prob, variable_names, alpha_bands=0.5)

def plot_estimated_components(trend_draws, stationary_draws, hdi_prob=0.9, 
                            trend_variable_names=None, stationary_variable_names=None):
    """Drop-in replacement with enhanced visibility"""
    plot_estimated_components_enhanced(trend_draws, stationary_draws, hdi_prob,
                                     trend_variable_names, stationary_variable_names, 
                                     alpha_bands=0.5)