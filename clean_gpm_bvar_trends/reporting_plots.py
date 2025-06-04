# clean_gpm_bvar_trends/reporting_plots.py - Simplified and Fixed

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, List, Union, Tuple, Dict, Any
import pandas as pd

from .common_types import SmootherResults


def compute_hdi_robust(draws: Union[jnp.ndarray, np.ndarray],
                      hdi_prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the Highest Density Interval for a given probability."""
    if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 2:
        dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

    draws_np = np.asarray(draws)
    original_shape = draws_np.shape # This is a tuple
    
    # if original_shape.ndim > 1 and original_shape[1] == 0: # original_shape is a tuple
    # Corrected logic:
    if len(original_shape) > 1 and original_shape[1] == 0: # Check if second dimension is zero (no timesteps)
        return (np.full(original_shape[1:], np.nan), np.full(original_shape[1:], np.nan))

    try:
        hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
        hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
        return hdi_lower, hdi_upper
    except Exception as e:
        print(f"HDI computation failed: {e}")
        dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else () # Use draws_np here
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))


def compute_summary_statistics(draws: Union[jnp.ndarray, np.ndarray]) -> dict:
    """Computes basic summary statistics (mean, median, std)."""
    if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 1:
        dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
        return {
            'mean': np.full(dummy_shape, np.nan),
            'median': np.full(dummy_shape, np.nan),
            'mode': np.full(dummy_shape, np.nan),
            'std': np.full(dummy_shape, np.nan)
        }

    draws_np = np.asarray(draws)
    return {
        'mean': np.nanmean(draws_np, axis=0),
        'median': np.nanmedian(draws_np, axis=0),
        'mode': np.nanmedian(draws_np, axis=0),  # Approximate mode with median
        'std': np.nanstd(draws_np, axis=0)
    }


def _format_datetime_axis(fig, ax, time_index):
    """Format x-axis for datetime indices."""
    try:
        if hasattr(time_index, 'to_pydatetime'):  # pandas DatetimeIndex
            fig.autofmt_xdate()
        elif hasattr(pd, 'DatetimeIndex') and isinstance(time_index, pd.DatetimeIndex):
            fig.autofmt_xdate()
    except:
        pass


def _add_info_box(ax, n_draws, hdi_prob=None, additional_info=None):
    """Add information box to plot."""
    info_lines = []
    
    if n_draws is not None and n_draws > 0:
        info_lines.append(f'Draws: {n_draws}')
    
    if hdi_prob is not None and n_draws > 1:
        info_lines.append(f'HDI Prob: {hdi_prob:.2f}')
    
    if additional_info:
        info_lines.extend(additional_info)
    
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)


def plot_time_series_with_uncertainty(
    draws: Union[jnp.ndarray, np.ndarray],
    variable_names: List[str],
    hdi_prob: float = 0.9,
    title_prefix: str = "",
    show_mean: bool = True,
    show_median: bool = False,
    show_hdi: bool = True,
    alpha_fill: float = 0.3,
    time_index: Optional[Any] = None,
    show_info_box: bool = False
) -> Optional[plt.Figure]:
    """
    Plots multiple time series from simulation draws with uncertainty bands.
    """
    draws_np = np.asarray(draws)

    # Validate input
    if (draws_np.ndim != 3 or draws_np.shape[0] < 1 or 
        draws_np.shape[2] != len(variable_names)):
        print(f"Invalid draws shape: {draws_np.shape}, expected (n_draws, T, {len(variable_names)})")
        return None

    n_draws, T_timesteps, n_series = draws_np.shape
    if T_timesteps == 0:
        return None

    # Set up time index
    if time_index is None or len(time_index) != T_timesteps:
        time_index_plot = np.arange(T_timesteps)
    else:
        time_index_plot = time_index

    # Calculate statistics
    stats = compute_summary_statistics(draws_np)
    mean_lines = stats.get('mean')
    median_lines = stats.get('median')
    
    hdi_lower, hdi_upper = None, None
    if show_hdi and n_draws > 1:
        hdi_lower, hdi_upper = compute_hdi_robust(draws_np, hdi_prob)

    # Check for valid data
    if mean_lines is None or not np.any(np.isfinite(mean_lines)):
        print("No valid data to plot")
        return None

    # Create figure
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series), squeeze=False)

    for i in range(n_series):
        ax = axes[i, 0]
        var_name = variable_names[i]

        # Plot HDI band
        if (hdi_lower is not None and hdi_upper is not None and
            hdi_lower.ndim == 2 and hdi_lower.shape == (T_timesteps, n_series) and
            hdi_upper.ndim == 2 and hdi_upper.shape == (T_timesteps, n_series)):
            
            if not (np.all(np.isnan(hdi_lower[:, i])) or np.all(np.isnan(hdi_upper[:, i]))):
                ax.fill_between(time_index_plot, hdi_lower[:, i], hdi_upper[:, i],
                               alpha=alpha_fill, color='royalblue', 
                               label=f'{int(hdi_prob*100)}% HDI')

        # Plot mean or median line
        line_to_plot = None
        line_label = None
        
        if (show_mean and mean_lines is not None and 
            mean_lines.ndim == 2 and mean_lines.shape == (T_timesteps, n_series)):
            if not np.all(np.isnan(mean_lines[:, i])):
                line_to_plot = mean_lines[:, i]
                line_label = 'Mean'
        
        if (show_median and median_lines is not None and 
            median_lines.ndim == 2 and median_lines.shape == (T_timesteps, n_series)):
            if not np.all(np.isnan(median_lines[:, i])):
                line_to_plot = median_lines[:, i]
                line_label = 'Median'

        if line_to_plot is not None and len(line_to_plot) == T_timesteps:
            color = 'b-' if line_label == 'Mean' else 'g-'
            ax.plot(time_index_plot, line_to_plot, color, linewidth=2, label=line_label)

        # Formatting
        ax.set_title(f'{title_prefix}: {var_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        _format_datetime_axis(fig, ax, time_index_plot)
        
        if show_info_box:
            _add_info_box(ax, n_draws, hdi_prob)

    plt.tight_layout()
    return fig



def plot_custom_series_comparison(
    plot_title: str,
    series_specs: List[Dict],
    results: SmootherResults,
    save_path: Optional[str] = None,
    show_info_box: bool = False
) -> Optional[plt.Figure]:
    """
    Plots multiple specified series from SmootherResults on a single axes.
    
    Args:
        plot_title: Title for the plot
        series_specs: List of dicts with keys: 'type', 'name', 'label', 'style', 'color', 'show_hdi'
        results: SmootherResults object
        save_path: Optional path to save plot
        show_info_box: Whether to show info box
    """
    # Extract data from results
    data_sources = {
        'observed': (results.observed_data, results.observed_variable_names),
        'trend': (results.trend_draws, results.trend_names),
        'stationary': (results.stationary_draws, results.stationary_names)
    }
    
    time_index_plot = results.time_index
    hdi_prob = results.hdi_prob

    # Determine time dimension
    T = 0
    for data, _ in data_sources.values():
        if data is not None and data.ndim > 0:
            T = data.shape[1] if data.ndim > 1 else data.shape[0]
            break

    if T == 0:
        print(f"Warning: No data for custom plot '{plot_title}'. Skipping plot.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    if time_index_plot is None or len(time_index_plot) != T:
        time_index_plot = np.arange(T)

    plotted_labels = set()

    for spec in series_specs:
        series_type = spec.get('type')
        series_name = spec.get('name')
        label = spec.get('label', series_name if series_name else series_type)
        
        if not label or not series_type or not series_name:
            continue

        # Parse style and color
        style = spec.get('style', '-')
        color = spec.get('color')
        show_hdi = spec.get('show_hdi', False if series_type == 'observed' else True)
        alpha_fill = spec.get('hdi_alpha', 0.2)

        # Get data
        if series_type not in data_sources:
            print(f"Warning: Unsupported series type '{series_type}'. Skipping.")
            continue

        data_array, name_list = data_sources[series_type]
        if data_array is None or not name_list:
            print(f"Warning: No data available for type '{series_type}'. Skipping.")
            continue

        try:
            idx = name_list.index(series_name)
        except ValueError:
            print(f"Warning: Series '{series_name}' not found in {series_type} names. Skipping.")
            continue

        # Extract series data
        if series_type == 'observed':
            if idx >= data_array.shape[1]:
                continue
            current_series_data = data_array[:, idx]
        else:
            if data_array.ndim != 3 or idx >= data_array.shape[2]:
                continue
            current_series_data = data_array[:, :, idx]

        # Plot the series
        if (current_series_data is not None and 
            len(current_series_data) == T and label not in plotted_labels):
            
            plotted_labels.add(label)
            
            if current_series_data.ndim == 1:  # Observed data
                if not np.all(np.isnan(current_series_data)):
                    plot_args = {'label': label, 'linestyle': style}
                    if color:
                        plot_args['color'] = color
                    ax.plot(time_index_plot, current_series_data, **plot_args)
            
            elif current_series_data.ndim == 2:  # Draw data
                if current_series_data.shape[0] > 0:
                    stats = compute_summary_statistics(current_series_data)
                    line_data = (stats.get('median') if current_series_data.shape[0] > 1 
                               else stats.get('mean'))
                    
                    if (line_data is not None and len(line_data) == T and 
                        not np.all(np.isnan(line_data))):
                        
                        plot_args = {'label': label, 'linestyle': style}
                        if color:
                            plot_args['color'] = color
                        ax.plot(time_index_plot, line_data, **plot_args)
                        
                        # Plot HDI
                        if show_hdi and current_series_data.shape[0] > 1:
                            hdi_lower, hdi_upper = compute_hdi_robust(current_series_data, hdi_prob)
                            
                            if (hdi_lower is not None and hdi_upper is not None and
                                hdi_lower.ndim == 1 and hdi_lower.shape == (T,) and
                                not (np.all(np.isnan(hdi_lower)) or np.all(np.isnan(hdi_upper)))):
                                
                                fill_args = {'alpha': alpha_fill}
                                fill_color = (color if color 
                                            else (ax.lines[-1].get_color() if ax.lines 
                                                  else 'blue'))
                                fill_args['color'] = fill_color
                                ax.fill_between(time_index_plot, hdi_lower, hdi_upper, **fill_args)

    # Format plot
    ax.set_title(plot_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    if plotted_labels:
        ax.legend()
    
    _format_datetime_axis(fig, ax, time_index_plot)
    
    if show_info_box:
        _add_info_box(ax, results.n_draws, results.hdi_prob)
    
    plt.tight_layout()

    # Save if requested
    if save_path and fig:
        try:
            safe_title = (plot_title.lower().replace(' ', '_').replace('/', '_')
                         .replace('(', '').replace(')', '').replace('=', '_')
                         .replace(':', '_').replace('.', ''))
            fig.savefig(f"{save_path}_{safe_title}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        except Exception as e:
            print(f"Error saving custom plot '{plot_title}': {e}")

    return fig


def plot_smoother_results(
    results: SmootherResults,
    save_path: Optional[str] = None,
    show_info_box: bool = False
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Plots individual trend and stationary components with uncertainty bands.
    """
    trend_fig = None
    if (results.trend_draws is not None and 
        results.trend_draws.shape[2] > 0 and results.trend_draws.shape[0] > 0):
        
        trend_fig = plot_time_series_with_uncertainty(
            results.trend_draws,
            variable_names=results.trend_names,
            hdi_prob=results.hdi_prob,
            title_prefix="Trend Component",
            show_mean=True, show_median=False, show_hdi=True,
            alpha_fill=0.3,
            time_index=results.time_index,
            show_info_box=show_info_box
        )

    stationary_fig = None
    if (results.stationary_draws is not None and 
        results.stationary_draws.shape[2] > 0 and results.stationary_draws.shape[0] > 0):
        
        stationary_fig = plot_time_series_with_uncertainty(
            results.stationary_draws,
            variable_names=results.stationary_names,
            hdi_prob=results.hdi_prob,
            title_prefix="Stationary Component",
            show_mean=True, show_median=False, show_hdi=True,
            alpha_fill=0.3,
            time_index=results.time_index,
            show_info_box=show_info_box
        )

    # Handle saving
    if save_path:
        if trend_fig:
            try:
                trend_fig.savefig(f"{save_path}_trends_components.png", 
                                dpi=150, bbox_inches='tight')
                plt.close(trend_fig)
                trend_fig = None
            except Exception as e:
                print(f"Error saving trend components plot: {e}")
        
        if stationary_fig:
            try:
                stationary_fig.savefig(f"{save_path}_stationary_components.png", 
                                     dpi=150, bbox_inches='tight')
                plt.close(stationary_fig)
                stationary_fig = None
            except Exception as e:
                print(f"Error saving stationary components plot: {e}")

    return trend_fig, stationary_fig


def plot_observed_vs_single_trend_component(
    results: SmootherResults,
    save_path: Optional[str] = None,
    show_info_box: bool = False,
    use_median_for_trend_line: bool = True
) -> Optional[plt.Figure]:
    """
    Plots each observed variable against its corresponding trend component.
    This is a simplified version that assumes 1-to-1 mapping between observed and trend variables.
    """
    if (results.observed_data is None or results.trend_draws is None or
        results.trend_draws.shape[0] == 0):
        print("Warning: No data available for observed vs trend component plots.")
        return None

    observed_np = results.observed_data
    trend_draws_np = results.trend_draws
    
    T_timesteps, n_obs = observed_np.shape
    n_trends = trend_draws_np.shape[2]
    
    # Use minimum of observed and trend variables
    n_to_plot = min(n_obs, n_trends)
    if n_to_plot == 0:
        return None

    time_plot = (results.time_index if results.time_index is not None and 
                len(results.time_index) == T_timesteps else np.arange(T_timesteps))

    # Create figure
    fig, axes = plt.subplots(n_to_plot, 1, figsize=(12, 4 * n_to_plot), squeeze=False)

    # Compute trend statistics
    trend_stats = compute_summary_statistics(trend_draws_np)
    trend_line_data = (trend_stats.get('median') if use_median_for_trend_line 
                      else trend_stats.get('mean'))
    
    trend_hdi_lower, trend_hdi_upper = None, None
    if trend_draws_np.shape[0] > 1:
        trend_hdi_lower, trend_hdi_upper = compute_hdi_robust(trend_draws_np, results.hdi_prob)

    for i in range(n_to_plot):
        ax = axes[i, 0]
        
        obs_name = (results.observed_variable_names[i] 
                   if i < len(results.observed_variable_names) else f"Obs{i+1}")
        trend_name = (results.trend_names[i] 
                     if i < len(results.trend_names) else f"Trend{i+1}")

        # Plot trend HDI
        if (trend_hdi_lower is not None and trend_hdi_upper is not None and
            not (np.all(np.isnan(trend_hdi_lower[:, i])) or np.all(np.isnan(trend_hdi_upper[:, i])))):
            ax.fill_between(time_plot, trend_hdi_lower[:, i], trend_hdi_upper[:, i],
                           alpha=0.3, color='blue', 
                           label=f'{trend_name} {int(results.hdi_prob*100)}% HDI')

        # Plot trend line
        if (trend_line_data is not None and trend_line_data.ndim == 2 and
            not np.all(np.isnan(trend_line_data[:, i]))):
            line_label = f'{trend_name} ({"Median" if use_median_for_trend_line else "Mean"})'
            ax.plot(time_plot, trend_line_data[:, i], 'b-', 
                   linewidth=2, label=line_label)

        # Plot observed data
        if not np.all(np.isnan(observed_np[:, i])):
            ax.plot(time_plot, observed_np[:, i], 'ko-', 
                   linewidth=1.5, markersize=2, label=f'Observed {obs_name}', alpha=0.8)

        # Formatting
        ax.set_title(f'Observed {obs_name} vs Trend {trend_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        _format_datetime_axis(fig, ax, time_plot)
        
        if show_info_box:
            _add_info_box(ax, results.n_draws, results.hdi_prob)

    plt.tight_layout()
    
    # Save if requested
    if save_path and fig:
        try:
            fig.savefig(f"{save_path}_observed_vs_trend_components.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        except Exception as e:
            print(f"Error saving observed vs trend components plot: {e}")

    return fig


# Utility function for easy access to all plotting functionality
def create_all_standard_plots(
    results: SmootherResults,
    save_path_prefix: Optional[str] = None,
    show_info_boxes: bool = False
) -> Dict[str, Optional[plt.Figure]]:
    """
    Creates all standard plots for SmootherResults and optionally saves them.
    
    Returns:
        Dictionary with plot names as keys and figure objects as values (or None if saved)
    """
    plots = {}
    
    # # Observed vs Fitted
    # plots['observed_vs_fitted'] = plot_observed_vs_fitted(
    #     results, save_path=save_path_prefix, show_info_box=show_info_boxes
    # )
    
    # Component plots
    trend_fig, stat_fig = plot_smoother_results(
        results, save_path=save_path_prefix, show_info_box=show_info_boxes
    )
    plots['trend_components'] = trend_fig
    plots['stationary_components'] = stat_fig
    
    # Observed vs individual trend components
    plots['observed_vs_trend_components'] = plot_observed_vs_single_trend_component(
        results, save_path=save_path_prefix, show_info_box=show_info_boxes
    )
    
    return plots


if __name__ == "__main__":
    # Example usage would go here
    pass