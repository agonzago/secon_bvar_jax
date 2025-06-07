# clean_gpm_bvar_trends/reporting_plots.py - Simplified and Fixed

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, List, Union, Tuple, Dict, Any
import pandas as pd

from .common_types import SmootherResults
import os 

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False



def compute_hdi_robust(draws: Union[jnp.ndarray, np.ndarray],
                      hdi_prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Highest Density Interval.
    Uses ArviZ if available.
    Falls back to percentiles if ArviZ is not available or fails.
    """
    
    if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 2:
        dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

    draws_np = np.asarray(draws) # Ensure it's a NumPy array
    
    # Handle cases with no features/timesteps to compute HDI for
    if draws_np.ndim > 1 and draws_np.shape[1] == 0:
        out_shape = list(draws_np.shape[1:]) 
        return (np.full(out_shape, np.nan), np.full(out_shape, np.nan))

    try:
        if ARVIZ_AVAILABLE:
            if draws_np.ndim == 2:  # Input shape (n_draws, n_features)
                # ArviZ expects (chain, draw, *shape) format to avoid the warning
                # Reshape from (n_draws, n_features) to (1, n_draws, n_features)
                if np.all(np.isnan(draws_np)): # Check for all-NaN input
                     return (np.full(draws_np.shape[1:], np.nan), np.full(draws_np.shape[1:], np.nan))
                
                # Reshape to (chain, draw, features) format
                draws_reshaped = draws_np[np.newaxis, :, :]  # Add chain dimension
                hdi_result = az.hdi(draws_reshaped, hdi_prob=hdi_prob)
                return hdi_result[:, 0], hdi_result[:, 1]
            
            elif draws_np.ndim == 3:  # Input shape (n_draws, n_timesteps, n_variables)
                n_draws, n_timesteps, n_variables = draws_np.shape
                if n_timesteps == 0 or n_variables == 0: 
                    return (np.full((n_timesteps, n_variables), np.nan), 
                            np.full((n_timesteps, n_variables), np.nan))

                hdi_lower_all_vars = np.zeros((n_timesteps, n_variables))
                hdi_upper_all_vars = np.zeros((n_timesteps, n_variables))
                
                for v_idx in range(n_variables):
                    # variable_draws_at_v is (n_draws, n_timesteps)
                    variable_draws_at_v = draws_np[:, :, v_idx]
                    if variable_draws_at_v.shape[1] == 0: 
                        hdi_lower_all_vars[:, v_idx] = np.nan
                        hdi_upper_all_vars[:, v_idx] = np.nan
                        continue
                    
                    if np.all(np.isnan(variable_draws_at_v)): # Check for all-NaN slice
                        hdi_lower_all_vars[:, v_idx] = np.nan
                        hdi_upper_all_vars[:, v_idx] = np.nan
                        continue

                    # Reshape to (chain, draw, timesteps) format to avoid warning
                    variable_draws_reshaped = variable_draws_at_v[np.newaxis, :, :]
                    hdi_result_var = az.hdi(variable_draws_reshaped, hdi_prob=hdi_prob)
                    hdi_lower_all_vars[:, v_idx] = hdi_result_var[:, 0]
                    hdi_upper_all_vars[:, v_idx] = hdi_result_var[:, 1]
                return hdi_lower_all_vars, hdi_upper_all_vars
            else: # For 1D array (n_draws,) or other unexpected shapes, fall back to percentile
                pass # Fall through to percentile calculation

        # Fallback to percentiles if ArviZ not available or for non-2D/3D cases handled by ArviZ part
        if np.all(np.isnan(draws_np)): # Check before percentile calculation
            dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
            return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
            
        hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
        hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
        return hdi_lower, hdi_upper
        
    except Exception as e:
        # print(f"HDI computation failed: {e}. Falling back to percentiles if possible.") # Keep for debug
        try: # Attempt percentile fallback one last time
            if np.all(np.isnan(draws_np)):
                dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
                return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
            hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
            hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
            return hdi_lower, hdi_upper
        except Exception as e_fallback:
            # print(f"Percentile fallback for HDI also failed: {e_fallback}") # Keep for debug
            dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
            return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

def compute_summary_statistics(draws: Union[jnp.ndarray, np.ndarray]) -> dict:
    """Computes basic summary statistics (mean, median, std)."""
    if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 1:
        dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
        return {
            'mean': np.full(dummy_shape, np.nan),
            'median': np.full(dummy_shape, np.nan),
            'mode': np.full(dummy_shape, np.nan), # Mode is complex, median is a robust proxy
            'std': np.full(dummy_shape, np.nan)
        }

    draws_np = np.asarray(draws)
    return {
        'mean': np.nanmean(draws_np, axis=0),
        'median': np.nanmedian(draws_np, axis=0),
        'mode': np.nanmedian(draws_np, axis=0),  # Approximate mode with median for simplicity
        'std': np.nanstd(draws_np, axis=0)
    }


def _format_datetime_axis(fig, ax, time_index):
    """Format x-axis for datetime indices."""
    try:
        # Check if time_index is like a pandas DatetimeIndex
        if hasattr(time_index, 'to_pydatetime') and callable(time_index.to_pydatetime):
            if not isinstance(time_index, pd.RangeIndex) or len(time_index) > 0 :
                 fig.autofmt_xdate()
    except Exception:
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

    if draws_np.ndim != 3 or draws_np.shape[0] < 1:
        print(f"Invalid draws shape for plot_time_series_with_uncertainty: {draws_np.shape}, expected (n_draws > 0, T, n_series). Skipping plot.")
        return None
    
    n_draws, T_timesteps, n_series = draws_np.shape

    if n_series != len(variable_names):
        print(f"Mismatch between number of series in draws ({n_series}) and variable_names ({len(variable_names)}). Skipping plot.")
        return None
        
    if T_timesteps == 0:
        print(f"No timesteps to plot for {title_prefix}. Skipping plot.")
        return None

    if time_index is None or len(time_index) != T_timesteps:
        time_index_plot = np.arange(T_timesteps)
    else:
        time_index_plot = time_index

    stats = compute_summary_statistics(draws_np) 
    mean_lines = stats.get('mean')
    median_lines = stats.get('median')
    
    hdi_lower, hdi_upper = None, None
    if show_hdi and n_draws > 1:
        hdi_lower, hdi_upper = compute_hdi_robust(draws_np, hdi_prob) 

    if mean_lines is None or not np.any(np.isfinite(mean_lines)):
        print(f"No valid data (mean_lines are None or all NaN) for {title_prefix}. Skipping plot.")
        return None

    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series), squeeze=False)

    for i in range(n_series):
        ax = axes[i, 0]
        var_name = variable_names[i]

        current_hdi_lower = hdi_lower[:, i] if hdi_lower is not None and hdi_lower.ndim == 2 and hdi_lower.shape[1] == n_series else None
        current_hdi_upper = hdi_upper[:, i] if hdi_upper is not None and hdi_upper.ndim == 2 and hdi_upper.shape[1] == n_series else None

        if show_hdi and current_hdi_lower is not None and current_hdi_upper is not None:
            if not (np.all(np.isnan(current_hdi_lower)) or np.all(np.isnan(current_hdi_upper))):
                ax.fill_between(time_index_plot, current_hdi_lower, current_hdi_upper,
                               alpha=alpha_fill, color='royalblue', 
                               label=f'{int(hdi_prob*100)}% HDI')

        line_to_plot_data = None
        line_label = None
        color_for_line = 'royalblue' 

        if show_mean and mean_lines is not None and mean_lines.ndim == 2 and mean_lines.shape[1] == n_series:
            if not np.all(np.isnan(mean_lines[:, i])):
                line_to_plot_data = mean_lines[:, i]
                line_label = 'Mean'
                color_for_line = 'blue'
        
        if show_median and median_lines is not None and median_lines.ndim == 2 and median_lines.shape[1] == n_series:
             if not np.all(np.isnan(median_lines[:, i])):
                line_to_plot_data = median_lines[:, i]
                line_label = 'Median'
                color_for_line = 'green'


        if line_to_plot_data is not None and len(line_to_plot_data) == T_timesteps:
            ax.plot(time_index_plot, line_to_plot_data, color=color_for_line, linestyle='-', linewidth=2, label=line_label)

        ax.set_title(f'{title_prefix}: {var_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        if line_label or (show_hdi and current_hdi_lower is not None): 
             ax.legend()
        
        _format_datetime_axis(fig, ax, time_index_plot)
        
        if show_info_box:
            _add_info_box(ax, n_draws, hdi_prob)

    plt.tight_layout()
    return fig


# def plot_custom_series_comparison(
#     plot_title: str,
#     series_specs: List[Dict],
#     results: SmootherResults,
#     save_path: Optional[str] = None,
#     show_info_box: bool = False
# ) -> Optional[plt.Figure]:
#     """
#     Plots multiple specified series from SmootherResults on a single axes.
#     COMPLETE WORKING VERSION - plots custom series combinations properly.
    
#     Args:
#         plot_title: Title for the plot
#         series_specs: List of dicts with keys: 'type', 'name', 'label', 'style', 'color', 'show_hdi'
#         results: SmootherResults object
#         save_path: Optional path to save plot
#         show_info_box: Whether to show info box
#     """
#     # Extract data from results
#     data_sources = {
#         'observed': (results.observed_data, results.observed_variable_names),
#         'trend': (results.trend_draws, results.trend_names),
#         'stationary': (results.stationary_draws, results.stationary_names)
#     }
    
#     time_index_plot = results.time_index
#     hdi_prob = results.hdi_prob

#     # Determine time dimension
#     T = 0
#     for data, _ in data_sources.values():
#         if data is not None and data.ndim > 0:
#             T = data.shape[1] if data.ndim > 1 else data.shape[0]
#             break

#     if T == 0:
#         print(f"Warning: No data for custom plot '{plot_title}'. Skipping plot.")
#         return None

#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 6))

#     if time_index_plot is None or len(time_index_plot) != T:
#         time_index_plot = np.arange(T)

#     plotted_labels = set()

#     for spec in series_specs:
#         series_type = spec.get('type')
#         series_name = spec.get('name')
#         label = spec.get('label', series_name if series_name else series_type)
        
#         if not label or not series_type or not series_name:
#             continue

#         # Parse style and color
#         style = spec.get('style', '-')
#         color = spec.get('color')
#         show_hdi = spec.get('show_hdi', False if series_type == 'observed' else True)
#         alpha_fill = spec.get('hdi_alpha', 0.2)

#         # Get data
#         if series_type not in data_sources:
#             print(f"Warning: Unsupported series type '{series_type}'. Skipping.")
#             continue

#         data_array, name_list = data_sources[series_type]
#         if data_array is None or not name_list:
#             print(f"Warning: No data available for type '{series_type}'. Skipping.")
#             continue

#         try:
#             idx = name_list.index(series_name)
#         except ValueError:
#             print(f"Warning: Custom plot series '{series_name}' (type: {series_type}) not found. Available names for type '{series_type}': {name_list}. Skipping series.")
#             continue

#         # Extract series data
#         if series_type == 'observed':
#             if idx >= data_array.shape[1]:
#                 continue
#             current_series_data = data_array[:, idx]
#         else:
#             if data_array.ndim != 3 or idx >= data_array.shape[2]:
#                 continue
#             current_series_data = data_array[:, :, idx]

#         # Plot the series
#         if (current_series_data is not None and 
#             len(current_series_data) == T and label not in plotted_labels):
            
#             plotted_labels.add(label)
            
#             if current_series_data.ndim == 1:  # Observed data
#                 if not np.all(np.isnan(current_series_data)):
#                     plot_args = {'label': label, 'linestyle': style}
#                     if color:
#                         plot_args['color'] = color
#                     ax.plot(time_index_plot, current_series_data, **plot_args)
            
#             elif current_series_data.ndim == 2:  # Draw data (trends/stationary)
#                 if current_series_data.shape[0] > 0:
#                     stats = compute_summary_statistics(current_series_data)
#                     line_data = (stats.get('median') if current_series_data.shape[0] > 1 
#                                else stats.get('mean'))
                    
#                     if (line_data is not None and len(line_data) == T and 
#                         not np.all(np.isnan(line_data))):
                        
#                         plot_args = {'label': label, 'linestyle': style}
#                         if color:
#                             plot_args['color'] = color
#                         ax.plot(time_index_plot, line_data, **plot_args)
                        
#                         # Plot HDI
#                         if show_hdi and current_series_data.shape[0] > 1:
#                             hdi_lower, hdi_upper = compute_hdi_robust(current_series_data, hdi_prob)
                            
#                             if (hdi_lower is not None and hdi_upper is not None and
#                                 hdi_lower.ndim == 1 and hdi_lower.shape == (T,) and
#                                 not (np.all(np.isnan(hdi_lower)) or np.all(np.isnan(hdi_upper)))):
                                
#                                 fill_color = color if color else 'blue'
#                                 ax.fill_between(time_index_plot, hdi_lower, hdi_upper,
#                                               alpha=alpha_fill, color=fill_color,
#                                               label=f'{label} {int(hdi_prob*100)}% HDI')

#     # Formatting
#     ax.set_title(plot_title)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.grid(True, alpha=0.3)
#     if plotted_labels:
#         ax.legend()
    
#     _format_datetime_axis(fig, ax, time_index_plot)
    
#     if show_info_box:
#         _add_info_box(ax, results.n_draws, hdi_prob)

#     plt.tight_layout()
    
#     # Save if requested
#     if save_path and fig:
#         try:
#             safe_title = plot_title.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('=', '_').replace(':', '_').replace('.', '')
#             fig.savefig(f"{save_path}_{safe_title}.png", dpi=150, bbox_inches='tight')
#             plt.close(fig)
#             return None
#         except Exception as e:
#             print(f"Error saving custom plot '{plot_title}': {e}")

#     return fig

# def plot_custom_series_comparison(
#     plot_title: str,
#     series_specs: List[Dict],
#     results: SmootherResults,
#     save_path: Optional[str] = None,
#     show_info_box: bool = False
# ) -> Optional[plt.Figure]:
#     """
#     Plots multiple specified series from SmootherResults on a single axes.
#     COMPLETE WORKING VERSION - plots custom series combinations properly.
    
#     Args:
#         plot_title: Title for the plot
#         series_specs: List of dicts with keys: 'type', 'name', 'label', 'style', 'color', 'show_hdi'
#         results: SmootherResults object
#         save_path: Optional path to save plot
#         show_info_box: Whether to show info box
#     """
#     # --- No changes needed in this section ---
#     data_sources = {
#         'observed': (results.observed_data, results.observed_variable_names),
#         'trend': (results.trend_draws, results.trend_names),
#         'stationary': (results.stationary_draws, results.stationary_names)
#     }
#     time_index_plot = results.time_index
#     hdi_prob = results.hdi_prob
#     T = 0
#     for data, _ in data_sources.values():
#         if data is not None and data.ndim > 0:
#             T = data.shape[1] if data.ndim > 1 else data.shape[0]
#             if T > 0: break
#     if T == 0:
#         print(f"Warning: No data for custom plot '{plot_title}'. Skipping plot.")
#         return None
#     fig, ax = plt.subplots(figsize=(12, 6))
#     if time_index_plot is None or len(time_index_plot) != T:
#         time_index_plot = np.arange(T)
#     plotted_labels = set()

#     for spec in series_specs:
#         series_type = spec.get('type')
#         series_name = spec.get('name')
#         label = spec.get('label', series_name if series_name else series_type)
#         if not label or not series_type or not series_name: continue
#         style = spec.get('style', '-')
#         color = spec.get('color')
#         show_hdi = spec.get('show_hdi', False if series_type == 'observed' else True)
#         alpha_fill = spec.get('hdi_alpha', 0.2)
#         if series_type not in data_sources: continue
#         data_array, name_list = data_sources[series_type]
#         if data_array is None or not name_list: continue
#         try:
#             idx = name_list.index(series_name)
#         except ValueError:
#             print(f"Warning: Custom plot series '{series_name}' (type: {series_type}) not found. Available names for type '{series_type}': {name_list}. Skipping series.")
#             continue
#         if series_type == 'observed':
#             if idx >= data_array.shape[1]: continue
#             current_series_data = data_array[:, idx]
#         else:
#             if data_array.ndim != 3 or idx >= data_array.shape[2]: continue
#             current_series_data = data_array[:, :, idx]

#         # --- FIX IS HERE ---
#         # The condition to check if the data is valid before plotting.
        
#         # Determine if the data is valid for plotting based on its shape
#         is_valid_for_plotting = False
#         if current_series_data is not None:
#             if current_series_data.ndim == 1 and current_series_data.shape[0] == T:
#                 # This is for 1D data like observed series
#                 is_valid_for_plotting = True
#             elif current_series_data.ndim == 2 and current_series_data.shape[1] == T:
#                 # This is for 2D draw data (n_draws, n_timesteps)
#                 is_valid_for_plotting = True

#         if is_valid_for_plotting and label not in plotted_labels:
#         # --- END OF FIX ---
            
#             plotted_labels.add(label)
            
#             if current_series_data.ndim == 1:  # Observed data
#                 if not np.all(np.isnan(current_series_data)):
#                     plot_args = {'label': label, 'linestyle': style}
#                     if color:
#                         plot_args['color'] = color
#                     ax.plot(time_index_plot, current_series_data, **plot_args)
            
#             elif current_series_data.ndim == 2:  # Draw data (trends/stationary)
#                 if current_series_data.shape[0] > 0:
#                     stats = compute_summary_statistics(current_series_data)
#                     line_data = (stats.get('median') if current_series_data.shape[0] > 1 
#                                else stats.get('mean'))
                    
#                     if (line_data is not None and len(line_data) == T and 
#                         not np.all(np.isnan(line_data))):
                        
#                         plot_args = {'label': label, 'linestyle': style}
#                         if color:
#                             plot_args['color'] = color
#                         ax.plot(time_index_plot, line_data, **plot_args)
                        
#                         if show_hdi and current_series_data.shape[0] > 1:
#                             hdi_lower, hdi_upper = compute_hdi_robust(current_series_data, hdi_prob)
                            
#                             if (hdi_lower is not None and hdi_upper is not None and
#                                 hdi_lower.ndim == 1 and hdi_lower.shape == (T,) and
#                                 not (np.all(np.isnan(hdi_lower)) or np.all(np.isnan(hdi_upper)))):
                                
#                                 fill_color = color if color else 'blue'
#                                 ax.fill_between(time_index_plot, hdi_lower, hdi_upper,
#                                               alpha=alpha_fill, color=fill_color,
#                                               label=f'{label} {int(hdi_prob*100)}% HDI')

#     # --- No changes needed in this section ---
#     ax.set_title(plot_title)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.grid(True, alpha=0.3)
#     if plotted_labels:
#         ax.legend()
#     _format_datetime_axis(fig, ax, time_index_plot)
#     if show_info_box:
#         _add_info_box(ax, results.n_draws, hdi_prob)
#     plt.tight_layout()
#     if save_path and fig:
#         try:
#             safe_title = plot_title.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('=', '_').replace(':', '_').replace('.', '')
#             fig.savefig(f"{save_path}_{safe_title}.png", dpi=150, bbox_inches='tight')
#             plt.close(fig)
#             return None
#         except Exception as e:
#             print(f"Error saving custom plot '{plot_title}': {e}")
#     return fig

def plot_custom_series_comparison(
    plot_title: str,
    series_specs: List[Dict],
    results: SmootherResults,
    save_path: Optional[str] = None,
    show_info_box: bool = False
) -> Optional[plt.Figure]:
    """
    Plots multiple specified series from SmootherResults on a single axes.
    This version uses pre-computed statistics for reliability and efficiency.
    """
    data_sources = {
        'observed': results.observed_data,
        'trend_stats': results.trend_stats,
        'stationary_stats': results.stationary_stats,
        'trend_hdi_lower': results.trend_hdi_lower,
        'trend_hdi_upper': results.trend_hdi_upper,
        'stationary_hdi_lower': results.stationary_hdi_lower,
        'stationary_hdi_upper': results.stationary_hdi_upper,
        'trend_names': results.trend_names,
        'stationary_names': results.stationary_names,
        'observed_names': results.observed_variable_names
    }
    
    T = results.observed_data.shape[0] if results.observed_data is not None else 0
    if T == 0:
        print(f"Warning: No data for custom plot '{plot_title}'. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    time_index_plot = results.time_index if results.time_index is not None and len(results.time_index) == T else np.arange(T)
    plotted_labels = set()

    for spec in series_specs:
        series_type = spec.get('type')
        series_name = spec.get('name')
        label = spec.get('label', series_name)
        if not all([series_type, series_name, label]) or label in plotted_labels:
            continue

        style = spec.get('style', '-')
        color = spec.get('color')
        show_hdi = spec.get('show_hdi', series_type != 'observed')
        alpha_fill = spec.get('hdi_alpha', 0.2)
        
        plotted_labels.add(label)

        if series_type == 'observed':
            name_list = data_sources['observed_names']
            data_array = data_sources['observed']
            if name_list and series_name in name_list:
                idx = name_list.index(series_name)
                series_data = data_array[:, idx]
                if not np.all(np.isnan(series_data)):
                    ax.plot(time_index_plot, series_data, label=label, linestyle=style, color=color)
        else: # For 'trend' or 'stationary'
            stats_dict = data_sources.get(f'{series_type}_stats')
            name_list = data_sources.get(f'{series_type}_names')
            
            if stats_dict and name_list and series_name in name_list:
                idx = name_list.index(series_name)
                
                # Plot the median line
                line_data = stats_dict.get('median')
                if line_data is not None and not np.all(np.isnan(line_data[:, idx])):
                    ax.plot(time_index_plot, line_data[:, idx], label=label, linestyle=style, color=color)

                # Plot the HDI bands
                hdi_lower = data_sources.get(f'{series_type}_hdi_lower')
                hdi_upper = data_sources.get(f'{series_type}_hdi_upper')
                if show_hdi and hdi_lower is not None and hdi_upper is not None:
                    lower_band = hdi_lower[:, idx]
                    upper_band = hdi_upper[:, idx]
                    if not (np.all(np.isnan(lower_band)) or np.all(np.isnan(upper_band))):
                        fill_color = color if color else 'gray'
                        ax.fill_between(time_index_plot, lower_band, upper_band, 
                                      alpha=alpha_fill, color=fill_color, 
                                      label=f'{label} {int(results.hdi_prob*100)}% HDI')
            else:
                 print(f"Warning: Could not find data for '{series_name}' of type '{series_type}'.")


    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        # Saving logic here...
        plt.close(fig)
        
        return None
        
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
        results.trend_draws.shape[0] > 0 and 
        results.trend_draws.shape[2] > 0):   
        
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
        results.stationary_draws.shape[0] > 0 and 
        results.stationary_draws.shape[2] > 0):   
        
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

    if save_path:
        save_dir = os.path.dirname(save_path) 
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

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
    FIXED: Plots each observed variable against its corresponding trend component.
    Uses original measurement equations to properly match observables to trends instead of index-based mapping.
    """
    if (results.observed_data is None or results.trend_draws is None or
        results.trend_draws.shape[0] == 0):
        print("Warning: No data available for observed vs trend component plots.")
        return None

    observed_np = results.observed_data
    trend_draws_np = results.trend_draws
    
    T_timesteps, n_obs = observed_np.shape
    n_trends = trend_draws_np.shape[2]
    
    if T_timesteps == 0 or n_obs == 0 or n_trends == 0:
        print("Warning: No data dimensions available for plotting.")
        return None

    time_plot = (results.time_index if results.time_index is not None and 
                len(results.time_index) == T_timesteps else np.arange(T_timesteps))

    # --- FIX STARTS HERE ---
    # Use original equations, not reduced ones, to find the correct trend component for each observable.
    if not results.gpm_model or not hasattr(results.gpm_model, 'all_original_measurement_equations'):
        print("Warning: Original measurement equations not available in SmootherResults. Cannot plot observed vs trend.")
        return None

    original_meas_eqs = {eq.lhs: eq for eq in results.gpm_model.all_original_measurement_equations}

    if not original_meas_eqs:
        print("Warning: No original measurement equations found. Cannot plot observed vs trend.")
        return None

    # Find valid observable-trend pairs using the ORIGINAL measurement equations
    valid_pairs = []
    for obs_idx, obs_name in enumerate(results.observed_variable_names):
        if obs_name not in original_meas_eqs:
            print(f"Warning: Observable '{obs_name}' not found in original measurement equations. Skipping.")
            continue
            
        # Parse the original measurement equation to find trend terms
        eq = original_meas_eqs[obs_name]
        trend_candidates = []
        
        for term in eq.rhs_terms:
            # In the original ME, the trend variable should be a single term
            var_name = term.variable
            lag = term.lag
            
            # We are looking for a contemporaneous trend variable
            if lag == 0 and var_name in results.trend_names:
                trend_candidates.append(var_name)
        
        # Choose the primary trend component from the candidates
        if len(trend_candidates) == 1:
            primary_trend = trend_candidates[0]
        elif len(trend_candidates) > 1:
            # If an observable is defined by multiple trends (e.g., y = trend1 + trend2 + cycle),
            # this logic defaults to the first one found. This is a reasonable fallback.
            primary_trend = trend_candidates[0]
            print(f"Info: Observable '{obs_name}' has multiple trend components in its original ME. Defaulting to the first: '{primary_trend}'.")
        else:
            print(f"Warning: No trend components found for observable '{obs_name}' in original ME. Skipping.")
            continue
        
        # Find the index of the chosen trend in the reconstructed data
        try:
            trend_idx = results.trend_names.index(primary_trend)
            valid_pairs.append((obs_idx, obs_name, trend_idx, primary_trend))
        except ValueError:
            print(f"Warning: Trend '{primary_trend}' (from original ME) not found in reconstructed trend names list. Skipping '{obs_name}'.")
            continue

    if not valid_pairs:
        print("Warning: No valid observable-trend pairs found for plotting.")
        return None
    # --- FIX ENDS HERE ---

    # Create figure with subplots for each valid pair
    n_pairs = len(valid_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4 * n_pairs), squeeze=False)

    # Compute trend statistics
    trend_stats = compute_summary_statistics(trend_draws_np)
    line_data_key = 'median' if use_median_for_trend_line and results.n_draws > 1 else 'mean'
    trend_line_data_all_vars = trend_stats.get(line_data_key)
    
    trend_hdi_lower_all_vars, trend_hdi_upper_all_vars = None, None
    if results.n_draws > 1:
        trend_hdi_lower_all_vars, trend_hdi_upper_all_vars = compute_hdi_robust(trend_draws_np, results.hdi_prob)

    # Plot each valid pair
    for plot_idx, (obs_idx, obs_name, trend_idx, trend_name) in enumerate(valid_pairs):
        ax = axes[plot_idx, 0]
        
        current_trend_line_data = (trend_line_data_all_vars[:, trend_idx] 
                                 if trend_line_data_all_vars is not None and trend_line_data_all_vars.ndim == 2 
                                 else None)
        
        # Plot trend HDI if available
        if (trend_hdi_lower_all_vars is not None and trend_hdi_upper_all_vars is not None and
            not (np.all(np.isnan(trend_hdi_lower_all_vars[:, trend_idx])) or 
                 np.all(np.isnan(trend_hdi_upper_all_vars[:, trend_idx])))):
            ax.fill_between(time_plot, 
                           trend_hdi_lower_all_vars[:, trend_idx], 
                           trend_hdi_upper_all_vars[:, trend_idx],
                           alpha=0.3, color='blue', 
                           label=f'{trend_name} {int(results.hdi_prob*100)}% HDI')

        # Plot trend line
        if (current_trend_line_data is not None and 
            not np.all(np.isnan(current_trend_line_data))):
            line_label = f'{trend_name} ({"Median" if use_median_for_trend_line else "Mean"})'
            ax.plot(time_plot, current_trend_line_data, 'b-', 
                   linewidth=2, label=line_label)

        # Plot observed data
        if not np.all(np.isnan(observed_np[:, obs_idx])):
            ax.plot(time_plot, observed_np[:, obs_idx], 'ko-', 
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


def _parse_variable_key_for_plotting(var_key: str) -> tuple[str, int]:
    """
    Helper function to parse variable key like 'var_name(-1)' into ('var_name', -1).
    For contemporaneous variables like 'var_name', returns ('var_name', 0).
    """
    if '(' in var_key and ')' in var_key:
        # Extract lag from parentheses
        base_name = var_key[:var_key.index('(')]
        lag_str = var_key[var_key.index('(')+1:var_key.index(')')]
        try:
            lag = int(lag_str)
        except ValueError:
            lag = 0
        return base_name.strip(), lag
    else:
        return var_key.strip(), 0

# ... (rest of reporting_plots.py remains the same) ...

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
    
    trend_fig, stat_fig = plot_smoother_results(
        results, save_path=save_path_prefix, show_info_box=show_info_boxes
    )
    plots['trend_components'] = trend_fig
    plots['stationary_components'] = stat_fig
    
    plots['observed_vs_trend_components'] = plot_observed_vs_single_trend_component(
        results, save_path=save_path_prefix, show_info_box=show_info_boxes
    )
    
    return plots


if __name__ == "__main__":
    pass