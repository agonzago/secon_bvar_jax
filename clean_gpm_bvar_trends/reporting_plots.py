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

# def compute_hdi_robust(draws: Union[jnp.ndarray, np.ndarray],
#                       hdi_prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Computes the Highest Density Interval.
#     Uses ArviZ if available, with transposition to align with future ArviZ 2D input interpretation (chain, draw).
#     Falls back to percentiles if ArviZ is not available or fails.
#     """
    
#     if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 2:
#         dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
#         return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

#     draws_np = np.asarray(draws) # Ensure it's a NumPy array
    
#     # Handle cases with no features/timesteps to compute HDI for
#     if draws_np.ndim > 1 and draws_np.shape[1] == 0:
#         out_shape = list(draws_np.shape[1:]) 
#         return (np.full(out_shape, np.nan), np.full(out_shape, np.nan))

#     try:
#         if ARVIZ_AVAILABLE:
#             if draws_np.ndim == 2:  # Input shape (n_draws, n_features)
#                 # Transpose to (n_features, n_draws) for az.hdi if it expects (chain, draw)
#                 # az.hdi will return (n_features, 2)
#                 hdi_result = az.hdi(draws_np.T, hdi_prob=hdi_prob)
#                 return hdi_result[:, 0], hdi_result[:, 1]
            
#             elif draws_np.ndim == 3:  # Input shape (n_draws, n_timesteps, n_variables)
#                 n_draws, n_timesteps, n_variables = draws_np.shape
#                 if n_timesteps == 0 or n_variables == 0: # Check if features or timesteps are zero
#                     return (np.full((n_timesteps, n_variables), np.nan), 
#                             np.full((n_timesteps, n_variables), np.nan))

#                 hdi_lower_all_vars = np.zeros((n_timesteps, n_variables))
#                 hdi_upper_all_vars = np.zeros((n_timesteps, n_variables))
                
#                 for v_idx in range(n_variables):
#                     # variable_draws_at_v is (n_draws, n_timesteps)
#                     variable_draws_at_v = draws_np[:, :, v_idx]
#                     if variable_draws_at_v.shape[1] == 0: # No timesteps for this variable slice
#                         hdi_lower_all_vars[:, v_idx] = np.nan
#                         hdi_upper_all_vars[:, v_idx] = np.nan
#                         continue
                    
#                     # Transpose to (n_timesteps, n_draws) for az.hdi
#                     # az.hdi will return (n_timesteps, 2)
#                     hdi_result_var = az.hdi(variable_draws_at_v.T, hdi_prob=hdi_prob)
#                     hdi_lower_all_vars[:, v_idx] = hdi_result_var[:, 0]
#                     hdi_upper_all_vars[:, v_idx] = hdi_result_var[:, 1]
#                 return hdi_lower_all_vars, hdi_upper_all_vars
#             else: # For 1D array (n_draws,) or other unexpected shapes, fall back to percentile
#                 pass # Fall through to percentile calculation

#         # Fallback to percentiles if ArviZ not available or for non-2D/3D cases handled by ArviZ part
#         hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
#         hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
#         return hdi_lower, hdi_upper
        
#     except Exception as e:
#         print(f"HDI computation failed: {e}. Falling back to percentiles if possible.")
#         try: # Attempt percentile fallback one last time
#             hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
#             hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
#             return hdi_lower, hdi_upper
#         except Exception as e_fallback:
#             print(f"Percentile fallback for HDI also failed: {e_fallback}")
#             dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
#             return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))


# def compute_hdi_robust(draws: Union[jnp.ndarray, np.ndarray],
#                       hdi_prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Computes the Highest Density Interval.
#     Uses ArviZ if available.
#     Falls back to percentiles if ArviZ is not available or fails.
#     """
    
#     if not hasattr(draws, 'shape') or draws.ndim == 0 or draws.shape[0] < 2:
#         dummy_shape = draws.shape[1:] if hasattr(draws, 'shape') and draws.ndim > 0 else ()
#         return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

#     draws_np = np.asarray(draws) # Ensure it's a NumPy array
    
#     # Handle cases with no features/timesteps to compute HDI for
#     if draws_np.ndim > 1 and draws_np.shape[1] == 0:
#         out_shape = list(draws_np.shape[1:]) 
#         return (np.full(out_shape, np.nan), np.full(out_shape, np.nan))

#     try:
#         if ARVIZ_AVAILABLE:
#             if draws_np.ndim == 2:  # Input shape (n_draws, n_features)
#                 # ArviZ expects (draws, features) for this type of HDI calculation.
#                 # Output will be (n_features, 2).
#                 if np.all(np.isnan(draws_np)): # Check for all-NaN input
#                      return (np.full(draws_np.shape[1:], np.nan), np.full(draws_np.shape[1:], np.nan))
#                 hdi_result = az.hdi(draws_np, hdi_prob=hdi_prob) # Pass directly
#                 return hdi_result[:, 0], hdi_result[:, 1]
            
#             elif draws_np.ndim == 3:  # Input shape (n_draws, n_timesteps, n_variables)
#                 n_draws, n_timesteps, n_variables = draws_np.shape
#                 if n_timesteps == 0 or n_variables == 0: 
#                     return (np.full((n_timesteps, n_variables), np.nan), 
#                             np.full((n_timesteps, n_variables), np.nan))

#                 hdi_lower_all_vars = np.zeros((n_timesteps, n_variables))
#                 hdi_upper_all_vars = np.zeros((n_timesteps, n_variables))
                
#                 for v_idx in range(n_variables):
#                     # variable_draws_at_v is (n_draws, n_timesteps)
#                     variable_draws_at_v = draws_np[:, :, v_idx]
#                     if variable_draws_at_v.shape[1] == 0: 
#                         hdi_lower_all_vars[:, v_idx] = np.nan
#                         hdi_upper_all_vars[:, v_idx] = np.nan
#                         continue
                    
#                     if np.all(np.isnan(variable_draws_at_v)): # Check for all-NaN slice
#                         hdi_lower_all_vars[:, v_idx] = np.nan
#                         hdi_upper_all_vars[:, v_idx] = np.nan
#                         continue

#                     # Pass (n_draws, n_timesteps) directly to az.hdi.
#                     # az.hdi will interpret as (draws, features=timesteps) and return (n_timesteps, 2).
#                     hdi_result_var = az.hdi(variable_draws_at_v, hdi_prob=hdi_prob)
#                     hdi_lower_all_vars[:, v_idx] = hdi_result_var[:, 0]
#                     hdi_upper_all_vars[:, v_idx] = hdi_result_var[:, 1]
#                 return hdi_lower_all_vars, hdi_upper_all_vars
#             else: # For 1D array (n_draws,) or other unexpected shapes, fall back to percentile
#                 pass # Fall through to percentile calculation

#         # Fallback to percentiles if ArviZ not available or for non-2D/3D cases handled by ArviZ part
#         if np.all(np.isnan(draws_np)): # Check before percentile calculation
#             dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
#             return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
            
#         hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
#         hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
#         return hdi_lower, hdi_upper
        
#     except Exception as e:
#         # print(f"HDI computation failed: {e}. Falling back to percentiles if possible.") # Keep for debug
#         try: # Attempt percentile fallback one last time
#             if np.all(np.isnan(draws_np)):
#                 dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
#                 return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
#             hdi_lower = np.percentile(draws_np, (1 - hdi_prob) / 2 * 100, axis=0)
#             hdi_upper = np.percentile(draws_np, (1 + hdi_prob) / 2 * 100, axis=0)
#             return hdi_lower, hdi_upper
#         except Exception as e_fallback:
#             # print(f"Percentile fallback for HDI also failed: {e_fallback}") # Keep for debug
#             dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
#             return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))

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



def plot_custom_series_comparison(
    plot_title: str,
    series_specs: List[Dict],
    results: SmootherResults,
    save_path: Optional[str] = None,
    show_info_box: bool = False
) -> Optional[plt.Figure]:
    """
    Plots multiple specified series from SmootherResults on a single axes.
    FIXED: Corrected data processing and plotting conditions.
    """
    data_sources = {
        'observed': (results.observed_data, results.observed_variable_names),
        'trend': (results.trend_draws, results.trend_names),
        'stationary': (results.stationary_draws, results.stationary_names)
    }
    
    time_index_plot = results.time_index
    hdi_prob = results.hdi_prob
    n_total_draws_for_info = results.n_draws

    T = 0
    if results.observed_data is not None and results.observed_data.ndim > 0:
        T = results.observed_data.shape[0]
    elif results.trend_draws is not None and results.trend_draws.ndim > 1:
        T = results.trend_draws.shape[1]
    elif results.stationary_draws is not None and results.stationary_draws.ndim > 1:
        T = results.stationary_draws.shape[1]

    if T == 0:
        print(f"Warning: No time dimension found for custom plot '{plot_title}'. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    if time_index_plot is None or len(time_index_plot) != T:
        time_index_plot = np.arange(T)

    plotted_labels = set()

    for spec_idx, spec in enumerate(series_specs):
        series_type = spec.get('type')
        series_name = spec.get('name')
        label = spec.get('label', series_name if series_name else series_type)
        
        if not label or not series_type or not series_name:
            continue

        style = spec.get('style', '-')
        color = spec.get('color')
        show_hdi_flag = spec.get('show_hdi', False if series_type == 'observed' else True)
        alpha_fill = spec.get('hdi_alpha', 0.2)

        if series_type not in data_sources:
            continue

        data_array_source, name_list_source = data_sources[series_type]
        if data_array_source is None or not name_list_source:
            continue

        try:
            var_idx_in_source = name_list_source.index(series_name)
        except ValueError:
            continue
        
        series_data_to_plot = None
        is_draw_data = False

        if series_type == 'observed':
            if data_array_source.ndim == 2 and 0 <= var_idx_in_source < data_array_source.shape[1]:
                series_data_to_plot = data_array_source[:, var_idx_in_source] 
            is_draw_data = False
        else: 
            if data_array_source.ndim == 3 and 0 <= var_idx_in_source < data_array_source.shape[2]:
                series_data_to_plot = data_array_source[:, :, var_idx_in_source] 
            is_draw_data = True
        
        if series_data_to_plot is None or series_data_to_plot.size == 0:
            continue
            
        if np.all(np.isnan(series_data_to_plot)):
            continue

        actual_label_for_plot = label if label not in plotted_labels else "_nolegend_"

        if not is_draw_data: 
            if series_data_to_plot.ndim == 1 and series_data_to_plot.shape[0] == T:
                plot_args = {'label': actual_label_for_plot, 'linestyle': style}
                if color: plot_args['color'] = color
                ax.plot(time_index_plot, series_data_to_plot, **plot_args)
                if actual_label_for_plot != "_nolegend_": plotted_labels.add(label)
        
        elif is_draw_data: 
            if series_data_to_plot.ndim == 2 and series_data_to_plot.shape[1] == T and series_data_to_plot.shape[0] > 0:
                n_draws_this_series = series_data_to_plot.shape[0]
                stats = compute_summary_statistics(series_data_to_plot) 
                line_data_key = 'median' if n_draws_this_series > 1 else 'mean'
                line_data_for_plot = stats.get(line_data_key)
                
                if line_data_for_plot is not None and line_data_for_plot.shape == (T,) and not np.all(np.isnan(line_data_for_plot)):
                    plot_args = {'label': actual_label_for_plot, 'linestyle': style}
                    if color: plot_args['color'] = color
                    ax.plot(time_index_plot, line_data_for_plot, **plot_args)
                    if actual_label_for_plot != "_nolegend_": plotted_labels.add(label)
                    
                    if show_hdi_flag and n_draws_this_series > 1:
                        hdi_lower, hdi_upper = compute_hdi_robust(series_data_to_plot, hdi_prob) 
                        if (hdi_lower is not None and hdi_upper is not None and
                            hdi_lower.shape == (T,) and hdi_upper.shape == (T,) and
                            not (np.all(np.isnan(hdi_lower)) or np.all(np.isnan(hdi_upper)))):
                            
                            fill_args_hdi = {'alpha': alpha_fill}
                            fill_color_val_hdi = color if color else (ax.lines[-1].get_color() if ax.lines else 'blue')
                            fill_args_hdi['color'] = fill_color_val_hdi
                            ax.fill_between(time_index_plot, hdi_lower, hdi_upper, **fill_args_hdi)

    ax.set_title(plot_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    if plotted_labels:
        ax.legend()
    
    _format_datetime_axis(fig, ax, time_index_plot)
    
    if show_info_box:
        _add_info_box(ax, n_total_draws_for_info, hdi_prob)
    
    plt.tight_layout()

    if save_path and fig:
        try:
            safe_title = (plot_title.lower().replace(' ', '_').replace('/', '_')
                         .replace('(', '').replace(')', '').replace('=', '_')
                         .replace(':', '_').replace('.', ''))
            
            save_dir = os.path.dirname(save_path) 
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            full_save_filename = f"{save_path}_{safe_title}.png"
            fig.savefig(full_save_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        except Exception as e:
            print(f"Error saving custom plot '{plot_title}': {e}")
            return fig

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
    Plots each observed variable against its corresponding trend component.
    This is a simplified version that assumes 1-to-1 mapping between observed and trend variables.
    """
    if (results.observed_data is None or results.trend_draws is None or
        results.trend_draws.shape[0] == 0): 
        print("Warning: No data available for observed vs trend component plots.")
        return None

    observed_np = results.observed_data 
    trend_draws_np = results.trend_draws 
    
    T_timesteps = observed_np.shape[0]
    n_obs = observed_np.shape[1]
    n_trends_available = trend_draws_np.shape[2]
    
    if T_timesteps == 0: return None

    n_to_plot = min(n_obs, n_trends_available)
    if n_to_plot == 0:
        print("Warning: No matching observed/trend variables to plot.")
        return None

    time_plot = (results.time_index if results.time_index is not None and 
                len(results.time_index) == T_timesteps else np.arange(T_timesteps))

    fig, axes = plt.subplots(n_to_plot, 1, figsize=(12, 4 * n_to_plot), squeeze=False)

    trend_stats = compute_summary_statistics(trend_draws_np) 
    line_data_key = 'median' if use_median_for_trend_line and results.n_draws > 1 else 'mean'
    trend_line_data_all_vars = trend_stats.get(line_data_key)
    
    trend_hdi_lower_all_vars, trend_hdi_upper_all_vars = None, None
    if results.n_draws > 1: 
        trend_hdi_lower_all_vars, trend_hdi_upper_all_vars = compute_hdi_robust(trend_draws_np, results.hdi_prob)

    for i in range(n_to_plot):
        ax = axes[i, 0]
        
        obs_name = (results.observed_variable_names[i] 
                   if i < len(results.observed_variable_names) else f"Obs{i+1}")
        trend_name = (results.trend_names[i] 
                     if i < len(results.trend_names) else f"Trend{i+1}")

        current_trend_line_data = trend_line_data_all_vars[:, i] if trend_line_data_all_vars is not None and trend_line_data_all_vars.ndim == 2 else None
        current_hdi_lower = trend_hdi_lower_all_vars[:, i] if trend_hdi_lower_all_vars is not None and trend_hdi_lower_all_vars.ndim == 2 else None
        current_hdi_upper = trend_hdi_upper_all_vars[:, i] if trend_hdi_upper_all_vars is not None and trend_hdi_upper_all_vars.ndim == 2 else None
        
        if current_hdi_lower is not None and current_hdi_upper is not None:
            if not (np.all(np.isnan(current_hdi_lower)) or np.all(np.isnan(current_hdi_upper))):
                ax.fill_between(time_plot, current_hdi_lower, current_hdi_upper,
                               alpha=0.3, color='blue', 
                               label=f'{trend_name} {int(results.hdi_prob*100)}% HDI')

        if current_trend_line_data is not None and not np.all(np.isnan(current_trend_line_data)):
            line_label_text = f'{trend_name} ({line_data_key.capitalize()})'
            ax.plot(time_plot, current_trend_line_data, 'b-', 
                   linewidth=2, label=line_label_text)

        if not np.all(np.isnan(observed_np[:, i])):
            ax.plot(time_plot, observed_np[:, i], 'ko-', 
                   linewidth=1.5, markersize=2, label=f'Observed {obs_name}', alpha=0.8)

        ax.set_title(f'Observed {obs_name} vs Trend {trend_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        _format_datetime_axis(fig, ax, time_plot)
        
        if show_info_box:
            _add_info_box(ax, results.n_draws, results.hdi_prob)

    plt.tight_layout()
    
    if save_path and fig:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        try:
            fig.savefig(f"{save_path}_observed_vs_trend_components.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        except Exception as e:
            print(f"Error saving observed vs trend components plot: {e}")
            return fig
    return fig


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