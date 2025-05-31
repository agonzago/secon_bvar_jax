# fixed_reporting_plots.py - Robust plotting for simulation smoother results

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, List, Union, Tuple, Dict, Any
import arviz as az

# Import ReducedExpression from gpm_model_parser if it's not already implicitly available
# from gpm_model_parser import ReducedExpression # Add this if type hinting is strict and file is separate

def compute_hdi_robust(draws: Union[jnp.ndarray, np.ndarray], 
                      hdi_prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HDI using ArviZ with robust error handling.
    
    Args:
        draws: Array of shape (n_draws, T, n_variables) or (n_draws, T)
        hdi_prob: HDI probability level
        
    Returns:
        Tuple of (hdi_lower, hdi_upper) arrays with shape (T, n_variables) or (T,)
    """
    if hasattr(draws, 'shape') and hasattr(draws, '__array__'):
        draws_np = np.asarray(draws)
    else:
        draws_np = draws
    
    if draws_np.ndim == 0 or draws_np.shape[0] < 2:
        dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
    
    if np.all(np.isnan(draws_np)):
        dummy_shape = draws_np.shape[1:]
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))
    
    try:
        lower_percentile = (1 - hdi_prob) / 2 * 100
        upper_percentile = (1 + hdi_prob) / 2 * 100
        
        hdi_lower = np.percentile(draws_np, lower_percentile, axis=0)
        hdi_upper = np.percentile(draws_np, upper_percentile, axis=0)
        
        return hdi_lower, hdi_upper
        
    except Exception as e:
        print(f"HDI computation failed: {e}")
        dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
        return (np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan))


def compute_summary_statistics(draws: Union[jnp.ndarray, np.ndarray]) -> dict:
    """
    Compute mean, median, mode (approximated) and other statistics.
    
    Args:
        draws: Array of shape (n_draws, T, n_variables) or (n_draws, T)
        
    Returns:
        Dictionary with summary statistics
    """
    draws_np = np.asarray(draws)
    
    if draws_np.ndim == 0 or draws_np.shape[0] < 1:
        dummy_shape = draws_np.shape[1:] if draws_np.ndim > 0 else ()
        return {
            'mean': np.full(dummy_shape, np.nan),
            'median': np.full(dummy_shape, np.nan),
            'mode': np.full(dummy_shape, np.nan), # Mode is approximated by median
            'std': np.full(dummy_shape, np.nan)
        }
    
    mean_vals = np.nanmean(draws_np, axis=0)
    median_vals = np.nanmedian(draws_np, axis=0)
    std_vals = np.nanstd(draws_np, axis=0)
    mode_vals = median_vals # Approximate mode with median
    
    return {
        'mean': mean_vals,
        'median': median_vals, 
        'mode': mode_vals,
        'std': std_vals
    }


def plot_time_series_with_uncertainty(
    draws: Union[jnp.ndarray, np.ndarray],
    variable_names: Optional[List[str]] = None,
    hdi_prob: float = 0.9,
    title_prefix: str = "Time Series",
    figsize_per_var: Tuple[float, float] = (12, 4),
    show_mean: bool = True,
    show_median: bool = True,
    show_mode: bool = False, # Mode usually less useful for time series draws
    show_hdi: bool = True,
    alpha_fill: float = 0.3,
    time_index: Optional[Any] = None, 
    show_info_box: bool = True 
) -> plt.Figure:
    """
    Plot time series with uncertainty bands (HDI) for multiple variables.
    """
    draws_np = np.asarray(draws)
    
    if draws_np.ndim == 2: # (n_draws, T) -> (n_draws, T, 1)
        draws_np = draws_np[:, :, np.newaxis]
    
    if draws_np.ndim != 3 or draws_np.shape[0] == 0 or draws_np.shape[1] == 0:
        print(f"Warning: Invalid shape for draws in plot_time_series_with_uncertainty: {draws_np.shape}. Skipping plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid data for plotting", ha='center', va='center')
        return fig

    n_draws, T, n_variables = draws_np.shape
    
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_variables)]
    elif len(variable_names) != n_variables:
        variable_names = [f'Var {i+1}' for i in range(n_variables)] # Fallback
    
    if time_index is None:
        time_index_plot = np.arange(T)
    else:
        time_index_plot = time_index if len(time_index) == T else np.arange(T)
    
    stats = compute_summary_statistics(draws_np)
    hdi_lower, hdi_upper = None, None
    if show_hdi and n_draws > 1 :
        hdi_lower, hdi_upper = compute_hdi_robust(draws_np, hdi_prob)
    
    fig, axes = plt.subplots(n_variables, 1, 
                            figsize=(figsize_per_var[0], figsize_per_var[1] * n_variables),
                            squeeze=False)
    
    for i in range(n_variables):
        ax = axes[i, 0]
        
        current_hdi_lower = hdi_lower[:, i] if hdi_lower is not None and hdi_lower.ndim == 2 and i < hdi_lower.shape[1] else (hdi_lower if hdi_lower is not None and hdi_lower.ndim == 1 and n_variables == 1 else None)
        current_hdi_upper = hdi_upper[:, i] if hdi_upper is not None and hdi_upper.ndim == 2 and i < hdi_upper.shape[1] else (hdi_upper if hdi_upper is not None and hdi_upper.ndim == 1 and n_variables == 1 else None)

        if show_hdi and current_hdi_lower is not None and current_hdi_upper is not None:
            if len(current_hdi_lower) == T and not (np.all(np.isnan(current_hdi_lower)) or np.all(np.isnan(current_hdi_upper))):
                ax.fill_between(time_index_plot, current_hdi_lower, current_hdi_upper, 
                               alpha=alpha_fill, color='lightblue', label=f'{int(hdi_prob*100)}% HDI')
        
        mean_line = stats['mean'][:, i] if stats['mean'].ndim == 2 and i < stats['mean'].shape[1] else (stats['mean'] if stats['mean'].ndim == 1 and n_variables == 1 else None)
        if show_mean and mean_line is not None:
            if len(mean_line) == T and not np.all(np.isnan(mean_line)):
                ax.plot(time_index_plot, mean_line, 'b-', linewidth=2, label='Mean')
        
        median_line = stats['median'][:, i] if stats['median'].ndim == 2 and i < stats['median'].shape[1] else (stats['median'] if stats['median'].ndim == 1 and n_variables == 1 else None)
        if show_median and median_line is not None:
             if len(median_line) == T and not np.all(np.isnan(median_line)):
                ax.plot(time_index_plot, median_line, 'r-', linewidth=2, label='Median')
        
        ax.set_title(f'{title_prefix}: {variable_names[i]}')
        ax.set_xlabel('Time'); ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3); ax.legend()
        
        if show_info_box:
            n_finite_draws_at_t0 = np.sum(~np.isnan(draws_np[:, 0, i])) if T > 0 else 0
            info_text = f'Draws: {n_draws}\nValid (t=0): {n_finite_draws_at_t0}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_smoother_results(
    trend_draws: Union[jnp.ndarray, np.ndarray],
    stationary_draws: Union[jnp.ndarray, np.ndarray],
    trend_names: Optional[List[str]] = None,
    stationary_names: Optional[List[str]] = None,
    hdi_prob: float = 0.9,
    save_path: Optional[str] = None,
    time_index: Optional[Any] = None, 
    show_info_box: bool = False 
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Main function to plot simulation smoother results.
    """
    trend_fig = plot_time_series_with_uncertainty(
        np.asarray(trend_draws), variable_names=trend_names, hdi_prob=hdi_prob,
        title_prefix="Trend Component", show_mean=True, show_median=False, show_hdi=True,
        time_index=time_index, show_info_box=show_info_box
    )
    
    stationary_fig = plot_time_series_with_uncertainty(
        np.asarray(stationary_draws), variable_names=stationary_names, hdi_prob=hdi_prob,
        title_prefix="Stationary Component", show_mean=True, show_median=False, show_hdi=True,
        time_index=time_index, show_info_box=show_info_box
    )
    
    if save_path:
        trend_fig.savefig(f"{save_path}_trends.png", dpi=300, bbox_inches='tight')
        stationary_fig.savefig(f"{save_path}_stationary.png", dpi=300, bbox_inches='tight')
    
    return trend_fig, stationary_fig


def plot_observed_vs_fitted(
    observed_data: Union[jnp.ndarray, np.ndarray],
    trend_draws: Union[jnp.ndarray, np.ndarray],        # (N_draws, T, N_trends)
    stationary_draws: Union[jnp.ndarray, np.ndarray],   # (N_draws, T, N_stat)
    variable_names: Optional[List[str]] = None,         # Names for observed_data columns
    trend_names: Optional[List[str]] = None,            # Names for 3rd dim of trend_draws
    stationary_names: Optional[List[str]] = None,       # Names for 3rd dim of stationary_draws
    reduced_measurement_equations: Optional[Dict[str, Any]] = None, # Parsed GPM ME definitions
    hdi_prob: float = 0.9,
    save_path: Optional[str] = None,
    time_index: Optional[Any] = None,
    show_info_box: bool = False
) -> plt.Figure:
    observed_np = np.asarray(observed_data)
    trend_draws_np = np.asarray(trend_draws)
    stationary_draws_np = np.asarray(stationary_draws)
    
    T, n_obs_to_plot = observed_np.shape
    
    # Determine number of draws from components, ensure consistency or handle absence
    n_plot_draws = 0
    if trend_draws_np.ndim == 3 and trend_draws_np.shape[0] > 0:
        n_plot_draws = trend_draws_np.shape[0]
    elif stationary_draws_np.ndim == 3 and stationary_draws_np.shape[0] > 0:
        n_plot_draws = stationary_draws_np.shape[0]

    if n_plot_draws == 0:
        print("Warning: No valid trend or stationary draws provided for observed vs. fitted plot. Fitted series will be empty.")

    # Default names if not provided
    if variable_names is None or len(variable_names) != n_obs_to_plot:
        variable_names = [f'Obs Var {i+1}' for i in range(n_obs_to_plot)]
    
    # Ensure trend_names and stationary_names are lists, even if empty
    # These correspond to the names of the 3rd dimension of trend_draws_np / stationary_draws_np
    if trend_names is None:
        trend_names = [f'TrendComp {i+1}' for i in range(trend_draws_np.shape[2])] if trend_draws_np.ndim == 3 else []
    if stationary_names is None:
        stationary_names = [f'StatComp {i+1}' for i in range(stationary_draws_np.shape[2])] if stationary_draws_np.ndim == 3 else []

    # --- REFACTORED FITTED_DRAWS CONSTRUCTION ---
    fitted_draws = np.full((n_plot_draws, T, n_obs_to_plot), np.nan) # Initialize with NaNs

    if reduced_measurement_equations and n_plot_draws > 0:
        trend_name_to_idx = {name: i for i, name in enumerate(trend_names)}
        stat_name_to_idx = {name: i for i, name in enumerate(stationary_names)}

        for i_obs_col, obs_col_name in enumerate(variable_names):
            if obs_col_name not in reduced_measurement_equations:
                print(f"Warning: No measurement equation definition found for '{obs_col_name}'. Fitted series will be NaN.")
                continue

            equation_expr = reduced_measurement_equations[obs_col_name]
            current_obs_fitted_sum_draws = np.zeros((n_plot_draws, T))

            # Handle constant part of the equation (assuming it's numeric or zero)
            try:
                const_val = float(equation_expr.constant_str)
                if const_val != 0:
                    current_obs_fitted_sum_draws += const_val
            except ValueError:
                if equation_expr.constant_str != "0":
                    print(f"Warning: Non-numeric constant '{equation_expr.constant_str}' in ME for '{obs_col_name}' ignored for plotting sum.")

            # Sum terms from the equation
            for term_var_name, coeff_str in equation_expr.terms.items():
                component_draws_slice = None
                
                if term_var_name in trend_name_to_idx:
                    idx = trend_name_to_idx[term_var_name]
                    if trend_draws_np.ndim == 3 and idx < trend_draws_np.shape[2]:
                        component_draws_slice = trend_draws_np[:, :, idx]
                elif term_var_name in stat_name_to_idx:
                    idx = stat_name_to_idx[term_var_name]
                    if stationary_draws_np.ndim == 3 and idx < stationary_draws_np.shape[2]:
                        component_draws_slice = stationary_draws_np[:, :, idx]
                else:
                    print(f"Warning: Term '{term_var_name}' in ME for '{obs_col_name}' not found in available trend/stationary component names. Skipping term.")
                    continue
                
                if component_draws_slice is not None:
                    # For plotting, assume coefficients are +1 or -1 if not simple numbers.
                    # GPM example `OBS = TREND_D + CYCLE` has implicit +1.
                    multiplier = 1.0
                    if coeff_str == "-1": # Example: OBS = TREND - CYCLE
                        multiplier = -1.0
                    elif coeff_str not in ["1", "0", None, ""]: # If coeff is something else, try float or warn
                        try:
                            multiplier = float(coeff_str)
                        except ValueError:
                            print(f"Warning: Non-numeric/non-unitary coefficient '{coeff_str}' for '{term_var_name}' in ME of '{obs_col_name}'. Assuming 1 for plotting sum.")
                            multiplier = 1.0
                    
                    if multiplier != 0.0:
                         current_obs_fitted_sum_draws += multiplier * component_draws_slice
            
            fitted_draws[:, :, i_obs_col] = current_obs_fitted_sum_draws
    elif not reduced_measurement_equations:
        print("Warning: `reduced_measurement_equations` not provided. Fitted series will be NaN.")
    # If n_plot_draws is 0, fitted_draws remains as initialized (NaNs or empty if T or n_obs_to_plot is 0)

    time_index_plot = np.arange(T) if time_index is None else time_index
    if len(time_index_plot) != T : time_index_plot = np.arange(T) 

    fitted_stats = compute_summary_statistics(fitted_draws)
    fitted_hdi_lower, fitted_hdi_upper = None, None
    if n_plot_draws > 1:
         fitted_hdi_lower, fitted_hdi_upper = compute_hdi_robust(fitted_draws, hdi_prob)
    
    fig, axes = plt.subplots(n_obs_to_plot, 1, figsize=(12, 4 * n_obs_to_plot), squeeze=False)
    
    for i in range(n_obs_to_plot):
        ax = axes[i, 0]
        
        current_hdi_lower = fitted_hdi_lower[:, i] if fitted_hdi_lower is not None and fitted_hdi_lower.ndim == 2 and i < fitted_hdi_lower.shape[1] else (fitted_hdi_lower if fitted_hdi_lower is not None and fitted_hdi_lower.ndim == 1 and n_obs_to_plot == 1 else None)
        current_hdi_upper = fitted_hdi_upper[:, i] if fitted_hdi_upper is not None and fitted_hdi_upper.ndim == 2 and i < fitted_hdi_upper.shape[1] else (fitted_hdi_upper if fitted_hdi_upper is not None and fitted_hdi_upper.ndim == 1 and n_obs_to_plot == 1 else None)
        
        if current_hdi_lower is not None and current_hdi_upper is not None:
            if len(current_hdi_lower) == T and not (np.all(np.isnan(current_hdi_lower)) or np.all(np.isnan(current_hdi_upper))):
                 ax.fill_between(time_index_plot, current_hdi_lower, current_hdi_upper, alpha=0.3, color='green', label=f'Fitted {int(hdi_prob*100)}% HDI')
        
        fitted_mean_line = fitted_stats['mean'][:, i] if fitted_stats['mean'].ndim == 2 and i < fitted_stats['mean'].shape[1] else (fitted_stats['mean'] if fitted_stats['mean'].ndim == 1 and n_obs_to_plot == 1 else None)
        if fitted_mean_line is not None :
            if len(fitted_mean_line) == T and not np.all(np.isnan(fitted_mean_line)):
                ax.plot(time_index_plot, fitted_mean_line, 'g-', linewidth=2, label='Fitted Mean')
        
        ax.plot(time_index_plot, observed_np[:, i], 'ko-', linewidth=1.5, markersize=2, label='Observed Data', alpha=0.8)
        
        ax.set_title(f'Observed vs Fitted: {variable_names[i]}')
        ax.set_xlabel('Time'); ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3); ax.legend()
        
        if show_info_box:
            rmse_val = np.nan
            if fitted_mean_line is not None and len(fitted_mean_line) == T:
                rmse_val = np.sqrt(np.nanmean((observed_np[:, i] - fitted_mean_line)**2))
            info_text_lines = [f'RMSE: {rmse_val:.3f}' if not np.isnan(rmse_val) else 'RMSE: N/A']
            info_text_lines.append(f'Draws: {n_plot_draws}')
            
            # Display terms from the measurement equation used for this obs_name
            if reduced_measurement_equations and variable_names[i] in reduced_measurement_equations:
                eq_terms = reduced_measurement_equations[variable_names[i]].terms
                terms_str = ", ".join(f"{c}*{v}" if c not in ["1", "0", None] else v for v,c in eq_terms.items())
                if not terms_str and reduced_measurement_equations[variable_names[i]].constant_str != "0":
                    terms_str = reduced_measurement_equations[variable_names[i]].constant_str
                elif terms_str and reduced_measurement_equations[variable_names[i]].constant_str != "0":
                    terms_str += f" + {reduced_measurement_equations[variable_names[i]].constant_str}"

                info_text_lines.append(f'Fit from: {terms_str if terms_str else "(No terms found)"}')

            ax.text(0.02, 0.98, "\n".join(info_text_lines), transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=8) # Smaller font for info box
    
    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}_observed_vs_fitted.png", dpi=300, bbox_inches='tight')
    return fig

def plot_custom_series_comparison(
    plot_title: str,
    series_specs: List[Dict],
    observed_data: Optional[np.ndarray] = None, # This is the input, ensure it's used
    trend_draws: Optional[np.ndarray] = None,
    stationary_draws: Optional[np.ndarray] = None,
    observed_names: Optional[List[str]] = None,
    trend_names: Optional[List[str]] = None,
    stationary_names: Optional[List[str]] = None,
    time_index: Optional[Any] = None,
    hdi_prob: float = 0.9,
    ax: Optional[plt.Axes] = None,
    default_fig_size: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Plots a custom combination of observed data, trend components, stationary components,
    or their sums on a single axes.
    """

    if ax is None:
        fig, ax_new = plt.subplots(figsize=default_fig_size)
    else:
        ax_new = ax
        fig = ax_new.get_figure()

    T = 0
    # Ensure observed_data is a NumPy array for consistent processing
    observed_np_internal = np.asarray(observed_data) if observed_data is not None else None

    if observed_np_internal is not None and observed_np_internal.ndim > 0: T = observed_np_internal.shape[0]
    elif trend_draws is not None and trend_draws.ndim > 1: T = trend_draws.shape[1]
    elif stationary_draws is not None and stationary_draws.ndim > 1: T = stationary_draws.shape[1]


    if T == 0 and not series_specs:
        ax_new.set_title(f"{plot_title} (No data or specs)")
        return fig
    
    time_idx_plot = np.arange(T) if time_index is None else time_index
    if len(time_idx_plot) != T:
        time_idx_plot = np.arange(T)

    for spec in series_specs:
        series_type = spec.get('type')
        series_name = spec.get('name') 
        label = spec.get('label', series_name if series_name else series_type)
        raw_style = spec.get('style', '-')
        user_color = spec.get('color')

        final_style = raw_style; final_color = user_color
        if isinstance(raw_style, str) and len(raw_style) > 1 and raw_style[0].isalpha() and raw_style[1] in ['-', '--', '-.', ':']:
            if user_color is None: final_color = raw_style[0]
            final_style = raw_style[1:]
        elif isinstance(raw_style, str) and raw_style[0].isalpha() and user_color is None:
            final_color = raw_style; final_style = '-'

        show_hdi = spec.get('show_hdi', False if series_type == 'observed' else True)
        hdi_alpha = spec.get('hdi_alpha', 0.2)
        
        data_to_plot_mean = None
        hdi_lower_plot, hdi_upper_plot = None, None

        current_series_draws_for_stats = None 

        if series_type == 'observed':
            if observed_np_internal is not None and observed_names: # Use observed_np_internal
                try:
                    idx = observed_names.index(series_name)
                    # FIX IS HERE: use observed_np_internal
                    data_to_plot_mean = observed_np_internal[:, idx] if observed_np_internal.ndim == 2 and idx < observed_np_internal.shape[1] else None
                except ValueError: 
                    print(f"Warning in '{plot_title}': Observed series '{series_name}' not found in observed_names.")
                    continue 
            else:
                print(f"Warning in '{plot_title}': observed_data or observed_names not provided for type 'observed'.")
                continue
        
        elif series_type in ['trend', 'stationary', 'combined']:
            # ... (rest of the logic for trend, stationary, combined) ...
            # This part seemed okay, the error was specific to 'observed'
            temp_draws_list_for_combination = [] 

            if series_type == 'trend':
                if trend_draws is not None and trend_names and trend_draws.ndim ==3:
                    try: 
                        idx = trend_names.index(series_name)
                        if idx < trend_draws.shape[2]: current_series_draws_for_stats = trend_draws[:, :, idx]
                    except ValueError: 
                        print(f"Warning in '{plot_title}': Trend series '{series_name}' not found in trend_names.")
                        continue
                else: 
                    print(f"Warning in '{plot_title}': trend_draws or trend_names not properly provided for type 'trend'.")
                    continue
            
            elif series_type == 'stationary':
                if stationary_draws is not None and stationary_names and stationary_draws.ndim == 3:
                    try: 
                        idx = stationary_names.index(series_name)
                        if idx < stationary_draws.shape[2]: current_series_draws_for_stats = stationary_draws[:, :, idx]
                    except ValueError: 
                        print(f"Warning in '{plot_title}': Stationary series '{series_name}' not found in stationary_names.")
                        continue
                else: 
                    print(f"Warning in '{plot_title}': stationary_draws or stationary_names not properly provided for type 'stationary'.")
                    continue

            elif series_type == 'combined':
                component_specs = spec.get('components', [])
                if not component_specs: 
                    print(f"Warning in '{plot_title}': 'components' not specified for 'combined' type '{series_name}'.")
                    continue
# ... (rest of the file, example_usage, etc.)
                for comp_spec in component_specs:
                    comp_type, comp_name = comp_spec.get('type'), comp_spec.get('name')
                    found_comp_draws_slice = None
                    if comp_type == 'trend' and trend_draws is not None and trend_names and trend_draws.ndim == 3:
                        try: 
                            idx = trend_names.index(comp_name)
                            if idx < trend_draws.shape[2]: found_comp_draws_slice = trend_draws[:, :, idx]
                        except ValueError: print(f"Warning in '{plot_title}' (combined for '{series_name}'): Trend component '{comp_name}' not found.")
                    elif comp_type == 'stationary' and stationary_draws is not None and stationary_names and stationary_draws.ndim == 3:
                        try: 
                            idx = stationary_names.index(comp_name)
                            if idx < stationary_draws.shape[2]: found_comp_draws_slice = stationary_draws[:, :, idx]
                        except ValueError: print(f"Warning in '{plot_title}' (combined for '{series_name}'): Stationary component '{comp_name}' not found.")
                    
                    if found_comp_draws_slice is not None:
                        temp_draws_list_for_combination.append(found_comp_draws_slice)
                
                if temp_draws_list_for_combination:
                    current_series_draws_for_stats = np.sum(np.stack(temp_draws_list_for_combination, axis=-1), axis=-1)
                else: 
                    print(f"Warning in '{plot_title}': No valid components found for 'combined' series '{series_name}'.")
                    continue
            
            if current_series_draws_for_stats is not None and current_series_draws_for_stats.shape[0] > 0:
                stats = compute_summary_statistics(current_series_draws_for_stats)
                data_to_plot_mean = stats['median'] # Use median for drawn components as it's robust to outliers
                if show_hdi and current_series_draws_for_stats.shape[0] > 1: # Need >1 draw for HDI
                    hdi_lower_plot, hdi_upper_plot = compute_hdi_robust(current_series_draws_for_stats, hdi_prob)
            else: 
                # This path can be reached if, for example, a trend or stationary component was named but had no draws.
                print(f"Warning in '{plot_title}': No draws available for component/combined series '{series_name}'.")
                continue

        # Plotting the series
        if data_to_plot_mean is not None and len(data_to_plot_mean) == T:
            plot_args = {'label': label, 'linestyle': final_style}
            if final_color: plot_args['color'] = final_color
            line_artist, = ax_new.plot(time_idx_plot, data_to_plot_mean, **plot_args)

            if show_hdi and hdi_lower_plot is not None and hdi_upper_plot is not None:
                if len(hdi_lower_plot) == T and not (np.all(np.isnan(hdi_lower_plot)) or np.all(np.isnan(hdi_upper_plot))):
                    fill_args = {'alpha': hdi_alpha}
                    # Determine fill color
                    fill_color_to_use = final_color if final_color else line_artist.get_color()
                    try:
                        import matplotlib.colors as mcolors
                        rgb_color = mcolors.to_rgb(fill_color_to_use)
                        light_rgb_color = tuple([(1 - 0.6) * c + 0.6 * 1.0 for c in rgb_color]) # 60% white mix
                        fill_args['color'] = light_rgb_color
                    except ValueError: # Fallback if color name is not recognized by mcolors
                        fill_args['color'] = fill_color_to_use 
                    ax_new.fill_between(time_idx_plot, hdi_lower_plot, hdi_upper_plot, **fill_args)
        elif data_to_plot_mean is None :
            # This means the series could not be constructed (e.g., name not found, no draws)
            # Warnings should have been printed above.
            pass 
        else : # data_to_plot_mean is not None BUT length mismatch
            print(f"Warning in '{plot_title}': Data length mismatch for '{label}'. Expected T={T}, got {len(data_to_plot_mean)}. Skipping this series.")

    ax_new.set_title(plot_title)
    ax_new.set_xlabel("Time"); ax_new.set_ylabel("Value")
    ax_new.grid(True, alpha=0.3)
    if any(s.get('label') for s in series_specs): ax_new.legend()
    
    if ax is None: plt.tight_layout()
    return fig



def debug_and_plot_simple(
    draws: Union[jnp.ndarray, np.ndarray],
    title: str = "Debug Plot"
) -> plt.Figure:
    """
    Simple debug plotting function to test with your exact data structure.
    """
    print(f"\n=== DEBUG PLOTTING: {title} ===")
    
    # Convert to numpy
    draws_np = np.asarray(draws)
    print(f"Input shape: {draws_np.shape}")
    print(f"Input dtype: {draws_np.dtype}")
    
    if draws_np.ndim == 2:
        draws_np = draws_np[:, :, np.newaxis]
        print(f"Expanded to 3D: {draws_np.shape}")
    
    n_draws, T, n_variables = draws_np.shape
    print(f"Dimensions: n_draws={n_draws}, T={T}, n_variables={n_variables}")
    
    # Create time index
    time_index = np.arange(T)
    print(f"Time index length: {len(time_index)}")
    
    # Compute simple statistics
    mean_vals = np.mean(draws_np, axis=0)  # Shape: (T, n_variables)
    print(f"Mean shape: {mean_vals.shape}")
    
    # Compute percentiles instead of HDI for debugging
    lower_vals = np.percentile(draws_np, 5, axis=0)  # Shape: (T, n_variables)
    upper_vals = np.percentile(draws_np, 95, axis=0)  # Shape: (T, n_variables)
    print(f"Percentile bounds shape: lower={lower_vals.shape}, upper={upper_vals.shape}")
    
    # Create figure
    fig, axes = plt.subplots(n_variables, 1, figsize=(10, 4 * n_variables), squeeze=False)
    
    for i in range(n_variables):
        ax = axes[i, 0]
        
        # Extract data for this variable
        if mean_vals.ndim == 2:
            mean_line = mean_vals[:, i]
            lower_line = lower_vals[:, i]
            upper_line = upper_vals[:, i]
        else:
            mean_line = mean_vals
            lower_line = lower_vals
            upper_line = upper_vals
        
        print(f"Variable {i}: mean_line length={len(mean_line)}, time_index length={len(time_index)}")
        
        # Plot
        ax.fill_between(time_index, lower_line, upper_line, alpha=0.3, color='lightblue', label='90% Range')
        ax.plot(time_index, mean_line, 'b-', linewidth=2, label='Mean')
        
        ax.set_title(f'{title} - Variable {i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"=== END DEBUG: {title} ===\n")
    return fig


# Example usage function
def example_usage():
    """Example of how to use the plotting functions."""
    
    # Simulate some example data
    np.random.seed(42)
    n_draws, T, n_trend_vars = 100, 50, 2
    n_stat_vars = 2
    
    # Generate fake draws
    trend_draws = np.random.randn(n_draws, T, n_trend_vars).cumsum(axis=1) * 0.1
    stationary_draws = np.random.randn(n_draws, T, n_stat_vars) * 0.5
    
    # Add some trend
    for i in range(n_trend_vars):
        trend_draws[:, :, i] += np.linspace(0, 2, T)
    
    # Generate observed data
    observed_data = np.mean(trend_draws + stationary_draws, axis=0) + np.random.randn(T, n_trend_vars) * 0.1
    
    print("Testing with debug function first...")
    
    # Test with debug function first
    debug_and_plot_simple(trend_draws, "Trend Components Debug")
    debug_and_plot_simple(stationary_draws, "Stationary Components Debug")
    
    print("Now testing main plotting functions...")
    
    # Plot results
    trend_names = ['GDP Trend', 'Inflation Trend']
    stat_names = ['GDP Cycle', 'Inflation Cycle']
    
    # Plot smoother results
    trend_fig, stat_fig = plot_smoother_results(
        trend_draws, stationary_draws,
        trend_names=trend_names,
        stationary_names=stat_names,
        hdi_prob=0.9
    )
    
    # Plot observed vs fitted
    fitted_fig = plot_observed_vs_fitted(
        observed_data, trend_draws, stationary_draws,
        variable_names=trend_names,
        hdi_prob=0.9
    )
    
    return trend_fig, stat_fig, fitted_fig


if __name__ == "__main__":
    # Run example
    example_usage()
