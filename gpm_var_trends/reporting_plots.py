# --- START OF FILE reporting_plots.py ---

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, List, Tuple

# Import HDI computation utility
try:
    # Assuming compute_hdi_with_percentiles is in simulation_smoothing.py
    from simulation_smoothing import compute_hdi_with_percentiles
except ImportError:
    print("Warning: Could not import compute_hdi_with_percentiles. HDI plotting may be limited.")
    # Define a dummy function if import fails
    def compute_hdi_with_percentiles(draws, hdi_prob=0.94):
        print("compute_hdi_with_percentiles not available.")
        return None # Indicate failure


def plot_observed_vs_true_decomposition(
    y_np: np.ndarray,               # Observed data (NumPy: T x n_vars)
    trends_true_np: Optional[np.ndarray],       # True trend component (NumPy: T x n_vars)
    stationary_true_np: Optional[np.ndarray],   # True stationary component (NumPy: T x n_vars)
    trend_draws: jnp.ndarray,         # Sampled trend draws (JAX: num_draws x T x n_vars)
    stationary_draws: jnp.ndarray,    # Sampled stationary draws (JAX: num_draws x T x n_vars)
    hdi_prob: float = 0.9,            # Probability for HDI intervals
    variable_names: Optional[List[str]] = None # Optional list of variable names
):
    """
    Plots the estimated (mean/median + HDI) and true trend and stationary
    components for each variable. Plots only estimated if true components are None.
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Convert JAX draws to NumPy for plotting and mean calculation
    trend_draws_np = np.asarray(trend_draws)
    stationary_draws_np = np.asarray(stationary_draws)

    # Compute mean and median of the draws if available
    has_draws = trend_draws_np.shape[0] > 1 # Need at least 2 draws for percentiles
    if has_draws:
        trend_mean_np = np.mean(trend_draws_np, axis=0)
        trend_median_np = np.percentile(trend_draws_np, 50, axis=0)
        stationary_mean_np = np.mean(stationary_draws_np, axis=0)
        stationary_median_np = np.percentile(stationary_draws_np, 50, axis=0)

        # Compute HDI
        trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=hdi_prob)
        stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=hdi_prob)

    else:
        print("Not enough simulation smoother draws to compute median/HDI (need at least 2). Plotting means only.")
        trend_mean_np = np.mean(trend_draws_np, axis=0) if trend_draws_np.shape[0] > 0 else np.full((T, n_vars), np.nan)
        trend_median_np = np.full((T, n_vars), np.nan)
        stationary_mean_np = np.mean(stationary_draws_np, axis=0) if stationary_draws_np.shape[0] > 0 else np.full((T, n_vars), np.nan)
        stationary_median_np = np.full((T, n_vars), np.nan)
        trend_hdi = None
        stationary_hdi = None


    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    elif len(variable_names) != n_vars:
         print(f"Warning: Number of variable names ({len(variable_names)}) does not match number of variables ({n_vars}). Using default names.")
         variable_names = [f'Variable {i+1}' for i in range(n_vars)]

    # Determine if true components are available
    has_true_trends = trends_true_np is not None and trends_true_np.shape == (T, n_vars)
    has_true_stationary = stationary_true_np is not None and stationary_true_np.shape == (T, n_vars)


    # Create subplots: one row per variable, two columns (Trend, Stationary)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3 * n_vars), squeeze=False)

    for i in range(n_vars):
        # --- Trend Plot ---
        ax1 = axes[i, 0]
        if has_true_trends:
            ax1.plot(time_index, trends_true_np[:, i], label='True Trend', color='blue', linestyle='--')

        if has_draws:
             ax1.plot(time_index, trend_median_np[:, i], label='Estimated Trend (median)', color='red')
             # Plot Trend HDI if available
             if trend_hdi is not None and 'low' in trend_hdi and 'high' in trend_hdi:
                 ax1.fill_between(
                     time_index,
                     trend_hdi['low'][:, i],
                     trend_hdi['high'][:, i],
                     color='red',
                     alpha=0.2,
                     label=f'Trend ({int(hdi_prob*100)}% HDI)'
                 )
        else: # Plot mean if no draws
             ax1.plot(time_index, trend_mean_np[:, i], label='Estimated Trend (mean)', color='red')


        ax1.set_title(f'{variable_names[i]} - Trend Component')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- Stationary Plot ---
        ax2 = axes[i, 1]
        if has_true_stationary:
            ax2.plot(time_index, stationary_true_np[:, i], label='True Stationary', color='blue', linestyle='--')

        if has_draws:
             ax2.plot(time_index, stationary_median_np[:, i], label='Estimated Stationary (median)', color='red')
             # Plot Stationary HDI if available
             if stationary_hdi is not None and 'low' in stationary_hdi and 'high' in stationary_hdi:
                 ax2.fill_between(
                     time_index,
                     stationary_hdi['low'][:, i],
                     stationary_hdi['high'][:, i],
                     color='red',
                     alpha=0.2,
                     label=f'Stationary ({int(hdi_prob*100)}% HDI)'
                 )
        else: # Plot mean if no draws
             ax2.plot(time_index, stationary_mean_np[:, i], label='Estimated Stationary (mean)', color='red')

        ax2.set_title(f'{variable_names[i]} - Stationary Component')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_observed_and_trend(
    y_np: np.ndarray,               # Observed data (NumPy: T x n_vars)
    trend_draws: jnp.ndarray,         # Sampled trend draws (JAX: num_draws x T x n_vars)
    hdi_prob: float = 0.9,            # Probability for HDI intervals
    variable_names: Optional[List[str]] = None # Optional list of variable names
):
    """
    Plots the observed data and the estimated trend component (median + HDI) for each variable.
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Convert JAX draws to NumPy for plotting and mean calculation
    trend_draws_np = np.asarray(trend_draws)

    # Compute mean and median of the trend draws if available
    has_draws = trend_draws_np.shape[0] > 1
    if has_draws:
        trend_median_np = np.percentile(trend_draws_np, 50, axis=0)
        # Compute HDI
        trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=hdi_prob)
    else:
        print("Not enough simulation smoother draws to compute median/HDI (need at least 2). Plotting mean only.")
        trend_mean_np = np.mean(trend_draws_np, axis=0) if trend_draws_np.shape[0] > 0 else np.full((T, n_vars), np.nan)
        trend_median_np = np.full((T, n_vars), np.nan) # Plot nan if no draws
        trend_hdi = None


    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    elif len(variable_names) != n_vars:
         print(f"Warning: Number of variable names ({len(variable_names)}) does not match number of variables ({n_vars}). Using default names.")
         variable_names = [f'Variable {i+1}' for i in range(n_vars)]


    # Create subplots: one row per variable
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), squeeze=False)

    for i in range(n_vars):
        ax = axes[i, 0]
        ax.plot(time_index, y_np[:, i], label='Observed Data', color='black', alpha=0.7)

        if has_draws:
             ax.plot(time_index, trend_median_np[:, i], label='Estimated Trend (median)', color='red')
             # Plot Trend HDI if available
             if trend_hdi is not None and 'low' in trend_hdi and 'high' in trend_hdi:
                 ax.fill_between(
                     time_index,
                     trend_hdi['low'][:, i],
                     trend_hdi['high'][:, i],
                     color='red',
                     alpha=0.2,
                     label=f'Trend ({int(hdi_prob*100)}% HDI)'
                 )
        else:
             ax.plot(time_index, trend_mean_np[:, i], label='Estimated Trend (mean)', color='red') # Plot mean if no draws


        ax.set_title(f'{variable_names[i]} - Observed Data and Trend')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_observed_and_fitted(
    y_np: np.ndarray,               # Observed data (NumPy: T x n_vars)
    trend_draws: jnp.ndarray,         # Sampled trend draws (JAX: num_draws x T x n_vars)
    stationary_draws: jnp.ndarray,    # Sampled stationary draws (JAX: num_draws x T x n_vars)
    hdi_prob: float = 0.9,            # Probability for HDI intervals
    variable_names: Optional[List[str]] = None # Optional list of variable names
):
    """
    Plots the observed data and the estimated fitted component (trend + stationary, median + HDI)
    for each variable. Requires both trend and stationary draws.
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Check if dimensions match
    if trend_draws.shape[0] != stationary_draws.shape[0] or \
       trend_draws.shape[1] != stationary_draws.shape[1] or \
       trend_draws.shape[2] != stationary_draws.shape[2]:
        print(f"Error: Trend and stationary draws shapes do not match for fitting.")
        print(f"  Trend draws shape: {trend_draws.shape}")
        print(f"  Stationary draws shape: {stationary_draws.shape}")
        return # Cannot compute fitted

    # Compute fitted draws
    fitted_draws = trend_draws + stationary_draws # num_draws x T x n_vars

    # Convert JAX draws to NumPy for plotting and calculation
    fitted_draws_np = np.asarray(fitted_draws)

    # Compute median of the fitted draws if available
    has_draws = fitted_draws_np.shape[0] > 1
    if has_draws:
        fitted_median_np = np.percentile(fitted_draws_np, 50, axis=0)
        # Compute HDI
        fitted_hdi = compute_hdi_with_percentiles(fitted_draws, hdi_prob=hdi_prob)
    else:
        print("Not enough simulation smoother draws to compute median/HDI for fitted (need at least 2). Plotting mean only.")
        fitted_mean_np = np.mean(fitted_draws_np, axis=0) if fitted_draws_np.shape[0] > 0 else np.full((T, n_vars), np.nan)
        fitted_median_np = np.full((T, n_vars), np.nan) # Plot nan if no draws
        fitted_hdi = None


    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    elif len(variable_names) != n_vars:
         print(f"Warning: Number of variable names ({len(variable_names)}) does not match number of variables ({n_vars}). Using default names.")
         variable_names = [f'Variable {i+1}' for i in range(n_vars)]


    # Create subplots: one row per variable
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), squeeze=False)

    for i in range(n_vars):
        ax = axes[i, 0]
        ax.plot(time_index, y_np[:, i], label='Observed Data', color='black', alpha=0.7)

        if has_draws:
             ax.plot(time_index, fitted_median_np[:, i], label='Estimated Fitted (median)', color='green')
             # Plot Fitted HDI if available
             if fitted_hdi is not None and 'low' in fitted_hdi and 'high' in fitted_hdi:
                 ax.fill_between(
                     time_index,
                     fitted_hdi['low'][:, i],
                     fitted_hdi['high'][:, i],
                     color='green',
                     alpha=0.2,
                     label=f'Fitted ({int(hdi_prob*100)}% HDI)'
                 )
        else:
             ax.plot(time_index, fitted_mean_np[:, i], label='Estimated Fitted (mean)', color='green') # Plot mean if no draws


        ax.set_title(f'{variable_names[i]} - Observed Data and Fitted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_estimated_components(
    trend_draws: jnp.ndarray,         # Sampled trend draws (JAX: num_draws x T x n_trends)
    stationary_draws: jnp.ndarray,    # Sampled stationary draws (JAX: num_draws x T x n_stationary)
    hdi_prob: float = 0.9,            # Probability for HDI intervals
    trend_variable_names: Optional[List[str]] = None, # Optional list of trend variable names
    stationary_variable_names: Optional[List[str]] = None # Optional list of stationary variable names
):
    """
    Plots the estimated trend and stationary components (median + HDI) in separate subplots.
    """
    num_draws_t, T, n_trends = trend_draws.shape
    num_draws_s, T_s, n_stationary = stationary_draws.shape

    if T != T_s:
        print(f"Error: Time dimensions of trend ({T}) and stationary ({T_s}) draws do not match.")
        return

    time_index = np.arange(T)

    # Convert JAX draws to NumPy for plotting and calculation
    trend_draws_np = np.asarray(trend_draws)
    stationary_draws_np = np.asarray(stationary_draws)

    # Compute median and HDI for trends
    has_trend_draws = num_draws_t > 1
    if has_trend_draws:
        trend_median_np = np.percentile(trend_draws_np, 50, axis=0)
        trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=hdi_prob)
    else:
        print("Not enough trend draws to compute median/HDI (need at least 2). Plotting mean only.")
        trend_mean_np = np.mean(trend_draws_np, axis=0) if num_draws_t > 0 else np.full((T, n_trends), np.nan)
        trend_median_np = np.full((T, n_trends), np.nan)
        trend_hdi = None

    # Compute median and HDI for stationary components
    has_stationary_draws = num_draws_s > 1
    if has_stationary_draws:
        stationary_median_np = np.percentile(stationary_draws_np, 50, axis=0)
        stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=hdi_prob)
    else:
        print("Not enough stationary draws to compute median/HDI (need at least 2). Plotting mean only.")
        stationary_mean_np = np.mean(stationary_draws_np, axis=0) if num_draws_s > 0 else np.full((T, n_stationary), np.nan)
        stationary_median_np = np.full((T, n_stationary), np.nan)
        stationary_hdi = None


    if trend_variable_names is None:
        trend_variable_names = [f'Trend Var {i+1}' for i in range(n_trends)]
    elif len(trend_variable_names) != n_trends:
        print(f"Warning: Number of trend variable names ({len(trend_variable_names)}) does not match number of trend variables ({n_trends}). Using default names.")
        trend_variable_names = [f'Trend Var {i+1}' for i in range(n_trends)]

    if stationary_variable_names is None:
        stationary_variable_names = [f'Stationary Var {i+1}' for i in range(n_stationary)]
    elif len(stationary_variable_names) != n_stationary:
        print(f"Warning: Number of stationary variable names ({len(stationary_variable_names)}) does not match number of stationary variables ({n_stationary}). Using default names.")
        stationary_variable_names = [f'Stationary Var {i+1}' for i in range(n_stationary)]


    # Create subplots: one row per variable, two columns (Trend, Stationary)
    # Total variables to plot = n_trends + n_stationary
    num_plots = n_trends + n_stationary
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), squeeze=False)


    # Plot Trend Components
    for i in range(n_trends):
        ax = axes[i, 0]
        if has_trend_draws:
             ax.plot(time_index, trend_median_np[:, i], label='Estimated Trend (median)', color='red')
             if trend_hdi is not None and 'low' in trend_hdi and 'high' in trend_hdi:
                 ax.fill_between(
                     time_index,
                     trend_hdi['low'][:, i],
                     trend_hdi['high'][:, i],
                     color='red',
                     alpha=0.2,
                     label=f'Trend ({int(hdi_prob*100)}% HDI)'
                 )
        else:
             ax.plot(time_index, trend_mean_np[:, i], label='Estimated Trend (mean)', color='red')

        ax.set_title(f'{trend_variable_names[i]} - Estimated Trend Component')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)


    # Plot Stationary Components
    for i in range(n_stationary):
        ax = axes[n_trends + i, 0] # Plot in the rows after trends

        # Add zero line for stationary components
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        if has_stationary_draws:
            ax.plot(time_index, stationary_median_np[:, i], label='Estimated Stationary (median)', color='blue')
            if stationary_hdi is not None and 'low' in stationary_hdi and 'high' in stationary_hdi:
                ax.fill_between(
                    time_index,
                    stationary_hdi['low'][:, i],
                    stationary_hdi['high'][:, i],
                    color='blue', # Using blue for stationary in this plot
                    alpha=0.2,
                    label=f'Stationary ({int(hdi_prob*100)}% HDI)'
                )
        else:
             ax.plot(time_index, stationary_mean_np[:, i], label='Estimated Stationary (mean)', color='blue')


        ax.set_title(f'{stationary_variable_names[i]} - Estimated Stationary Component')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)


    plt.tight_layout()
    plt.show()


# --- End of new functions ---

# Rename the original plot_decomposition_results for clarity
#plot_estimated_vs_true_decomposition = plot_decomposition_results
# ... (the original plot_decomposition_results code follows the header) ...
# Ensure the original code is still here below this comment block


# The original plot_decomposition_results is renamed above.
# Copy the original definition here if it was removed.
# Keeping the original body below allows external code that still
# references plot_decomposition_results to potentially still work,
# though using the new name plot_estimated_vs_true_decomposition is preferred.
# def plot_decomposition_results(...):
#    ... (original code) ...
# This is commented out assuming you just added the new functions.
# If you are replacing the file content, make sure the body of
# plot_decomposition_results is copied here under the new name.


# The body of plot_observed_and_trend is already above.


# --- END OF FILE reporting_plots.py ---