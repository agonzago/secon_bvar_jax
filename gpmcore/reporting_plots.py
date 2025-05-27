# Add these imports at the beginning of your script:
import matplotlib.pyplot as plt
import numpy as np # Ensure numpy is imported as np

import jax.numpy as jnp
from typing import Optional, Dict, List

def plot_decomposition_results(
    y_np: np.ndarray,               # Observed data (NumPy: T x n_vars)
    trends_true_np: np.ndarray,       # True trend component (NumPy: T x n_vars)
    stationary_true_np: np.ndarray,   # True stationary component (NumPy: T x n_vars)
    trend_draws: jnp.ndarray,         # Sampled trend draws (JAX: num_draws x T x n_vars)
    stationary_draws: jnp.ndarray,    # Sampled stationary draws (JAX: num_draws x T x n_vars)
    trend_hdi: Optional[Dict[str, np.ndarray]], # Trend HDI (dict of NumPy: {'low': T x n_vars, 'high': T x n_vars})
    stationary_hdi: Optional[Dict[str, np.ndarray]], # Stationary HDI (dict of NumPy: {'low': T x n_vars, 'high': T x n_vars})
    variable_names: Optional[List[str]] = None # Optional list of variable names
):
    """
    Plots the estimated and true trend and stationary components for each variable, including HDIs.
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Convert JAX draws to NumPy for plotting and mean calculation
    trend_draws_np = np.asarray(trend_draws)
    stationary_draws_np = np.asarray(stationary_draws)

    # Compute mean of the draws if available
    if trend_draws_np.shape[0] > 0:
        trend_mean_np = np.mean(trend_draws_np, axis=0) # Shape (T, n_vars)
        stationary_mean_np = np.mean(stationary_draws_np, axis=0) # Shape (T, n_vars)
    else:
        # Return NaN means if no draws
        trend_mean_np = np.full_like(trends_true_np, np.nan)
        stationary_mean_np = np.full_like(stationary_true_np, np.nan)


    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    elif len(variable_names) != n_vars:
         print(f"Warning: Number of variable names ({len(variable_names)}) does not match number of variables ({n_vars}). Using default names.")
         variable_names = [f'Variable {i+1}' for i in range(n_vars)]


    # Create subplots: one row per variable, two columns (Trend, Stationary)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3 * n_vars), squeeze=False)

    for i in range(n_vars):
        # --- Trend Plot ---
        ax1 = axes[i, 0]
        ax1.plot(time_index, trends_true_np[:, i], label='True Trend', color='blue', linestyle='--')
        ax1.plot(time_index, trend_mean_np[:, i], label='Estimated Mean Trend', color='red')

        # Plot Trend HDI if available
        if trend_hdi is not None and 'low' in trend_hdi and 'high' in trend_hdi:
            ax1.fill_between(
                time_index,
                trend_hdi['low'][:, i],
                trend_hdi['high'][:, i],
                color='red',
                alpha=0.2,
                label='Trend HDI'
            )

        ax1.set_title(f'{variable_names[i]} - Trend Component')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- Stationary Plot ---
        ax2 = axes[i, 1]
        ax2.plot(time_index, stationary_true_np[:, i], label='True Stationary', color='blue', linestyle='--')
        ax2.plot(time_index, stationary_mean_np[:, i], label='Estimated Mean Stationary', color='red')

        # Plot Stationary HDI if available
        if stationary_hdi is not None and 'low' in stationary_hdi and 'high' in stationary_hdi:
            ax2.fill_between(
                time_index,
                stationary_hdi['low'][:, i],
                stationary_hdi['high'][:, i],
                color='red',
                alpha=0.2,
                label='Stationary HDI'
            )

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
    trend_hdi: Optional[Dict[str, np.ndarray]], # Trend HDI (dict of NumPy: {'low': T x n_vars, 'high': T x n_vars})
    variable_names: Optional[List[str]] = None # Optional list of variable names
):
    """
    Plots the observed data and the estimated trend component (mean + HDI) for each variable.
    """
    T, n_vars = y_np.shape
    time_index = np.arange(T)

    # Convert JAX draws to NumPy for plotting and mean calculation
    trend_draws_np = np.asarray(trend_draws)

    # Compute mean of the trend draws if available
    if trend_draws_np.shape[0] > 0:
        trend_mean_np = np.mean(trend_draws_np, axis=0) # Shape (T, n_vars)
    else:
        # Return NaN means if no draws
        trend_mean_np = np.full_like(y_np, np.nan) # Use y_np shape as placeholder


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
        ax.plot(time_index, trend_mean_np[:, i], label='Estimated Mean Trend', color='red')

        # Plot Trend HDI if available
        if trend_hdi is not None and 'low' in trend_hdi and 'high' in trend_hdi:
            ax.fill_between(
                time_index,
                trend_hdi['low'][:, i],
                trend_hdi['high'][:, i],
                color='red',
                alpha=0.2,
                label='Trend HDI'
            )

        ax.set_title(f'{variable_names[i]} - Observed Data and Trend')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# # --- Update example_usage function to call these plotting functions ---

# # Modify the end of your example_usage function:

# def example_usage():
#     # ... (previous code for data generation, MCMC, extraction remains the same) ...

#     print("Computing HDI intervals using percentiles...") # Updated print statement

#     # Compute HDI using the percentile-based function
#     if trend_draws.shape[0] > 1:
#         trend_hdi = compute_hdi_with_percentiles(trend_draws, hdi_prob=0.9)
#         stationary_hdi = compute_hdi_with_percentiles(stationary_draws, hdi_prob=0.9)
#         print("HDI computed successfully using percentiles!") # Updated print statement
#     else:
#         trend_hdi = None
#         stationary_hdi = None
#         print("Not enough simulation smoother draws to compute HDI (need at least 2).")


#     print(f"Synthetic data (NumPy) shape: {y_np.shape}")
#     print(f"Synthetic data (JAX) shape: {y_jax.shape}")
#     print(f"Trend component draws shape: {trend_draws.shape}")
#     print(f"Stationary component draws shape: {stationary_draws.shape}")

#     # --- Add calls to plotting functions here ---
#     print("Generating plots...")

#     # You can provide variable names if you have them, otherwise defaults are used.
#     # For the synthetic data example, let's just use the defaults.
#     variable_names = [f'Var {i+1}' for i in range(n_vars)] # Example variable names

#     # Plot estimated vs true components
#     plot_decomposition_results(
#         y_np=y_np,
#         trends_true_np=trends_np,
#         stationary_true_np=stationary_np,
#         trend_draws=trend_draws,
#         stationary_draws=stationary_draws,
#         trend_hdi=trend_hdi,
#         stationary_hdi=stationary_hdi,
#         variable_names=variable_names
#     )

#     # Plot observed data and estimated trend
#     plot_observed_and_trend(
#         y_np=y_np,
#         trend_draws=trend_draws,
#         trend_hdi=trend_hdi,
#         variable_names=variable_names
#     )

#     print("Plotting complete.")
#     # --- End plotting calls ---


#     return {
#         'mcmc': mcmc,
#         'y_synthetic_np': y_np,
#         'y_synthetic_jax': y_jax,
#         'trends_true_np': trends_np,
#         'stationary_true_np': stationary_np,
#         'trend_draws': trend_draws,
#         'stationary_draws': stationary_draws,
#         'trend_hdi': trend_hdi,
#         'stationary_hdi': stationary_hdi
#     }

# # if __name__ == "__main__":
# #     results = example_usage() # Keep this outside the function definition