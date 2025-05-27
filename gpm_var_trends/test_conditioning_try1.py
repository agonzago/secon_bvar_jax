# TESTING FRAMEWORK FOR CONDITIONAL SAMPLING

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

def test_conditional_vs_independent_sampling(data_file: str = 'sim_data.csv',
                                           gpm_file: str = 'test_model.gpm',
                                           num_warmup: int = 200,
                                           num_samples: int = 400,
                                           num_chains: int = 2):
    """
    Compare independent vs conditional sampling approaches.
    
    Tests:
    1. Convergence (divergences, R-hat, n_eff)
    2. Parameter estimates (bias, posterior intervals)
    3. Likelihood values
    4. Computational efficiency
    """
    
    print("=== TESTING CONDITIONAL VS INDEPENDENT SAMPLING ===\n")
    
    # Load data
    import pandas as pd
    dta = pd.read_csv(data_file)
    y = jnp.asarray(dta.values)
    
    results = {}
    
    # Test 1: Independent sampling (current baseline)
    print("1. Testing INDEPENDENT sampling (baseline)...")
    start_time = time.time()
    
    try:
        results['independent'] = run_model_test(
            y, gpm_file, 
            use_conditional_init=False,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            test_name="Independent"
        )
        results['independent']['runtime'] = time.time() - start_time
        print(f"✓ Independent sampling completed in {results['independent']['runtime']:.1f}s")
        
    except Exception as e:
        print(f"✗ Independent sampling failed: {e}")
        results['independent'] = {'failed': True, 'error': str(e)}
    
    print()
    
    # Test 2: Conditional sampling with very weak conditioning
    print("2. Testing CONDITIONAL sampling (weak conditioning=0.01)...")
    start_time = time.time()
    
    try:
        results['conditional_weak'] = run_model_test(
            y, gpm_file,
            use_conditional_init=True,
            conditioning_strength=0.01,  # Very weak
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            test_name="Conditional_Weak"
        )
        results['conditional_weak']['runtime'] = time.time() - start_time
        print(f"✓ Conditional (weak) sampling completed in {results['conditional_weak']['runtime']:.1f}s")
        
    except Exception as e:
        print(f"✗ Conditional (weak) sampling failed: {e}")
        results['conditional_weak'] = {'failed': True, 'error': str(e)}
    
    print()
    
    # Test 3: Conditional sampling with moderate conditioning  
    print("3. Testing CONDITIONAL sampling (moderate conditioning=0.1)...")
    start_time = time.time()
    
    try:
        results['conditional_moderate'] = run_model_test(
            y, gpm_file,
            use_conditional_init=True,
            conditioning_strength=0.1,  # Moderate
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            test_name="Conditional_Moderate"
        )
        results['conditional_moderate']['runtime'] = time.time() - start_time
        print(f"✓ Conditional (moderate) sampling completed in {results['conditional_moderate']['runtime']:.1f}s")
        
    except Exception as e:
        print(f"✗ Conditional (moderate) sampling failed: {e}")
        results['conditional_moderate'] = {'failed': True, 'error': str(e)}
    
    print()
    
    # Compare results
    print("=== COMPARISON RESULTS ===")
    compare_sampling_approaches(results)
    
    return results


def run_model_test(y: jnp.ndarray, gpm_file: str,
                   use_conditional_init: bool,
                   conditioning_strength: float = 0.1,
                   num_warmup: int = 200,
                   num_samples: int = 400, 
                   num_chains: int = 2,
                   test_name: str = "Test") -> Dict:
    """Run a single model test and return diagnostics"""
    
    from gpm_parser import GPMParser
    from gpm_bvar_trends import GPMStateSpaceBuilder, create_gpm_based_model_with_conditional_init
    from numpyro.infer import MCMC, NUTS
    import jax.random as random
    
    # Create model
    model_fn, gpm_model, ss_builder = create_gpm_based_model_with_conditional_init(
        gpm_file,
        use_conditional_init=use_conditional_init,
        conditioning_strength=conditioning_strength
    )
    
    # Run MCMC
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    mcmc_key = random.PRNGKey(42)  # Fixed seed for comparison
    mcmc.run(mcmc_key, y=y)
    
    # Extract diagnostics
    samples = mcmc.get_samples()
    
    # Check for divergences
    try:
        divergences = mcmc.get_extra_fields()['diverging'].sum()
    except:
        divergences = 0
    
    # Compute parameter summaries
    param_summary = {}
    for param_name, param_values in samples.items():
        if len(param_values.shape) <= 2:  # Skip high-dimensional parameters for summary
            param_summary[param_name] = {
                'mean': float(jnp.mean(param_values)),
                'std': float(jnp.std(param_values)),
                'rhat': compute_rhat_simple(param_values, num_chains),
                'n_eff': compute_neff_simple(param_values)
            }
    
    # Compute log likelihood
    try:
        log_likelihood = jnp.mean(samples.get('loglik', jnp.array([jnp.nan])))
    except:
        log_likelihood = jnp.nan
    
    return {
        'mcmc': mcmc,
        'samples': samples,
        'divergences': int(divergences),
        'param_summary': param_summary,
        'log_likelihood': float(log_likelihood),
        'num_samples': num_samples * num_chains,
        'test_name': test_name,
        'use_conditional': use_conditional_init,
        'conditioning_strength': conditioning_strength if use_conditional_init else None
    }


def compare_sampling_approaches(results: Dict):
    """Compare the different sampling approaches"""
    
    approaches = ['independent', 'conditional_weak', 'conditional_moderate']
    
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    # Divergences comparison
    print(f"{'Approach':<20} {'Divergences':<12} {'Runtime(s)':<12} {'Status':<10}")
    print("-" * 60)
    
    for approach in approaches:
        if approach in results and not results[approach].get('failed', False):
            r = results[approach]
            print(f"{r['test_name']:<20} {r['divergences']:<12} {r.get('runtime', 0):<12.1f} {'✓ OK':<10}")
        else:
            error = results.get(approach, {}).get('error', 'Not run')
            print(f"{approach:<20} {'N/A':<12} {'N/A':<12} {'✗ FAIL':<10}")
    
    print()
    
    # Parameter comparison for key parameters
    key_params = ['sigma_SHK_TREND1', 'sigma_SHK_STAT1', 'Amu_0', 'Amu_1']
    
    print("PARAMETER ESTIMATES COMPARISON")
    print("="*60)
    
    for param in key_params:
        print(f"\nParameter: {param}")
        print(f"{'Approach':<20} {'Mean':<10} {'Std':<10} {'R-hat':<10} {'N_eff':<10}")
        print("-" * 60)
        
        for approach in approaches:
            if (approach in results and 
                not results[approach].get('failed', False) and
                param in results[approach]['param_summary']):
                
                p = results[approach]['param_summary'][param]
                print(f"{results[approach]['test_name']:<20} "
                      f"{p['mean']:<10.3f} {p['std']:<10.3f} "
                      f"{p['rhat']:<10.3f} {p['n_eff']:<10.0f}")
    
    print()
    
    # Model fit comparison
    print("MODEL FIT COMPARISON")
    print("="*40)
    print(f"{'Approach':<20} {'Log Likelihood':<15}")
    print("-" * 40)
    
    for approach in approaches:
        if approach in results and not results[approach].get('failed', False):
            ll = results[approach]['log_likelihood']
            print(f"{results[approach]['test_name']:<20} {ll:<15.2f}")
    
    print()
    
    # Summary recommendations
    print("RECOMMENDATIONS")
    print("="*40)
    
    # Find best approach based on divergences and convergence
    best_approach = None
    min_divergences = float('inf')
    
    for approach in approaches:
        if approach in results and not results[approach].get('failed', False):
            div = results[approach]['divergences']
            if div < min_divergences:
                min_divergences = div
                best_approach = approach
    
    if best_approach:
        print(f"✓ Best approach: {results[best_approach]['test_name']}")
        print(f"  - Divergences: {results[best_approach]['divergences']}")
        print(f"  - Runtime: {results[best_approach].get('runtime', 0):.1f}s")
        
        if results[best_approach]['use_conditional']:
            strength = results[best_approach]['conditioning_strength'] 
            print(f"  - Conditioning strength: {strength}")
            print(f"  - Theoretical justification: Uses VAR-conditional initial conditions")
        else:
            print(f"  - Uses independent initial condition priors")
    else:
        print("✗ No approach completed successfully")


def compute_rhat_simple(samples: jnp.ndarray, num_chains: int) -> float:
    """Simple R-hat computation"""
    if len(samples.shape) > 1:
        samples = samples.flatten()
    
    n_total = len(samples)
    n_per_chain = n_total // num_chains
    
    if n_per_chain < 2:
        return jnp.nan
    
    # Reshape into chains
    chains = samples[:n_per_chain * num_chains].reshape(num_chains, n_per_chain)
    
    # Between and within chain variance
    chain_means = jnp.mean(chains, axis=1)
    overall_mean = jnp.mean(chain_means)
    
    B = n_per_chain * jnp.var(chain_means, ddof=1)
    W = jnp.mean(jnp.var(chains, axis=1, ddof=1))
    
    var_plus = ((n_per_chain - 1) * W + B) / n_per_chain
    rhat = jnp.sqrt(var_plus / W)
    
    return float(rhat)


def compute_neff_simple(samples: jnp.ndarray) -> float:
    """Simple effective sample size computation"""
    if len(samples.shape) > 1:
        samples = samples.flatten()
    
    n = len(samples)
    if n < 4:
        return float(n)
    
    # Simple autocorrelation-based estimate
    mean_sample = jnp.mean(samples)
    var_sample = jnp.var(samples)
    
    if var_sample == 0:
        return float(n)
    
    # Estimate autocorrelation at lag 1
    autocorr_1 = jnp.corrcoef(samples[:-1], samples[1:])[0, 1]
    
    if jnp.isnan(autocorr_1) or autocorr_1 <= 0:
        return float(n)
    
    # Simple effective sample size estimate
    n_eff = n * (1 - autocorr_1) / (1 + autocorr_1)
    
    return float(jnp.clip(n_eff, 1, n))


def diagnostic_plots(results: Dict, save_plots: bool = True):
    """Generate diagnostic plots comparing approaches"""
    
    approaches = ['independent', 'conditional_weak', 'conditional_moderate']
    successful_approaches = [a for a in approaches if a in results and not results[a].get('failed', False)]
    
    if len(successful_approaches) < 2:
        print("Need at least 2 successful approaches for comparison plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Conditional vs Independent Sampling Comparison', fontsize=14)
    
    # Plot 1: Divergences
    ax1 = axes[0, 0]
    approach_names = [results[a]['test_name'] for a in successful_approaches]
    divergences = [results[a]['divergences'] for a in successful_approaches]
    colors = ['blue', 'orange', 'green'][:len(successful_approaches)]
    
    bars = ax1.bar(approach_names, divergences, color=colors, alpha=0.7)
    ax1.set_title('Number of Divergences')
    ax1.set_ylabel('Divergences')
    
    # Add value labels on bars
    for bar, div in zip(bars, divergences):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(divergences)*0.01,
                str(div), ha='center', va='bottom')
    
    # Plot 2: Runtime
    ax2 = axes[0, 1]
    runtimes = [results[a].get('runtime', 0) for a in successful_approaches]
    bars = ax2.bar(approach_names, runtimes, color=colors, alpha=0.7)
    ax2.set_title('Runtime (seconds)')
    ax2.set_ylabel('Time (s)')
    
    for bar, time in zip(bars, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtimes)*0.01,
                f'{time:.1f}', ha='center', va='bottom')
    
    # Plot 3: Parameter comparison (trend shock std)
    ax3 = axes[1, 0]
    param_name = 'sigma_SHK_TREND1'
    
    if all(param_name in results[a]['param_summary'] for a in successful_approaches):
        means = [results[a]['param_summary'][param_name]['mean'] for a in successful_approaches]
        stds = [results[a]['param_summary'][param_name]['std'] for a in successful_approaches]
        
        x_pos = range(len(successful_approaches))
        ax3.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, capthick=2)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(approach_names)
        ax3.set_title(f'{param_name} Estimates')
        ax3.set_ylabel('Parameter Value')
    
    # Plot 4: R-hat comparison
    ax4 = axes[1, 1]
    
    if all(param_name in results[a]['param_summary'] for a in successful_approaches):
        rhats = [results[a]['param_summary'][param_name]['rhat'] for a in successful_approaches]
        
        bars = ax4.bar(approach_names, rhats, color=colors, alpha=0.7)
        ax4.axhline(y=1.1, color='red', linestyle='--', alpha=0.7, label='R-hat = 1.1')
        ax4.set_title(f'{param_name} R-hat Values')
        ax4.set_ylabel('R-hat')
        ax4.legend()
        
        for bar, rhat in zip(bars, rhats):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rhat:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('conditional_sampling_comparison.png', dpi=300, bbox_inches='tight')
        print("Diagnostic plots saved as 'conditional_sampling_comparison.png'")
    
    plt.show()


# SIMPLE TEST RUNNER

def quick_conditional_test():
    """Quick test to see if conditional sampling works"""
    
    print("Quick Conditional Sampling Test")
    print("=" * 40)
    
    # Generate test data if needed
    if not os.path.exists('sim_data.csv'):
        print("Generating test data...")
        from gpm_bar_smoother import generate_synthetic_data
        generate_synthetic_data()
    
    # Create test GPM file if needed  
    if not os.path.exists('test_conditional.gpm'):
        print("Creating test GPM file...")
        from gpm_bar_smoother import create_default_gpm_file
        create_default_gpm_file('test_conditional.gpm', 3)
    
    # Run comparison test
    results = test_conditional_vs_independent_sampling(
        data_file='sim_data.csv',
        gpm_file='test_conditional.gpm',
        num_warmup=100,  # Quick test
        num_samples=200,
        num_chains=2
    )
    
    # Generate plots
    try:
        diagnostic_plots(results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    return results


if __name__ == "__main__":
    import os
    results = quick_conditional_test()