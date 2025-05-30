# simple_test_jax_fix.py - Test the JAX tracing fix

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import os

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def test_jax_compatible_p0():
    """Test that the P0 initialization works with JAX tracing"""
    print("=== Testing JAX-Compatible P0 Initialization ===")
    
    # Create a minimal test case
    state_dim = 4
    n_dynamic_trends = 2
    n_stationary = 2
    var_order = 1
    
    # Mock gamma matrix (would come from VAR sampling)
    gamma_0 = jnp.array([[0.25, 0.05], [0.05, 0.20]], dtype=jnp.float64)
    gamma_scaling = 0.1
    
    def test_model():
        """Simple NumPyro model to test P0 initialization"""
        
        # Initialize arrays (no NaN!)
        init_mean_base = jnp.zeros(state_dim, dtype=jnp.float64)
        init_std_for_sampling = jnp.ones(state_dim, dtype=jnp.float64)
        
        # Set values for dynamic trends (indices 0, 1)
        init_std_for_sampling = init_std_for_sampling.at[0].set(1.0)  # TREND1
        init_std_for_sampling = init_std_for_sampling.at[1].set(1.0)  # TREND2
        
        # Set values for stationary vars (indices 2, 3) using gamma
        theoretical_std_stat = jnp.sqrt(jnp.maximum(jnp.diag(gamma_0), 1e-9)) * jnp.sqrt(gamma_scaling)
        stat_block_start_idx = n_dynamic_trends
        stat_block_end_idx = stat_block_start_idx + n_stationary
        
        init_std_for_sampling = init_std_for_sampling.at[stat_block_start_idx:stat_block_end_idx].set(theoretical_std_stat)
        
        # This should work without JAX tracing errors
        init_mean_draw = numpyro.sample("init_mean_full", dist.Normal(init_mean_base, init_std_for_sampling).to_event(1))
        
        return init_mean_draw
    
    try:
        # Test that the model can be traced and run
        from numpyro.infer import MCMC, NUTS
        
        kernel = NUTS(test_model)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5, num_chains=1)
        
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key)
        
        samples = mcmc.get_samples()
        init_mean_samples = samples['init_mean_full']
        
        print(f"✓ JAX tracing test PASSED")
        print(f"  Sample shape: {init_mean_samples.shape}")
        print(f"  Sample mean: {jnp.mean(init_mean_samples, axis=0)}")
        print(f"  Sample std: {jnp.std(init_mean_samples, axis=0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX tracing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_problem_fix():
    """Test the specific original problem with a complete GPM model"""
    print("\n=== Testing Original Problem Fix ===")
    
    test_gpm_content = """
parameters rho;

estimated_params;
    stderr SHK_TREND1, inv_gamma_pdf, 2.3, 0.5;
    stderr SHK_TREND2, inv_gamma_pdf, 2.3, 0.5;
    stderr shk_stat1, inv_gamma_pdf, 2.3, 1.5;
    stderr shk_stat2, inv_gamma_pdf, 2.3, 1.5;
    rho, normal_pdf, 0.5, 0.1;
end;

trends_vars TREND1, TREND2;
stationary_variables stat1, stat2;

trend_shocks;
    var SHK_TREND1;
    var SHK_TREND2;
end;

shocks;
    var shk_stat1;
    var shk_stat2;
end;

trend_model;
    TREND1 = TREND1(-1) + rho*TREND2(-1) + SHK_TREND1;
    TREND2 = TREND2(-1) + SHK_TREND2;
end;

varobs OBS1, OBS2;

measurement_equations;
    OBS1 = TREND1 + stat1;
    OBS2 = TREND2 + stat2;
end;

var_prior_setup;
    var_order = 1;
    es = 0.7,0.1;
    fs = 0.5,0.5;
    gs = 3,2;
    hs = 1,0.5;
    eta = 2;
end;

initval;
    TREND1, normal_pdf, 0, 1;
    TREND2, normal_pdf, 0, 1;
end;
"""
    
    test_file = "test_original_fix.gpm"
    with open(test_file, "w") as f:
        f.write(test_gpm_content)
    
    try:
        # Generate some test data
        T_data, n_obs = 20, 2  # Smaller for faster testing
        key = random.PRNGKey(456)
        y_test = random.normal(key, (T_data, n_obs)) * 0.3
        
        # Import your fixed modules
        from gpm_numpyro_models import fit_gpm_numpyro_model
        
        print("Testing with very small MCMC run...")
        
        # This should now work without the NaN error OR the JAX tracing error
        mcmc_obj, reduced_model, ss_builder = fit_gpm_numpyro_model(
            gpm_file_path=test_file,
            y_data=y_test,
            num_warmup=3,  # Minimal for testing
            num_samples=3,
            num_chains=1,
            use_gamma_init_for_P0=True,  # This was causing both errors
            gamma_init_scaling_for_P0=0.1,
            target_accept_prob=0.9
        )
        
        print("✓ Original problem fix PASSED")
        print(f"  Model ran successfully")
        print(f"  State dimension: {ss_builder.state_dim}")
        print(f"  Dynamic trends: {ss_builder.n_dynamic_trends}")
        
        # Try to get samples to verify everything worked
        samples = mcmc_obj.get_samples()
        if 'init_mean_full' in samples:
            init_samples = samples['init_mean_full']
            print(f"  Init mean samples shape: {init_samples.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Original problem fix FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """Run all tests"""
    print("Testing JAX Tracing Fixes")
    print("=" * 50)
    
    # Test 1: Simple JAX compatibility
    test1_passed = test_jax_compatible_p0()
    
    # Test 2: Original problem fix
    test2_passed = test_original_problem_fix()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"JAX tracing compatibility: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Original problem fix: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All JAX tracing fixes work! The original error should be resolved.")
    else:
        print("❌ Some tests still failing. More fixes needed.")


if __name__ == "__main__":
    main()