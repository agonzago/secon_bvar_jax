"""
Test New Integration System
===========================

This script tests the complete new integration pipeline and provides
examples of how to migrate from the old system to the new one.
"""

import sys
import os
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, Any

# Import the new integration system
try:
    from parameter_contract import get_parameter_contract, PARAMETER_CONTRACT
    from mcmc_adapter import create_mcmc_adapter
    from integration_coordinator import create_integration_coordinator
    from gmp_bvar_trends_fixed import create_gmp_based_model_fixed, fit_gmp_model_fixed
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all new integration files are in the Python path")
    sys.exit(1)

def test_parameter_contract():
    """Test the parameter contract system"""
    
    print("="*60)
    print("TESTING PARAMETER CONTRACT")
    print("="*60)
    
    contract = get_parameter_contract()
    
    # Print contract summary
    print(contract.get_contract_summary())
    
    # Test parameter name mapping
    test_cases = [
        ("sigma_shk_trend_r_world", "shk_trend_r_world"),
        ("sigma_shk_cycle_y_us", "shk_cycle_y_us"),
        ("var_phi", "var_phi")
    ]
    
    print("\nTesting parameter name mapping:")
    for mcmc_name, expected_builder_name in test_cases:
        try:
            builder_name = contract.get_builder_name(mcmc_name)
            if builder_name == expected_builder_name:
                print(f"✓ {mcmc_name} → {builder_name}")
            else:
                print(f"✗ {mcmc_name} → {builder_name} (expected {expected_builder_name})")
        except Exception as e:
            print(f"✗ {mcmc_name} → ERROR: {e}")
    
    # Test validation
    print("\nTesting parameter validation:")
    
    # Valid parameters
    valid_params = {
        "sigma_shk_trend_r_world": 0.1,
        "sigma_shk_cycle_y_us": 0.2, 
        "var_phi": 1.5
    }
    
    try:
        contract.validate_mcmc_parameters(valid_params)
        print("✓ Valid parameter validation passed")
    except Exception as e:
        print(f"✗ Valid parameter validation failed: {e}")
    
    # Invalid parameters (missing required)
    invalid_params = {
        "sigma_shk_trend_r_world": 0.1,
        # Missing many required parameters
    }
    
    try:
        contract.validate_mcmc_parameters(invalid_params)
        print("✗ Invalid parameter validation should have failed")
    except Exception as e:
        print("✓ Invalid parameter validation correctly failed")
    
    return True


def test_mcmc_adapter():
    """Test the MCMC adapter system"""
    
    print("\n" + "="*60)
    print("TESTING MCMC ADAPTER")
    print("="*60)
    
    adapter = create_mcmc_adapter()
    
    # Test parameter dictionary transformation
    mcmc_params = {
        "sigma_shk_trend_r_world": jnp.array(0.1),
        "sigma_shk_cycle_y_us": jnp.array(0.2),
        "var_phi": jnp.array(1.5),
        "_var_coefficients": jnp.zeros((2, 3, 3)),
        "_var_innovation_cov": jnp.eye(3) * 0.1
    }
    
    try:
        transformed = adapter.transform_parameter_dict(mcmc_params)
        print("✓ Parameter transformation successful:")
        for key, value in transformed.items():
            if isinstance(value, jnp.ndarray):
                print(f"  {key}: array{value.shape}")
            else:
                print(f"  {key}: {value}")
        return True
    except Exception as e:
        print(f"✗ Parameter transformation failed: {e}")
        return False


def test_integration_coordinator():
    """Test the integration coordinator"""
    
    print("\n" + "="*60)
    print("TESTING INTEGRATION COORDINATOR")
    print("="*60)
    
    # Create a simple test GPM file
    test_gmp_content = """
parameters var_phi;

estimated_params;
    stderr shk_trend_r_world, inv_gamma_pdf, 2.1, 0.81;
    stderr shk_trend_pi_world, inv_gamma_pdf, 2.1, 0.81;
    stderr shk_cycle_y_us, inv_gamma_pdf, 2.1, 0.38;
    var_phi, normal_pdf, 1.0, 0.2;
end;

trends_vars
    trend_r_world,
    trend_pi_world
;

stationary_variables
    cycle_y_us
;

trend_shocks;
    shk_trend_r_world,
    shk_trend_pi_world
end;

shocks;
    shk_cycle_y_us
end;

trend_model;
    trend_r_world = trend_r_world(-1) + shk_trend_r_world;
    trend_pi_world = trend_pi_world(-1) + shk_trend_pi_world;
end;

varobs y_us;

measurement_equations;
    y_us = trend_r_world + cycle_y_us;
end;

var_prior_setup;
    var_order = 1;
    es = 0.5, 0.3;
    fs = 0.5, 0.5;
    gs = 2.0, 2.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;
"""
    
    # Save test file
    with open('test_integration.gmp', 'w') as f:
        f.write(test_gmp_content)
    
    try:
        # Create coordinator
        coordinator = create_integration_coordinator('test_integration.gmp')
        print("✓ Coordinator created successfully")
        
        # Print model summary
        print(coordinator.get_model_summary())
        
        # Test state space construction
        success = coordinator.test_state_space_construction()
        
        if success:
            print("✓ Integration coordinator test passed")
            return True
        else:
            print("✗ Integration coordinator test failed")
            return False
            
    except Exception as e:
        print(f"✗ Integration coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test file
        if os.path.exists('test_integration.gmp'):
            os.remove('test_integration.gmp')


def test_fixed_mcmc_system():
    """Test the fixed MCMC system with a complete example"""
    
    print("\n" + "="*60)
    print("TESTING FIXED MCMC SYSTEM")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    T = 100
    n_vars = 2
    y_synthetic = np.random.normal(0, 1, (T, n_vars))
    y_jax = jnp.asarray(y_synthetic)
    
    # Create test GPM file
    test_gmp_content = """
parameters var_phi;

estimated_params;
    stderr shk_trend_r_world, inv_gamma_pdf, 2.1, 0.81;
    stderr shk_trend_pi_world, inv_gamma_pdf, 2.1, 0.81;
    stderr shk_cycle_y_us, inv_gamma_pdf, 2.1, 0.38;
    stderr shk_cycle_r_us, inv_gamma_pdf, 2.1, 0.38;
    var_phi, normal_pdf, 1.0, 0.2;
end;

trends_vars
    trend_r_world,
    trend_pi_world
;

stationary_variables
    cycle_y_us,
    cycle_r_us
;

trend_shocks;
    shk_trend_r_world,
    shk_trend_pi_world
end;

shocks;
    shk_cycle_y_us,
    shk_cycle_r_us
end;

trend_model;
    trend_r_world = trend_r_world(-1) + shk_trend_r_world;
    trend_pi_world = trend_pi_world(-1) + shk_trend_pi_world;
end;

varobs y_us, r_us;

measurement_equations;
    y_us = trend_r_world + cycle_y_us;
    r_us = trend_pi_world + cycle_r_us;
end;

var_prior_setup;
    var_order = 1;
    es = 0.5, 0.3;
    fs = 0.5, 0.5;
    gs = 2.0, 2.0;
    hs = 1.0, 1.0;
    eta = 2.0;
end;
"""
    
    # Save test file
    with open('test_mcmc_fixed.gmp', 'w') as f:
        f.write(test_gmp_content)
    
    try:
        print("Testing MCMC with minimal settings...")
        
        # Run MCMC with minimal settings for testing
        mcmc, gmp_model, ss_builder = fit_gmp_model_fixed(
            'test_mcmc_fixed.gmp', 
            y_jax,
            num_warmup=50,    # Minimal for testing
            num_samples=100,  # Minimal for testing  
            num_chains=1,     # Single chain for testing
            rng_key=random.PRNGKey(42)
        )
        
        if mcmc is not None:
            print("✓ FIXED MCMC system test passed")
            
            # Test parameter extraction
            samples = mcmc.get_samples()
            print(f"✓ Extracted {len(samples)} parameter types from MCMC")
            print(f"  Sample parameters: {list(samples.keys())}")
            
            return True
        else:
            print("✗ FIXED MCMC system test failed - no MCMC results")
            return False
            
    except Exception as e:
        print(f"✗ FIXED MCMC system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test file
        if os.path.exists('test_mcmc_fixed.gmp'):
            os.remove('test_mcmc_fixed.gmp')


def run_migration_example():
    """Show how to migrate existing code"""
    
    print("\n" + "="*60)
    print("MIGRATION EXAMPLE")
    print("="*60)
    
    print("OLD CODE (before):")
    print("-" * 20)
    print("""
# OLD imports
from gmp_bvar_trends import create_gmp_based_model, fit_gmp_model
from integration_helper import create_reduced_gmp_model

# OLD usage
model_fn, gmp_model, ss_builder = create_gmp_based_model(gmp_file)
integration, reduced_model, builder = create_reduced_gmp_model(gmp_file)
mcmc, gmp_model, ss_builder = fit_gmp_model(gmp_file, y_data)
""")
    
    print("NEW CODE (after):")
    print("-" * 20)
    print("""
# NEW imports  
from gmp_bvar_trends_fixed import create_gmp_based_model_fixed, fit_gmp_model_fixed
from gmp_bvar_trends_fixed import create_reduced_gmp_model_fixed

# NEW usage (SAME INTERFACE!)
model_fn, gmp_model, ss_builder = create_gmp_based_model_fixed(gmp_file)
integration, reduced_model, builder = create_reduced_gmp_model_fixed(gmp_file)
mcmc, gmp_model, ss_builder = fit_gmp_model_fixed(gmp_file, y_data)
""")
    
    print("BENEFITS OF NEW SYSTEM:")
    print("-" * 25)
    benefits = [
        "✓ No more parameter name guessing or fallbacks",
        "✓ Clear error messages when parameters are missing", 
        "✓ Contract-driven validation ensures consistency",
        "✓ Robust type conversion handles JAX arrays properly",
        "✓ Same interface - minimal code changes required",
        "✓ Better debugging with explicit error contexts"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")


def main():
    """Run all tests"""
    
    print("TESTING NEW INTEGRATION SYSTEM")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Parameter Contract", test_parameter_contract),
        ("MCMC Adapter", test_mcmc_adapter), 
        ("Integration Coordinator", test_integration_coordinator),
        ("Fixed MCMC System", test_fixed_mcmc_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
            results.append((test_name, False))
    
    # Show migration example
    run_migration_example()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The new integration system is ready for production use.")
        print("\nTo migrate your code:")
        print("1. Replace imports with _fixed versions")
        print("2. Test with your existing GPM files")
        print("3. Enjoy robust parameter handling!")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please check the error messages above and fix issues before migration.")


if __name__ == "__main__":
    main()