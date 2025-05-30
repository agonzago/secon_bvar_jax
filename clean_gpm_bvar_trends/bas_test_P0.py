# test_p0_fixes.py - Test script to verify the P0 initialization fixes

import jax
import jax.numpy as jnp
import jax.random as random
import os

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def test_state_space_builder_dimensions():
    """Test that StateSpaceBuilder calculates dimensions correctly"""
    print("=== Testing StateSpaceBuilder Dimensions ===")
    
    # Create a test GPM content with known structure
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
    
    test_gpm_file = "test_dimensions.gpm"
    with open(test_gpm_file, "w") as f:
        f.write(test_gpm_content)
    
    try:
        # Import your fixed modules here
        from gpm_model_parser import GPMModelParser
        from state_space_builder import StateSpaceBuilder  # Your fixed version
        
        # Parse the GPM
        parser = GPMModelParser()
        reduced_model = parser.parse_file(test_gpm_file)
        
        # Create StateSpaceBuilder with fixed logic
        ss_builder = StateSpaceBuilder(reduced_model)
        
        print(f"Core variables: {reduced_model.core_variables}")
        print(f"Stationary variables: {reduced_model.stationary_variables}")
        print(f"n_core: {ss_builder.n_core}")
        print(f"n_stationary: {ss_builder.n_stationary}")
        print(f"n_dynamic_trends: {ss_builder.n_dynamic_trends}")
        print(f"var_order: {ss_builder.var_order}")
        print(f"state_dim: {ss_builder.state_dim}")
        print(f"core_var_map: {ss_builder.core_var_map}")
        
        # Expected values
        expected_n_core = 4  # TREND1, TREND2, stat1, stat2
        expected_n_stationary = 2  # stat1, stat2
        expected_n_dynamic_trends = 2  # TREND1, TREND2
        expected_state_dim = 4  # TREND1, TREND2, stat1_t, stat2_t (VAR(1))
        
        assert ss_builder.n_core == expected_n_core, f"n_core: got {ss_builder.n_core}, expected {expected_n_core}"
        assert ss_builder.n_stationary == expected_n_stationary, f"n_stationary: got {ss_builder.n_stationary}, expected {expected_n_stationary}"
        assert ss_builder.n_dynamic_trends == expected_n_dynamic_trends, f"n_dynamic_trends: got {ss_builder.n_dynamic_trends}, expected {expected_n_dynamic_trends}"
        assert ss_builder.state_dim == expected_state_dim, f"state_dim: got {ss_builder.state_dim}, expected {expected_state_dim}"
        
        # Check state vector mapping
        expected_core_var_map = {
            'TREND1': 0,    # First dynamic trend
            'TREND2': 1,    # Second dynamic trend
            'stat1': 2,     # First stationary var (current period)
            'stat2': 3      # Second stationary var (current period)
        }
        
        for var_name, expected_idx in expected_core_var_map.items():
            actual_idx = ss_builder.core_var_map.get(var_name)
            assert actual_idx == expected_idx, f"core_var_map['{var_name}']: got {actual_idx}, expected {expected_idx}"
        
        print("✓ StateSpaceBuilder dimensions test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ StateSpaceBuilder dimensions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_gpm_file):
            os.remove(test_gpm_file)


def test_p0_initialization():
    """Test the P0 initialization functions with correct state dimensions"""
    print("\n=== Testing P0 Initialization ===")
    
    # Create synthetic data for testing
    T_data, n_obs = 50, 2
    key = random.PRNGKey(123)
    y_test = random.normal(key, (T_data, n_obs)) * 0.5
    
    test_gpm_file = "test_p0.gpm"
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
    
    with open(test_gpm_file, "w") as f:
        f.write(test_gpm_content)
    
    try:
        # Test with your fixed modules
        from gpm_numpyro_models import fit_gpm_numpyro_model  # With fixes applied
        
        print("Testing gamma-based P0 initialization...")
        
        # This should now work without the NaN error
        mcmc_obj, reduced_model, ss_builder = fit_gpm_numpyro_model(
            gpm_file_path=test_gpm_file,
            y_data=y_test,
            num_warmup=10,  # Very small for testing
            num_samples=10,
            num_chains=1,
            use_gamma_init_for_P0=True,  # This was causing the error
            gamma_init_scaling_for_P0=0.1,
            target_accept_prob=0.9
        )
        
        print("✓ Gamma-based P0 initialization test PASSED")
        
        # Test standard P0 as well
        print("Testing standard P0 initialization...")
        mcmc_obj2, _, _ = fit_gpm_numpyro_model(
            gpm_file_path=test_gpm_file,
            y_data=y_test,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            use_gamma_init_for_P0=False,  # Standard initialization
            target_accept_prob=0.9
        )
        
        print("✓ Standard P0 initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ P0 initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_gpm_file):
            os.remove(test_gpm_file)


def main():
    """Run all tests"""
    print("Running P0 Initialization Fix Tests")
    print("=" * 50)
    
    # Test 1: State space dimensions
    test1_passed = test_state_space_builder_dimensions()
    
    # Test 2: P0 initialization 
    test2_passed = test_p0_initialization()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"StateSpaceBuilder dimensions: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"P0 initialization: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests PASSED! The fixes should resolve the original error.")
    else:
        print("❌ Some tests FAILED. Please review the fixes.")


if __name__ == "__main__":
    main()