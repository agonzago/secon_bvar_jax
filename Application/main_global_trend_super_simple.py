# main_global_trend_new.py (Modified)
import sys
import os
import numpy as np
import pandas as pd
import jax

# Ensure the package root is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the main workflow function from your package
# This function internally calls complete_gpm_workflow_with_smoother_fixed or evaluate_gpm_at_parameters
from clean_gpm_bvar_trends import run_complete_gpm_analysis 

import multiprocessing
# JAX Configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# Consider setting XLA_FLAGS outside the script if specific CPU counts are needed for JAX/NumPyro parallelism
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count() or 1}"

## DEBUGGING COE
# Add this diagnostic function to your main script to debug reconstruction

def debug_variable_reconstruction(results):
    """
    Debug the variable reconstruction process step by step.
    """
    print("\n" + "="*80)
    print("VARIABLE RECONSTRUCTION DIAGNOSTIC")
    print("="*80)
    
    # Check a specific trend to see if it's being reconstructed correctly
    # Let's focus on the US variables since those are what you're plotting
    
    # 1. Check core variables (should be correct after the ordering fix)
    print("\n1. CORE VARIABLES CHECK:")
    print("-" * 40)
    
    core_vars_to_check = [
        'r_US_idio_trend', 'pi_US_idio_trend', 'y_US_trend'
    ]
    
    for var in core_vars_to_check:
        if var in results.trend_names:
            idx = results.trend_names.index(var)
            data = results.trend_draws[:, :, idx]
            median_series = np.median(data, axis=0)
            
            print(f"  {var}:")
            print(f"    Index in results: {idx}")
            print(f"    Shape: {data.shape}")
            print(f"    Median first 5 values: {median_series[:5]}")
            print(f"    Median last 5 values: {median_series[-5:]}")
            print(f"    Range: [{np.min(median_series):.3f}, {np.max(median_series):.3f}]")
        else:
            print(f"  ❌ {var} not found in trend_names")
    
    # 2. Check derived variables
    print("\n2. DERIVED VARIABLES CHECK:")
    print("-" * 40)
    
    # Check if derived variables are constructed correctly
    derived_vars_to_check = [
        'rr_US_full_trend', 'pi_US_full_trend', 'R_US_short_trend'
    ]
    
    for var in derived_vars_to_check:
        if var in results.trend_names:
            idx = results.trend_names.index(var)
            data = results.trend_draws[:, :, idx]
            median_series = np.median(data, axis=0)
            
            print(f"  {var}:")
            print(f"    Index in results: {idx}")
            print(f"    Shape: {data.shape}")
            print(f"    Median first 5 values: {median_series[:5]}")
            print(f"    Median last 5 values: {median_series[-5:]}")
            print(f"    Range: [{np.min(median_series):.3f}, {np.max(median_series):.3f}]")
        else:
            print(f"  ❌ {var} not found in trend_names")
    
    # 3. Check mathematical relationships
    print("\n3. MATHEMATICAL RELATIONSHIPS CHECK:")
    print("-" * 40)
    
    # According to your GPM file:
    # rr_US_full_trend = r_US_idio_trend  
    # pi_US_full_trend = pi_US_idio_trend
    # R_US_short_trend = rr_US_full_trend + pi_US_full_trend
    
    try:
        # Get the variables
        r_idio_idx = results.trend_names.index('r_US_idio_trend')
        pi_idio_idx = results.trend_names.index('pi_US_idio_trend')
        rr_full_idx = results.trend_names.index('rr_US_full_trend')
        pi_full_idx = results.trend_names.index('pi_US_full_trend')
        r_short_idx = results.trend_names.index('R_US_short_trend')
        
        # Get median series for easier comparison
        r_idio = np.median(results.trend_draws[:, :, r_idio_idx], axis=0)
        pi_idio = np.median(results.trend_draws[:, :, pi_idio_idx], axis=0)
        rr_full = np.median(results.trend_draws[:, :, rr_full_idx], axis=0)
        pi_full = np.median(results.trend_draws[:, :, pi_full_idx], axis=0)
        r_short = np.median(results.trend_draws[:, :, r_short_idx], axis=0)
        
        # Check relationship 1: rr_US_full_trend should equal r_US_idio_trend
        diff1 = np.abs(rr_full - r_idio)
        max_diff1 = np.max(diff1)
        print(f"  rr_US_full_trend vs r_US_idio_trend:")
        print(f"    Max absolute difference: {max_diff1:.6f}")
        if max_diff1 < 1e-10:
            print(f"    ✅ CORRECT: rr_US_full_trend = r_US_idio_trend")
        else:
            print(f"    ❌ WRONG: rr_US_full_trend ≠ r_US_idio_trend")
            print(f"    rr_full sample: {rr_full[:5]}")
            print(f"    r_idio sample:  {r_idio[:5]}")
        
        # Check relationship 2: pi_US_full_trend should equal pi_US_idio_trend
        diff2 = np.abs(pi_full - pi_idio)
        max_diff2 = np.max(diff2)
        print(f"  pi_US_full_trend vs pi_US_idio_trend:")
        print(f"    Max absolute difference: {max_diff2:.6f}")
        if max_diff2 < 1e-10:
            print(f"    ✅ CORRECT: pi_US_full_trend = pi_US_idio_trend")
        else:
            print(f"    ❌ WRONG: pi_US_full_trend ≠ pi_US_idio_trend")
            print(f"    pi_full sample: {pi_full[:5]}")
            print(f"    pi_idio sample: {pi_idio[:5]}")
        
        # Check relationship 3: R_US_short_trend should equal rr_US_full_trend + pi_US_full_trend
        expected_r_short = rr_full + pi_full
        diff3 = np.abs(r_short - expected_r_short)
        max_diff3 = np.max(diff3)
        print(f"  R_US_short_trend vs (rr_US_full_trend + pi_US_full_trend):")
        print(f"    Max absolute difference: {max_diff3:.6f}")
        if max_diff3 < 1e-10:
            print(f"    ✅ CORRECT: R_US_short_trend = rr_US_full_trend + pi_US_full_trend")
        else:
            print(f"    ❌ WRONG: R_US_short_trend ≠ rr_US_full_trend + pi_US_full_trend")
            print(f"    r_short sample:   {r_short[:5]}")
            print(f"    expected sample:  {expected_r_short[:5]}")
        
    except Exception as e:
        print(f"  ❌ Error checking relationships: {e}")
    
    # 4. Compare with observed data
    print("\n4. COMPARISON WITH OBSERVED DATA:")
    print("-" * 40)
    
    try:
        # Compare observed y_us with y_US_trend
        if 'y_us' in results.observed_variable_names and 'y_US_trend' in results.trend_names:
            obs_idx = results.observed_variable_names.index('y_us')
            trend_idx = results.trend_names.index('y_US_trend')
            
            obs_data = results.observed_data[:, obs_idx]
            trend_data = np.median(results.trend_draws[:, :, trend_idx], axis=0)
            
            print(f"  y_us (observed) vs y_US_trend:")
            print(f"    Observed range: [{np.min(obs_data):.3f}, {np.max(obs_data):.3f}]")
            print(f"    Trend range:    [{np.min(trend_data):.3f}, {np.max(trend_data):.3f}]")
            print(f"    Correlation: {np.corrcoef(obs_data, trend_data)[0,1]:.3f}")
            
            # The trend should be somewhat close to the observed data
            if np.corrcoef(obs_data, trend_data)[0,1] > 0.5:
                print(f"    ✅ Good correlation between observed and trend")
            else:
                print(f"    ⚠️  Low correlation - possible issue")
        
        # Check inflation as well
        if 'pi_us' in results.observed_variable_names and 'pi_US_full_trend' in results.trend_names:
            obs_idx = results.observed_variable_names.index('pi_us')
            trend_idx = results.trend_names.index('pi_US_full_trend')
            
            obs_data = results.observed_data[:, obs_idx]
            trend_data = np.median(results.trend_draws[:, :, trend_idx], axis=0)
            
            print(f"  pi_us (observed) vs pi_US_full_trend:")
            print(f"    Observed range: [{np.min(obs_data):.3f}, {np.max(obs_data):.3f}]")
            print(f"    Trend range:    [{np.min(trend_data):.3f}, {np.max(trend_data):.3f}]")
            print(f"    Correlation: {np.corrcoef(obs_data, trend_data)[0,1]:.3f}")
            
    except Exception as e:
        print(f"  ❌ Error comparing with observed data: {e}")
    
    print("\n" + "="*80)

def diagnose_variable_names(results, custom_plot_specs):
    """
    Comprehensive diagnostic of variable names in SmootherResults.
    """
    print("\n" + "="*80)
    print("VARIABLE NAMES DIAGNOSTIC")
    print("="*80)
    
    # 1. Check what's in SmootherResults
    print("\n1. AVAILABLE VARIABLES IN SMOOTHER RESULTS:")
    print("-" * 50)
    
    if hasattr(results, 'trend_draws') and results.trend_draws is not None:
        print(f"📊 TREND VARIABLES ({len(results.trend_names) if results.trend_names else 0}):")
        if results.trend_names:
            for i, name in enumerate(results.trend_names):
                print(f"   [{i:2d}] {name}")
        else:
            print("   No trend variable names available")
        print(f"   Trend draws shape: {results.trend_draws.shape if results.trend_draws is not None else 'None'}")
    else:
        print("📊 TREND VARIABLES: None available")
    
    if hasattr(results, 'stationary_draws') and results.stationary_draws is not None:
        print(f"\n📊 STATIONARY VARIABLES ({len(results.stationary_names) if results.stationary_names else 0}):")
        if results.stationary_names:
            for i, name in enumerate(results.stationary_names):
                print(f"   [{i:2d}] {name}")
        else:
            print("   No stationary variable names available")
        print(f"   Stationary draws shape: {results.stationary_draws.shape if results.stationary_draws is not None else 'None'}")
    else:
        print("📊 STATIONARY VARIABLES: None available")
    
    if hasattr(results, 'observed_data') and results.observed_data is not None:
        print(f"\n📊 OBSERVED VARIABLES ({len(results.observed_variable_names) if results.observed_variable_names else 0}):")
        if results.observed_variable_names:
            for i, name in enumerate(results.observed_variable_names):
                print(f"   [{i:2d}] {name}")
        else:
            print("   No observed variable names available")
        print(f"   Observed data shape: {results.observed_data.shape if results.observed_data is not None else 'None'}")
    else:
        print("📊 OBSERVED VARIABLES: None available")
    
    # 2. Check custom plot specs
    print("\n\n2. CUSTOM PLOT SPECIFICATIONS ANALYSIS:")
    print("-" * 50)
    
    if not custom_plot_specs:
        print("❌ No custom plot specifications provided")
        return
    
    # Collect all available variable names
    all_trend_names = set(results.trend_names) if results.trend_names else set()
    all_stationary_names = set(results.stationary_names) if results.stationary_names else set()
    all_observed_names = set(results.observed_variable_names) if results.observed_variable_names else set()
    
    for i, plot_spec in enumerate(custom_plot_specs):
        print(f"\n🎨 CUSTOM PLOT {i+1}: '{plot_spec.get('title', 'Untitled')}'")
        
        series_specs = plot_spec.get('series_to_plot', [])
        if not series_specs:
            print("   ⚠️  No series specified for this plot")
            continue
            
        for j, series in enumerate(series_specs):
            series_type = series.get('type', 'unknown')
            series_name = series.get('name', 'unnamed')
            series_label = series.get('label', 'unlabeled')
            
            print(f"   Series {j+1}: {series_label}")
            print(f"     Type: {series_type}")
            print(f"     Name: {series_name}")
            
            # Check if variable exists
            if series_type == 'trend':
                if series_name in all_trend_names:
                    print(f"     ✅ FOUND in trend variables")
                else:
                    print(f"     ❌ NOT FOUND in trend variables")
                    print(f"        Available trend names: {sorted(all_trend_names)}")
            elif series_type == 'stationary':
                if series_name in all_stationary_names:
                    print(f"     ✅ FOUND in stationary variables")
                else:
                    print(f"     ❌ NOT FOUND in stationary variables")
                    print(f"        Available stationary names: {sorted(all_stationary_names)}")
            elif series_type == 'observed':
                if series_name in all_observed_names:
                    print(f"     ✅ FOUND in observed variables")
                else:
                    print(f"     ❌ NOT FOUND in observed variables")
                    print(f"        Available observed names: {sorted(all_observed_names)}")
            else:
                print(f"     ❓ UNKNOWN TYPE: {series_type}")



# Add this debug to your main script to confirm the parser issue

def debug_parser_order_preservation():
    """
    Debug whether the parser is preserving the GPM file order.
    """
    print("\n" + "="*80)
    print("PARSER ORDER PRESERVATION DEBUG")
    print("="*80)
    
    print("GPM FILE DECLARED ORDER:")
    expected_gpm_order = [
        'r_US_idio_trend', 'r_EA_idio_trend', 'r_JP_idio_trend',
        'pi_US_idio_trend', 'pi_EA_idio_trend', 'pi_JP_idio_trend', 
        'y_US_trend', 'y_EA_trend', 'y_JP_trend',
        'rr_US_full_trend', 'pi_US_full_trend', 'R_US_short_trend',
        'rr_EA_full_trend', 'pi_EA_full_trend', 'R_EA_short_trend',
        'rr_JP_full_trend', 'pi_JP_full_trend', 'R_JP_short_trend'
    ]
    
    for i, var in enumerate(expected_gmp_order):
        print(f"  {i:2d}: {var}")
    
    print("\nACTUAL PARSER OUTPUT ORDER:")
    # This will come from your results
    print("  (Add this debug print to see what the parser actually produces)")
    
    print("\nTO FIX: Add this debug print to gmp_model_parser.py:")
    print("In the parse() method, after parsing trends_vars, add:")
    print("print(f'PARSER trends_vars order: {self.model_data[\"gmp_trend_variables_original\"]}')")

# The core issue is likely in the GPM parser's _parse_trends_vars method
# Let's check how it parses the trends_vars block

def identify_parser_bug():
    """
    Identify the specific bug in the parser.
    """
    print("\n" + "="*60)
    print("PARSER BUG IDENTIFICATION")
    print("="*60)
    
    print("The issue is likely in gmp_model_parser.py in one of these methods:")
    print("1. _parse_trends_vars() - might not preserve order when parsing")
    print("2. _identify_core_variables() - might reorder variables")
    print()
    
    print("LIKELY BUG in _identify_core_variables():")
    print("The method does this:")
    print("1. Creates a SET of core variables (sets don't preserve order!)")
    print("2. Then tries to rebuild order from gmp_trend_variables_original")
    print("3. But if gmp_trend_variables_original is wrong, the order is wrong")
    print()
    
    print("The fix is to ensure _parse_trends_vars() preserves the EXACT order from the GPM file.")

# Here's the likely fix for the parser
def proposed_parser_fix():
    """
    Show the proposed fix for the parser.
    """
    print("\n" + "="*60)
    print("PROPOSED PARSER FIX")
    print("="*60)
    
    print("In gmp_model_parser.py, in the _parse_trends_vars() method:")
    print()
    print("CURRENT CODE (probably wrong):")
    print("```python")
    print("# Probably something like:")
    print("trends_vars = []")
    print("for line in block_lines:")
    print("    vars_in_line = line.split(',')")
    print("    for var in vars_in_line:")
    print("        var_clean = var.strip().rstrip(';').strip()")
    print("        if var_clean and not var_clean.startswith('//'):")
    print("            trends_vars.append(var_clean)")
    print("```")
    print()
    print("FIXED CODE should be:")
    print("```python") 
    print("trends_vars = []")
    print("for line in block_lines:")
    print("    # Remove comments first")
    print("    line_no_comments = line.split('//')[0].strip()")
    print("    if not line_no_comments:")
    print("        continue")
    print("    ")
    print("    # Split by comma and process each variable")
    print("    vars_in_line = line_no_comments.split(',')")
    print("    for var in vars_in_line:")
    print("        var_clean = var.strip().rstrip(';').strip()")
    print("        if var_clean:")
    print("            trends_vars.append(var_clean)")
    print("            print(f'PARSER: Adding trend var {len(trends_vars)}: {var_clean}')")
    print("```")
    print()
    print("The key is to preserve the EXACT order as declared in the GPM file.")

# Add this to your main function
def main():
    # ... existing code ...
    
    if results_fixed:
        # ... existing diagnostics ...
        
        # ADD THESE NEW DIAGNOSTICS:
        debug_parser_order_preservation()
        identify_parser_bug()
        proposed_parser_fix()
        
        print("\n" + "="*80)
        print("IMMEDIATE ACTION REQUIRED")
        print("="*80)
        print("1. Find gmp_model_parser.py in your codebase")
        print("2. Add debug print to see what order the parser creates")
        print("3. Find the _parse_trends_vars() method") 
        print("4. Check if it's preserving the GPM file order correctly")
        print("5. If not, fix it to preserve the exact order")
        print()
        print("This will fix the root cause without any hardcoding!")

# Critical debugging step
def add_parser_debug_instructions():
    """
    Exact instructions for adding the parser debug.
    """
    print("\n" + "="*80)
    print("EXACT DEBUG INSTRUCTIONS")
    print("="*80)
    
    print("STEP 1: Find gmp_model_parser.py")
    print("STEP 2: In the parse() method, find where it calls _parse_trends_vars()")
    print("STEP 3: Right after that call, add this line:")
    print("print(f'PARSER RESULT: gmp_trend_variables_original = {self.model_data.get(\"gmp_trend_variables_original\", [])}')")
    print()
    print("STEP 4: Run your script and check if the parser output matches:")
    expected = [
        'r_US_idio_trend', 'r_EA_idio_trend', 'r_JP_idio_trend',
        'pi_US_idio_trend', 'pi_EA_idio_trend', 'pi_JP_idio_trend',
        'y_US_trend', 'y_EA_trend', 'y_JP_trend', 
        'rr_US_full_trend', 'pi_US_full_trend', 'R_US_short_trend',
        'rr_EA_full_trend', 'pi_EA_full_trend', 'R_EA_short_trend',
        'rr_JP_full_trend', 'pi_JP_full_trend', 'R_JP_short_trend'
    ]
    print("Expected order:", expected)
    print()
    print("If the parser output doesn't match this exactly, that's your bug!")


def debug_data_content_vs_names(results):
    """
    Debug whether the data content matches the variable names.
    The names are correct, but the data might be scrambled.
    """
    print("\n" + "="*80)
    print("DATA CONTENT vs NAMES MISMATCH DIAGNOSTIC")
    print("="*80)
    
    print("HYPOTHESIS: Variable names are correct, but wrong data is assigned to them.")
    print()
    
    # Test: Try to match each trend variable with observed data to see what it actually contains
    print("MATCHING TREND VARIABLES WITH OBSERVED DATA:")
    print("-" * 50)
    
    # For each trend variable, find which observed variable it correlates best with
    for trend_idx, trend_name in enumerate(results.trend_names):
        trend_data = np.median(results.trend_draws[:, :, trend_idx], axis=0)
        
        best_corr = -1
        best_obs = None
        correlations = {}
        
        for obs_idx, obs_name in enumerate(results.observed_variable_names):
            obs_data = results.observed_data[:, obs_idx]
            corr = np.corrcoef(obs_data, trend_data)[0, 1]
            correlations[obs_name] = corr
            
            if corr > best_corr:
                best_corr = corr
                best_obs = obs_name
        
        print(f"  {trend_name:<20} -> best match: {best_obs:<8} (corr={best_corr:.3f})")
        
        # Check if this is the expected pairing
        expected_pairs = {
            'y_US_trend': 'y_us', 'y_EA_trend': 'y_ea', 'y_JP_trend': 'y_jp',
            'pi_US_full_trend': 'pi_us', 'pi_EA_full_trend': 'pi_ea', 'pi_JP_full_trend': 'pi_jp',
            'R_US_short_trend': 'r_us', 'R_EA_short_trend': 'r_ea', 'R_JP_short_trend': 'r_jp'
        }
        
        if trend_name in expected_pairs:
            expected_obs = expected_pairs[trend_name]
            if best_obs == expected_obs:
                print(f"    ✅ CORRECT: {trend_name} matches {expected_obs}")
            else:
                print(f"    ❌ WRONG: {trend_name} should match {expected_obs}, but matches {best_obs}")
                print(f"    🔍 All correlations: {correlations}")
    
    print("\nSUMMARY OF MISMATCHES:")
    print("-" * 30)
    
    mismatches = []
    expected_pairs = {
        'y_US_trend': 'y_us', 'y_EA_trend': 'y_ea', 'y_JP_trend': 'y_jp',
        'pi_US_full_trend': 'pi_us', 'pi_EA_full_trend': 'pi_ea', 'pi_JP_full_trend': 'pi_jp', 
        'R_US_short_trend': 'r_us', 'R_EA_short_trend': 'r_ea', 'R_JP_short_trend': 'r_jp'
    }
    
    for trend_idx, trend_name in enumerate(results.trend_names):
        if trend_name in expected_pairs:
            trend_data = np.median(results.trend_draws[:, :, trend_idx], axis=0)
            expected_obs = expected_pairs[trend_name]
            
            # Find the expected observed data
            if expected_obs in results.observed_variable_names:
                obs_idx = results.observed_variable_names.index(expected_obs)
                obs_data = results.observed_data[:, obs_idx]
                expected_corr = np.corrcoef(obs_data, trend_data)[0, 1]
                
                # Find what it actually matches best
                best_corr = -1
                best_obs = None
                for obs_idx2, obs_name2 in enumerate(results.observed_variable_names):
                    obs_data2 = results.observed_data[:, obs_idx2]
                    corr = np.corrcoef(obs_data2, trend_data)[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_obs = obs_name2
                
                if best_obs != expected_obs and abs(best_corr - expected_corr) > 0.1:
                    mismatches.append({
                        'trend_name': trend_name,
                        'expected_obs': expected_obs,
                        'actual_best_obs': best_obs,
                        'expected_corr': expected_corr,
                        'actual_corr': best_corr
                    })
    
    if mismatches:
        print("🚨 CONFIRMED MISMATCHES:")
        for mismatch in mismatches:
            print(f"  {mismatch['trend_name']} should correlate with {mismatch['expected_obs']} (corr={mismatch['expected_corr']:.3f})")
            print(f"    but actually correlates best with {mismatch['actual_best_obs']} (corr={mismatch['actual_corr']:.3f})")
        
        print(f"\nRoot cause: Data reconstruction is putting wrong time series into correctly named variables.")
        
    else:
        print("✅ No significant mismatches found. Data content matches variable names.")

def debug_core_state_extraction(results):
    """
    Debug if the core state extraction is working correctly.
    """
    print("\n" + "="*60)
    print("CORE STATE EXTRACTION DEBUG")
    print("="*60)
    
    print("The issue might be in how core states are extracted in variable reconstruction.")
    print()
    print("From your StateSpaceBuilder output:")
    print("Core var map (CANONICAL ORDER): {'r_US_idio_trend': 0, 'r_EA_idio_trend': 1, ...}")
    print()
    print("This means:")
    print("  State index 0 should contain r_US_idio_trend data")
    print("  State index 1 should contain r_EA_idio_trend data")
    print("  State index 6 should contain y_US_trend data")
    print("  etc.")
    print()
    print("But if the reconstruction is extracting from wrong state indices,")
    print("then wrong data gets assigned to correct names.")
    print()
    print("TO DEBUG: Add this to variable_reconstruction.py:")
    print("print(f'Extracting {var_name} from state index {state_idx}')")
    print("print(f'Data sample: {core_states_draw[:5, state_idx]}')")

def identify_reconstruction_bug_location():
    """
    Identify where in the reconstruction the bug occurs.
    """
    print("\n" + "="*60)
    print("RECONSTRUCTION BUG LOCATION")
    print("="*60)
    
    print("Since variable names are correct but data is wrong, the bug is in:")
    print()
    print("OPTION A: _reconstruct_original_variables() function")
    print("  - Wrong extraction from core_states_draw array")
    print("  - Wrong use of core_var_map indices")
    print()
    print("OPTION B: State space simulation itself")
    print("  - Wrong data in the core_states_draw array from the start")
    print("  - State space matrices have wrong structure")
    print()
    print("OPTION C: Final assembly in SmootherResults")
    print("  - Correct reconstruction but wrong stacking order")
    print()
    print("Most likely: OPTION A - wrong extraction in reconstruction")


def split_data_for_presample(data, split_ratio=0.15, method='first'):
    """Split data into pre-sample and main sample."""
    n_total = len(data)
    n_presample = int(n_total * split_ratio)
    n_main = n_total - n_presample
    
    print(f"\n=== DATA SPLITTING ===")
    print(f"Total observations: {n_total}")
    print(f"Pre-sample size: {n_presample} ({split_ratio*100:.1f}%)")
    print(f"Main sample size: {n_main} ({(1-split_ratio)*100:.1f}%)")
    
    if method == 'first':
        presample_data = data.iloc[:n_presample].copy()
        main_data = data.iloc[n_presample:].copy()
        print(f"Pre-sample period: {presample_data.index[0]} to {presample_data.index[-1]}")
        print(f"Main sample period: {main_data.index[0]} to {main_data.index[-1]}")
        
    elif method == 'last':
        presample_data = data.iloc[-n_presample:].copy()
        main_data = data.iloc[:-n_presample].copy()
        print(f"Pre-sample period: {presample_data.index[0]} to {presample_data.index[-1]}")
        print(f"Main sample period: {main_data.index[0]} to {main_data.index[-1]}")
        
    elif method == 'random':
        np.random.seed(42)
        presample_indices = np.random.choice(n_total, n_presample, replace=False)
        main_indices = np.setdiff1d(np.arange(n_total), presample_indices)
        
        presample_data = data.iloc[presample_indices].copy()
        main_data = data.iloc[main_indices].copy()
        print(f"Random split - Pre-sample: {n_presample} observations")
        print(f"Random split - Main sample: {n_main} observations")
    
    split_info = {
        'method': method,
        'split_ratio': split_ratio,
        'n_presample': n_presample,
        'n_main': n_main,
        'presample_period': (presample_data.index[0], presample_data.index[-1]),
        'main_period': (main_data.index[0], main_data.index[-1])
    }
    
    return presample_data, main_data, split_info


def main():
    # --- Data Loading ---
    # Assuming data_m5.csv is in the same directory as this script or a 'data' subdirectory
    data_file_name = "data_m5.csv"
    data_file_path = os.path.join(SCRIPT_DIR, data_file_name) 
    if not os.path.exists(data_file_path):
        data_file_path = os.path.join(SCRIPT_DIR, "data", data_file_name) # Try a 'data' subdirectory
        if not os.path.exists(data_file_path):
            print(f"FATAL ERROR: Data file {data_file_name} not found in {SCRIPT_DIR} or {os.path.join(SCRIPT_DIR, 'data')}")
            sys.exit(1)
    
    print(f"Loading data from: {data_file_path}")
    dta = pd.read_csv(data_file_path)
    dta['Date'] = pd.to_datetime(dta['Date'])
    dta.set_index('Date', inplace=True)
    dta = dta.asfreq('QE') # Ensure quarterly frequency

    # --- Model Configuration ---
    # These must match the 'varobs' block in your GPM file
    observed_vars_model = [
        'y_us', 'y_ea', 'y_jp',
        'pi_us', 'pi_ea', 'pi_jp',
        'r_us', 'r_ea', 'r_jp'
    ]
    data_sub = dta[observed_vars_model].copy()
    data_sub = data_sub.dropna() 
   
    print(f"Data shape after selecting observed variables and dropping NaNs: {data_sub.shape}")
    if data_sub.empty:
        print("FATAL ERROR: Data is empty after processing. Check observed_vars_model and data content.")
        sys.exit(1)


    ## Split the data 
    presample_data, main_data, split_info = split_data_for_presample(data_sub, split_ratio=0.15)

    gpm_file_name = 'gpm_y_pi_rshort_simple_rw.gpm'
    # Assuming the GPM file is in ../clean_gpm_bvar_trends/models/ relative to this script
    gpm_file_path = os.path.join(PROJECT_ROOT, 'clean_gpm_bvar_trends', 'models', gpm_file_name)

    if not os.path.exists(gpm_file_path):
        print(f"FATAL ERROR: GPM file {gpm_file_path} not found.")
        sys.exit(1)

    output_base_dir = os.path.join(SCRIPT_DIR, "estimation_results_super_simple")
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Custom Plot Specifications for the Factor Model ---
    custom_plot_specs_factor_model = [
        {
            "title": "US Output vs. Trend & Real Rate",
            "series_to_plot": [
                {'type': 'observed', 'name': 'y_us', 'label': 'Observed Output (y_us)', 'style': '-'},
                {'type': 'trend', 'name': 'y_US_trend', 'label': 'Output Trend (y_US_trend)', 'show_hdi': True, 'color': 'blue'},
                #{'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'Real Rate Trend (rr_US_full_trend)', 'show_hdi': True, 'color': 'green', 'style': '--'}
            ]
        },
        {
            "title": "US Inflation vs. Trend Components",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_us', 'label': 'Observed Inflation (pi_us)', 'style': '-'},
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Full Inflation Trend (pi_US_full_trend)', 'show_hdi': True, 'color': 'red'},
                #{'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend (pi_w_trend)', 'show_hdi': True, 'color': 'magenta', 'style': ':'},                
            ]
        },
        {
            "title": "US Inflation vs. Trend Components Idio",
            "series_to_plot": [
                {'type': 'observed', 'name': 'pi_us', 'label': 'Observed Short Rate (pi_us)', 'style': '-'},
                {'type': 'trend', 'name': 'pi_US_full_trend', 'label': 'Pi full (pi_US_full_trend)', 'show_hdi': True, 'color': 'orange'},
                {'type': 'trend', 'name': 'pi_US_idio_trend', 'label': 'Pi Idio (pi_US_idio_trend)', 'show_hdi': True, 'color': 'grey', 'style': ':'},
            ]
        },
        # #EA
        # {
        #     "title": "EA Output vs. Trend & Real Rate",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'y_ea', 'label': 'Observed Output (y_ea)', 'style': '-'},
        #         {'type': 'trend', 'name': 'y_EA_trend', 'label': 'Output Trend (y_EA_trend)', 'show_hdi': True, 'color': 'blue'},
        #         {'type': 'trend', 'name': 'rr_EA_full_trend', 'label': 'Real Rate Trend (rr_EA_full_trend)', 'show_hdi': True, 'color': 'green', 'style': '--'}
        #     ]
        # },
        # {
        #     "title": "EA Inflation vs. Trend Components",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'pi_ea', 'label': 'Observed Inflation (pi_ea)', 'style': '-'},
        #         {'type': 'trend', 'name': 'pi_EA_full_trend', 'label': 'Full Inflation Trend (pi_EA_full_trend)', 'show_hdi': True, 'color': 'red'},
        #         #{'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend (pi_w_trend)', 'show_hdi': True, 'color': 'magenta', 'style': ':'},             
        #     ]
        # },
        # {
        #     "title": "EA Short Rate vs. Trend Components",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'r_ea', 'label': 'Observed Short Rate (r_ea)', 'style': '-'},
        #         {'type': 'trend', 'name': 'R_EA_short_trend', 'label': 'Nominal Short Rate Trend (R_EA_short_trend)', 'show_hdi': True, 'color': 'orange'},
        #         {'type': 'trend', 'name': 'rr_EA_full_trend', 'label': 'Real Rate Trend (rr_EA_full_trend)', 'show_hdi': True, 'color': 'grey', 'style': ':'},
        #     ]
        # },
        # #JAPAN
        # {
        #     "title": "Japan Output vs. Trend & Real Rate",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'y_jp', 'label': 'Observed Output (y_jp)', 'style': '-'},
        #         {'type': 'trend', 'name': 'y_JP_trend', 'label': 'Output Trend (y_JP_trend)', 'show_hdi': True, 'color': 'blue'},
        #         {'type': 'trend', 'name': 'rr_JP_full_trend', 'label': 'Real Rate Trend (rr_JP_full_trend)', 'show_hdi': True, 'color': 'green', 'style': '--'}
        #     ]
        # },
        # {
        #     "title": "JP Inflation vs. Trend Components",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'pi_jp', 'label': 'Observed Inflation (pi_jp)', 'style': '-'},
        #         {'type': 'trend', 'name': 'pi_JP_full_trend', 'label': 'Full Inflation Trend (pi_JP_full_trend)', 'show_hdi': True, 'color': 'red'},
        #         #{'type': 'trend', 'name': 'pi_w_trend', 'label': 'World Inflation Trend (pi_w_trend)', 'show_hdi': True, 'color': 'magenta', 'style': ':'},            
        #     ]
        # },
        # {
        #     "title": "JP Short Rate vs. Trend Components",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'r_jp', 'label': 'Observed Short Rate (r_jp)', 'style': '-'},
        #         {'type': 'trend', 'name': 'R_JP_short_trend', 'label': 'Nominal Short Rate Trend (R_JP_short_trend)', 'show_hdi': True, 'color': 'orange'},
        #         {'type': 'trend', 'name': 'rr_JP_full_trend', 'label': 'Real Rate Trend (rr_JP_full_trend)', 'show_hdi': True, 'color': 'grey', 'style': ':'},
        #     ]
        # },
        # {
        #     "title": "US Real Rate Full Trend Decomposition",
        #     "series_to_plot": [
        #         {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'US Full Real Rate Trend', 'show_hdi': True, 'color': 'blue'},
        #         {'type': 'trend', 'name': 'r_w_trend', 'label': 'World Real Rate Trend', 'show_hdi': True, 'color': 'green', 'style': ':'},
        #         #{'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Deviation Trend', 'show_hdi': True, 'color': 'purple', 'style': '-.'}
        #     ]
        # },
        # {
        #     "title": "US Real Rate Deviation Trend Decomposition",
        #     "series_to_plot": [
        #         {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Deviation Trend', 'show_hdi': True, 'color': 'purple'},
        #         {'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (r_devs)', 'show_hdi': True, 'color': 'black', 'style': '--'},
        #         {'type': 'trend', 'name': 'r_US_idio_trend', 'label': 'US Idiosyncratic Trend', 'show_hdi': True, 'color': 'brown', 'style': ':'}
        #     ]
        # },
        # {
        #     "title": "Nominal and Real Rate (US)",
        #     "series_to_plot": [
        #         {'type': 'observed', 'name': 'r_us', 'label': 'US Real Rate', 'show_hdi': True, 'color': 'red'},
        #         {'type': 'trend', 'name': 'r_US_full_trend', 'label': 'Nominal US Rate trend', 'show_hdi': True, 'color': 'blue'},
        #         {'type': 'trend', 'name': 'pi_JP_full_trend', 'label': 'Inflation trens', 'show_hdi': True, 'color': 'green'},
        #         {'type': 'trend', 'name': 'rr_US_full_trend', 'label': 'Real Rate trend', 'show_hdi': True, 'color': 'black', 'style': '--'}
        #     ]
        # }
        # Add similar plots for EA and JP if desired
        # {
        #     "title": "Comparison of Real Rate Deviation Trends (US, EA, Factor)",
        #     "series_to_plot": [
        #         {'type': 'trend', 'name': 'r_US_dev_trend', 'label': 'US Real Rate Dev Trend', 'show_hdi': True, 'color': 'blue'},
        #         {'type': 'trend', 'name': 'r_EA_dev_trend', 'label': 'EA Real Rate Dev Trend', 'show_hdi': True, 'color': 'green'},
        #         #{'type': 'trend', 'name': 'factor_r_devs', 'label': 'Common Factor (r_devs)', 'show_hdi': True, 'color': 'black', 'style': '--'}
        #     ]
        # }
    ]

    run_mcmc = False
    if run_mcmc:
        # --- MCMC Estimation ---
        print(f"\n--- Scenario 1: MCMC Estimation with Custom P0 Scales ---")
        mcmc_output_dir = os.path.join(output_base_dir, "mcmc_estimation_factor_model")
        os.makedirs(mcmc_output_dir, exist_ok=True)

        results_mcmc = run_complete_gpm_analysis(
            data=main_data.copy(),
            gpm_file=gpm_file_path,
            analysis_type="mcmc",
            num_warmup=50,  # Adjust for real runs
            num_samples=5, # Adjust for real runs
            num_chains=2,    
            target_accept_prob=0.85,
            use_gamma_init=True, # Ensure Gamma P0 for stationary components
            gamma_scale_factor=1.0, 
            num_extract_draws=10, # Number of draws for smoother from MCMC posterior
            generate_plots=True, 
            hdi_prob_plot=0.68,
            show_plot_info_boxes=False,
            custom_plot_specs=custom_plot_specs_factor_model, 
            plot_save_path=mcmc_output_dir, # Save plots in the specific MCMC output directory
            save_plots=True,
            variable_names_override=observed_vars_model, # From data loading
            data_file_source_for_summary=data_file_path,
            # P0 Overrides for MCMC estimation phase
            #mcmc_trend_P0_scales={"pi_w_trend": 1e-3}, # Example: specific scales for world trends
            # mcmc_trend_P0_scavolatilityscales={"pi_w_trend": 1e-3}, 
            smoother_stationary_P0_scale=1.0
        )

        if results_mcmc:
            print(f"\nMCMC Workflow for {gpm_file_name} successfully completed!")
            # Access results, e.g., results_mcmc.trend_draws, results_mcmc.stationary_draws
        else:
            print(f"\nMCMC Workflow for {gpm_file_name} failed.")

        
    if not run_mcmc:
        # --- Fixed Parameter Estimation ---
        print(f"\n--- Scenario 2: Fixed Parameter Estimation ---")
        fixed_params_output_dir = os.path.join(output_base_dir, "fixed_param_simple_model")
        os.makedirs(fixed_params_output_dir, exist_ok=True)

        # Define the fixed parameter values. These MUST match parameter names in the GPM.
        # And for shocks, use the direct shock name (e.g., "shk_r_w") for std. dev.
        # or provide _var_coefficients, _var_innovation_corr_chol directly.
        # fixed_parameter_values = {
        #     'var_phi': 2.0, 
            
        #     # Shock standard deviations (these are what _resolve_parameter_value expects)
        #     'shk_r_w': 0.5, 
        #     'shk_pi_w': 0.3,
            
        #     'shk_r_US_idio': 0.2, 
        #     'shk_pi_US_idio': 0.2,
        #     'shk_r_EA_idio': 0.2,
        #     'shk_pi_EA_idio': 0.2,
        #     'shk_r_JP_idio': 0.2, 
        #     'shk_pi_JP_idio': 0.2,
        #     'shk_y_US': 0.1, 
        #     'shk_y_EA': 0.2, 
        #     'shk_y_JP': 0.1,
        #     # For VAR cycles, you can provide _var_coefficients and _var_innovation_corr_chol
        #     # or let them default based on var_prior_setup in GPM and shock std devs below.
        #     # If providing _var_coefficients:
        #     # num_stat_vars = 9 (cycle_Y_US, ..., cycle_Rshort_JP)
        #     # A_example = np.eye(num_stat_vars) * 0.8 
        #     # A_example = A_example.reshape(1, num_stat_vars, num_stat_vars) # Assuming var_order = 1
        #     # '_var_coefficients': jax.numpy.array(A_example),
        #     #'_var_innovation_corr_chol': jax.numpy.eye(num_stat_vars),
        #     # If relying on individual shock std devs for VAR:
        #     'shk_cycle_Y_US': 1.005, 'shk_cycle_PI_US': 1.003, 'shk_cycle_Rshort_US': 1.002,
        #     'shk_cycle_Y_EA': 1.005, 'shk_cycle_PI_EA': 1.003, 'shk_cycle_Rshort_EA': 1.002,
        #     'shk_cycle_Y_JP': 1.005, 'shk_cycle_PI_JP': 1.003, 'shk_cycle_Rshort_JP': 1.002,
        # }
        fixed_parameter_values = {}
        results_fixed = run_complete_gpm_analysis(
            data=data_sub.copy(),
            gpm_file=gpm_file_path,
            analysis_type="fixed_params", # Specify fixed parameter analysis
            param_values=fixed_parameter_values,
            num_sim_draws=10, # Number of draws for smoother with fixed params
            plot_results=True,
            plot_default_observed_vs_trend_components=True, # Plot default OvT plots
            custom_plot_specs=custom_plot_specs_factor_model,
            variable_names=observed_vars_model, # From data loading
            use_gamma_init_for_test=True, # Ensure Gamma P0 for stationary components
            gamma_init_scaling=1.0,
            hdi_prob=0.68, # HDI for plots from fixed param simulation draws
            trend_P0_var_scale=0.01, # P0 scale for trend components in fixed param eval
            var_P0_var_scale=1.0,  # P0 scale for VAR components in fixed param eval
            save_plots_path_prefix=os.path.join(fixed_params_output_dir, "fixed_eval_plot"), # Path prefix for saving plots
            show_plot_info_boxes=False,
            # initial_state_prior_overrides can be added here if needed
        )

        if results_fixed:
            print(f"\nFixed Parameter Workflow for {gpm_file_name} successfully completed!")
            print(f"  Log-likelihood: {results_fixed.log_likelihood if results_fixed.log_likelihood is not None else 'N/A'}")
            
            # Existing diagnostic
            diagnose_variable_names(results_fixed, custom_plot_specs_factor_model)
            
            # ADD THIS NEW DIAGNOSTIC:
            debug_variable_reconstruction(results_fixed)
            
            # ADD THESE NEW DIAGNOSTICS:
            debug_data_content_vs_names(results_fixed)
            debug_core_state_extraction(results_fixed)
            identify_reconstruction_bug_location()            
            add_parser_debug_instructions()
            print(f"\nFixed Parameter Workflow for {gpm_file_name} successfully completed!")
            print(f"  Log-likelihood: {results_fixed.log_likelihood if results_fixed.log_likelihood is not None else 'N/A'}")
        else:
            print(f"\nFixed Parameter Workflow for {gpm_file_name} failed.")

if __name__ == "__main__":
    main()


