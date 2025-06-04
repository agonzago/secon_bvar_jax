import sys
import os

# --- Path Setup ---
SCRIPT_DIR_CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ORCH = os.path.abspath(os.path.join(SCRIPT_DIR_CURRENT_FILE, '..'))
if PROJECT_ROOT_ORCH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_ORCH)

# Standard library imports
import subprocess
import json
import re
import time
import shutil
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import argparse

# --- JAX Configuration ---
from clean_gpm_bvar_trends.jax_config import configure_jax
configure_jax()

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: google.generativeai not available. AI calls will be simulated.")
    GEMINI_AVAILABLE = False

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = 'gemini-2.0-flash-exp'

if not GOOGLE_API_KEY or not GEMINI_AVAILABLE:
    print("WARNING: GOOGLE_API_KEY not set or Gemini not available. AI calls will be SIMULATED.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"✓ Using Gemini model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Could not configure Gemini API: {e}. AI calls will be SIMULATED.")
        gemini_model = None

# Paths
BASE_GPM_FILE_PATH = os.path.join(PROJECT_ROOT_ORCH, "Application", "Models", "gpm_factor_y_pi_rshort.gpm")
MODEL_RUNNER_SCRIPT = os.path.join(PROJECT_ROOT_ORCH, "Application", "main_custom_plots.py")
DATA_FILE_PATH = os.path.join(PROJECT_ROOT_ORCH, "Application", "data_m5.csv")

AUTONOMOUS_RUN_BASE_DIR = os.path.join(PROJECT_ROOT_ORCH, "Application", "autonomous_gpm_calibration_output")
os.makedirs(AUTONOMOUS_RUN_BASE_DIR, exist_ok=True)

PRIOR_STATE_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "current_priors_config.json")
CALIBRATION_HISTORY_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "calibration_history.json")

# --- Prior Definition and GPM Modification ---
def get_initial_tunable_priors() -> Dict[str, Dict]:
    """
    Defines the set of prior hyperparameters to be tuned.
    Updated for new codebase structure.
    """
    return {
        # === STRUCTURAL PARAMETERS ===
        "var_phi_US_mean": {
            "gpm_keyword_for_line": "var_phi_US", "gpm_param_name": "var_phi_US",
            "dist_type": "normal_pdf", "hyper_name": "mean", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 0.5, "max_val": 4.0,
            "description": "Mean for EIS parameter var_phi_US (Normal prior)."
        },
        "var_phi_US_std": {
            "gpm_keyword_for_line": "var_phi_US", "gpm_param_name": "var_phi_US",
            "dist_type": "normal_pdf", "hyper_name": "std", "value_index_in_gpm_line": 1,
            "current_value": 0.5, "min_val": 0.1, "max_val": 1.5,
            "description": "Std dev for EIS parameter var_phi_US (Normal prior)."
        },
        "lambda_pi_US_mean": {
            "gpm_keyword_for_line": "lambda_pi_US", "gpm_param_name": "lambda_pi_US",
            "dist_type": "normal_pdf", "hyper_name": "mean", "value_index_in_gpm_line": 0,
            "current_value": 1.0, "min_val": 0.3, "max_val": 1.5,
            "description": "Mean for US loading on world inflation trend (Normal prior)."
        },

        # === SHOCK STANDARD DEVIATIONS ===
        "stderr_shk_r_w_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_w",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_w."
        },
        "stderr_shk_r_w_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_w",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.001, "max_val": 0.05,
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_w."
        },
        
        # Add similar patterns for other shocks...
        "stderr_shk_r_US_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_US_idio."
        },
        "stderr_shk_r_US_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_US_idio."
        },

        # Continue pattern for other countries and variables...
        "stderr_shk_y_US_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_US",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_y_US."
        },
        "stderr_shk_y_US_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_US",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.002, "max_val": 0.05,
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_y_US."
        },

        # === VAR PRIOR SETUP ===
        "var_es_diag": {
            "gpm_keyword_for_line": "es", "gpm_param_name": "es",
            "dist_type": "var_prior_setup", "hyper_name": "diag", "value_index_in_gpm_line": 0,
            "current_value": 0.6, "min_val": 0.1, "max_val": 0.9,
            "description": "VAR diagonal coefficient prior mean."
        },
        "var_fs_diag": {
            "gpm_keyword_for_line": "fs", "gpm_param_name": "fs",
            "dist_type": "var_prior_setup", "hyper_name": "diag", "value_index_in_gpm_line": 0,
            "current_value": 0.15, "min_val": 0.05, "max_val": 0.5,
            "description": "VAR diagonal coefficient prior std dev."
        },
        "var_eta": {
            "gpm_keyword_for_line": "eta", "gpm_param_name": "eta",
            "dist_type": "var_prior_setup", "hyper_name": "scalar", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.1, "max_val": 10.0,
            "description": "LKJ concentration parameter for VAR innovation correlations."
        }
    }

def modify_gpm_file_content(base_gpm_content: str, current_priors_config: Dict[str, Dict]) -> str:
    """
    Modifies GPM file content based on current prior configuration.
    Updated for new codebase structure.
    """
    modified_content = base_gpm_content
    line_updates: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}

    # Group updates by GPM line
    for _tunable_key, spec in current_priors_config.items():
        key = (spec["gpm_keyword_for_line"], spec["gpm_param_name"], spec["dist_type"])
        if key not in line_updates:
            line_updates[key] = []
        line_updates[key].append((spec["value_index_in_gpm_line"], spec["current_value"]))

    # Apply updates
    for (gpm_keyword, gpm_name, dist_type), updates in line_updates.items():
        updates.sort()
        new_values_str_list = [f"{val:.6g}" for _idx, val in updates]

        if dist_type in ["inv_gamma_pdf", "normal_pdf"]:
            # Handle estimated_params section
            base_pattern_str = re.escape(gpm_keyword)
            if gpm_keyword.lower() == "stderr":
                base_pattern_str += r"\s+" + re.escape(gpm_name)
            
            pattern_str = rf"({base_pattern_str}\s*,\s*{re.escape(dist_type)}\s*,\s*)([\d\.\s,eE+-]+)(\s*;)"
            pattern = re.compile(pattern_str)
            replacer = lambda m: m.group(1) + ", ".join(new_values_str_list) + m.group(3)
            modified_content, num_subs = pattern.subn(replacer, modified_content)
            
            if num_subs == 0:
                print(f"Warning: No substitution for {gpm_keyword} {gpm_name} {dist_type}")
                
        elif dist_type == "var_prior_setup":
            # Handle var_prior_setup section
            base_pattern_str = re.escape(gpm_keyword)
            pattern_str = rf"({base_pattern_str}\s*=\s*)([\d\.\s,eE+-]+)(\s*;)"
            pattern = re.compile(pattern_str)
            replacer_var_setup = lambda m: m.group(1) + ", ".join(new_values_str_list) + m.group(3)
            modified_content, num_subs = pattern.subn(replacer_var_setup, modified_content)
            
            if num_subs == 0:
                print(f"Warning: No substitution for var_prior_setup '{gpm_keyword}'")

    return modified_content

def run_model_with_new_api(
    gpm_file_path: str, 
    iteration_output_dir: str, 
    data_file: str, 
    eval_mode: str = "mcmc"
) -> Tuple[Optional[Dict], Optional[str], List[str]]:
    """
    Runs the model using the new codebase API structure.
    """
    print(f"  Running model with GPM: {gpm_file_path}")
    print(f"  Output directory: {iteration_output_dir}")
    print(f"  Data file: {data_file}")
    print(f"  Evaluation mode: {eval_mode}")

    # Import the new codebase functions
    try:
        from clean_gpm_bvar_trends import run_complete_gpm_analysis
        
        # Load data
        import pandas as pd
        data_df = pd.read_csv(data_file)
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'])
            data_df.set_index('Date', inplace=True)
        
        print(f"  Data loaded: {data_df.shape}")
        
        # Create output directory
        os.makedirs(iteration_output_dir, exist_ok=True)
        plot_save_path = os.path.join(iteration_output_dir, "plots") if eval_mode == "mcmc" else None
        
        # Run analysis using new API
        if eval_mode == "mcmc":
            results = run_complete_gpm_analysis(
                data=data_df,
                gpm_file=gpm_file_path,
                analysis_type="mcmc",
                num_warmup=100,
                num_samples=200,
                num_chains=1,
                num_extract_draws=50,
                generate_plots=True,
                save_plots=True,
                plot_save_path=plot_save_path,
                use_gamma_init=True,
                gamma_scale_factor=1.0,
                target_accept_prob=0.80
            )
        else:  # fixed mode
            # For fixed mode, extract parameter values from GPM priors
            from clean_gpm_bvar_trends.integration_orchestrator import create_integration_orchestrator
            from clean_gpm_bvar_trends.gpm_prior_evaluator import _resolve_parameter_value
            
            orchestrator = create_integration_orchestrator(gpm_file_path)
            parsed_model = orchestrator.reduced_model
            
            # Extract fixed parameters
            fixed_params = {}
            for param_name in parsed_model.parameters:
                try:
                    fixed_params[param_name] = _resolve_parameter_value(
                        param_name, {}, parsed_model.estimated_params, False)
                except:
                    print(f"Warning: Could not resolve parameter {param_name}")
            
            # Shock standard deviations
            all_shocks = parsed_model.trend_shocks + parsed_model.stationary_shocks
            for shock_name in all_shocks:
                try:
                    fixed_params[shock_name] = _resolve_parameter_value(
                        shock_name, {}, parsed_model.estimated_params, True)
                except:
                    print(f"Warning: Could not resolve shock {shock_name}")
            
            # VAR correlation matrix (default to identity)
            if parsed_model.stationary_variables:
                n_stat = len(parsed_model.stationary_variables)
                if n_stat > 0:
                    import jax.numpy as jnp
                    from clean_gpm_bvar_trends.constants import _DEFAULT_DTYPE
                    fixed_params['_var_innovation_corr_chol'] = jnp.eye(n_stat, dtype=_DEFAULT_DTYPE)
            
            results = run_complete_gpm_analysis(
                data=data_df,
                gpm_file=gpm_file_path,
                analysis_type="fixed_params",
                param_values=fixed_params,
                num_sim_draws=50,
                plot_results=True,
                save_plots_path_prefix=os.path.join(iteration_output_dir, "plots", "fixed_plot"),
                use_gamma_init_for_test=True
            )
        
        # Process results
        if results is not None:
            # Extract metrics
            metrics = {
                "log_likelihood": getattr(results, 'log_likelihood', None),
                "n_draws": getattr(results, 'n_draws', 0),
                "eval_mode": eval_mode
            }
            
            # For MCMC mode, we could add more diagnostics here if available
            if eval_mode == "mcmc" and hasattr(results, 'mcmc_results'):
                # Add MCMC diagnostics if available
                pass
            
            print(f"  ✓ Model run completed successfully")
            print(f"    Log-likelihood: {metrics.get('log_likelihood', 'N/A')}")
            print(f"    Number of draws: {metrics.get('n_draws', 0)}")
            
        else:
            metrics = {"error": "Model run returned None"}
            print(f"  ✗ Model run failed: returned None")
        
        # Collect generated images
        image_paths = []
        plots_dir = os.path.join(iteration_output_dir, "plots")
        if os.path.exists(plots_dir):
            for fname in os.listdir(plots_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(plots_dir, fname))
        
        print(f"  Found {len(image_paths)} generated images")
        
        # Save metrics
        metrics_file = os.path.join(iteration_output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        return metrics, gpm_file_path, image_paths
        
    except Exception as e:
        import traceback
        error_msg = f"Model execution failed: {str(e)}"
        print(f"  ✗ {error_msg}")
        traceback.print_exc()
        
        metrics = {
            "error": error_msg,
            "traceback": traceback.format_exc()
        }
        
        # Save error metrics
        metrics_file = os.path.join(iteration_output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        return metrics, gpm_file_path, []

def get_ai_suggestions(
    priors_config: Dict[str, Dict],
    metrics: Optional[Dict],
    history: List[Dict],
    image_paths: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Gets new prior suggestions from AI based on current state, metrics, and history.
    Enhanced with detailed model description and economic interpretation.
    """
    if gemini_model is None:
        print("  AI SIMULATION: Gemini model not available. Generating random adjustments.")
        suggestions = {}
        for key, spec in priors_config.items():
            change = np.random.uniform(-0.1, 0.1) * (spec["max_val"] - spec["min_val"])
            suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
        return suggestions

    prompt_parts = ["You are an expert Bayesian econometrician optimizing prior hyperparameters for a GPM/BVAR model."]
    
    # Add detailed model description
    prompt_parts.append("\n=== MODEL DESCRIPTION ===")
    prompt_parts.append("This is a Global Vector Autoregression (GVAR) model with factor structure for three major economies: US, Euro Area (EA), and Japan (JP).")
    prompt_parts.append("\nECONOMIC STRUCTURE:")
    prompt_parts.append("1. OBSERVABLES: Output growth (y), Inflation (pi), Short-term interest rates (r) for each country")
    prompt_parts.append("2. DECOMPOSITION: Each observable = Trend + Cycle")
    prompt_parts.append("   - Trends capture long-run, permanent movements")
    prompt_parts.append("   - Cycles capture short-run, temporary fluctuations")
    
    prompt_parts.append("\n3. TREND STRUCTURE:")
    prompt_parts.append("   a) WORLD FACTORS:")
    prompt_parts.append("      - r_w_trend: Global real interest rate trend")
    prompt_parts.append("      - pi_w_trend: Global inflation trend")
    prompt_parts.append("   b) DEVIATION FACTORS:")
    prompt_parts.append("      - factor_r_devs: Common factor for real rate deviations")
    prompt_parts.append("      - factor_pi_devs: Common factor for inflation deviations")
    prompt_parts.append("   c) IDIOSYNCRATIC DEVIATIONS:")
    prompt_parts.append("      - Country-specific trend deviations")
    prompt_parts.append("   d) OUTPUT TRENDS: Determined by Euler equations using EIS parameters")
    
    prompt_parts.append("\n4. KEY PARAMETERS:")
    prompt_parts.append("   - var_phi_XX: Elasticity of Intertemporal Substitution (EIS)")
    prompt_parts.append("   - lambda_pi_XX: Country loading on world inflation")
    prompt_parts.append("   - stderr parameters: Shock volatilities")
    prompt_parts.append("   - VAR parameters: es, fs, gs, hs, eta")
    
    # Current parameters
    prompt_parts.append("\n=== CURRENT PRIOR HYPERPARAMETERS ===")
    for key, spec in priors_config.items():
        param_type = ""
        if "var_phi" in key:
            param_type = " [EIS - output response to rates]"
        elif "lambda_pi" in key:
            param_type = " [Inflation pass-through]"
        elif "stderr" in key:
            param_type = " [Shock volatility]"
        elif "var_" in key:
            param_type = " [VAR prior]"
            
        prompt_parts.append(f"- {key}: {spec['current_value']:.4g} (range: {spec['min_val']:.4g}-{spec['max_val']:.4g}){param_type}")
        prompt_parts.append(f"  {spec['description']}")
    
    # Metrics and history
    if metrics:
        prompt_parts.append("\n=== LAST RUN METRICS ===")
        simple_metrics = {
            "log_likelihood": metrics.get("log_likelihood"),
            "min_ess_bulk": metrics.get("min_ess_bulk"),
            "max_rhat": metrics.get("max_rhat"),
            "error": metrics.get("error")
        }
        prompt_parts.append(json.dumps({k: v for k, v in simple_metrics.items() if v is not None}, indent=2))
        
        # Add interpretation
        if simple_metrics.get("error"):
            prompt_parts.append("⚠️ ERROR: Model failed - may need parameter adjustments")
        elif simple_metrics.get("min_ess_bulk") and simple_metrics["min_ess_bulk"] < 400:
            prompt_parts.append("⚠️ LOW ESS: Poor MCMC mixing - consider wider priors")
        elif simple_metrics.get("max_rhat") and simple_metrics["max_rhat"] > 1.05:
            prompt_parts.append("⚠️ HIGH R-HAT: Poor convergence")
    
    if history:
        prompt_parts.append("\n=== CALIBRATION HISTORY ===")
        for i, entry in enumerate(history[-3:]):  # Last 3 attempts
            entry_metrics = entry.get("metrics", {})
            attempt_num = len(history) - len(history[-3:]) + i + 1
            prompt_parts.append(f"Attempt {attempt_num}: LL={entry_metrics.get('log_likelihood', 'N/A')}")
    
    # Handle images
    loaded_images = []
    if image_paths:
        prompt_parts.append("\n=== PLOT IMAGES ===")
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                loaded_images.append(img)
                prompt_parts.append(f"📊 {os.path.basename(img_path)}")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
    
    # Task instructions
    prompt_parts.append("\n=== TASK ===")
    prompt_parts.append("Suggest new hyperparameter values to improve model performance.")
    prompt_parts.append("Consider:")
    prompt_parts.append("- Statistical fit (log-likelihood)")
    prompt_parts.append("- MCMC diagnostics (if applicable)")
    prompt_parts.append("- Economic sensibility from plots")
    prompt_parts.append("- Parameter bounds")
    
    prompt_parts.append("\nOutput ONLY a JSON object with parameter names and new values.")
    prompt_parts.append("Example: {\"var_phi_US_mean\": 1.8, \"stderr_shk_r_w_beta\": 0.008}")
    
    # Prepare API call
    final_prompt = []
    if prompt_parts:
        final_prompt.append("\n".join(prompt_parts))
    if loaded_images:
        final_prompt.extend(loaded_images)
    
    print("--- Sending to AI ---")
    print(f"Images: {len(loaded_images)}")
    
    try:
        response = gemini_model.generate_content(final_prompt)
        response_text = response.text
        print(f"AI Response: {response_text}")
        
        # Extract JSON
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            suggestions = json.loads(json_str)
            print(f"Parsed suggestions: {suggestions}")
            return suggestions
        else:
            print("ERROR: No valid JSON in AI response")
            return {}
            
    except Exception as e:
        print(f"ERROR: AI suggestion failed: {e}")
        # Fallback random
        suggestions = {}
        for key, spec in priors_config.items():
            change = np.random.uniform(-0.05, 0.05) * (spec["max_val"] - spec["min_val"])
            suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
        return suggestions

def apply_ai_suggestions(priors_config: Dict[str, Dict], suggestions: Dict[str, float]) -> Dict[str, Dict]:
    """Applies AI suggestions with bounds checking."""
    updated_config = priors_config.copy()
    
    for key, suggested_val in suggestions.items():
        if key in updated_config:
            spec = updated_config[key]
            clipped_val = np.clip(float(suggested_val), spec["min_val"], spec["max_val"])
            
            if updated_config[key]["current_value"] != clipped_val:
                print(f"  {key}: {updated_config[key]['current_value']:.4g} → {clipped_val:.4g}")
            
            updated_config[key]["current_value"] = clipped_val
        else:
            print(f"  Warning: Unknown parameter '{key}' ignored")
    
    return updated_config

def load_json_state(file_path: str, default_factory=None):
    """Load JSON state with fallback."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {file_path}. Using default.")
    return default_factory() if default_factory else {}

def save_json_state(data: Any, file_path: str):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def autonomous_calibration_loop(max_iterations: int = 10, eval_mode: str = "mcmc"):
    """
    Main autonomous calibration loop using the new codebase.
    """
    print("=== Starting Autonomous GPM Prior Calibration ===")
    print(f"Max iterations: {max_iterations}")
    print(f"Evaluation mode: {eval_mode}")
    
    # Load state
    current_priors = load_json_state(PRIOR_STATE_FILE, get_initial_tunable_priors)
    calibration_history = load_json_state(CALIBRATION_HISTORY_FILE, list)
    
    # Validate base GPM file
    if not os.path.exists(BASE_GPM_FILE_PATH):
        print(f"ERROR: Base GPM file not found: {BASE_GPM_FILE_PATH}")
        return
    
    # Load base GPM content
    with open(BASE_GPM_FILE_PATH, 'r') as f:
        base_gpm_content = f.read()
    
    print(f"✓ Base GPM file loaded: {BASE_GPM_FILE_PATH}")
    print(f"✓ Current priors loaded: {len(current_priors)} parameters")
    print(f"✓ History loaded: {len(calibration_history)} previous runs")
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")
        
        iteration_start_time = time.time()
        
        # 1. Create modified GPM file
        iteration_gpm_filename = f"gpm_iter_{iteration + 1}.gpm"
        iteration_gpm_file_path = os.path.join(AUTONOMOUS_RUN_BASE_DIR, iteration_gpm_filename)
        
        try:
            modified_gpm_content = modify_gpm_file_content(base_gpm_content, current_priors)
            with open(iteration_gpm_file_path, 'w') as f:
                f.write(modified_gpm_content)
            print(f"✓ Modified GPM saved: {iteration_gpm_filename}")
        except Exception as e:
            print(f"✗ Failed to create modified GPM: {e}")
            break
        
        # 2. Run model
        iteration_output_dir = os.path.join(AUTONOMOUS_RUN_BASE_DIR, f"iteration_{iteration + 1}")
        os.makedirs(iteration_output_dir, exist_ok=True)
        
        print(f"Running model with {eval_mode} mode...")
        metrics, used_gpm_file, image_paths = run_model_with_new_api(
            iteration_gpm_file_path, 
            iteration_output_dir, 
            DATA_FILE_PATH, 
            eval_mode
        )
        
        # 3. Store history
        history_entry = {
            "iteration": iteration + 1,
            "priors_used": {k: v["current_value"] for k, v in current_priors.items()},
            "metrics": metrics,
            "gpm_file": iteration_gpm_filename,
            "image_paths": image_paths,
            "timestamp": time.time(),
            "iteration_time": time.time() - iteration_start_time
        }
        calibration_history.append(history_entry)
        save_json_state(calibration_history, CALIBRATION_HISTORY_FILE)
        
        print(f"✓ History updated: {len(calibration_history)} total runs")
        
        # 4. Check for errors
        if metrics and metrics.get("error"):
            print(f"⚠️ Model error: {metrics['error']}")
        elif not metrics:
            print("⚠️ No metrics returned from model run")
        else:
            print(f"✓ Model completed successfully")
            if "log_likelihood" in metrics:
                print(f"  Log-likelihood: {metrics['log_likelihood']}")
        
        # 5. Get AI suggestions
        print("Getting AI suggestions...")
        ai_suggestions = get_ai_suggestions(current_priors, metrics, calibration_history, image_paths)
        
        if not ai_suggestions:
            print("⚠️ No AI suggestions received. Stopping calibration.")
            break
        
        print(f"✓ Received {len(ai_suggestions)} AI suggestions")
        
        # 6. Apply suggestions
        current_priors = apply_ai_suggestions(current_priors, ai_suggestions)
        save_json_state(current_priors, PRIOR_STATE_FILE)
        
        iteration_time = time.time() - iteration_start_time
        print(f"✓ Iteration {iteration + 1} completed in {iteration_time:.1f}s")
        
        # 7. Check stopping criteria
        if metrics and not metrics.get("error"):
            # Could add more sophisticated stopping criteria here
            min_ess = metrics.get("min_ess_bulk", 0)
            max_rhat = metrics.get("max_rhat", 100)
            
            if min_ess > 400 and max_rhat < 1.05:
                print("🎯 Convergence criteria met! Stopping early.")
                break
    
    print(f"\n{'='*60}")
    print("AUTONOMOUS CALIBRATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total iterations: {len(calibration_history)}")
    
    # Print summary
    if calibration_history:
        best_iteration = max(calibration_history, 
                           key=lambda x: x.get("metrics", {}).get("log_likelihood", -np.inf))
        print(f"Best iteration: {best_iteration['iteration']}")
        print(f"Best log-likelihood: {best_iteration.get('metrics', {}).get('log_likelihood', 'N/A')}")

def print_usage():
    """Print usage instructions."""
    print("GPM AI Calibrator - Usage:")
    print("1. Run calibration loop:")
    print("   python gpm_ai_calibrator.py --run_calibration_loop --max_iterations 5")
    print("")
    print("2. Set evaluation mode:")
    print("   python gpm_ai_calibrator.py --run_calibration_loop --eval_mode mcmc")
    print("   python gpm_ai_calibrator.py --run_calibration_loop --eval_mode fixed")
    print("")
    print("3. Environment setup:")
    print("   export GEMINI_API_KEY='your_api_key_here'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPM AI Calibrator - Updated for New Codebase")
    
    parser.add_argument("--run_calibration_loop", action="store_true",
                        help="Run the autonomous calibration loop")
    parser.add_argument("--eval_mode", type=str, default="mcmc", choices=["mcmc", "fixed"],
                        help="Evaluation mode for calibration iterations")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Maximum number of calibration iterations")
    
    # Test/debug arguments
    parser.add_argument("--test_gpm_modification", action="store_true",
                        help="Test GPM file modification without running model")
    parser.add_argument("--test_single_run", action="store_true",
                        help="Test a single model run")
    
    args = parser.parse_args()
    
    if args.test_gpm_modification:
        print("Testing GPM file modification...")
        if not os.path.exists(BASE_GPM_FILE_PATH):
            print(f"ERROR: Base GPM file not found: {BASE_GPM_FILE_PATH}")
        else:
            with open(BASE_GPM_FILE_PATH, 'r') as f:
                base_content = f.read()
            
            test_priors = get_initial_tunable_priors()
            modified_content = modify_gpm_file_content(base_content, test_priors)
            
            test_file = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "test_modified.gpm")
            with open(test_file, 'w') as f:
                f.write(modified_content)
            
            print(f"✓ Test modification saved to: {test_file}")
            print("Check the file to verify modifications worked correctly.")
    
    elif args.test_single_run:
        print("Testing single model run...")
        test_output_dir = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "test_run")
        os.makedirs(test_output_dir, exist_ok=True)
        
        metrics, _, images = run_model_with_new_api(
            BASE_GPM_FILE_PATH, 
            test_output_dir, 
            DATA_FILE_PATH, 
            args.eval_mode
        )
        
        print("Test run results:")
        print(json.dumps(metrics, indent=2, default=str))
        print(f"Generated images: {len(images)}")
    
    elif args.run_calibration_loop:
        print(f"Starting calibration with {args.eval_mode} mode...")
        autonomous_calibration_loop(
            max_iterations=args.max_iterations, 
            eval_mode=args.eval_mode
        )
    
    else:
        print_usage()