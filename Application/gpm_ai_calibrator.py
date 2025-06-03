

import sys  # Standard library, always available
import os   # Standard library, always available

# --- Path Setup ---
# This block should be the VERY FIRST part of your script logic,
# immediately after standard library imports that don't depend on your project.
SCRIPT_DIR_CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))
# For scripts in Application/, '..' goes up to secon_bvar_jax/
PROJECT_ROOT_ORCH = os.path.abspath(os.path.join(SCRIPT_DIR_CURRENT_FILE, '..'))
if PROJECT_ROOT_ORCH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_ORCH) # Add secon_bvar_jax to the beginning of sys.path
# Optional: print for confirmation, good for debugging
# print(f"DEBUG ({os.path.basename(__file__)}): sys.path now includes: {PROJECT_ROOT}")

# NOW, import other standard libraries and THEN your custom packages/modules
import subprocess
import json
import re
import time
import shutil
from PIL import Image
import numpy as np
import json
import os
from PIL import Image # For loading images
import numpy as np # For np.clip and random if needed for AI fallback
import sys
from typing import Dict, List, Tuple, Any, Optional # Added Optional

# --- JAX Configuration ---
# This import should now reliably find your package's jax_config WITHOUT prior warning
from clean_gpm_bvar_trends.jax_config import configure_jax
configure_jax() # Explicit call

import google.genai as genai # If using Gemini (ensure installed)

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = 'gemini-2.5-pro-preview-05-06'

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. AI calls will be SIMULATED.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Could not configure Gemini API or model: {e}. AI calls will be SIMULATED.")
        gemini_model = None

# Paths
BASE_GPM_FILE_PATH = os.path.join(PROJECT_ROOT_ORCH, "Application", "Models", "gpm_factor_y_pi_rshort.gpm") # Using the provided GPM
MODEL_RUNNER_SCRIPT = os.path.join(PROJECT_ROOT_ORCH, "Application", "main_custom_plots.py")
DATA_FILE_PATH = os.path.join(PROJECT_ROOT_ORCH, "Application", "data_m5.csv") # Default data file

AUTONOMOUS_RUN_BASE_DIR = os.path.join(PROJECT_ROOT_ORCH, "Application", "autonomous_gpm_calibration_output")
os.makedirs(AUTONOMOUS_RUN_BASE_DIR, exist_ok=True)

PRIOR_STATE_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "current_priors_config.json")
CALIBRATION_HISTORY_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "calibration_history.json")

# --- Prior Definition and GPM Modification ---
def get_initial_tunable_priors() -> Dict[str, Dict]:
    """
    Defines the set of prior hyperparameters to be tuned for gpm_factor_y_pi_rshort.gpm
    """
    # Adapted from your example, focusing on parameters in gpm_factor_y_pi_rshort.gpm
    return {
        # === STRUCTURAL PARAMETERS ===
        "var_phi_US_mean": {
            "gpm_keyword_for_line": "var_phi_US", "gpm_param_name": "var_phi_US",
            "dist_type": "normal_pdf", "hyper_name": "mean", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 0.5, "max_val": 4.0,  # Adjusted for realistic EIS range
            "description": "Mean for EIS parameter var_phi_US (Normal prior)."
        },
        "var_phi_US_std": {
            "gpm_keyword_for_line": "var_phi_US", "gpm_param_name": "var_phi_US",
            "dist_type": "normal_pdf", "hyper_name": "std", "value_index_in_gpm_line": 1,
            "current_value": 0.5, "min_val": 0.1, "max_val": 1.5,  # Reasonable uncertainty
            "description": "Std dev for EIS parameter var_phi_US (Normal prior)."
        },
        "lambda_pi_US_mean": {
            "gpm_keyword_for_line": "lambda_pi_US", "gpm_param_name": "lambda_pi_US",
            "dist_type": "normal_pdf", "hyper_name": "mean", "value_index_in_gpm_line": 0,
            "current_value": 1.0, "min_val": 0.3, "max_val": 1.5,  # Inflation pass-through range
            "description": "Mean for US loading on world inflation trend (Normal prior)."
        },

        # === INVERSE GAMMA ALPHA PARAMETERS (SHAPE) ===
        "stderr_shk_r_w_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_w",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,  # CORRECTED RANGE
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_w."  # CORRECTED DESCRIPTION
        },
        "stderr_shk_r_US_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_US_idio."  # CORRECTED
        },
        "stderr_shk_pi_US_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_pi_US_idio."  # CORRECTED
        },
        "stderr_shk_r_EA_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_EA_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_EA_idio."  # CORRECTED
        },
        "stderr_shk_pi_EA_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_EA_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_pi_EA_idio."  # CORRECTED
        },
        "stderr_shk_r_JP_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_JP_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_r_JP_idio."  # CORRECTED
        },
        "stderr_shk_pi_JP_idio_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_JP_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_pi_JP_idio."  # CORRECTED
        },
        "stderr_shk_y_US_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_US",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_y_US."  # CORRECTED
        },
        "stderr_shk_y_EA_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_EA",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_y_EA."  # CORRECTED
        },
        "stderr_shk_y_JP_alpha": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_JP",
            "dist_type": "inv_gamma_pdf", "hyper_name": "alpha", "value_index_in_gpm_line": 0,
            "current_value": 2.0, "min_val": 1.5, "max_val": 5.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_y_JP."  # CORRECTED
        },

        # === INVERSE GAMMA BETA PARAMETERS (SCALE) ===
        "stderr_shk_r_w_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_w",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.001, "max_val": 0.05,  # CORRECTED RANGE
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_w."
        },
        "stderr_shk_r_US_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_US_idio."
        },
        "stderr_shk_pi_US_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_US_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_pi_US_idio."
        },
        "stderr_shk_r_EA_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_EA_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_EA_idio."
        },
        "stderr_shk_pi_EA_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_EA_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_pi_EA_idio."
        },
        "stderr_shk_r_JP_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_r_JP_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_r_JP_idio."
        },
        "stderr_shk_pi_JP_idio_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_pi_JP_idio",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.005, "min_val": 0.001, "max_val": 0.02,  # CORRECTED MIN
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_pi_JP_idio."
        },
        "stderr_shk_y_US_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_US",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.002, "max_val": 0.05,  # CORRECTED
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_y_US."
        },
        "stderr_shk_y_EA_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_EA",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.002, "max_val": 0.05,  # CORRECTED
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_y_EA."
        },
        "stderr_shk_y_JP_beta": {
            "gpm_keyword_for_line": "stderr", "gpm_param_name": "shk_y_JP",
            "dist_type": "inv_gamma_pdf", "hyper_name": "beta", "value_index_in_gpm_line": 1,
            "current_value": 0.01, "min_val": 0.002, "max_val": 0.05,  # CORRECTED
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_y_JP."
        }
        # REMOVED DUPLICATES
    }

def modify_gpm_file_content(base_gpm_content: str, current_priors_config: Dict[str, Dict]) -> str:
    modified_content = base_gpm_content
    line_updates: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}

    for _tunable_key, spec in current_priors_config.items():
        key = (spec["gpm_keyword_for_line"], spec["gpm_param_name"], spec["dist_type"])
        if key not in line_updates: line_updates[key] = []
        line_updates[key].append((spec["value_index_in_gpm_line"], spec["current_value"]))

    for (gpm_keyword, gpm_name, dist_type), updates in line_updates.items():
        updates.sort()
        new_values_str_list = [f"{val:.6g}" for _idx, val in updates]

        if dist_type in ["inv_gamma_pdf", "normal_pdf"]:
            base_pattern_str = re.escape(gpm_keyword)
            if gpm_keyword.lower() == "stderr":
                base_pattern_str += r"\s+" + re.escape(gpm_name)
            pattern_str = rf"({base_pattern_str}\s*,\s*{re.escape(dist_type)}\s*,\s*)([\d\.\s,eE+-]+)(\s*;)"
            pattern = re.compile(pattern_str)
            replacer = lambda m: m.group(1) + ", ".join(new_values_str_list) + m.group(3)
            modified_content, num_subs = pattern.subn(replacer, modified_content)
            if num_subs == 0: print(f"Warning: No substitution for {gpm_keyword} {gpm_name} {dist_type}. Pattern: {pattern_str}")
        elif dist_type == "var_prior_setup":
            base_pattern_str = re.escape(gpm_keyword)
            pattern_str = rf"({base_pattern_str}\s*=\s*)([\d\.\s,eE+-]+)(\s*;)"
            pattern = re.compile(pattern_str)
            replacer_var_setup = lambda m: m.group(1) + ", ".join(new_values_str_list) + m.group(3)
            modified_content, num_subs = pattern.subn(replacer_var_setup, modified_content)
            if num_subs == 0: print(f"Warning: No substitution for var_prior_setup '{gpm_keyword}'. Pattern: {pattern_str}")
    return modified_content

def run_andres_bvar_model(gpm_file_path: str, iteration_output_dir: str, data_file: str, eval_mode: str = "mcmc") -> Tuple[Optional[Dict], Optional[str], List[str]]:
    """
    Runs the Andres BVAR model using the specified GPM file and output directory.
    Returns metrics dictionary, path to the GPM file used, and paths to generated images.
    """
    print(f"  Running BVAR model with GPM: {gpm_file_path}")
    print(f"  Output directory for this run: {iteration_output_dir}")
    print(f"  Data file: {data_file}")
    print(f"  Evaluation mode: {eval_mode}")

    metrics_file = os.path.join(iteration_output_dir, "metrics.json")
    plot_dir = os.path.join(iteration_output_dir, "plots") # Defined plot subdirectory

    # Ensure the specific iteration's plot directory exists if plots are to be saved there by main_custom_plots.py
    os.makedirs(plot_dir, exist_ok=True)

    cmd = [
        sys.executable, MODEL_RUNNER_SCRIPT,
        "--gpm_file", gpm_file_path,
        "--output_dir", iteration_output_dir, # main_custom_plots.py will save plots in iteration_output_dir/plots
        "--data_file", data_file,
        "--eval_mode", eval_mode
    ]
    if eval_mode == "fixed": # Add fixed mode specific params if any
        cmd.extend(["--fixed_trend_p0_scale", "0.01", "--fixed_var_p0_scale", "0.1"])


    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=1200) # Increased timeout
        print("--- Model Runner STDOUT ---")
        print(process.stdout)
        print("--- Model Runner STDERR ---")
        print(process.stderr)
        process.check_returncode() # Raise error if return code is non-zero

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(f"  Metrics loaded successfully from {metrics_file}")
        else:
            print(f"  ERROR: Metrics file not found at {metrics_file}")
            metrics = {"error": "Metrics file not found."}

        # Collect image paths
        generated_image_paths = []
        if os.path.isdir(plot_dir):
            for fname in os.listdir(plot_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    generated_image_paths.append(os.path.join(plot_dir, fname))
            print(f"  Found {len(generated_image_paths)} images in {plot_dir}")
        else:
            print(f"  Warning: Plot directory {plot_dir} not found.")

        return metrics, gpm_file_path, generated_image_paths

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Model run failed with return code {e.returncode}.")
        return {"error": f"Model run failed: {e.returncode}", "stdout": e.stdout, "stderr": e.stderr}, gpm_file_path, []
    except subprocess.TimeoutExpired:
        print("  ERROR: Model run timed out.")
        return {"error": "Model run timed out"}, gpm_file_path, []
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred during model run: {e}")
        return {"error": f"Unexpected error: {str(e)}"}, gpm_file_path, []


# def get_ai_suggestions(
#     priors_config: Dict[str, Dict],
#     metrics: Optional[Dict],
#     history: List[Dict],
#     image_paths: Optional[List[str]] = None # New parameter for image paths
# ) -> Dict[str, float]:
#     """
#     Gets new prior suggestions from the AI based on current state, metrics, and history.
#     Now includes images in the prompt to the AI.
#     """
#     if gemini_model is None:
#         print("  AI SIMULATION: Gemini model not available. Generating random adjustments.")
#         suggestions = {}
#         for key, spec in priors_config.items():
#             change = np.random.uniform(-0.1, 0.1) * (spec["max_val"] - spec["min_val"])
#             suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
#         return suggestions

#     prompt_parts = ["You are an expert Bayesian econometrician optimizing prior hyperparameters for a GPM/BVAR model."]
#     prompt_parts.append("Objective: Improve model fit (e.g., log-likelihood, MCMC diagnostics like ESS, R-hat) and ensure economically sensible posterior outcomes (visible in plots). " \
#     "That is trends should not be very volatile or smooth")
#     prompt_parts.append("\nCURRENT PRIOR HYPERPARAMETERS TO TUNE (with current values, bounds, and descriptions):")
#     for key, spec in priors_config.items():
#         prompt_parts.append(f"- {key}: current_value={spec['current_value']:.4g}, min={spec['min_val']:.4g}, max={spec['max_val']:.4g}. ({spec['hyper_name']} for {spec['gpm_param_name']}, {spec['dist_type']}). Desc: {spec['description']}")

#     if metrics:
#         prompt_parts.append("\nLAST RUN METRICS:")
#         # Simplify metrics for the prompt if too verbose
#         simple_metrics = {
#             "log_likelihood": metrics.get("log_likelihood_estimate") or metrics.get("log_likelihood"),
#             "min_ess_bulk": metrics.get("min_ess_bulk"),
#             "max_rhat": metrics.get("max_rhat"),
#             "fitting_time": metrics.get("fitting_time_seconds"),
#             "error": metrics.get("error")
#         }
#         prompt_parts.append(json.dumps({k: v for k, v in simple_metrics.items() if v is not None}, indent=2))
#     else:
#         prompt_parts.append("\nLAST RUN METRICS: Not available (e.g., first run or error).")

#     if history:
#         prompt_parts.append("\nCALIBRATION HISTORY (last few attempts):")
#         for i, entry in enumerate(history[-3:]): # Show last 3
#             entry_metrics = entry.get("metrics", {})
#             simple_entry_metrics = {
#                 "log_likelihood": entry_metrics.get("log_likelihood_estimate") or entry_metrics.get("log_likelihood"),
#                 "min_ess_bulk": entry_metrics.get("min_ess_bulk"),
#                 "max_rhat": entry_metrics.get("max_rhat"),
#                 "error": entry_metrics.get("error")
#             }
#             prompt_parts.append(f"  Attempt {len(history)-len(history[-3:])+i}: Priors={ {k:f'{v:.3g}' for k,v in entry['priors_used'].items()} }, Metrics={ {k: (f'{v:.3g}' if isinstance(v,float) else v) for k,v in simple_entry_metrics.items() if v is not None} }")

#     # Add image handling
#     loaded_images = []
#     if image_paths:
#         prompt_parts.append("\nPLOT IMAGES FROM LAST RUN (consider these for economic sensibility and model fit):")
#         for img_path in image_paths:
#             try:
#                 img = Image.open(img_path)
#                 # Optional: Resize if images are too large, though Gemini handles large images well.
#                 # img.thumbnail((1024, 1024)) # Example resize
#                 loaded_images.append(img)
#                 prompt_parts.append(f"(Image: {os.path.basename(img_path)} provided)") # Placeholder text for the image
#             except Exception as e:
#                 print(f"  Warning: Could not load image {img_path}: {e}")
#                 prompt_parts.append(f"(Could not load image: {os.path.basename(img_path)})")
#     else:
#         prompt_parts.append("\nPLOT IMAGES FROM LAST RUN: No images provided for this iteration.")


#     prompt_parts.append("\nYOUR TASK: Provide new values for the tunable prior hyperparameters listed above.")
#     prompt_parts.append("Consider the metrics, history, and PLOT IMAGES. Aim for plausible economic behavior and good statistical fit.")
#     prompt_parts.append("If the last run had errors, try to suggest changes that might fix them (e.g., wider priors if ESS is low, different means if likelihood is poor).")
#     prompt_parts.append("If the model is fitting well, suggest smaller, exploratory changes.")
#     prompt_parts.append("Output ONLY a JSON object with keys matching the tunable prior hyperparameter names and their new suggested numeric values. Do not include any other text or explanations.")
#     prompt_parts.append("Example JSON output: {\"stderr_shk_trend_y_world_alpha\": 2.6, \"var_phi_US_mean\": 1.9, ...}")


#     # Construct the multimodal prompt
#     # The Gemini API expects a list where text and image parts alternate or are grouped.
#     # If only text, prompt_parts can be joined. If images, they need to be interspersed.
#     final_prompt_for_api = []
#     text_buffer = []
#     for part in prompt_parts:
#         if isinstance(part, str):
#             text_buffer.append(part)
#         # In this structure, images are implicitly referred to after the "PLOT IMAGES..." text.
#         # The Gemini API call structure handles this by having a list of [text, image, text, image...]
#         # Here, we'll append all text, then all images. Or, intersperse more directly if needed.

#     if text_buffer:
#         final_prompt_for_api.append("\n".join(text_buffer))
#     if loaded_images:
#         final_prompt_for_api.extend(loaded_images) # Add PIL Image objects to the list

#     print("\n--- Sending to AI ---")
#     print("Prompt (text part initial section):")
#     print("\n".join(prompt_parts[:10])) # Print first few lines of text part
#     if loaded_images:
#         print(f"Number of images being sent: {len(loaded_images)}")


#     try:
#         # Use the `contents` argument for multimodal input
#         response = gemini_model.generate_content(final_prompt_for_api)
#         response_text = response.text
#         print(f"  AI Raw Response Text: {response_text}")

#         # Extract JSON from the response
#         # The regex tries to find a JSON object within the response text.
#         # It looks for text starting with '{' and ending with '}'
#         match = re.search(r"\{.*\}", response_text, re.DOTALL)
#         if match:
#             json_str = match.group(0)
#             suggestions = json.loads(json_str)
#             print(f"  AI Parsed Suggestions: {suggestions}")
#             return suggestions
#         else:
#             print("  ERROR: AI response did not contain a valid JSON object.")
#             return {} # Fallback to empty dict
#     except Exception as e:
#         print(f"  ERROR: AI suggestion generation failed: {e}")
#         # Fallback: simple random adjustment if AI fails
#         suggestions = {}
#         for key, spec in priors_config.items():
#             change = np.random.uniform(-0.05, 0.05) * (spec["max_val"] - spec["min_val"]) # Smaller change
#             suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
#         print(f"  Fallback random suggestions: {suggestions}")
#         return suggestions


# Enhanced get_ai_suggestions function with detailed model description

def get_ai_suggestions(
    priors_config: Dict[str, Dict],
    metrics: Optional[Dict],
    history: List[Dict],
    image_paths: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Gets new prior suggestions from the AI based on current state, metrics, and history.
    Now includes detailed model description and economic interpretation.
    """
    if gemini_model is None:
        print("  AI SIMULATION: Gemini model not available. Generating random adjustments.")
        suggestions = {}
        for key, spec in priors_config.items():
            change = np.random.uniform(-0.1, 0.1) * (spec["max_val"] - spec["min_val"])
            suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
        return suggestions

    prompt_parts = ["You are an expert Bayesian econometrician optimizing prior hyperparameters for a GPM/BVAR model."]
    
    # ADD DETAILED MODEL DESCRIPTION
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
    prompt_parts.append("   b) DEVIATION FACTORS (common across countries):")
    prompt_parts.append("      - factor_r_devs: Common factor for real rate deviations from world trend")
    prompt_parts.append("      - factor_pi_devs: Common factor for inflation deviations from world trend")
    prompt_parts.append("   c) IDIOSYNCRATIC DEVIATIONS (country-specific):")
    prompt_parts.append("      - r_XX_idio_trend, pi_XX_idio_trend for each country XX")
    prompt_parts.append("   d) OUTPUT TRENDS: Determined by Euler equations using EIS parameters")
    prompt_parts.append("      - y_XX_trend = (1/var_phi_XX) * real_rate_trend + shock")
    
    prompt_parts.append("\n4. KEY RELATIONSHIPS:")
    prompt_parts.append("   - Real Rate: rr_XX = r_w_trend + country_deviations")
    prompt_parts.append("   - Inflation: pi_XX = lambda_pi_XX * pi_w_trend + country_deviations")
    prompt_parts.append("   - Nominal Rate: R_XX = rr_XX + pi_XX")
    prompt_parts.append("   - Output: Euler equation links output growth to real rates via EIS")
    
    prompt_parts.append("\n5. PARAMETER INTERPRETATION:")
    prompt_parts.append("   - var_phi_XX: Elasticity of Intertemporal Substitution (EIS) - higher = more responsive output to rates")
    prompt_parts.append("   - lambda_pi_XX: Country's loading on world inflation (1.0 = full pass-through)")
    prompt_parts.append("   - loading_XX_on_factor: How much country XX responds to common factors")
    prompt_parts.append("   - stderr parameters: Volatility of shocks (smaller = smoother trends)")
    
    prompt_parts.append("\n6. ECONOMIC PRIORS:")
    prompt_parts.append("   - EIS (var_phi) typically 0.5-3.0 (literature consensus)")
    prompt_parts.append("   - Inflation pass-through (lambda_pi) typically 0.5-1.5")
    prompt_parts.append("   - World trends should be less volatile than country-specific trends")
    prompt_parts.append("   - Idiosyncratic shocks should be smaller than common factor shocks")
    prompt_parts.append("   - Cycles should be more volatile than trends")
    
    prompt_parts.append("\n=== OPTIMIZATION OBJECTIVE ===")
    prompt_parts.append("Optimize for:")
    prompt_parts.append("1. STATISTICAL FIT: High log-likelihood, good MCMC diagnostics (ESS > 400, R-hat < 1.05)")
    prompt_parts.append("2. ECONOMIC SENSIBILITY: Trends should be smooth but not too smooth, reasonable parameter values")
    prompt_parts.append("3. EMPIRICAL PLAUSIBILITY: Estimated trends should match economic intuition")
    
    prompt_parts.append("\n=== TUNING GUIDELINES ===")
    prompt_parts.append("- If trends are too volatile: DECREASE stderr parameters (smaller beta for inv_gamma)")
    prompt_parts.append("- If trends are too smooth: INCREASE stderr parameters (larger beta for inv_gamma)")
    prompt_parts.append("- If MCMC has poor mixing: Consider wider priors (larger std for normal, smaller alpha for inv_gamma)")
    prompt_parts.append("- If parameters hit bounds: Widen the prior ranges")
    prompt_parts.append("- For inv_gamma(α,β): Mode = β/(α+1), Mean = β/(α-1) for α>1")
    
    # CURRENT PARAMETERS SECTION
    prompt_parts.append("\n=== CURRENT PRIOR HYPERPARAMETERS TO TUNE ===")
    for key, spec in priors_config.items():
        # Add parameter interpretation
        param_type = ""
        if "var_phi" in key:
            param_type = " [EIS parameter - controls output response to rates]"
        elif "lambda_pi" in key:
            param_type = " [Inflation pass-through from world trend]"
        elif "loading" in key:
            param_type = " [Factor loading - cross-country spillovers]"
        elif "stderr" in key and "shk_r_w" in key:
            param_type = " [World real rate trend volatility]"
        elif "stderr" in key and "shk_pi_w" in key:
            param_type = " [World inflation trend volatility]"
        elif "stderr" in key and "factor" in key:
            param_type = " [Common factor volatility]"
        elif "stderr" in key and "idio" in key:
            param_type = " [Country-specific deviation volatility]"
        elif "stderr" in key and "shk_y" in key:
            param_type = " [Output trend shock volatility]"
        elif "stderr" in key and "cycle" in key:
            param_type = " [Business cycle volatility]"
            
        prompt_parts.append(f"- {key}: current_value={spec['current_value']:.4g}, min={spec['min_val']:.4g}, max={spec['max_val']:.4g}{param_type}")
        prompt_parts.append(f"  ({spec['hyper_name']} for {spec['gpm_param_name']}, {spec['dist_type']}) - {spec['description']}")
    
    # METRICS AND HISTORY (existing code)
    if metrics:
        prompt_parts.append("\n=== LAST RUN METRICS ===")
        simple_metrics = {
            "log_likelihood": metrics.get("log_likelihood_estimate") or metrics.get("log_likelihood"),
            "min_ess_bulk": metrics.get("min_ess_bulk"),
            "max_rhat": metrics.get("max_rhat"),
            "fitting_time": metrics.get("fitting_time_seconds"),
            "error": metrics.get("error")
        }
        prompt_parts.append(json.dumps({k: v for k, v in simple_metrics.items() if v is not None}, indent=2))
        
        # Add interpretation
        if simple_metrics.get("error"):
            prompt_parts.append("⚠️ ERROR DETECTED: Model failed to run - may need parameter adjustments")
        elif simple_metrics.get("min_ess_bulk") and simple_metrics["min_ess_bulk"] < 400:
            prompt_parts.append("⚠️ LOW ESS: Poor MCMC mixing - consider wider priors or different parameter values")
        elif simple_metrics.get("max_rhat") and simple_metrics["max_rhat"] > 1.05:
            prompt_parts.append("⚠️ HIGH R-HAT: Poor MCMC convergence - model may be overparameterized")
    else:
        prompt_parts.append("\n=== LAST RUN METRICS ===")
        prompt_parts.append("Not available (first run or error).")

    if history:
        prompt_parts.append("\n=== CALIBRATION HISTORY ===")
        prompt_parts.append("Previous attempts (most recent first):")
        for i, entry in enumerate(history[-3:]):
            entry_metrics = entry.get("metrics", {})
            simple_entry_metrics = {
                "log_likelihood": entry_metrics.get("log_likelihood_estimate") or entry_metrics.get("log_likelihood"),
                "min_ess_bulk": entry_metrics.get("min_ess_bulk"),
                "max_rhat": entry_metrics.get("max_rhat"),
                "error": entry_metrics.get("error")
            }
            attempt_num = len(history) - len(history[-3:]) + i + 1
            prompt_parts.append(f"  Attempt {attempt_num}:")
            prompt_parts.append(f"    Priors: {{{', '.join([f'{k}:{v:.3g}' for k,v in entry['priors_used'].items()])}}}")
            result_items = []
            for k, v in simple_entry_metrics.items():
                if v is not None:
                    if isinstance(v, float):
                        result_items.append(f"{k}:{v:.3g}")
                    else:
                        result_items.append(f"{k}:{v}")
            prompt_parts.append(f"    Result: {{{', '.join(result_items)}}}")

    # IMAGE HANDLING (existing code)
    loaded_images = []
    if image_paths:
        prompt_parts.append("\n=== PLOT IMAGES FROM LAST RUN ===")
        prompt_parts.append("Examine these plots for economic sensibility:")
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                loaded_images.append(img)
                prompt_parts.append(f"📊 {os.path.basename(img_path)}")
            except Exception as e:
                print(f"  Warning: Could not load image {img_path}: {e}")
                prompt_parts.append(f"❌ Could not load: {os.path.basename(img_path)}")
    else:
        prompt_parts.append("\n=== PLOT IMAGES FROM LAST RUN ===")
        prompt_parts.append("No images provided for this iteration.")

    # TASK INSTRUCTIONS
    prompt_parts.append("\n=== YOUR TASK ===")
    prompt_parts.append("Based on the model structure, current metrics, history, and plots:")
    prompt_parts.append("1. Identify what's wrong (if anything) with the current calibration")
    prompt_parts.append("2. Suggest new hyperparameter values that will improve model performance")
    prompt_parts.append("3. Consider economic plausibility - parameters should make economic sense")
    prompt_parts.append("4. Focus on the most impactful parameters first")
    
    prompt_parts.append("\nSTRATEGY:")
    prompt_parts.append("- If errors: Fix fundamental issues (parameter bounds, numerical stability)")
    prompt_parts.append("- If poor fit: Adjust parameters controlling trend smoothness and factor loadings")
    prompt_parts.append("- If poor MCMC: Widen priors or adjust scale parameters")
    prompt_parts.append("- If good fit: Make smaller refinements")
    
    prompt_parts.append("\nRESTRICTIONS:")
    prompt_parts.append("- Only suggest parameters from the tunable list above")
    prompt_parts.append("- Stay within the specified min/max bounds")
    prompt_parts.append("- Output ONLY a JSON object with parameter names and values")
    prompt_parts.append("- No explanations or additional text")
    
    prompt_parts.append("\nExample output: {\"var_phi_US_mean\": 1.8, \"stderr_shk_r_w_beta\": 0.008, ...}")

    # Construct multimodal prompt (existing code)
    final_prompt_for_api = []
    if prompt_parts:
        final_prompt_for_api.append("\n".join(prompt_parts))
    if loaded_images:
        final_prompt_for_api.extend(loaded_images)

    print("\n--- Sending Enhanced Prompt to AI ---")
    print("Model description included: ✓")
    print("Parameter interpretations included: ✓")
    print("Economic guidelines included: ✓")
    if loaded_images:
        print(f"Images attached: {len(loaded_images)}")

    # API call (existing code)
    try:
        response = gemini_model.generate_content(final_prompt_for_api)
        response_text = response.text
        print(f"  AI Raw Response Text: {response_text}")

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            suggestions = json.loads(json_str)
            print(f"  AI Parsed Suggestions: {suggestions}")
            return suggestions
        else:
            print("  ERROR: AI response did not contain a valid JSON object.")
            return {}
    except Exception as e:
        print(f"  ERROR: AI suggestion generation failed: {e}")
        suggestions = {}
        for key, spec in priors_config.items():
            change = np.random.uniform(-0.05, 0.05) * (spec["max_val"] - spec["min_val"])
            suggestions[key] = np.clip(spec["current_value"] + change, spec["min_val"], spec["max_val"])
        print(f"  Fallback random suggestions: {suggestions}")
        return suggestions
    
def apply_ai_suggestions_auto(priors_config: Dict[str, Dict], suggestions: Dict[str, float]) -> Dict[str, Dict]:
    """Applies AI suggestions, respecting min/max bounds."""
    updated_config = priors_config.copy()
    for key, suggested_val in suggestions.items():
        if key in updated_config:
            spec = updated_config[key]
            clipped_val = np.clip(float(suggested_val), spec["min_val"], spec["max_val"])
            if updated_config[key]["current_value"] != clipped_val:
                 print(f"  Updating {key}: {updated_config[key]['current_value']:.4g} -> {clipped_val:.4g} (Suggested: {float(suggested_val):.4g})")
            updated_config[key]["current_value"] = clipped_val
        else:
            print(f"  Warning: AI suggested unknown prior '{key}'. Ignoring.")
    return updated_config

def load_json_state(file_path: str, default_factory=None):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Using default.")
    return default_factory() if default_factory else {}

def save_json_state(data: Any, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def autonomous_calibration_loop(max_iterations: int = 10, eval_mode_for_iteration: str = "mcmc"):
    """Main loop for autonomous prior calibration."""
    print("=== Starting Autonomous GPM Prior Calibration Loop ===")

    current_priors = load_json_state(PRIOR_STATE_FILE, get_initial_tunable_priors)
    calibration_history = load_json_state(CALIBRATION_HISTORY_FILE, list)

    # Load the base GPM file content once
    if not os.path.exists(BASE_GPM_FILE_PATH):
        print(f"ERROR: Base GPM file not found at {BASE_GPM_FILE_PATH}")
        return
    with open(BASE_GPM_FILE_PATH, 'r') as f:
        base_gpm_content = f.read()

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        iteration_start_time = time.time()

        # 1. Create a unique GPM file for this iteration
        iteration_gpm_filename = f"gpm_iter_{iteration + 1}.gpm"
        iteration_gpm_file_path = os.path.join(AUTONOMOUS_RUN_BASE_DIR, iteration_gpm_filename)
        modified_gpm_content = modify_gpm_file_content(base_gpm_content, current_priors)
        with open(iteration_gpm_file_path, 'w') as f:
            f.write(modified_gpm_content)
        print(f"  Modified GPM file saved to: {iteration_gpm_file_path}")

        # 2. Run the BVAR model
        iteration_output_dir = os.path.join(AUTONOMOUS_RUN_BASE_DIR, f"iteration_{iteration + 1}")
        os.makedirs(iteration_output_dir, exist_ok=True)
        # Pass the DATA_FILE_PATH to the model runner
        metrics, _used_gpm_file, image_paths = run_andres_bvar_model(iteration_gpm_file_path, iteration_output_dir, DATA_FILE_PATH, eval_mode=eval_mode_for_iteration)

        # 3. Store history
        history_entry = {
            "iteration": iteration + 1,
            "priors_used": {k: v["current_value"] for k, v in current_priors.items()},
            "metrics": metrics,
            "gpm_file": iteration_gpm_filename,
            "image_paths": image_paths, # Store paths to images for this iteration
            "timestamp": time.time()
        }
        calibration_history.append(history_entry)
        save_json_state(calibration_history, CALIBRATION_HISTORY_FILE)

        if metrics and metrics.get("error"):
            print(f"  Error in model run: {metrics['error']}. Attempting AI suggestion for fix.")
        elif not metrics:
            print("  Model run did not produce metrics. Halting iteration or trying fallback.")
            # Could add a simple fallback here if needed, or just proceed to AI with no metrics

        # 4. Get AI suggestions (now with images)
        print("  Getting AI suggestions for next priors...")
        ai_suggestions = get_ai_suggestions(current_priors, metrics, calibration_history, image_paths)

        if not ai_suggestions:
            print("  No valid suggestions from AI. Halting calibration.")
            break

        # 5. Apply suggestions and save state
        current_priors = apply_ai_suggestions_auto(current_priors, ai_suggestions)
        save_json_state(current_priors, PRIOR_STATE_FILE)

        iteration_time = time.time() - iteration_start_time
        print(f"  Iteration {iteration + 1} completed in {iteration_time:.2f}s.")

        # Optional: Add a stopping condition based on metrics or number of iterations
        if metrics and metrics.get("min_ess_bulk", 0) is not None and metrics.get("min_ess_bulk",0) > 400 and \
           metrics.get("max_rhat", 100) is not None and metrics.get("max_rhat",100) < 1.05 and not metrics.get("error"):
            print("  Convergence criteria met (example: ESS > 400, R-hat < 1.05). Stopping.")
            # break # Uncomment to stop early if criteria are met

    print("\n=== Autonomous GPM Prior Calibration Loop Finished ===")


if __name__ == "__main__":
    import argparse

    main_parser = argparse.ArgumentParser(description="GPM Calibration Orchestrator or Model Runner")
    main_parser.add_argument("--run_calibration_loop", action="store_true",
                        help="Run the main autonomous calibration loop.")
    main_parser.add_argument("--eval_mode_loop", type=str, default="mcmc", choices=["mcmc", "fixed"],
                        help="Evaluation mode ('mcmc' or 'fixed') to use within the calibration loop iterations.")
    main_parser.add_argument("--max_calib_iterations", type=int, default=3,
                        help="Maximum number of iterations for the calibration loop.")

    # Args for when this script itself is called as the MODEL_RUNNER_SCRIPT
    # These are parsed by main_custom_plots.py, but this script can also be the target.
    main_parser.add_argument("--gpm_file", type=str, help="Path to GPM file for model execution (if run as model runner).")
    main_parser.add_argument("--output_dir", type=str, help="Output directory for model execution (if run as model runner).")
    main_parser.add_argument("--data_file", type=str, default=DATA_FILE_PATH, help="Path to data CSV for model execution.")
    main_parser.add_argument("--eval_mode", type=str, default="mcmc", choices=["mcmc", "fixed"], help="Eval mode for direct model run.")
    # Add other args that main_custom_plots.py might expect if this script is the runner
    main_parser.add_argument("--fixed_trend_p0_scale", type=float, default=1e4)
    main_parser.add_argument("--fixed_var_p0_scale", type=float, default=1.0)
    main_parser.add_argument("--initial_state_overrides_json", type=str, default=None)


    cli_args = main_parser.parse_args()

    if cli_args.run_calibration_loop:
        print(f"Starting autonomous calibration loop with eval_mode='{cli_args.eval_mode_loop}' for {cli_args.max_calib_iterations} iterations.")
        autonomous_calibration_loop(max_iterations=cli_args.max_calib_iterations, eval_mode_for_iteration=cli_args.eval_mode_loop)

    elif cli_args.gpm_file and cli_args.output_dir:
        # This block means this script is being run AS the MODEL_RUNNER_SCRIPT by the calibration loop.
        # It should essentially do what main_custom_plots.py does.
        print(f"--- gpm_ai_calibrator.py running in MODEL EXECUTION mode ---")
        print(f"  GPM file: {cli_args.gpm_file}")
        print(f"  Output dir: {cli_args.output_dir}")
        print(f"  Data file: {cli_args.data_file}")
        print(f"  Eval mode: {cli_args.eval_mode}")

        from clean_gpm_bvar_trends.main_custom_plots import run_model_and_generate_outputs as main_custom_plots_runner

        # Parse initial state overrides if provided
        parsed_initial_state_overrides = None
        if cli_args.initial_state_overrides_json:
            try:
                parsed_initial_state_overrides = json.loads(cli_args.initial_state_overrides_json)
            except json.JSONDecodeError as e:
                print(f"Error decoding initial_state_overrides_json for model runner: {e}")

        # Call the main logic from main_custom_plots.py
        main_custom_plots_runner(
            gpm_file_arg=cli_args.gpm_file,
            output_dir_arg=cli_args.output_dir,
            data_path_arg=cli_args.data_file,
            eval_mode=cli_args.eval_mode,
            initial_state_overrides_fixed=parsed_initial_state_overrides,
            trend_p0_scale_fixed=cli_args.fixed_trend_p0_scale,
            var_p0_scale_fixed=cli_args.fixed_var_p0_scale
        )
        print(f"--- gpm_ai_calibrator.py (as model runner) finished. ---")

    else:
        print("Script called without '--run_calibration_loop' or sufficient model execution arguments.")
        print("To run calibration: python gpm_ai_calibrator.py --run_calibration_loop [--eval_mode_loop <mcmc|fixed>] [--max_calib_iterations <N>]")