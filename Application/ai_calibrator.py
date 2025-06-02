import subprocess
import json
import os
import re
import time
import shutil
from PIL import Image
import google.generativeai as genai # Assuming you'll use this SDK

# --- Configuration ---
# IMPORTANT: Store your API key securely (e.g., environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. AI calls will be simulated.")
    # raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Model for Gemini API
# Use a model that supports multimodal input (text and image)
# gemini_model_name = 'gemini-1.5-flash-latest' # Or 'gemini-1.5-pro-latest'
# gemini_model = genai.GenerativeModel(gemini_model_name) if GOOGLE_API_KEY else None
# print(f"Using Gemini model: {gemini_model_name if GOOGLE_API_KEY else 'SIMULATED'}")

# Paths (adjust as needed)
BASE_GPM_FILE_PATH = "gdps_1_actual_gpm_file.txt" # Your original GPM file
# This is the script we will call as a subprocess
MODEL_RUNNER_SCRIPT = "main_custom_plots.py"
# Base directory for all outputs of this autonomous calibration
AUTONOMOUS_RUN_BASE_DIR = "autonomous_bvar_calibration_runs"
os.makedirs(AUTONOMOUS_RUN_BASE_DIR, exist_ok=True)

# History and current prior state files
PRIOR_STATE_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "current_priors_config.json")
CALIBRATION_HISTORY_FILE = os.path.join(AUTONOMOUS_RUN_BASE_DIR, "calibration_history.json")

# --- Prior Definition and GPM Modification ---

# Define which priors in the GPM file are tunable and how to find/replace them.
# Format: "user_friendly_name": {"gpm_line_pattern": r"line_regex", "param_indices": [idx1, idx2], "current_values": [val1, val2]}
# param_indices refers to the numerical parameters in the GPM line after the distribution type.
# Example: stderr shk_trend_y_world, inv_gamma_pdf, 2.5, 0.025;
#   user_friendly_name: "shk_trend_y_world_stderr_inv_gamma"
#   gpm_line_pattern: r"stderr\s+shk_trend_y_world\s*,\s*inv_gamma_pdf\s*,"
#   param_indices: [0, 1] (for the 2.5 and 0.025 respectively)
#   current_values: [2.5, 0.025]

def get_initial_tunable_priors():
    return {
        "shk_trend_y_world_ig_alpha": {
            "gpm_param_name": "shk_trend_y_world", "dist_type": "inv_gamma_pdf",
            "hyper_name": "alpha", "current_value": 2.5, "min_val": 0.1, "max_val": 10.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_trend_y_world. Higher is tighter."
        },
        "shk_trend_y_world_ig_beta": {
            "gpm_param_name": "shk_trend_y_world", "dist_type": "inv_gamma_pdf",
            "hyper_name": "beta", "current_value": 0.025, "min_val": 0.001, "max_val": 1.0,
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_trend_y_world. Relates to mode."
        },
        "shk_trend_y_jp_ig_alpha": {
            "gpm_param_name": "shk_trend_y_jp", "dist_type": "inv_gamma_pdf",
            "hyper_name": "alpha", "current_value": 1.5, "min_val": 0.1, "max_val": 10.0,
            "description": "Shape (alpha) for inv_gamma prior on stderr of shk_trend_y_jp."
        },
        "shk_trend_y_jp_ig_beta": {
            "gpm_param_name": "shk_trend_y_jp", "dist_type": "inv_gamma_pdf",
            "hyper_name": "beta", "current_value": 0.25, "min_val": 0.001, "max_val": 1.0,
            "description": "Scale (beta) for inv_gamma prior on stderr of shk_trend_y_jp."
        },
        # Add more priors you want to tune from your gdps_1_actual_gpm_file.txt
        # For VAR priors (es, fs, gs, hs, eta):
        "var_prior_es_diag": {
            "gpm_param_name": "es", "dist_type": "var_prior_setup", # Special type
            "hyper_name": "mean_diag_A", "current_value": 0.5, "min_val": -1.0, "max_val": 1.0,
             "description": "Mean for diagonal elements of VAR coefficient matrix A."
        },
        "var_prior_es_offdiag": {
            "gpm_param_name": "es", "dist_type": "var_prior_setup",
            "hyper_name": "mean_offdiag_A", "current_value": 0.3, "min_val": -1.0, "max_val": 1.0,
             "description": "Mean for off-diagonal elements of VAR coefficient matrix A."
        },
        "var_prior_eta": {
            "gpm_param_name": "eta", "dist_type": "var_prior_setup",
            "hyper_name": "lkj_concentration", "current_value": 2.0, "min_val": 0.5, "max_val": 10.0,
             "description": "LKJ prior concentration for VAR innovation correlation matrix."
        }
        # Add fs, gs, hs similarly if desired.
    }

def modify_gpm_file_content(base_gpm_content: str, current_prior_hypervalues: Dict[str, Dict]) -> str:
    """
    Modifies the GPM file content in memory based on current_prior_hypervalues.
    Each key in current_prior_hypervalues corresponds to a defined tunable prior.
    """
    modified_content = base_gpm_content
    
    # Group by GPM parameter name and distribution type for easier processing
    grouped_params = {}
    for tunable_key, spec in current_prior_hypervalues.items():
        gpm_name = spec["gpm_param_name"]
        dist_type = spec["dist_type"]
        hyper_name = spec["hyper_name"]
        value = spec["current_value"]
        
        if (gpm_name, dist_type) not in grouped_params:
            grouped_params[(gpm_name, dist_type)] = {}
        grouped_params[(gpm_name, dist_type)][hyper_name] = value

    for (gpm_name, dist_type), hypers in grouped_params.items():
        if dist_type == "inv_gamma_pdf":
            # Pattern for: stderr shk_trend_y_world, inv_gamma_pdf, 2.5, 0.025;
            # We need to update alpha (2.5) and beta (0.025)
            alpha_val = hypers.get("alpha", None)
            beta_val = hypers.get("beta", None)
            
            if alpha_val is not None and beta_val is not None:
                # Regex to find the line and capture the numbers
                # Example: stderr shk_trend_y_world, inv_gamma_pdf, 2.5, 0.025;
                pattern = re.compile(
                    r"(stderr\s+" + re.escape(gpm_name) + r"\s*,\s*inv_gamma_pdf\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*;)"
                )
                replacement_string = rf"\g<1>{alpha_val:.4f}\g<3>{beta_val:.4f}\g<5>"
                modified_content = pattern.sub(replacement_string, modified_content)
            else:
                print(f"Warning: Missing alpha or beta for {gpm_name} with inv_gamma_pdf. Skipping modification.")

        elif dist_type == "var_prior_setup":
            # For var_prior_setup, parameters are on separate lines or part of a list
            if gpm_name == "es": # e.g., es = 0.5, 0.3;
                val1 = hypers.get("mean_diag_A")
                val2 = hypers.get("mean_offdiag_A")
                if val1 is not None and val2 is not None:
                    pattern = re.compile(r"(es\s*=\s*)([\d\.-]+)(\s*,\s*)([\d\.-]+)(\s*;)")
                    replacement_string = rf"\g<1>{val1:.3f}\g<3>{val2:.3f}\g<5>"
                    modified_content = pattern.sub(replacement_string, modified_content)
            elif gpm_name == "eta": # e.g., eta = 2.0;
                val_eta = hypers.get("lkj_concentration")
                if val_eta is not None:
                    pattern = re.compile(r"(eta\s*=\s*)([\d\.-]+)(\s*;)")
                    replacement_string = rf"\g<1>{val_eta:.2f}\g<3>"
                    modified_content = pattern.sub(replacement_string, modified_content)
            # Add more for fs, gs, hs if they are tuned
    
    return modified_content

# --- Model Execution ---
def run_andres_bvar_model(gpm_file_path_for_run: str, output_dir_for_run: str) -> Tuple[Optional[Dict], Optional[Dict[str, str]]]:
    """
    Runs Andres's main_custom_plots.py script as a subprocess.
    The script is expected to:
    1. Take --gpm_file <path> as an argument.
    2. Take --output_dir <path> as an argument.
    3. Save all plots to <output_dir>/plots/
    4. Save quantitative metrics to <output_dir>/metrics.json
    """
    plot_subdir = os.path.join(output_dir_for_run, "plots")
    os.makedirs(plot_subdir, exist_ok=True)
    metrics_file = os.path.join(output_dir_for_run, "metrics.json")

    # We need to modify `main_custom_plots.py` to accept these arguments
    # and save outputs accordingly. For now, let's assume it's modified.
    cmd = [
        "python", MODEL_RUNNER_SCRIPT,
        "--gpm_file", gpm_file_path_for_run,
        "--output_dir", output_dir_for_run # main_custom_plots needs to use this for its plot_save_path and metrics
    ]
    print(f"Executing: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800) # 30 min timeout
        print("Model run successful.")
        # print("STDOUT:", process.stdout) # Can be verbose
        
        # Load metrics
        quant_metrics = None
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                quant_metrics = json.load(f)
        else:
            print(f"Warning: {metrics_file} not found after model run.")
            quant_metrics = {"error": "metrics_file_not_found"}

        # Get plot paths
        generated_plots = {}
        if os.path.exists(plot_subdir):
            for f_name in os.listdir(plot_subdir):
                if f_name.endswith(".png"): # Assuming PNG plots
                    # Use a simple key, e.g., the filename without extension
                    plot_key = os.path.splitext(f_name)[0]
                    generated_plots[plot_key] = os.path.join(plot_subdir, f_name)
        
        return quant_metrics, generated_plots

    except subprocess.CalledProcessError as e:
        print(f"Model run failed with error code {e.returncode}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return {"error": "model_run_failed", "details": e.stderr[:1000]}, None
    except subprocess.TimeoutExpired:
        print("Model run timed out.")
        return {"error": "model_run_timeout"}, None
    except Exception as e:
        print(f"An unexpected error occurred during model execution: {e}")
        return {"error": f"unexpected_model_run_error: {str(e)[:200]}"}, None

# --- AI Interaction ---
def get_ai_suggestions(current_priors_config: Dict,
                       quantitative_results: Optional[Dict],
                       plot_paths: Optional[Dict[str, str]],
                       history: List[Dict]) -> Dict:
    """
    Sends current state to Gemini and gets prior suggestions.
    """
    if not GOOGLE_API_KEY or gemini_model is None:
        print("SIMULATING AI RESPONSE (No API Key or Model)")
        # Simulate a response: suggest small changes to the first few parameters
        sim_suggestions = {}
        first_two_tunable = list(current_priors_config.keys())[:2]
        for key in first_two_tunable:
            current_val = current_priors_config[key]["current_value"]
            sim_suggestions[key] = round(current_val * (1.0 + np.random.uniform(-0.1, 0.1)), 3) # +/- 10%
        
        return {
            "ai_assessment_text": "Simulated: Trends look somewhat plausible. Suggesting minor tweaks.",
            "suggested_prior_values": sim_suggestions,
            "reasoning": "This is a simulated response for testing the loop.",
            "stop_signal": False # Add a stop signal from AI
        }

    prompt_parts = [
        "You are an expert Bayesian econometrician advising on setting priors for a BVAR model with a trend component and a stationary prior, specified in a GPM file.",
        "Your goal is to guide the priors towards values that result in economically sensible trends and cycles (smooth but not too flat, within plausible ranges, appropriate volatility and persistence for cycles) and good quantitative model fit (e.g., low divergences, good ESS, sensible posterior means for key parameters).",
        "You will be given the current prior hyperparameter values used in the GPM file, quantitative metrics from the MCMC run, and images of key plots (trends and cycles).",
        "Analyze these inputs comprehensively. You have previously demonstrated an ability to visually assess if trends/cycles look strange or implausible. Combine that with the metrics.",
        
        "\n--- Current Prior Hyperparameters ---",
        "The following hyperparameters were used in the GPM file for this iteration:"
    ]
    prior_details_for_prompt = []
    for name, spec in current_priors_config.items():
        prior_details_for_prompt.append(f"- {name} ({spec.get('description','N/A')}): {spec['current_value']:.4f} (Range: [{spec.get('min_val','N/A')}, {spec.get('max_val','N/A')}])")
    prompt_parts.append("\n".join(prior_details_for_prompt))

    prompt_parts.append(f"\n--- Quantitative Results ---")
    if quantitative_results:
        prompt_parts.append(json.dumps(quantitative_results, indent=2))
    else:
        prompt_parts.append("No quantitative results available for this iteration.")

    # Prepare image inputs
    image_input_parts = []
    if plot_paths:
        prompt_parts.append("\n--- Plot Images ---")
        prompt_parts.append("The following images are provided for your visual assessment:")
        for plot_key, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                prompt_parts.append(f"- {plot_key}")
                try:
                    img = Image.open(plot_path)
                    # Ensure images are not excessively large for the API
                    # img.thumbnail((1024, 1024)) # Optional resizing
                    image_input_parts.append(img)
                except Exception as e:
                    print(f"Warning: Could not load image {plot_path} for AI: {e}")
                    prompt_parts.append(f"  (Error loading image {plot_key})")
            else:
                prompt_parts.append(f"- {plot_key} (Image file not found at {plot_path})")
    else:
        prompt_parts.append("\nNo plot images available for this iteration.")

    # History (summary of last few attempts)
    if history:
        prompt_parts.append("\n--- Summary of Recent Attempts (latest first, max 3) ---")
        for i, entry in enumerate(reversed(history[-3:])):
            priors_summary = {k: v['current_value'] for k, v in entry.get('priors_used', {}).items()}
            assessment = entry.get('ai_assessment_text', 'N/A')
            metrics_summary = entry.get('quantitative_results', {}).get('mcmc_summary_short', 'N/A') # Assuming you add this
            prompt_parts.append(
                f"\nAttempt {len(history) - i}:"
                f"\n  Priors: {json.dumps(priors_summary, indent=1)}"
                f"\n  Metrics: {metrics_summary}"
                f"\n  AI Assessment then: {assessment[:150]}..."
            )

    prompt_parts.extend([
        "\n--- Your Task ---",
        "1.  **Overall Assessment (`ai_assessment_text` field):** Provide a concise textual assessment of the model's output based on the current priors, metrics, and plots. Note any issues with trends (too smooth, too volatile, outside plausible range), cycles, or quantitative metrics.",
        "2.  **Prior Suggestions (`suggested_prior_values` field):** Based on your assessment, suggest NEW NUMERICAL VALUES for some or all of the listed 'Current Prior Hyperparameters'. Output this as a JSON object where keys are the 'user_friendly_name' of the prior hyperparameter (e.g., 'shk_trend_y_world_ig_alpha') and values are the new proposed numerical values. Respect the provided min/max ranges.",
        "    - If a hyperparameter should not be changed, DO NOT include it in this JSON object.",
        "    - Make sensible, conservative adjustments. The goal is iterative improvement.",
        "    - For example: {\"shk_trend_y_world_ig_alpha\": 2.8, \"var_prior_es_diag\": 0.45}",
        "3.  **Reasoning (`reasoning` field):** Briefly explain why you are suggesting these changes, linking them to your assessment of the plots and metrics.",
        "4.  **Stop Signal (`stop_signal` field):** If you judge the current results to be good enough and stable, and further tuning is unlikely to yield significant improvements, set this field to `true`. Otherwise, set it to `false`.",
        "\n--- Output Format ---",
        "Return your response as a single JSON object with the keys: \"ai_assessment_text\", \"suggested_prior_values\", \"reasoning\", and \"stop_signal\".",
        "Example: {\"ai_assessment_text\": \"Trends are too flat, cycles lack volatility.\", \"suggested_prior_values\": {\"shk_trend_y_world_ig_beta\": 0.05, \"var_prior_es_diag\": 0.4}, \"reasoning\": \"Increased beta for world trend stderr to allow more variation. Reduced es_diag to encourage more cycle dynamics.\", \"stop_signal\": false}"
    ])

    # Combine text and image parts for the multimodal prompt
    # The Gemini API expects a list of parts, where text and images are distinct parts.
    request_content = []
    for part in prompt_parts:
        request_content.append(part)
    for img_part in image_input_parts:
        request_content.append(img_part) # Add PIL.Image objects

    print("\n--- Sending Request to Gemini ---")
    # print("Prompt (text parts only for brevity):") # For debugging
    # for p in prompt_parts: print(p)

    try:
        # response = gemini_model.generate_content(request_content)
        # ai_response_text = response.text
        # # Manually create a simulated response string for now
        # # In a real scenario, you would parse response.text
        sim_suggestions = {}
        first_two_tunable = list(current_priors_config.keys())[:2]
        for key in first_two_tunable:
            current_val = current_priors_config[key]["current_value"]
            min_val = current_priors_config[key].get("min_val", current_val * 0.1)
            max_val = current_priors_config[key].get("max_val", current_val * 10.0)
            sim_suggestions[key] = round(np.clip(current_val * (1.0 + np.random.uniform(-0.1, 0.1)), min_val, max_val) , 3)

        simulated_ai_json_response = {
            "ai_assessment_text": "Actual AI Simulated: Trends look somewhat plausible. Suggesting minor tweaks.",
            "suggested_prior_values": sim_suggestions,
            "reasoning": "This is an actual AI simulated response for testing the loop.",
            "stop_signal": False
        }
        ai_response_text = json.dumps(simulated_ai_json_response) # Remove when using API


        print(f"AI Raw Response Text (first 500 chars): {ai_response_text[:500]}")
        # Attempt to parse the JSON string (which might be within a larger text block)
        # A common pattern is that the JSON is enclosed in ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", ai_response_text)
        if match:
            json_str = match.group(1)
        else: # Assume the whole text is JSON or it's directly JSON
            json_str = ai_response_text

        parsed_ai_response = json.loads(json_str)

        if not all(k in parsed_ai_response for k in ["ai_assessment_text", "suggested_prior_values", "reasoning", "stop_signal"]):
            print("Error: AI response missing required keys.")
            # Fallback to a safe structure
            return {
                "ai_assessment_text": "AI response malformed, missing keys.",
                "suggested_prior_values": {}, "reasoning": "Malformed AI response.", "stop_signal": False
            }
        print(f"AI Assessment: {parsed_ai_response['ai_assessment_text']}")
        print(f"AI Suggested Prior Values: {json.dumps(parsed_ai_response['suggested_prior_values'], indent=2)}")
        return parsed_ai_response

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode AI JSON response: {e}")
        print(f"Problematic text: {ai_response_text[:1000]}") # Log more of the problematic text
        return {"ai_assessment_text": f"AI response JSON decode error: {e}", "suggested_prior_values": {}, "reasoning": "JSON decode error", "stop_signal": False}
    except Exception as e:
        print(f"Error communicating with AI or parsing response: {type(e).__name__}: {e}")
        return {"ai_assessment_text": f"AI interaction failed: {e}", "suggested_prior_values": {}, "reasoning": f"AI interaction failed: {e}", "stop_signal": False}

# --- Prior Application & History ---
def apply_ai_suggestions_auto(current_priors_config: Dict[str, Dict], ai_suggested_values: Dict[str, float]) -> Tuple[Dict[str, Dict], int]:
    """
    Automatically applies AI suggested new numerical values to the priors configuration.
    """
    updated_priors = current_priors_config.copy() # Work on a copy
    changes_applied_count = 0
    print("\n--- Automatically Applying AI Suggested Prior Values ---")

    if not isinstance(ai_suggested_values, dict):
        print("Warning: `ai_suggested_values` from AI is not a dictionary. No changes applied.")
        return updated_priors, changes_applied_count

    for prior_key_name, new_value_proposed in ai_suggested_values.items():
        if prior_key_name not in updated_priors:
            print(f"Warning: AI suggested changing '{prior_key_name}', which is not in current tunable priors. Skipping.")
            continue

        spec = updated_priors[prior_key_name]
        current_value = spec["current_value"]
        
        try:
            # Convert AI's proposed value to float (it should already be numeric)
            new_value_float = float(new_value_proposed)

            # Clamp to min/max if defined
            min_val = spec.get("min_val", -float('inf'))
            max_val = spec.get("max_val", float('inf'))
            final_new_value = np.clip(new_value_float, min_val, max_val)

            if abs(final_new_value - current_value) > 1e-5 : # Apply if meaningfully different
                print(f"Applying change: '{prior_key_name}' from {current_value:.4f} to {final_new_value:.4f}")
                updated_priors[prior_key_name]["current_value"] = final_new_value # Update the value in the spec
                changes_applied_count += 1
            else:
                print(f"AI suggested value for '{prior_key_name}' ({final_new_value:.4f}) is very close to current ({current_value:.4f}). No change applied.")
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not apply AI suggested value '{new_value_proposed}' for '{prior_key_name}'. Invalid numeric value: {e}. Skipping.")

    if changes_applied_count == 0:
        print("No valid prior changes were applied based on AI suggestions.")
    else:
        print(f"Applied {changes_applied_count} changes to priors.")
    return updated_priors, changes_applied_count


def load_json_state(filepath: str, default_value_func = None):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Using default.")
            return default_value_func() if default_value_func else ({} if "priors" in filepath else [])
    return default_value_func() if default_value_func else ({} if "priors" in filepath else [])

def save_json_state(data: Any, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# --- Main Loop ---
def autonomous_calibration_loop(max_iterations: int = 5):
    # Load base GPM content
    try:
        with open(BASE_GPM_FILE_PATH, 'r') as f:
            base_gpm_content = f.read()
    except FileNotFoundError:
        print(f"ERROR: Base GPM file '{BASE_GPM_FILE_PATH}' not found. Exiting.")
        return

    current_priors_config = load_json_state(PRIOR_STATE_FILE, get_initial_tunable_priors)
    history = load_json_state(CALIBRATION_HISTORY_FILE)
    start_iteration = len(history) + 1

    for i in range(start_iteration, start_iteration + max_iterations):
        print(f"\n{'='*20} ITERATION {i} {'='*20}")
        
        iter_output_dir = os.path.join(AUTONOMOUS_RUN_BASE_DIR, f"iteration_{i:03d}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # 1. Modify GPM file with current priors
        modified_gpm_iter_content = modify_gpm_file_content(base_gpm_content, current_priors_config)
        gpm_file_for_this_run = os.path.join(iter_output_dir, f"model_iter_{i:03d}.gpm")
        with open(gpm_file_for_this_run, 'w') as f:
            f.write(modified_gpm_iter_content)
        print(f"Generated GPM file for iteration: {gpm_file_for_this_run}")
        current_priors_display = {name: spec["current_value"] for name, spec in current_priors_config.items()}
        print(f"Priors used for this iteration: {json.dumps(current_priors_display, indent=2)}")

        # 2. Run Andres's BVAR model script
        quant_results, plot_paths = run_andres_bvar_model(gpm_file_for_this_run, iter_output_dir)
        
        # Simplify quant_results for logging and AI prompt if it's too verbose
        # e.g., if quant_results contains full MCMC summary text.
        mcmc_summary_short = "N/A"
        if quant_results and isinstance(quant_results.get("mcmc_summary"), str): # Example field
            mcmc_summary_short = quant_results["mcmc_summary"][:300] + "..." # Truncate for prompt
        elif quant_results:
            mcmc_summary_short = json.dumps({k:v for k,v in quant_results.items() if k not in ['full_mcmc_samples_path']})[:300] + "..."


        # 3. Get AI suggestions
        ai_output = get_ai_suggestions(current_priors_config, quant_results, plot_paths, history)

        # 4. Store this iteration's result
        history_entry = {
            "iteration": i,
            "priors_used_config": current_priors_config.copy(), # Save the full config with desc, min, max
            "priors_used_values": {k: v['current_value'] for k, v in current_priors_config.items()},
            "gpm_file_generated": gpm_file_for_this_run,
            "output_dir": iter_output_dir,
            "quantitative_results": quant_results,
            "mcmc_summary_short_for_ai_prompt": mcmc_summary_short, # For logging what AI saw
            "plot_paths": plot_paths,
            "ai_assessment_text": ai_output.get("ai_assessment_text"),
            "ai_suggested_prior_values": ai_output.get("suggested_prior_values"),
            "ai_reasoning": ai_output.get("reasoning"),
            "ai_stop_signal": ai_output.get("stop_signal", False)
        }
        history.append(history_entry)

        # 5. Apply AI suggestions for next iteration
        if ai_output.get("suggested_prior_values"):
            current_priors_config, changes_made = apply_ai_suggestions_auto(
                current_priors_config,
                ai_output["suggested_prior_values"]
            )
            history_entry["changes_applied_by_ai_count"] = changes_made
        else:
            print("AI provided no new prior suggestions.")
            history_entry["changes_applied_by_ai_count"] = 0

        # 6. Save state
        save_json_state(current_priors_config, PRIOR_STATE_FILE)
        save_json_state(history, CALIBRATION_HISTORY_FILE)

        # 7. Check stopping criteria
        if ai_output.get("stop_signal", False):
            print(f"\nAI signaled to stop at iteration {i}. Assessment: {ai_output.get('ai_assessment_text')}")
            break
        
        if i == start_iteration + max_iterations - 1:
            print("Reached maximum iterations.")
            break
        
        print(f"Pausing for a moment before next iteration...")
        time.sleep(5) # Small pause

    print("\n--- Autonomous Calibration Workflow Finished ---")
    print(f"Final priors config saved to: {PRIOR_STATE_FILE}")
    print(f"Full history saved to: {CALIBRATION_HISTORY_FILE}")

# --- Helper: Script to be called by subprocess (main_custom_plots_adapted.py) ---
# This is a conceptual adaptation of your main_custom_plots.py.
# You'll need to integrate this logic into your actual main_custom_plots.py
# or create a new script that wraps its core functionality.

if __name__ == "__main__":
    import argparse
    # Check if this script is being run as the main orchestrator
    # or if it's being called as the "MODEL_RUNNER_SCRIPT"
    
    parser = argparse.ArgumentParser(description="Main BVAR Model Runner or Autonomous Calibrator")
    parser.add_argument("--run_calibration_loop", action="store_true", help="Run the autonomous calibration loop.")
    
    # Arguments for when this script acts as the model runner (called by subprocess)
    parser.add_argument("--gpm_file", type=str, help="Path to the GPM file to use for model run.")
    parser.add_argument("--output_dir", type=str, help="Directory to save plots and metrics for model run.")

    args = parser.parse_args()

    if args.run_calibration_loop:
        print("Starting autonomous calibration loop...")
        autonomous_calibration_loop(max_iterations=3) # Set desired number of iterations
    elif args.gpm_file and args.output_dir:
        # This block executes if the script is called with --gpm_file and --output_dir
        # It means this script is NOW `main_custom_plots.py` (or its equivalent)
        # being run by the orchestrator.
        print(f"Running as model executor with GPM: {args.gpm_file} and Output: {args.output_dir}")
        
        # --- ADAPTED main_custom_plots.py LOGIC ---
        # Initialize paths based on args.output_dir
        plot_save_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_save_dir, exist_ok=True)
        metrics_output_file = os.path.join(args.output_dir, "metrics.json")

        # Data loading (remains the same as your main_custom_plots.py)
        dta_path = os.path.join(os.path.dirname(__file__), "data_m5.csv") # Assuming data_m5.csv is relative
        data_source = dta_path
        dta = pd.read_csv(data_source)
        dta['Date'] = pd.to_datetime(dta['Date'])
        dta.set_index('Date', inplace=True)
        dta = dta.asfreq('QE')  
        data_sub = dta[['y_us', 'y_ea', 'y_jp']]

        # GPM file path comes from the argument
        gpm_file_path_from_arg = args.gpm_file

        # Custom plot specifications (remains the same)
        custom_plot_specifications = [
            {"title": "World Trend vs US Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'blue'}, {'type': 'observed', 'name': 'y_us', 'label': 'Observed US', 'style': 'k--'}]},
            {"title": "World Trend vs EA Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'green'},{'type': 'observed', 'name': 'y_ea', 'label': 'Observed EA', 'style': 'k--'}]},
            {"title": "World Trend vs JP Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_world', 'label': 'World Trend', 'show_hdi': True, 'color': 'red'},{'type': 'observed', 'name': 'y_jp', 'label': 'Observed JP', 'style': 'k--'}]},
            {"title": "Country-Specific Trends", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_us', 'label': 'US Trend Comp', 'show_hdi': True, 'color': 'cyan'},{'type': 'trend', 'name': 'trend_y_ea', 'label': 'EA Trend Comp', 'show_hdi': True, 'color': 'magenta'},{'type': 'trend', 'name': 'trend_y_jp', 'label': 'JP Trend Comp', 'show_hdi': True, 'color': 'brown'}]},
            {"title": "US Trend vs US Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_us_d', 'label': 'US Trend (Full)', 'show_hdi': True, 'color': 'purple'},{'type': 'observed', 'name': 'y_us', 'label': 'Observed US', 'style': 'k--'}]},
            {"title": "EA Trend vs EA Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_ea_d', 'label': 'EA Trend (Full)', 'show_hdi': True, 'color': 'orange'},{'type': 'observed', 'name': 'y_ea', 'label': 'Observed EA', 'style': 'k--'}]},
            {"title": "JP Trend vs JP Data", "series_to_plot": [{'type': 'trend', 'name': 'trend_y_jp_d', 'label': 'JP Trend (Full)', 'show_hdi': True, 'color': 'gray'},{'type': 'observed', 'name': 'y_jp', 'label': 'Observed JP', 'style': 'k--'}]}
        ]
        
        metrics_to_save = {}
        try:
            # sys.path needs to be correctly set for these imports IF this script is run directly
            # If autonomous_calibration_script.py is in the same dir or clean_gpm_bvar_trends is in PYTHONPATH, it should be fine.
            # This assumes 'clean_gpm_bvar_trends' is a package sibling to 'applications' or in PYTHONPATH.
            # When called by subprocess from autonomous_calibration_script.py, sys.path is inherited.
            
            # Add parent of 'clean_gpm_bvar_trends' to sys.path if it's not already importable
            # This is often needed if the package structure is `project/clean_gpm_bvar_trends` and `project/applications`
            # And we are running from `project/applications`
            module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Goes up to 'project' level
            if module_path not in sys.path:
                sys.path.insert(0, module_path)

            from clean_gpm_bvar_trends.gpm_bar_smoother_old import complete_gpm_workflow_with_smoother_fixed
            # Import jax and numpyro here if not at top, or ensure they are loaded.
            import jax
            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
            import numpyro
            
            results = complete_gpm_workflow_with_smoother_fixed(
                data=data_sub,
                gpm_file=gpm_file_path_from_arg, # Use the GPM file passed as argument
                num_warmup=100, num_samples=100, num_chains=2, # Keep these relatively low for calibration speed
                target_accept_prob=0.85, use_gamma_init=True, gamma_scale_factor=1.0,
                num_extract_draws=50, generate_plots=True,
                hdi_prob_plot=0.9, show_plot_info_boxes=True,
                custom_plot_specs=custom_plot_specifications,
                plot_save_path=plot_save_dir, # Save plots to the iteration-specific plot subdir
                save_plots=True
            )

            if results and results.get('mcmc_object'):
                mcmc_summary = az.summary(results['mcmc_object'], hdi_prob=0.9)
                metrics_to_save["mcmc_summary_df"] = mcmc_summary.to_dict() # Save summary as dict
                # You might want to pull out specific metrics like min ESS, max r_hat
                metrics_to_save["min_ess_bulk"] = float(mcmc_summary['ess_bulk'].min()) if 'ess_bulk' in mcmc_summary else None
                metrics_to_save["max_rhat"] = float(mcmc_summary['r_hat'].max()) if 'r_hat' in mcmc_summary else None
                # Add any other specific metrics you want from the 'results' dictionary
                metrics_to_save["fitting_time_seconds"] = results.get('fitting_time_seconds')
                print("Model execution part completed successfully.")
            else:
                metrics_to_save["error"] = "Workflow did not return expected MCMC results."
                print("Workflow completed but MCMC results might be missing.")

        except Exception as e_model:
            import traceback
            print(f"Error during model execution in subprocess: {e_model}")
            metrics_to_save["error"] = f"Model execution error: {str(e_model)}"
            metrics_to_save["traceback"] = traceback.format_exc()
        
        # Save metrics to JSON
        with open(metrics_output_file, 'w') as f_metrics:
            json.dump(metrics_to_save, f_metrics, indent=4, default=str) # Use default=str for non-serializable
        print(f"Metrics saved to {metrics_output_file}")

        # Ensure plots are closed to free memory if many iterations
        plt.close('all')

    else:
        print("This script can be run in two modes:")
        print("1. As the main autonomous calibration loop: `python autonomous_calibration_script.py --run_calibration_loop`")
        print("2. As the model executor (called by the loop): `python autonomous_calibration_script.py --gpm_file <path> --output_dir <path>`")
        print("   (Typically, mode 2 is not run manually but by the calibration loop itself).")