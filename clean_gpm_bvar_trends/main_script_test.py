# From your main script outside the package
# Add the parent directory to the Python path if needed
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_gpm_bvar_trends import run_complete_gpm_analysis, run_quick_example

# Run a quick example
results = run_quick_example(analysis_type="mcmc", save_results=True)

# Or run with your own data
results = run_complete_gpm_analysis(
    data="my_data.csv",
    gpm_file="my_model.gpm", 
    analysis_type="mcmc"
)