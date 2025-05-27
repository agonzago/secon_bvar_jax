try:
    from gpmcore import GPMParser
    from gpmcore import KalmanFilter
    from gpmcore import fit_gpm_model_with_smoother
    print("Successfully imported GPMParser, KalmanFilter, and fit_gpm_model_with_smoother from gpmcore.")
    print("Basic package structure seems OK.")
except ImportError as e:
    print(f"Error importing from gpmcore: {e}")
    print("There might be an issue with the package setup or __init__.py.")

# Test a more complex import if possible, e.g., one that involves sub-dependencies within gpmcore
try:
    from gpmcore.gpm_bvar_trends import GPMStateSpaceBuilder # Depends on gpm_parser
    print("Successfully imported GPMStateSpaceBuilder.")
except ImportError as e:
    print(f"Error importing GPMStateSpaceBuilder: {e}")

try:
    from gpmcore.simulation_smoothing import jarocinski_corrected_simulation_smoother # Depends on KalmanFilter
    print("Successfully imported jarocinski_corrected_simulation_smoother.")
except ImportError as e:
    print(f"Error importing jarocinski_corrected_simulation_smoother: {e}")
