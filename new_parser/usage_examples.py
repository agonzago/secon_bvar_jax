
# Example 1: Replace existing parser in your MCMC code
# OLD CODE:
# from gpm_bvar_trends import create_gpm_based_model
# model_fn, gpm_model, ss_builder = create_gpm_based_model('model.gpm')

# NEW CODE:
from integration_helper import create_reduced_gpm_model
model_fn, gpm_model, ss_builder = create_reduced_gpm_model('model.gpm')

# The interface remains the same! Your existing MCMC code should work unchanged.

# Example 2: Enhanced version with fallback
from integration_helper import enhanced_create_gpm_based_model
model_fn, gpm_model, ss_builder = enhanced_create_gpm_based_model(
    'model.gpm', 
    use_reduction=True  # Set to False to use original parser
)

# Example 3: For simulation smoother compatibility
from integration_helper import ReducedModelWrapper
integration, reduced_model, builder = create_reduced_gpm_model('model.gpm')
wrapper = ReducedModelWrapper(integration)
compatible_interface = wrapper.get_compatible_interface()

# Use compatible_interface with your existing simulation smoother

# Example 4: Direct state space access
F, Q, C, H = integration.build_state_space_matrices(params)
# F, Q, C, H are ready for your Kalman filter

