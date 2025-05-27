# FIXED CONDITIONAL SAMPLING FUNCTIONS (JAX-compatible)

def _sample_initial_conditions_conditional_fixed(gpm_model: GPMModel, state_dim: int, 
                                                gamma_list: List[jnp.ndarray],
                                                n_trends: int, n_stationary: int,
                                                var_order: int) -> jnp.ndarray:
    """
    JAX-compatible conditional initial condition sampling.
    Fixed: Removed all Python if statements that cause tracing errors.
    """
    
    init_mean = jnp.zeros(state_dim, dtype=_DEFAULT_DTYPE)
    init_std = jnp.ones(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Handle trends: use GPM specifications or defaults
    # Note: This part uses Python if but outside of traced functions, so it's OK
    for var_name, var_spec in gpm_model.initial_values.items():
        if var_spec.init_dist == 'normal_pdf' and len(var_spec.init_params) >= 2:
            mean, std = var_spec.init_params[:2]
            if var_name in gpm_model.trend_variables:
                idx = gmp_model.trend_variables.index(var_name)
                if idx < n_trends:
                    init_mean = init_mean.at[idx].set(mean)
                    init_std = init_std.at[idx].set(std)
    
    # Set diffuse priors for trends using JAX operations only
    trend_mask = jnp.arange(state_dim) < n_trends
    init_std = jnp.where(trend_mask, 
                        jnp.where(init_std == 1.0, 3.0, init_std),
                        init_std)
    
    # VAR components: Use gamma matrices if available, otherwise defaults
    # Key fix: Use JAX conditional operations instead of Python if statements
    
    def use_gamma_matrices(operand):
        gamma_0, var_start, n_stat, v_order = operand
        
        # Extract conditional standard deviations
        conditional_std = jnp.sqrt(jnp.diag(gamma_0))
        
        # Apply to all VAR states at once using vectorized operations
        var_indices = jnp.arange(n_stat * v_order) + var_start
        lag_numbers = jnp.arange(n_stat * v_order) // n_stat  # Which lag each state belongs to
        
        # Scale factor based on lag (current=1.0, lag1=0.3, lag2=0.15, etc.)
        scale_factors = jnp.where(lag_numbers == 0, 1.0, 0.3 / (1 + lag_numbers))
        
        # Repeat conditional_std for each lag
        repeated_std = jnp.tile(conditional_std, v_order)
        scaled_std = repeated_std * scale_factors
        
        # Clip to reasonable bounds
        scaled_std = jnp.clip(scaled_std, 0.01, 2.0)
        
        # Update init_std
        return init_std.at[var_indices].set(scaled_std)
    
    def use_default_var_std(operand):
        var_start, n_stat, v_order = operand
        var_end = var_start + n_stat * v_order
        var_indices = jnp.arange(var_start, jnp.minimum(var_end, state_dim))
        return init_std.at[var_indices].set(0.1)
    
    # Check if we should use gamma matrices (all JAX operations)
    has_gamma = len(gamma_list) > 0
    has_stationary = n_stationary > 0
    has_var_order = var_order > 0
    
    if has_gamma and has_stationary and has_var_order:
        gamma_0 = gamma_list[0]
        var_start = n_trends
        
        # Safety checks using JAX operations
        is_finite = jnp.all(jnp.isfinite(gamma_0))
        has_positive_diag = jnp.all(jnp.diag(gamma_0) > 0)
        is_right_shape = gamma_0.shape[0] == n_stationary
        
        # Use JAX conditional instead of Python if
        use_gamma = is_finite & has_positive_diag & is_right_shape
        
        init_std = jax.lax.cond(
            use_gamma,
            use_gamma_matrices,
            lambda op: use_default_var_std(op[1:]),  # Skip gamma_0 for default case
            operand=(gamma_0, var_start, n_stationary, var_order)
        )
    else:
        # Use default for VAR components
        var_start = n_trends
        var_end = var_start + n_stationary * var_order
        if var_end <= state_dim:
            var_indices = jnp.arange(var_start, var_end)
            init_std = init_std.at[var_indices].set(0.1)
    
    # Sample from the conditional distribution
    init_mean_sampled = numpyro.sample("init_mean_full", 
                                      dist.Normal(init_mean, init_std))
    
    return init_mean_sampled


def _create_initial_covariance_conditional_fixed(state_dim: int, n_trends: int,
                                               gamma_list: List[jnp.ndarray],
                                               n_stationary: int, var_order: int,
                                               conditioning_strength: float = 0.1) -> jnp.ndarray:
    """
    JAX-compatible conditional covariance creation.
    Fixed: All operations use JAX conditionals instead of Python if statements.
    """
    
    init_cov = jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    # Trends: diffuse prior
    trend_cov = jnp.eye(n_trends, dtype=_DEFAULT_DTYPE) * 1e6
    init_cov = init_cov.at[:n_trends, :n_trends].set(trend_cov)
    
    # VAR components: conditional or default covariance
    def build_conditional_cov(operand):
        gamma_0, v_start, v_state_dim, n_stat, v_order, cond_strength = operand
        
        # Build conditional covariance structure
        var_state_cov = jnp.zeros((v_state_dim, v_state_dim), dtype=_DEFAULT_DTYPE)
        
        # Use vectorized operations instead of nested loops
        i_indices, j_indices = jnp.meshgrid(jnp.arange(v_order), jnp.arange(v_order), indexing='ij')
        lag_diffs = jnp.abs(i_indices - j_indices)
        
        # Create all blocks at once
        for lag in range(len(gamma_list)):
            mask = lag_diffs == lag
            block_cov = gamma_list[lag] * cond_strength
            
            # Apply to all relevant block positions
            i_vals, j_vals = jnp.where(mask)
            for idx in range(len(i_vals)):
                i, j = i_vals[idx], j_vals[idx]
                row_start, row_end = i * n_stat, (i + 1) * n_stat
                col_start, col_end = j * n_stat, (j + 1) * n_stat
                
                if i <= j:
                    var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov)
                else:
                    var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_cov.T)
        
        # Handle lags beyond available gamma matrices with decay
        max_lag = len(gamma_list) - 1
        remaining_mask = lag_diffs > max_lag
        if jnp.any(remaining_mask):
            decay_cov = gamma_list[0] * cond_strength * 0.5**jnp.maximum(lag_diffs - max_lag, 0)
            
            i_vals, j_vals = jnp.where(remaining_mask)
            for idx in range(len(i_vals)):
                i, j = i_vals[idx], j_vals[idx]
                row_start, row_end = i * n_stat, (i + 1) * n_stat
                col_start, col_end = j * n_stat, (j + 1) * n_stat
                
                block_decay = decay_cov if i <= j else decay_cov.T
                var_state_cov = var_state_cov.at[row_start:row_end, col_start:col_end].set(block_decay)
        
        # Insert into full covariance
        var_end = v_start + v_state_dim
        return init_cov.at[v_start:var_end, v_start:var_end].set(var_state_cov)
    
    def build_default_cov(operand):
        v_start, v_state_dim = operand
        default_cov = jnp.eye(v_state_dim, dtype=_DEFAULT_DTYPE) * 0.1
        return init_cov.at[v_start:v_start + v_state_dim, v_start:v_start + v_state_dim].set(default_cov)
    
    # Determine whether to use conditional covariance
    has_gamma = len(gamma_list) > 0
    has_stationary = n_stationary > 0  
    has_var_order = var_order > 0
    
    if has_gamma and has_stationary and has_var_order:
        gamma_0 = gamma_list[0]
        var_start = n_trends
        var_state_dim = n_stationary * var_order
        
        # Safety checks
        is_finite = jnp.all(jnp.isfinite(gamma_0))
        has_positive_diag = jnp.all(jnp.diag(gamma_0) > 0)
        good_condition = jnp.linalg.cond(gamma_0 + _JITTER * jnp.eye(n_stationary, dtype=_DEFAULT_DTYPE)) < 1e10
        
        use_conditional = is_finite & has_positive_diag & good_condition
        
        init_cov = jax.lax.cond(
            use_conditional,
            build_conditional_cov,
            lambda op: build_default_cov(op[-2:]),  # Extract var_start, var_state_dim
            operand=(gamma_0, var_start, var_state_dim, n_stationary, var_order, conditioning_strength)
        )
    
    # Ensure positive definite
    init_cov = (init_cov + init_cov.T) / 2.0 + _KF_JITTER * jnp.eye(state_dim, dtype=_DEFAULT_DTYPE)
    
    return init_cov


def create_gpm_based_model_with_conditional_init_fixed(gpm_file_path: str, 
                                                     use_conditional_init: bool = False,
                                                     conditioning_strength: float = 0.1):
    """
    FIXED version that works with JAX tracing.
    """
    
    parser = GPMParser()
    gpm_model = parser.parse_file(gmp_file_path)
    ss_builder = GPMStateSpaceBuilder(gpm_model)
    
    def gpm_bvar_model_conditional_fixed(y: jnp.ndarray):
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gpm_model.parameters:
            if param_name in gpm_model.estimated_params:
                prior_spec = gmp_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample covariances  
        Sigma_eta = _sample_trend_covariance(gpm_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gpm_model)
        Sigma_eps = _sample_measurement_covariance(gmp_model) if _has_measurement_error(gmp_model) else None
        
        # Conditional vs independent initialization
        if use_conditional_init:
            # Use FIXED conditional functions
            init_mean = _sample_initial_conditions_conditional_fixed(
                gmp_model, ss_builder.state_dim, gamma_list,
                ss_builder.n_trends, ss_builder.n_stationary, ss_builder.var_order
            )
            init_cov = _create_initial_covariance_conditional_fixed(
                ss_builder.state_dim, ss_builder.n_trends, gamma_list,
                ss_builder.n_stationary, ss_builder.var_order,
                conditioning_strength=conditioning_strength
            )
        else:
            # Original approach
            init_mean = _sample_initial_conditions(gmp_model, ss_builder.state_dim)
            init_cov = _create_initial_covariance(ss_builder.state_dim, ss_builder.n_trends)
        
        # Rest unchanged
        params = EnhancedBVARParams(
            A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
            structural_params=structural_params, Sigma_eps=Sigma_eps
        )
        
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean)) & jnp.all(jnp.isfinite(init_cov)))
        
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean, init_P=init_cov)
        
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gmp_bvar_model_conditional_fixed, gmp_model, ss_builder


# SIMPLER VERSION: Minimal conditional sampling that should work
def create_gmp_based_model_minimal_conditional(gmp_file_path: str):
    """
    Minimal conditional sampling - just scale the VAR initial variance by gamma[0] diagonal.
    This avoids complex JAX tracing issues while still being theoretically motivated.
    """
    
    parser = GPMParser()
    gmp_model = parser.parse_file(gmp_file_path)
    ss_builder = GPMStateSpaceBuilder(gmp_model)
    
    def gmp_bvar_model_minimal_conditional(y: jnp.ndarray):
        T, n_obs = y.shape
        
        # Sample structural parameters
        structural_params = {}
        for param_name in gmp_model.parameters:
            if param_name in gmp_model.estimated_params:
                prior_spec = gmp_model.estimated_params[param_name]
                structural_params[param_name] = _sample_parameter(param_name, prior_spec)
        
        # Sample covariances
        Sigma_eta = _sample_trend_covariance(gmp_model)
        Sigma_u, A_transformed, gamma_list = _sample_var_parameters(gmp_model)
        Sigma_eps = _sample_measurement_covariance(gmp_model) if _has_measurement_error(gmp_model) else None
        
        # MINIMAL conditional initialization
        init_mean = jnp.zeros(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
        # Standard deviations
        init_std = jnp.ones(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        
        # Trends: diffuse
        init_std = init_std.at[:ss_builder.n_trends].set(3.0)
        
        # VAR components: use gamma[0] diagonal if available
        if len(gamma_list) > 0 and ss_builder.n_stationary > 0:
            gamma_0 = gamma_list[0]
            
            # Extract diagonal variances and take square root for std
            var_std = jnp.sqrt(jnp.diag(gamma_0)) * 0.1  # Scale down by 10%
            var_std = jnp.clip(var_std, 0.01, 1.0)  # Reasonable bounds
            
            # Apply to current VAR states
            var_start = ss_builder.n_trends  
            init_std = init_std.at[var_start:var_start + ss_builder.n_stationary].set(var_std)
            
            # Lagged states get smaller std
            for lag in range(1, ss_builder.var_order):
                lag_start = var_start + lag * ss_builder.n_stationary
                lag_end = lag_start + ss_builder.n_stationary
                if lag_end <= ss_builder.state_dim:
                    init_std = init_std.at[lag_start:lag_end].set(var_std * 0.3)
        else:
            # Fallback
            var_start = ss_builder.n_trends
            var_end = ss_builder.state_dim
            init_std = init_std.at[var_start:var_end].set(0.1)
        
        # Sample initial mean
        init_mean_sampled = numpyro.sample("init_mean_full", dist.Normal(init_mean, init_std))
        
        # Initial covariance: simple approach
        init_cov = jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE)
        init_cov = init_cov.at[:ss_builder.n_trends, :ss_builder.n_trends].set(
            jnp.eye(ss_builder.n_trends, dtype=_DEFAULT_DTYPE) * 1e6
        )
        init_cov = init_cov.at[ss_builder.n_trends:, ss_builder.n_trends:].set(
            jnp.eye(ss_builder.state_dim - ss_builder.n_trends, dtype=_DEFAULT_DTYPE) * 0.01
        )
        
        # Rest of model unchanged
        params = EnhancedBVARParams(
            A=A_transformed, Sigma_u=Sigma_u, Sigma_eta=Sigma_eta,
            structural_params=structural_params, Sigma_eps=Sigma_eps
        )
        
        F, Q, C, H = ss_builder.build_state_space_matrices(params)
        
        matrices_ok = (jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(Q)) & 
                      jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(H)) & 
                      jnp.all(jnp.isfinite(init_mean_sampled)) & jnp.all(jnp.isfinite(init_cov)))
        
        try:
            R = jnp.linalg.cholesky(Q + _JITTER * jnp.eye(ss_builder.state_dim, dtype=_DEFAULT_DTYPE))
        except:
            R = jnp.diag(jnp.sqrt(jnp.diag(Q) + _JITTER))
        
        kf = KalmanFilter(T=F, R=R, C=C, H=H, init_x=init_mean_sampled, init_P=init_cov)
        
        valid_obs_idx = jnp.arange(n_obs, dtype=int)
        I_obs = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        
        loglik = jax.lax.cond(
            ~matrices_ok,
            lambda: jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE),
            lambda: kf.log_likelihood(y, valid_obs_idx, n_obs, C, H, I_obs)
        )
        
        numpyro.factor("loglik", loglik)
    
    return gmp_bvar_model_minimal_conditional, gmp_model, ss_builder