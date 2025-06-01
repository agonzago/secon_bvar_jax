# ... (inside run_parameter_sensitivity_workflow)
        try:
            eval_results = evaluate_gpm_at_parameters(
                gpm_file_path=base_config.gpm_file_path,
                y=y_jax_for_eval, # This is jnp.array(data_df.values)
                param_values=current_fixed_params,
                num_sim_draws=base_config.num_smoother_draws_for_fixed_params, # MODIFIED
                plot_results=False, # MODIFIED: Control plotting externally
                use_gamma_init_for_test=base_config.use_gamma_init,
                gamma_init_scaling=base_config.gamma_scale_factor,
                variable_names=base_config.observed_variable_names
            )
            study_results['all_eval_results'].append(eval_results)

            if eval_results and 'loglik' in eval_results and jnp.isfinite(eval_results['loglik']):
                loglik_val = float(eval_results['loglik'])
                study_results['log_likelihoods'].append(loglik_val)
                study_results['run_status'].append('success')
                print(f"    ✓ LogLik: {loglik_val:.3f}")

                # --- MODIFIED: Plotting for this sensitivity point ---
                if base_config.plot_sensitivity_point_results and \
                   eval_results.get('reconstructed_original_trends') is not None and \
                   eval_results['reconstructed_original_trends'].shape[0] > 0: # Check for actual draws
                    
                    print(f"    Generating plots for {parameter_name_to_vary}={p_val}...")
                    reconstructed_trends_np = np.asarray(eval_results['reconstructed_original_trends'])
                    reconstructed_stationary_np = np.asarray(eval_results['reconstructed_original_stationary'])
                    gpm_model_eval = eval_results['gpm_model']
                    trend_names_gpm = gpm_model_eval.gpm_trend_variables_original
                    stat_names_gpm = gpm_model_eval.gpm_stationary_variables_original
                    
                    fig_title_suffix = f"({parameter_name_to_vary}={p_val})"
                    
                    # Prepare save path if needed
                    plot_save_prefix_point = None
                    if base_config.save_plots and base_config.plot_save_path:
                        # Create a subdirectory for each sensitivity point's plots for organization
                        point_plot_dir = os.path.join(base_config.plot_save_path, "sensitivity_points", f"param_{parameter_name_to_vary}_val_{p_val}".replace('.', '_'))
                        os.makedirs(point_plot_dir, exist_ok=True)
                        plot_save_prefix_point = os.path.join(point_plot_dir, "plot")


                    # 1. Plot Trend Components
                    if reconstructed_trends_np.shape[2] > 0: # Check if there are trend variables
                        fig_trends = plot_time_series_with_uncertainty(
                            reconstructed_trends_np,
                            variable_names=trend_names_gpm,
                            hdi_prob=base_config.plot_hdi_prob,
                            title_prefix=f"Trend Components {fig_title_suffix}",
                            show_info_box=base_config.show_plot_info_boxes,
                            time_index=time_index_for_plots # Passed to function
                        )
                        if plot_save_prefix_point:
                            fig_trends.savefig(f"{plot_save_prefix_point}_trends.png", dpi=150, bbox_inches='tight')
                        plt.show()
                        plt.close(fig_trends)


                    # 2. Plot Observed vs. Sum of Trend Components (or custom specs)
                    actual_custom_specs_for_point = base_config.sensitivity_plot_custom_specs
                    if actual_custom_specs_for_point is None: # Generate default specs
                        default_sensitivity_custom_specs = []
                        for obs_name_iter in base_config.observed_variable_names:
                            if obs_name_iter in gpm_model_eval.reduced_measurement_equations:
                                me = gpm_model_eval.reduced_measurement_equations[obs_name_iter]
                                trend_components_in_me_for_sum = []
                                for term_var_name, coeff_str in me.terms.items():
                                    # Here, term_var_name is the name of the variable in the GPM (e.g., 'trend_y_us_d')
                                    if term_var_name in trend_names_gpm:
                                        # For 'combined' plot type, just list components. Assumes positive sum.
                                        # More complex coefficient handling would require enhancing plot_custom_series_comparison
                                        # or using a different plotting approach.
                                        trend_components_in_me_for_sum.append({'type': 'trend', 'name': term_var_name})
                                
                                if trend_components_in_me_for_sum:
                                    series_specs = [
                                        {'type': 'observed', 'name': obs_name_iter, 'label': f'Observed {obs_name_iter}', 'style': 'k-'},
                                        {'type': 'combined',
                                         'name': f'fitted_trends_{obs_name_iter}', # Name for the combined series
                                         'components': trend_components_in_me_for_sum,
                                         'label': f'Sum of Trends for {obs_name_iter}', 'show_hdi': True, 'color':'green'}
                                    ]
                                    default_sensitivity_custom_specs.append({
                                        "title": f"Observed vs. Sum of Trends for {obs_name_iter} {fig_title_suffix}",
                                        "series_to_plot": series_specs
                                    })
                        actual_custom_specs_for_point = default_sensitivity_custom_specs

                    if actual_custom_specs_for_point:
                        for spec_idx, spec_dict_item in enumerate(actual_custom_specs_for_point):
                            fig_custom = plot_custom_series_comparison(
                                plot_title=spec_dict_item.get("title", f"Custom Plot {spec_idx+1}") + f" {fig_title_suffix}",
                                series_specs=spec_dict_item.get("series_to_plot", []),
                                observed_data=np.asarray(data_df[base_config.observed_variable_names].values),
                                trend_draws=reconstructed_trends_np,
                                stationary_draws=reconstructed_stationary_np, # Pass for completeness
                                observed_names=base_config.observed_variable_names,
                                trend_names=trend_names_gpm,
                                stationary_names=stat_names_gpm,
                                time_index=time_index_for_plots,
                                hdi_prob=base_config.plot_hdi_prob
                            )
                            if plot_save_prefix_point:
                                safe_title_fig = spec_dict_item.get("title", f"custom_{spec_idx+1}").lower().replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('=','_').replace(':','')
                                fig_custom.savefig(f"{plot_save_prefix_point}_custom_{safe_title_fig}.png", dpi=150, bbox_inches='tight')
                            plt.show()
                            plt.close(fig_custom)
            # ... (rest of the loop)