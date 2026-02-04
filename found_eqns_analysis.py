import sys
sys.set_int_max_str_digits(10000)
from xgb_shap_pysr_fake_FG13 import *
from xgb_shap_ALL_FEATS import *
import numpy as np
import h5py
import os.path
import glob 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.colors as mcolors 
import pandas as pd
import sympy
import shap 
import pdb 
import pickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from astropy import units as u, constants as const
#def extract_data_for_analytic_models(df):

def find_best_model(df, sfr_type):
    gas_mass    = np.array(df["GasNeutMass"])    * u.M_sun
    gas_vdisp   = np.array(df["GasNeutVDisp_z"]) * u.km/u.second
    omega_dyn   = np.array(df["omega_dyn"])      * u.Gyr**(-1)
    pixel_width = 750                            * u.pc

    if (sfr_type==10):
        sfr          = np.array(df["SFR_ave_010"])/0.85*1e6 * u.M_sun/u.Myr
    if (sfr_type==100):
        sfr          = np.array(df["SFR_ave_100"])/0.70*1e6 * u.M_sun/u.Myr

    model_losses_zip, df_analytic_sfr_models = sfr_analytic_models(sfr, gas_mass, gas_vdisp, omega_dyn, pixel_width=pixel_width)
    model_names, model_losses_alldata        = map(np.array, zip(*model_losses_zip)) 

    best_base_loss  = np.nanmin(model_losses_alldata)
    best_model_name = model_names[np.where(model_losses_alldata == best_base_loss)][0]
    
    return best_base_loss, best_model_name, df_analytic_sfr_models

def sfr_analytic_models_plot(df_analytic_sfr_models, X_test, y_XGBreg, test_feature_array, sfr_plot_title, plot_figures=True, sfr_comp_fig=True, savefile="sfr_models_comp.pdf"):
    df_analytic_sfr_models.insert(1, "XGBoost", y_XGBreg)
    y_data_np = np.array(df_analytic_sfr_models["DATA"])
    y_units   = "$ [\mathrm{M}_{\odot}\mathrm{pc}^{-2} \mathrm{Myr}^{-1}]$"
    y_label   = rf"$\log\Sigma_{{\mathrm{{SFR,{sfr_type}Myr}}}}~$"    

    # ---- SET AXES FOR SFR COMPARISONS PLOT ---- #
    f_sfr    = plt.figure(figsize=(14, 14))
    n_cols   = 2
    n_rows   = (len(df_analytic_sfr_models.keys())-1)//2
    gs_whole = GridSpec(n_rows, 6, figure=f_sfr)
    
    gs_sfr   = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs_whole[:, 0:-1], wspace=0, hspace=0)
    
    gs_cbar        = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_whole[:, -1], wspace=0, hspace=0)
    ax_cbar        = f_sfr.add_subplot(gs_cbar[:, 0])

    # ---- SET AXES FOR KS PLOTS ---- #
    f_sd               = plt.figure(figsize=(12, 16))
    f_sd.text(0.5, 0.9, sfr_plot_title, ha='center', fontsize=22)
    
    gs_whole_sd        = GridSpec(n_rows, n_cols, figure=f_sd, wspace=0, hspace=0)
    test_feature       = "\log \Sigma_{\mathrm{gas}}"
    test_feature_units = "$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$"
    x_label            = r"$" + test_feature + "$" + test_feature_units
    plot_x             = np.squeeze(X_test[test_feature].to_numpy())
    plot_y             = y_XGBreg #np.squeeze(y_test.to_numpy())
    #n_levels           = 5 
    n_levels           = [0.05, 0.2, 0.5, 0.8, 0.9999]
    n_levels0          = [n_levels[0]]
    m_b                = 7100  # minimum baryonic mass [M_sol] 
    pixel_width        = 750   # [pc]
    sfr_floor          = np.log10(m_b/((pixel_width**2)*sfr_type)) 
    ylim               = [-4.4, 1.2]
    xlim               = [0.5,  2.9]
    gas_cut            = np.log10(10)    # [M_sol pc^-2]
    K98_FUNC           = lambda x: 1.4*x + np.log10(2.5e-4)
    
    perr_m_model_LIST  = [] 
    popt_m_FIRE, perr_m_FIRE = find_dispersion(plot_x, plot_y) 
    
    for m , model_name in enumerate(df_analytic_sfr_models.keys()):
        if m > 0:
            n = m-1
            y_model_pred = df_analytic_sfr_models[model_name].to_numpy() 
            popt_m_model, perr_m_model = find_dispersion(plot_x, y_model_pred)
            perr_m_model_LIST.append(perr_m_model)

            if plot_figures==True:
                ax_sfr       = f_sfr.add_subplot(gs_whole[n//n_cols, n%n_cols])
                #gs_eqn       = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[n//n_cols,  n%n_cols])
                ax_sd        = f_sd.add_subplot(gs_whole_sd[n//n_cols, n%n_cols])
                ax_sd.set_ylim(ylim[0], ylim[1])
                ax_sd.set_xlim(xlim[0], xlim[1])
                ax_sd.tick_params(axis='both', which='major', labelsize=20) 
                ax_sd.text(0.07, 0.9, model_name, fontsize = 20, fontweight ="bold", bbox=dict(facecolor='bisque'), transform=ax_sd.transAxes)
                if (n//n_cols == n_rows-1):
                    ax_sfr.set_xlabel(y_label + y_units, fontsize=20)
                    ax_sd.set_xlabel(x_label, fontsize=20)
                else:
                    ax_sfr.xaxis.set_tick_params(labelbottom=False)
                    ax_sd.xaxis.set_tick_params(labelbottom=False)
                if (n//n_cols == n_rows-2) & (n%n_cols==0):
                    ax_sfr.set_ylabel(y_label + " (Analytic Model) " + y_units, fontsize=22)
                    ax_sd.set_ylabel(y_label + y_units, fontsize=22)
                else:
                    if (n%n_cols==1):
                        ax_sfr.yaxis.set_tick_params(labelleft=False)
                        ax_sd.yaxis.set_tick_params(labelleft=False)
            
                if sfr_comp_fig==True:
                    sfr_scatter, sfr_contours = sfr_comparison_plots(ax_sfr, y_data_np, y_model_pred, y_label, test_feature_array, model_name=model_name)
            
                sns.kdeplot(x=plot_x, y=plot_y,          cmap="Greys", fill=True, ax=ax_sd, bw_adjust=0.5, levels=n_levels)
                sns.kdeplot(x=plot_x, y=plot_y,          cmap="Accent_r",         ax=ax_sd, bw_adjust=0.5, levels=n_levels0)
    
                #if int(perr_m_model) == 0:
                if (n==1):
                    accent_green = mpl.colormaps['Accent']([0, 1])[0]
                    ax_sd.plot(plot_x, y_model_pred, linestyle='solid', linewidth=3.0, c=accent_green)
                else:
                    sns.kdeplot(x=plot_x, y=y_model_pred, cmap="Greens",     fill=True, ax=ax_sd, bw_adjust=0.5, levels=n_levels)
                    sns.kdeplot(x=plot_x, y=y_model_pred, cmap="Accent",                ax=ax_sd, bw_adjust=0.5, levels=n_levels0)

                model_loss, model_r2 = r2_loss_calc(plot_y, y_model_pred)
            
                ax_sd.text(0.07, 0.78, "$R^2$={:.3f}".format(model_r2), transform=ax_sd.transAxes, fontsize=18)
                ax_sd.text(0.07, 0.71, "$\sigma_{{\mathrm{{plane}}}}$={:.3f}".format(perr_m_model), transform=ax_sd.transAxes, fontsize=18)
                ax_sd.plot(xlim, K98_FUNC(np.array(xlim)),  color='dimgray', linestyle='dashed', linewidth=3.0, label = "K98")
                ax_sd.axhline(y=sfr_floor, c="lightgrey", linestyle="dashdot")
                ax_sd.axvline(x=gas_cut,   c="lightgrey", linestyle="dashdot")
                #eqn_behaviour(ax_sd, eqn_latex, X_test, test_feature=test_feature, comp_obj=comp_obj)
    
    if plot_figures==True:
        ax_sd.legend(loc="lower right", fontsize=20)
        f_sd.savefig(savefile[:savefile.find(".")]+"_behaviour.pdf", bbox_inches='tight')
        print("SAVED " + savefile)
    
        if sfr_comp_fig==True: 
            cbar = f_sfr.colorbar(sfr_scatter, cax=ax_cbar, cmap="Greens", location='right')
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(x_label, size=14)
            f_sfr.savefig(savefile, bbox_inches='tight')
    
    
    sd_residuals     = np.abs(perr_m_FIRE - perr_m_model_LIST[1:])
    where_min_sd_res = int(np.where(sd_residuals == np.nanmin(sd_residuals))[0])
    min_sd_res       = perr_m_model_LIST[1:][where_min_sd_res]
    min_sd_res_model = df_analytic_sfr_models.keys()[2:][where_min_sd_res]
    return min_sd_res, min_sd_res_model


def make_seperate_loss_curves(ax_FULL_loss_train, ax_FULL_loss_test, ax_QUANT_loss_train, ax_QUANT_loss_test, df_top_eqns, 
                             df_train_FULL_loss_top_eqns,     df_test_FULL_loss_top_eqns,
                             df_train_QUANTILE_loss_top_eqns, df_test_QUANTILE_loss_top_eqns,
                             min_loss=None, max_loss=1, max_loss_model=None, is_right=True): 
    
    n_epochs     = df_train_FULL_loss_top_eqns.index[-1] + 1  
    epochs_array = np.arange(n_epochs)+1
    FULL_ylim     = [0.262, 0.32]
    QUANT_ylim   = [0.00, 0.049]
    ax_FULL_loss_train.set_ylim(FULL_ylim[0], FULL_ylim[1])
    ax_FULL_loss_test.set_ylim(FULL_ylim[0], FULL_ylim[1])
    ax_QUANT_loss_train.set_ylim(QUANT_ylim[0], QUANT_ylim[1])
    ax_QUANT_loss_test.set_ylim(QUANT_ylim[0], QUANT_ylim[1])

    if is_right==False:
        ax_FULL_loss_train.set_ylabel("Full Loss (MSE + Quantile Loss)")
        ax_QUANT_loss_train.set_ylabel("Quantile Loss")
    else:
        ax_FULL_loss_train.yaxis.set_tick_params(labelleft=False)
        ax_QUANT_loss_train.yaxis.set_tick_params(labelleft=False)
    ax_FULL_loss_test.yaxis.set_tick_params(labelleft=False)
    ax_FULL_loss_test.xaxis.set_tick_params(labelbottom=False)
    ax_QUANT_loss_test.yaxis.set_tick_params(labelleft=False)

    ax_FULL_loss_train.axhline(y=max_loss, color='lightgray', linestyle='dashed', label=max_loss_model)
    ax_FULL_loss_train.xaxis.set_tick_params(labelbottom=False)
    ax_QUANT_loss_train.set_xlabel("Epochs")
    ax_QUANT_loss_test.set_xlabel("Epochs")

    if min_loss != None: 
        ax_FULL_loss_test.axhline(y=min_loss, color='lightgray', linestyle='dashdot', label='XGBoost')

    for n in range(len(df_top_eqns)):
    #### ---- plot PySR found equation ---- ####
        eqn_complexity   = int((df_top_eqns["complexity"])[n]) 
        eqn_colour       = complexity_colour(n, num_top_eqns=len(df_top_eqns))
        train_FULL_loss_array = np.array(df_train_FULL_loss_top_eqns[eqn_complexity]) 
        test_FULL_loss_array  = np.array(df_test_FULL_loss_top_eqns[eqn_complexity])
        
        train_QUANTILE_loss_array = np.array(df_train_QUANTILE_loss_top_eqns[eqn_complexity]) 
        test_QUANTILE_loss_array  = np.array(df_test_QUANTILE_loss_top_eqns[eqn_complexity])
        
        label = "complexity = "+str(eqn_complexity)
            
        ax_FULL_loss_train.plot(epochs_array, train_FULL_loss_array,     c=eqn_colour, label=label)
        ax_FULL_loss_test.plot(epochs_array,  test_FULL_loss_array,      c=eqn_colour, label=label)
        ax_QUANT_loss_train.plot(epochs_array, train_QUANTILE_loss_array, c=eqn_colour, label=label)#, linestyle="dashed")
        ax_QUANT_loss_test.plot(epochs_array,  test_QUANTILE_loss_array,  c=eqn_colour, label=label)#, linestyle="dashed")

    ax_FULL_loss_train.legend(loc="upper right", fontsize=10)


def both_sfr_timescales_analysis_plots(eqns_save_obj_LIST, xbg_save_object_LIST, best_loss_LIST, best_model_LIST, sfr_title_LIST=None, min_sd_res_zip_LIST=None, num_top_eqns=6, savefile="FOUND_EQNS.pdf"): 
    n_cols = len(eqns_save_obj_LIST)

    # ---- DEFINE ALL PLOT OBJECTS ---- ####
    #f_loss, ax_loss = plt.subplots(nrows=2, ncols=n_cols, sharex=True, sharey=True, figsize=(10, 8))
    #f_loss.subplots_adjust(hspace=0, wspace=0.05)
    f_loss        = plt.figure(figsize=(16, 8))
    gs_loss_whole = GridSpec(2, 2, figure=f_loss, wspace=0.05, hspace=0)

    f_train_metrics, ax_train_metrics = plt.subplots(nrows=2, ncols=n_cols, sharex=True, sharey=False, figsize=(9, 6))
    f_train_metrics.subplots_adjust(hspace=0, wspace=0.05)

    f_sd, ax_sd = plt.subplots(nrows=3, ncols=n_cols, sharex=True, sharey=False, figsize=(9, 9))
    f_sd.subplots_adjust(hspace=0, wspace=0.05)
    
    n_rows         = num_top_eqns
    f_sfr          = plt.figure(figsize=(14, 14))
    f_sfr.text(0.07, 0.5, "$\Sigma_{\mathrm{SFR}}~\mathrm{(Found~Equation)}~[\mathrm{M}_{\odot}\mathrm{pc}^{-2} \mathrm{Myr}^{-1}]$", va='center', rotation='vertical', fontsize=14)
    
    gs_sfr_whole   = GridSpec(n_rows, 7, figure=f_sfr)
    gs_sfr_10myr   = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs_sfr_whole[:, 0:3], wspace=0, hspace=0)
    gs_sfr_100myr  = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs_sfr_whole[:, 3:6], wspace=0, hspace=0)
    gs_cbar        = GridSpecFromSubplotSpec(1,      3, subplot_spec=gs_sfr_whole[:, -1],  wspace=0, hspace=0)
    ax_cbar        = f_sfr.add_subplot(gs_cbar[:, 0])
    gs_sfr_LIST    = [gs_sfr_10myr, gs_sfr_100myr]
    sfr_timescales = [10, 100]
    
    f_all_eqn, ax_all_eqn = plt.subplots(nrows=1, ncols=1, figsize = (6, 6))
    r2_axes = []
    used_markers = set()

    for i, eqns_save_obj in enumerate(eqns_save_obj_LIST):
        # ---- SPECIFY AXES TO DRAW ON FOR EACH PLOT ---- # 
        ax_FULL = f_loss.add_subplot(gs_loss_whole[:, i])
        ax_FULL.set_title(sfr_title_LIST[i])
        ax_FULL.axis("off")

        gs_FULL = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_loss_whole[0, i], wspace=0, hspace=0)
        ax_FULL_loss_train = f_loss.add_subplot(gs_FULL[0])
        ax_FULL_loss_test  = f_loss.add_subplot(gs_FULL[1])
        ax_FULL_loss_train.text(0.09, 0.9, "Train loss", bbox=dict(facecolor='lightgrey'), transform=ax_FULL_loss_train.transAxes)
        ax_FULL_loss_test.text(0.09, 0.9, "Test loss", bbox=dict(facecolor='lightgrey'), transform=ax_FULL_loss_test.transAxes)

        gs_QUANT = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_loss_whole[1, i], wspace=0, hspace=0)
        ax_QUANT_loss_train = f_loss.add_subplot(gs_QUANT[0])
        ax_QUANT_loss_test  = f_loss.add_subplot(gs_QUANT[1]) 
        #ax_QUANT_loss_train.text(0.15, 0.85, "Train loss", transform=ax_QUANT_loss_train.transAxes)
        #ax_QUANT_loss_test.text(0.15, 0.85, "Test loss", transform=ax_QUANT_loss_test.transAxes)

        ax_train_metrics_sfr = ax_train_metrics[:, i]
        ax_sd_sfr            = ax_sd[:, i]

        ax_train_metrics_sfr[0].set_title(sfr_title_LIST[i])
        ax_sd_sfr[0].set_title(sfr_title_LIST[i])
        
        gs_sfr = gs_sfr_LIST[i]
        # ---- EXTRACT TRAINING DATA AND RESULTANT EQUATIONS ---- ####

        (model_trained, df_train_loss_all_eqns, df_test_loss_all_eqns) = eqns_save_obj
        n_epochs          = len(df_train_loss_all_eqns.index)
        test_feature_name = "\log \Sigma_{\mathrm{gas}}"

        [X_train,     X_test,     y_train,     y_test,   model_XBGreg, shap_values_reg] = xbg_save_object_LIST[i]
        
        xgb_test_loss, xgb_test_r2 = r2_loss_calc(np.squeeze(y_test.to_numpy()), model_XBGreg.predict(X_test))
        
        no_columns             = X_train.columns.shape[0] 
        temp_column_names      = list(map(lambda n: "x"+str(n), np.arange(no_columns)))
        test_feature_temp_name = temp_column_names[int(np.where(np.array(X_test.columns.to_list()) == test_feature_name)[0])]
        X_test_tempname        = X_test.set_axis(temp_column_names, axis='columns')   
        y_test_np              = np.squeeze(np.array(y_test))
        y_units              = "[\mathrm{M}_{\odot}\mathrm{pc}^{-2} \mathrm{Myr}^{-1}]"
        y_label              = r"$" + (y_test.keys())[0] + "$"
        
        # ---- GET TOP EQNS ---- #    

        df_top_eqns, df_found_eqns   = filter_top_eqns(model_trained, X_train, best_loss_LIST[i], num_top_eqns=num_top_eqns)
        model_indices                = df_top_eqns.index
        df_top_eqns                  = df_top_eqns.reset_index(drop=True) 

        is_right=True
        if (i == 0):
            is_right=False
        min_sd_res_zip = min_sd_res_zip_LIST[i]
       
        #CHANGE LOOP SYNTAX
        for j, (index, row) in enumerate(df_top_eqns.iterrows()):
            ax_sfr           = f_sfr.add_subplot(gs_sfr[j])
            if j == 0:
                #ax_sfr.set_title(r"$\langle t_{\mathrm{SFR}} \rangle = $"+str(sfr_timescales[i])+" Myr", fontsize=16)
                ax_sfr.set_title(sfr_title_LIST[i], fontsize=16)
            if j == df_top_eqns.shape[0]-1:
                ax_sfr.set_xlabel(y_label + "$~"+y_units+"$", fontsize=14)
            else:
                ax_sfr.xaxis.set_tick_params(labelbottom=False)
            if i == 1:
                ax_sfr.yaxis.set_tick_params(labelleft=False)
            eqn_complexity   = int(row["complexity"]) #int((df_top_eqns["complexity"])[j]) 
            model_index      = int(row["model_orig_index"]) #int((df_top_eqns["model_orig_index"])[j])
            eqn_latex        = row["latex equation"]#(df_top_eqns["latex equation"])[j]
        
            y_model_trained = model_trained.predict(X_test_tempname, index=model_index) 
            test_feature    = X_test[test_feature_name].to_numpy()    

            sfr_scatter, sfr_contours = sfr_comparison_plots(ax_sfr, y_test_np, y_model_trained, y_label, test_feature, complexity=str(eqn_complexity), model_eqn=r"$"+eqn_latex+"$")
            

        df_top_eqns_metric, r2_axes, used_markers = plot_complexity_vs_r2(ax_train_metrics_sfr, ax_sd_sfr, model_trained, df_found_eqns, X_test_tempname, y_test, r2_axes, used_markers, 
                                                                          test_feature_temp_name=test_feature_temp_name, min_loss=xgb_test_loss, max_r2=xgb_test_r2, df_top_eqns=df_top_eqns,  
                                                                          min_sd_res_zip=min_sd_res_zip, is_right=is_right) 

        
        df_train_FULL_loss_top_eqns = df_train_loss_all_eqns["FULL_CUSTOM_LOSS"][df_top_eqns["complexity"]]
        df_test_FULL_loss_top_eqns  = df_test_loss_all_eqns["FULL_CUSTOM_LOSS"][df_top_eqns["complexity"]]
        
        df_train_QUANTILE_loss_top_eqns = df_train_loss_all_eqns["QUANTILE_LOSS"][df_top_eqns["complexity"]]
        df_test_QUANTILE_loss_top_eqns  = df_test_loss_all_eqns["QUANTILE_LOSS"][df_top_eqns["complexity"]]
         
        make_seperate_loss_curves(ax_FULL_loss_train, ax_FULL_loss_test, ax_QUANT_loss_train, ax_QUANT_loss_test, df_top_eqns,
                             df_train_FULL_loss_top_eqns,     df_test_FULL_loss_top_eqns,
                             df_train_QUANTILE_loss_top_eqns, df_test_QUANTILE_loss_top_eqns,
                             max_loss=best_loss_LIST[i], max_loss_model=best_model_LIST[i], is_right=is_right) #min_loss=xgb_test_loss


    sfr_scatter.set_clim(np.nanmin(test_feature), np.nanmax(test_feature))
    cbar = f_sfr.colorbar(sfr_scatter, cax=ax_cbar, cmap="Greens", location='right')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("$\log \Sigma_{\mathrm{gas}} ~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$", size=14)
    
    handles, labels = [], []
    for ax in r2_axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    f_train_metrics.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.11))
    
    # ---- SAVE ALL PLOT OBJECTS ---- ####
    f_loss.savefig(savefile[:savefile.find(".")]+"_loss.pdf",                 bbox_inches='tight')
    f_train_metrics.savefig(savefile[:savefile.find(".")]+"_train_metrics.pdf", bbox_inches='tight')
    f_sd.savefig(savefile[:savefile.find(".")]+"_complexity.pdf",             bbox_inches='tight')  
    f_sfr.savefig(savefile[:savefile.find(".")]+"_sfr_comparison.pdf",             bbox_inches='tight')
    print("SAVED " + savefile)

def Sigma_star_Omega_dyn_relation(X_test, sfr_type, savepath):
    log_Sigma_stars = np.array(X_test['\log \Sigma_{*}'])
    log_Omega_dyn   = np.array(X_test['\log \Omega_{\mathrm{dyn}}'])
    _, r2           = r2_loss_calc(log_Sigma_stars, log_Omega_dyn)
    
    # ---- LINFUNC ---- #
    linfit_FUNC     = lambda x, m, c: m*x + c
    popt, pcov      = curve_fit(linfit_FUNC, log_Sigma_stars, log_Omega_dyn)
    y_vals_model    = linfit_FUNC(log_Sigma_stars, *popt)
    _, r2_lin       = r2_loss_calc(log_Sigma_stars, y_vals_model)
    
    # ---- QUADFUNC ---- #
    quadfit_FUNC    = lambda x, m, c: m*x**2. + c
    popt2, pcov2    = curve_fit(quadfit_FUNC, log_Sigma_stars, log_Omega_dyn)
    y_vals_model2   = quadfit_FUNC(log_Sigma_stars, *popt2)
    _, r2_quad      = r2_loss_calc(log_Sigma_stars, y_vals_model2)
    
    # ---- PLOT ---- #
    f, ax           = plt.subplots(figsize=(6,6))
    ax.set_xlabel(r"$\log \Sigma_{*}$")
    ax.set_ylabel(r"$\log \Omega_{\mathrm{dyn}}$")
    ax.scatter(log_Sigma_stars, log_Omega_dyn)
    ax.plot(log_Sigma_stars, y_vals_model, label = "{:.2f}$\log \Sigma_{{*}}$ + {:.2f}, $R^2={:.2f}$".format(popt[0], popt[1], r2_lin), c="orange")
    ax.scatter(log_Sigma_stars, y_vals_model2, label = "{:.2f}$\log \Sigma_{{*}}^2$ + {:.2f}, $R^2={:.2f}$".format(popt2[0], popt2[1], r2_quad), c="red")
    ax.legend()
    f.savefig(savepath+"Sigma_star_Omega_dyn_relation_"+str(sfr_type)+"MYR.pdf")


def run_found_eqns_analysis_plots(sfr_type, savepath, df, filtered=False):
    sfr_str          = "_z_ALL_SFR_"+str(sfr_type)+"Myr" 
    if filtered==True:
        sfr_str += "_FILTERED"

    sfr_str_savepath = savepath + sfr_str[1:]+"/"
    xgb_picklefile   = glob.glob(sfr_str_savepath + "xgb*.pickle")[0]
    with open(xgb_picklefile, "rb") as f:
        xbg_save_object     = pickle.load(f)
    [X_train,     X_test,     y_train,     y_test,   model_XBGreg, shap_values_reg] = xbg_save_object[0]
    [X_train_lin, X_test_lin, y_train_lin, y_test_lin] = list(map(lambda df: df.apply(power_10_FUNC).rename(lin_name_FUNC, axis='columns'), [X_train, X_test, y_train, y_test]))
        
    df_train = df.iloc[X_train.index]
    df_test  = df.iloc[X_test.index]

    #Sigma_star_Omega_dyn_relation(X_test, sfr_type, savepath)
    #pdb.set_trace()
    best_loss_train, best_model_train, df_sfr_model_ytrain = find_best_model(df_train, sfr_type)
    best_loss_test,  best_model_test,  df_sfr_model_ytest  = find_best_model(df_test,  sfr_type)
        
    y_XGBreg           = model_XBGreg.predict(X_test)
    test_feature_array = X_test["\log \Sigma_{\mathrm{gas}}"].to_numpy()
    sfr_comp_fig       = False #True
    plot_figures       = True
    sfr_plot_title     = sfr_title_LIST[i]
    min_sd_res_zip = sfr_analytic_models_plot(df_sfr_model_ytest, X_test, y_XGBreg, test_feature_array, sfr_plot_title, plot_figures=plot_figures, sfr_comp_fig=sfr_comp_fig, savefile=sfr_str_savepath+"sfr_models_comp"+sfr_str+".pdf")

    eqns_picklefile  = (glob.glob(sfr_str_savepath + "*"+str(n_epochs)+"EPOCHS.pickle"))[0]
    #for j, eqns_picklefile in enumerate(eqns_picklefile_list):
    with open(eqns_picklefile, "rb") as f:
        eqns_save_object = pickle.load(f)    

    savefile = eqns_picklefile[:eqns_picklefile.find(".")]+".pdf"
    found_eqns_analysis_plots(eqns_save_object, X_train, X_test, y_train, y_test, sfr_type=sfr_type, num_top_eqns=num_top_eqns, max_loss=best_loss_train, savefile = savefile)
    
    return eqns_save_object, xbg_save_object[0], best_loss_train, best_model_train, min_sd_res_zip
    
#### =========================================== ####
#### ---------- CALLING MAIN FUNCTION ---------- #### 
#### =========================================== ####

if __name__ == "__main__":
    
    # -------- second argument in terminal call is index for redshift bin range
    # -------- eg. run shap_all_parameters 2
    ######## LOAD IN RAW DATA ########

    galmap_mdir       = "/mnt/home/morr/ceph/analysis/sfr/galmap/"
    redshift_txt      = "/mnt/home/chayward/firesims/fire2/public_release/core/snapshot_times_public.txt"
    all_df_picklefile = "all_galaxies_all_params_redshift_bins_df.pickle"
    sfr_timescales    = [10, 100] #units: Myr
    #sfr_timescales    = [10] #units: Myr
    #savepath          = "/mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS/"
    savepath          = "/mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/"
    n_epochs_old      = 0
    n_epochs          = 4000
    #n_epochs          = 3000
    num_top_eqns      = 4 #6
    
    filename_FUNC           = (lambda sp: (lambda n, ex: (lambda zs: sp + zs[1:] + "/" + n + zs + ex)))(savepath) 

    with open(all_df_picklefile, "rb") as f:
        all_redshifts_df_save_obj     = pickle.load(f)
    df = pd.concat(all_redshifts_df_save_obj, ignore_index=True)

    power_10_FUNC  = lambda x: 10**(x)
    lin_name_FUNC  = lambda x: x.replace("\log ", "")
    sfr_title_FUNC = lambda x: f"$\\langle t_{{\\mathrm{{SFR}}}}\\rangle$ = {x} Myr"

    eqns_save_object_LIST = []
    xbg_save_object_LIST  = []
    best_loss_LIST        = []
    best_model_LIST       = []
    sfr_title_LIST        = list(map(sfr_title_FUNC, sfr_timescales))
    min_sd_res_zip_LIST   = []

    for i, sfr_type in enumerate(sfr_timescales):
        
        eqns_save_object, xbg_save_object, best_loss_train, best_model_train, min_sd_res_zip = run_found_eqns_analysis_plots(sfr_type, savepath, df, filtered=False)
        
        eqns_save_object_LIST.append(eqns_save_object)
        xbg_save_object_LIST.append(xbg_save_object)
        best_loss_LIST.append(best_loss_train)
        best_model_LIST.append(best_model_train)
        min_sd_res_zip_LIST.append(min_sd_res_zip)

    eqns_save_object_filt, xbg_save_object_filt, best_loss_train_filt, best_model_train_filt, min_sd_res_zip_filt = run_found_eqns_analysis_plots(sfr_type, savepath, df, filtered=True)

    pdb.set_trace()
    savefile_bothsfr = savepath+"FOUND_EQNS_"+str(n_epochs)+"EPOCHS.pdf"
    both_sfr_timescales_analysis_plots(eqns_save_object_LIST, xbg_save_object_LIST, best_loss_LIST, best_model_LIST, sfr_title_LIST=sfr_title_LIST, min_sd_res_zip_LIST=min_sd_res_zip_LIST, num_top_eqns=num_top_eqns, savefile=savefile_bothsfr)
