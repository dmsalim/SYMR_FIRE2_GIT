import sys
import numpy as np
import h5py
import os.path
import glob 
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import shap 
from scipy.optimize import curve_fit
import pdb
import pickle
from astropy.constants import G
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import random
import argparse
from xgb_shap_pysr_fake_FG13 import *
from xgb_shap_ALL_FEATS import *

#### ================================================== ####
#### ---------- GET ARGUMENTS FOR EXPERIMENT ---------- ####
#### ================================================== ####
def get_args(): 
    parser = argparse.ArgumentParser(description="Run PySR experiments")

    parser.add_argument("--input_df_picklefile", type=str,  default=None,
                            help="picklefile of DataFrames of input variables & target variable")
    parser.add_argument("--galmap_mdir",         type=str,  default="/mnt/home/morr/ceph/analysis/sfr/galmap/",
                            help="directory where galaxy folders live")
    parser.add_argument("--redshift_txt",        type=str,  default="/mnt/home/chayward/firesims/fire2/public_release/core/snapshot_times_public.txt",
                            help="directory and filename of txt file with info regarding redshift and snapshot index of galaxies")
    parser.add_argument("--all_df_picklefile",   type=str,  default="~/SYMR_FIRE2_GIT/all_galaxies_all_params_redshift_bins_df.pickle",
                            help="directory and filename of picklefile containing all extracted data from galaxy hdf files")
    parser.add_argument("--savepath",            type=str,  default="~/SYMR_FIRE2_GIT/RUN_XGB_SHAP_PYSR_ZBINS/", 
                            help="directory to save analysis plots")
    parser.add_argument("--xgb_picklefile",      type=str,  default="~/SYMR_FIRE2_GIT/RUN_XGB_SHAP_PYSR_ZBINS/xgb_shap_z0_z0pt5_SFR_10MYR.pickle",
                            help="directory and filename of picklefile containing model and data used in training XGBoost model")

    parser.add_argument("--redshift_index",      type=str,  default=None,           help="which redshift bin to train on,if an integer (default:None, to train on all z as one dataset)")
    parser.add_argument("--epochs",              type=int,  default=10000,          help="total number of epochs to train to (default: 10000)")
    parser.add_argument("--epochs_old",          type=int,  default=0,              help="number of epochs of model to train from (default: 0)") 
    parser.add_argument("--n_saves_train",       type=int,  default=1,              help="number of times to save training progress")
    parser.add_argument("--sfr_type",            type=int,  default=10,             help="short (10Myr)  or long (100Myr) SFR to consider")
    parser.add_argument("--pixel_width",         type=int,  default=750,            help="length in pc of pixel edge")
    parser.add_argument("--redshift_bins",       type=list, default=[0, 0.5, 1, 2], help="redshift bins")          
    
    args = parser.parse_args()
    return args

#### ===================================================================== ####
#### ---------- FUNCTIONS TO RUN PYSR AND ANALYSE SFR RELATIONS ---------- ####
#### ===================================================================== ####

def sfr_analytic_models_plot(df_sfr_model_ytest, y_XGBreg, test_feature, savefile="sfr_models_comp.pdf"):
    df_sfr_model_ytest.insert(1, "XGBoost", y_XGBreg)
    y_data_np = np.array(df_sfr_model_ytest["DATA"])
    y_label   = r"$\log\Sigma_{\mathrm{SFR}}$"    

    f_sfr    = plt.figure(figsize=(12, 12))
    gs_whole = GridSpec((len(df_sfr_model_ytest.keys())-1)//2, 2, figure=f_sfr)
    
    for n , model in enumerate(df_sfr_model_ytest.keys()):
        if n > 0:
            y_model_pred = df_sfr_model_ytest[model]
            ax_sfr       = f_sfr.add_subplot(gs_whole[(n-1)//2,  (n-1)%2])
            #gs_sfr       = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_whole[(n-1)//2,  (n-1)%2]) 
            sfr_comparison_plots(f_sfr, gs_sfr, y_data_np, y_model_pred, y_label, model_name=model)
    
    f_sfr.savefig(savefile, bbox_inches='tight')
    print("SAVED " + savefile)

def get_min_loss_analytic_model(df_all_params, y_train, y_test, y_train_XGBreg, pixel_width=750, sfr_type=10):
 
    if sfr_type == 10:
        (sfr_key, sfr_divider) = ("SFR_ave_010", 0.85)
    if sfr_type == 100:
        (sfr_key, sfr_divider) = ("SFR_ave_100", 0.70)
   
    pixel_width                 = pixel_width * u.pc
    analytic_model_params       = [sfr_key,       "GasNeutMass", "GasNeutVDisp_z", "omega_dyn"] #change from GasColdMass to GasNeutMass etc
    analytic_model_params_units = [u.M_sun/u.Myr, u.M_sun,       u.km/u.second,    u.Gyr**(-1) ] 
    sfr_raw, gas_cold_mass, gas_cold_vdisp, omega_dyn = list(map(lambda p, u: np.array(df_all_params[p])*u, analytic_model_params, analytic_model_params_units))
    sfr = sfr_raw/sfr_divider*1e6 #[M_sol/Myr^-1]

    # ---------- get sfr predictions from analytic models and their losses for ALL data points ---------- #
    model_MSE_losses_zip, df_analytic_sfr_models = sfr_analytic_models(sfr, gas_cold_mass, gas_cold_vdisp, omega_dyn, pixel_width=pixel_width)
    model_names, model_MSE_losses_alldata        = map(np.array, zip(*model_MSE_losses_zip)) 
    
    df_sfr_model_ytest       = df_analytic_sfr_models.loc[y_test.index]
    df_sfr_model_ytrain      = df_analytic_sfr_models.loc[y_train.index]
    df_sfr_model_ytrain.insert(1, "XGBoost", y_train_XGBreg)

    # ---------- work out which model has minimum FULL CUSTOM loss (MSE + quantile) ---------- #
    df_sfr_model_ytrain_list      = [row for row in np.array(df_sfr_model_ytrain.drop(['DATA'], axis=1).values.tolist()).transpose()]
    full_custom_loss_models_train = list(map((lambda i: (lambda j: full_custom_loss(np.squeeze(np.array(i)), j)))(y_train), df_sfr_model_ytrain_list))
    
    #loss_r2_models_train     = list(map((lambda i: (lambda j: r2_loss_calc(np.squeeze(np.array(i)), j)))(y_train), df_sfr_model_ytrain_list))   
    #model_losses_train, model_r2s_train = map(np.array, zip(*loss_r2_models_train)) 
    #XGBreg_LOSS_train, XGBreg_R2_train = model_losses_train[0], model_r2s_train[0]
    #best_sfr_model = list(map((lambda n:(lambda l: (np.nanmin(l), n[np.where(l == np.nanmin(l))][0])))(model_names), [model_losses_alldata, model_losses_train[1:]]))
    
    best_sfr_model = list(map((lambda n:(lambda l: (np.nanmin(l), n[np.where(l == np.nanmin(l))][0])))(model_names), [model_MSE_losses_alldata, full_custom_loss_models_train[1:]]))
    print("BEST ANALYTIC MODEL FOR ALL DATA IS "      + best_sfr_model[0][1] + " WITH MSE LOSS " + str(best_sfr_model[0][0]))
    print("BEST ANALYTIC MODEL FOR TRAINING DATA IS " + best_sfr_model[1][1] + " WITH FULL CUSTOM LOSS " + str(best_sfr_model[1][0]))
    
    (min_loss_sfr_laws, min_loss_model) = best_sfr_model[1]
    
    return df_sfr_model_ytest, df_sfr_model_ytrain, min_loss_sfr_laws, min_loss_model

def analyse_sfr_models_run_pysr(df_all_params, xbg_save_obj, sfr_type, n_epochs=3000, n_saves=5, pixel_width=750, eqns_picklefile="FOUND_EQNS.pickle", eqns_picklefile_old=None):

    # ---------- extract data to find best model on test set ---------- #
    [X_train, X_test, y_train, y_test, model_XGBreg, shap_values_reg] = xbg_save_obj[0]
    
    [y_test_XGBreg, y_train_XGBreg] = list(map((lambda m: (lambda X: m.predict(X)))(model_XGBreg), [X_test, X_train]))
    
    XGBreg_LOSS_test,  XGBreg_R2_test  = r2_loss_calc(np.squeeze(np.array(y_test)), y_test_XGBreg )

    # ---------- call function to calculate analytic model with minimium FULL CUSTOM loss (MSE + quantile)---------- #
    df_sfr_model_ytest, df_sfr_model_ytrain, min_loss_sfr_laws, min_loss_model = get_min_loss_analytic_model(df_all_params, y_train, y_test, y_train_XGBreg, pixel_width=pixel_width, sfr_type=sfr_type)
    
    # ---------- plot FIRE data against SFR predictions from analytic models for test set ---------- #
    #insert_index             = [i for i, x in enumerate(eqns_picklefile) if x == "/"][-1]
    #sfr_models_comp_savepath = eqns_picklefile[:insert_index] + "/sfr_models_comp" + eqns_picklefile[eqns_picklefile.find("_z"):eqns_picklefile.find("Myr")+3] + ".pdf"
    
    #sfr_analytic_models_plot(df_sfr_model_ytest, y_test_XGBreg, savefile=sfr_models_comp_savepath)

    # ---------- load in picklefile with trained model or run PySR to train model to n_epochs ---------- #
    if (os.path.isfile(eqns_picklefile)==True):
        with open(eqns_picklefile, "rb") as f:
            eqns_save_object     = pickle.load(f) 
    else: 
            eqns_save_object = train_pysr_save_loss(X_train, X_test, y_train, y_test, n_epochs=n_epochs, n_saves=n_saves, sfr_type=sfr_type, min_loss_sfr_laws=min_loss_sfr_laws, eqns_picklefile=eqns_picklefile)    
   
    print("FINISHED TRAINING")
    (model_trained, df_train_loss_all_eqns, df_test_loss_all_eqns) = eqns_save_object

    # ---------- run PySR (if not already trained) and plot FIRE data against SFR predictions from FOUND models for test set ---------- #
    #savefile = eqns_picklefile[:eqns_picklefile.find(".")]+".pdf"
    #make_loss_sfrcomp_plot(eqns_save_object, X_train, X_test, y_train, y_test, n_epochs=n_epochs, savefile=savefile,  min_loss=XGBreg_LOSS_test, max_loss=min_loss_sfr_laws, max_loss_model=min_loss_model)

#### =========================================== ####
#### ---------- CALLING MAIN FUNCTION ---------- #### 
#### =========================================== ####

if __name__ == "__main__":
    if '-f' in sys.argv:
        sys.argv.remove('-f')    
    
    args                = get_args()
    input_df_picklefile = args.input_df_picklefile
    xgb_picklefile      = args.xgb_picklefile
    ind                 = args.redshift_index
    n_epochs            = args.epochs
    pixel_width         = args.pixel_width
    sfr_type            = args.sfr_type
    n_saves             = args.n_saves_train
    savepath            = args.savepath
    folder_index        = [i for i, x in enumerate(savepath) if x == "/"]
    exp_id              = savepath[folder_index[-2]+1:folder_index[-1]] 

    ######## LOAD IN RAW DATA ########
    if (os.path.isfile(args.all_df_picklefile)==True):
        with open(args.all_df_picklefile, "rb") as f:
            all_redshifts_df_save_obj     = pickle.load(f) 
    else: 
        all_redshifts_df_save_obj = make_data_redshift_bin(galmap_mdir       = args.galmap_mdir, 
                                                           redshift_txt      = args.redshift_txt, 
                                                           redshift_bins     = args.redshift_bins, 
                                                           pixel_width       = args.pixel_width, 
                                                           all_df_picklefile = args.all_df_picklefile)
    
    if ind == "None": 
        df_all_params = pd.concat(all_redshifts_df_save_obj, ignore_index=True)
    else: 
        ind = int(ind)
        df_all_params = all_redshifts_df_save_obj[ind]


    ######## LOAD IN TRAINED XGBOOST MODEL (OR EXECUTE IF NOT) ########    
    if (os.path.isfile(xgb_picklefile)==True):
        with open(xgb_picklefile, "rb") as f:
            xbg_save_obj    = pickle.load(f)
    else:
        if (os.path.isfile(input_df_picklefile)==True): 
            with open(input_df_picklefile, "rb") as f:
                input_df_save_object     = pickle.load(f)
                df_log_filt_feats, df_log_Zsfr = input_df_save_object
        else:
            df_log_filt_feats, df_log_Zsfr = make_df_filt_feats(df_all_params, pixel_width, sfr_type)
        pdf_filename = "shap_summary_plot" + exp_id + ".pdf"
        xbg_save_obj = make_shap_plot(df_log_filt_feats, df_log_Zsfr, test_feature_name="\\log \\Sigma_{\\mathrm{gas}}", picklefile=xgb_picklefile, pdf_savefile=pdf_filename)

    df_all_params = df_all_params.loc[df_log_filt_feats.index]

    ######## RUN PYSR ########
    found_eqns_picklefile = savepath + "FOUND_EQNS_" + exp_id + "_"+str(n_epochs)+"EPOCHS.pickle" 
    analyse_sfr_models_run_pysr(df_all_params, 
                                xbg_save_obj, 
                                sfr_type,
                                n_epochs        = n_epochs, 
                                n_saves         = n_saves,
                                pixel_width     = pixel_width, 
                                eqns_picklefile = found_eqns_picklefile)
                                #eqns_picklefile_old = eqns_pdf_files_list_old)

