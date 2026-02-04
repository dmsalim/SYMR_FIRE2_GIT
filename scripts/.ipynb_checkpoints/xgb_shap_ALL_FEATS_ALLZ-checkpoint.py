import sys
import numpy as np
import h5py
import os.path
import glob 
import juliacall
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import shap 
from scipy.optimize import curve_fit
import pdb
import pickle
from astropy.constants import G
#from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import random
import argparse
from functions_run_xgb_shap import *
from functions_run_pysr import *
from xgb_shap_ALL_FEATS import *
#from shaphypetune import BoostRFE, BoostBoruta

#### =========================================== ####
#### ---------- CALLING MAIN FUNCTION ---------- #### 
#### =========================================== ####

if __name__ == "__main__":
    
    # -------- second argument in terminal call is index for redshift bin range
    # -------- eg. run shap_all_parameters 2
    ######## LOAD IN RAW DATA ########

    galmap_mdir       = "/mnt/home/morr/ceph/analysis/sfr/galmap/"
    redshift_txt      = "/mnt/home/chayward/firesims/fire2/public_release/core/snapshot_times_public.txt"
    pixel_width       = 750
    all_df_picklefile = "all_galaxies_all_params_redshift_bins_df.pickle"
    sfr_timescales    = [10, 100] #units: Myr
    disk_types        = ["INNER_DISK", "OUTER_DISK"]
    savepath          = "~/SYMR_FIRE2_GIT/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/"
    n_epochs_old      = 0
    n_epochs          = 4000
    n_saves           = 8
    
    if (os.path.isfile(all_df_picklefile)==True):
        with open(all_df_picklefile, "rb") as f:
            all_redshifts_df_save_obj     = pickle.load(f) 
    else: 
        all_redshifts_df_save_obj = make_data_redshift_bin(galmap_mdir=galmap_mdir, redshift_txt=redshift_txt, redshift_bins=redshift_bins, pixel_width=pixel_width, all_df_picklefile=all_df_picklefile) 
    
    df_all_params_all_z = pd.concat(all_redshifts_df_save_obj, ignore_index=True)

    filename_FUNC           = (lambda sp: (lambda n, ex: (lambda zs: sp + zs[1:] + "/" + n + zs + ex)))(savepath) 

    ######## RUN XGBOOST & SHAP ########
    df_log_Zsfr_list = []

    pdb.set_trace()
    for i, sfr_type in enumerate(sfr_timescales):
        sfr_str    = "_z_ALL_SFR_"+str(sfr_type)+"Myr" 
        
        xgb_file   = filename_FUNC("xgb_shap"         , ".pickle")(sfr_str)
        pdf_file   = filename_FUNC("shap_summary_plot", ".pdf")(sfr_str)

        df_log_filt_feats, df_log_Zsfr = make_df_filt_feats(df_all_params_all_z, pixel_width, sfr_type)

        df_log_Zsfr_list.append(df_log_Zsfr)
         
        write_sh_script(sfr_str, xgb_file, sfr_type, redshift_index=None, pixel_width=pixel_width, n_epochs=n_epochs, n_saves=n_saves, savepath=savepath)
        
        #xbg_save_obj = make_shap_plot(df_log_filt_feats, df_log_Zsfr, xgb_file, pdf_file)
        
    # ----- extract only pixels finite in 10 Myr SFR data -----
    df_combined_all_feats = pd.concat(df_log_Zsfr_list + [df_log_filt_feats], axis=1)
    df_combined_all_feats.replace({np.inf:np.nan, -np.inf:np.nan}, inplace=True)
    df_combined_all_feats = df_combined_all_feats.dropna()

    df_log_Zsfr_100_filt       = pd.DataFrame(np.array(df_combined_all_feats['\log \Sigma_{\mathrm{SFR, 100Myr}}']), columns=['\log \Sigma_{\mathrm{SFR, 100Myr}}'], index=df_combined_all_feats.index)
    df_combined_all_feats_filt = df_combined_all_feats.drop(['\log \Sigma_{\mathrm{SFR, 10Myr}}', '\log \Sigma_{\mathrm{SFR, 100Myr}}'], axis=1)

    sfr_str_filt      = sfr_str + "_FILTERED"
    xgb_file_filt     = filename_FUNC("xgb_shap"         , ".pickle")(sfr_str_filt)

    input_df_save_object         = [df_combined_all_feats_filt, df_log_Zsfr_100_filt]
    input_df_picklefile          = "df_all_feats_Zsfr_100_filt.pickle"
    input_df_picklefile_savepath = savepath + sfr_str_filt[1:] + "/" + input_df_picklefile

    pdb.set_trace()
    with open(input_df_picklefile_savepath, "wb") as f:
        pickle.dump(input_df_save_object, f)

    write_sh_script(sfr_str_filt, xgb_file_filt, sfr_type, redshift_index=None, pixel_width=pixel_width, n_epochs=n_epochs, n_saves=n_saves, input_df_picklefile=input_df_picklefile_savepath, savepath=savepath)

    #xbg_save_obj_filt = run_xgboost_shap_values(df_combined_all_feats_filt, df_log_Zsfr_100_filt, picklefile=xgb_file_filt)
    


         



