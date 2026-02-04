import numpy as np
from pysr import PySRRegressor, jl
#from numpy.lib.function_base import percentile
import h5py
import os.path
import glob 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import sympy
from functools import reduce
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import shap 
from numpy import linspace, meshgrid
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import gaussian_kde
import pdb 
import pickle
from astropy import units as u, constants as const
from astropy.table import QTable
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
from itertools import starmap
from sympy import simplify, sympify, symbols, latex
from latex2sympy2 import latex2sympy
from xgb_shap_pysr_functions import *

if __name__ == "__main__":
    ####### READ IN DATA FILE #######

    #df_all_params_filename     = "df_ALL_M12galaxies_all_params.csv" #put this in an if statement to call func if file not found
    #df_ALL_galaxies_all_params = pd.read_csv(df_all_params_filename)
    parent_path       = "~/SYMR_FIRE2_GIT/"
    all_df_picklefile = parent_path + "all_galaxies_all_params_redshift_bins_df.pickle"
    savepath          = parent_path + "SYNTHETIC_FG13/" 

    if (os.path.isfile(all_df_picklefile)==True):
        with open(all_df_picklefile, "rb") as f:
            all_redshifts_df_save_obj     = pickle.load(f) 
    else:
        all_redshifts_df_save_obj = make_data_redshift_bin()

    [df_all_params_z0_z0pt5,_,_] = all_redshifts_df_save_obj
     
    pixel_width    = 750 * u.pc #[pc], integer
    pixel_area     = (pixel_width)**2. #[pc^2]

    [Q_logZsfrFG13_var, Q_logZsfrFG13_var_noise, Q_logZsfrFG13, Q_logZsfr] = create_synthetic_FG13_dataset(df_all_params_z0_z0pt5)

    
    #df_logZsfrFG13_clean, df_logZsfr_clean = clean_dataframes(df_logZsfrFG13, df_logZsfr)
    
    #_               = make_shap_plot(df_ZsfrFG13_var,       df_ZsfrFG13, picklefile="xgb_shap_SYNTHETIC_FG13.pickle",       pdf_savefile="shap_summary_plot_SYNTHETIC_FG13.pdf")
    picklefile =  savepath+"xgb_shap_SYNTHETIC_FG13_noise.pickle"#_units.pickle"
    if (os.path.isfile(picklefile)==True):
        with open(picklefile, "rb") as f:
            xbg_save_object     = pickle.load(f)
    else:
        xbg_save_object = make_shap_plot(Q_logZsfrFG13_var_noise, Q_logZsfrFG13, picklefile=picklefile, pdf_savefile=savepath+"shap_summary_plot_SYNTHETIC_FG13_noise.pdf")

    [X_train,   X_test,   y_train,   y_test,   model_XBGreg, shap_values_reg] = xbg_save_object[0]
    [X_train_c, X_test_c, y_train_c, y_test_c, model_XBGcla, shap_values_cla] = xbg_save_object[1]

    filter_FUNC       = lambda df, y: np.squeeze(np.array(df.loc[y.index]))
    
    base_loss_FG13, base_r2_FG13, offset_fit_FG13, best_fit_FG13 = fixed_slope_calc_base_residuals(filter_FUNC(Q_logZsfrFG13.to_pandas(), y_train), filter_FUNC(Q_logZsfr.to_pandas(), y_train), 1)
    
    #y_XBGreg    = model_XBGreg.predict(X_test)
    #y_test_np   = np.squeeze(np.array(y_test))
    #XBGreg_LOSS, XBGreg_R2 = r2_loss_calc(y_test_np, y_XBGreg)

    y_XBGreg_train    = model_XBGreg.predict(X_train)
    y_train_np        = np.squeeze(np.array(y_train))
    XBGreg_LOSS_train, XBGreg_R2_train = r2_loss_calc(y_train_np, y_XBGreg_train)
    ############### DEFINE PYSR MODEL TO FIND EQUATIONS ######################
   
    n_epochs        = 3000
    
    #n_epochs        = 10
    eqns_picklefile = savepath + "SYNTHETIC_FG13_FOUND_EQNS_"+str(n_epochs)+"EPOCHS.pickle"
    
    rename_FUNC = lambda s: s.replace(", FG13", "")
    y_test      = y_test.rename(rename_FUNC, axis='columns')
    
    
    if (os.path.isfile(eqns_picklefile)==True):
        with open(eqns_picklefile, "rb") as f:
            eqns_save_object     = pickle.load(f) 
    else: 
            eqns_save_object = train_pysr_save_loss(X_train, X_test, y_train, y_test, n_epochs=n_epochs, eqns_picklefile=eqns_picklefile, eqns_picklefile_old=eqns_picklefile_old)

    test_feature = "\log\Sigma_{\mathrm{gas, cold}}"
    savefile = eqns_picklefile[:eqns_picklefile.find(".")]+".pdf"
    found_eqns_analysis_plots(eqns_save_object, X_train, X_test, y_train, y_test, savefile = savefile, max_loss = base_loss_FG13, test_feature=test_feature)