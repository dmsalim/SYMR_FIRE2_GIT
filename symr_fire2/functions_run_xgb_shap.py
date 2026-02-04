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

#### ============================================================ ####
#### ---------- FUNCTIONS TO CALCULATE DATA STATISTICS ---------- ####
#### ============================================================ ####

def r2_loss_calc(data, prediction):
    data_mean = np.nanmean(data)

    SStot = np.nansum((data - data_mean)**2.)
    SSres = np.nansum((data - prediction)**2.)
    R2    = 1. - (SSres/SStot)

    MSE_LOSS  = SSres/len(data)

    return MSE_LOSS, R2

def quantile_loss_calc(data, prediction):
    quantiles     = [0.1, 0.25, 0.5, 0.75, 0.9]
    q_data        = [np.percentile(data, q) for q in quantiles]
    q_pred        = [np.percentile(prediction, q) for q in quantiles]
    quantile_loss = np.nansum((np.array(q_data) - np.array(q_pred)) ** 2.) / len(quantiles)
    return quantile_loss

def full_custom_loss(data, prediction, gamma=1):
    mse_loss, _      = r2_loss_calc(data, prediction)
    quantile_loss    = quantile_loss_calc(data, prediction)
    full_custom_loss = mse_loss + gamma*quantile_loss
    return full_custom_loss

def fixed_slope_calc_base_residuals(x_vals, y_vals, gradient):
    x_vals, y_vals = list(map(lambda a: np.squeeze(np.array(a)), [d for d in clean_dataframes(pd.DataFrame(x_vals), pd.DataFrame(y_vals))]))
    #x_vals, y_vals = list(map(lambda a: a[0].tolist(), [d for d in clean_dataframes(pd.DataFrame(x_vals), pd.DataFrame(y_vals))]))
    fixed_slope_FUNC   = (lambda g: (lambda x, a: g * x + a))(gradient)
    popt, pcov         = curve_fit(fixed_slope_FUNC, x_vals.tolist(), y_vals.tolist())
    best_fit_y_vals    = fixed_slope_FUNC(x_vals, *popt)
    base_loss, base_r2 = r2_loss_calc(y_vals, best_fit_y_vals)
    offset_fit         = popt[0]
    #return_object      = [base_loss, base_r2, offset_fit, best_fit_y_vals]
    return base_loss, base_r2, offset_fit, best_fit_y_vals

def find_dispersion(x_vals, y_vals):
    mask         = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    linfit_FUNC  = lambda x, m, c: m*x + c
    popt, pcov   = curve_fit(linfit_FUNC, x_vals[mask].tolist(), y_vals[mask].tolist())
          #popt  = Optimal values for the parameters [m, c] so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
          #pcov  = The estimated approximate covariance of popt. 
    perr         = np.sqrt(np.diag(pcov)) #one standard deviation errors on the parameters
    y_vals_model = linfit_FUNC(x_vals, *popt) 
    return popt[0], perr[0]#, y_vals_model

def find_chi_squared(data, model):
    chi_squared = np.nansum(((data - model)**2.) / model)
    return chi_squared

def log10_units(quantity_with_units):
    if hasattr(quantity_with_units, 'unit') and quantity_with_units.unit is not None:
        output = np.log10(quantity_with_units.value) * quantity_with_units.unit
    else:
        output = np.log10(quantity_with_units.value) * u.dimensionless_unscaled
    return output

def log10_units_QTable(qtable):
    log_rename_FUNC        = lambda x: f"\log~{x}"
    log_data = {}
    for col_name in qtable.colnames:
        log_col_name           = log_rename_FUNC(col_name)
        log_data[log_col_name] = log10_units(qtable[col_name])
    # Create a new QTable with LogQuantity columns
    log_qtable = QTable(log_data)
    return log_qtable

    
#### ================================================================================= ####
#### ---------- FUNCTION FOR CALCULATING SIGMA_SFR BASED ON ANALYTIC MODELS ---------- ####
#### ================================================================================= ####

def sfr_analytic_models(sfr, gas_mass, gas_vdisp, omega_dyn, 
                        pixel_width =750*u.pc, P_on_mstar = 1500*u.km/u.s, c_speed=0.4*u.km/u.s, 
                        b_parameter = 0.3,     alpha_vir  = 10):#,            output_log_space=True):
    #ALL INPUT CONSTANTS MUST HAVE UNITS

    year       = 31557600. *u.second/u.year # year [sec/yr]
    phi_x      = 0.17                        # +/- 0.02; fixed “fudge factor” of order unity (introduced in Krumholz & McKee (2005))
    pixel_area = (pixel_width)**2. #[pc^2]
    
    #omega_dyn  = omega_dyn.to("Myr") #/ (1e3)         #[Myr^-1] (raw units in Gyr^-1) #*year) #units: s^-1
    Zsfr       = sfr / pixel_area                               #[M_sol/pc^2/Myr^-1]
    
    # ---- K 98 ----
    Zgas_mol                              = gas_mass / pixel_area                 #[M_sol/pc^2]        
    Zgas_cut                              =  10. * u.M_sun/u.pc**2 
    Zgas_mol[np.where(Zgas_mol<Zgas_cut)] = np.nan                                #apply gas cut
    K98_FUNC                              = lambda x: 1.4*x + np.log10(2.5e-4)
    
    # ---- KDM 12 ----- 
    H_gas        = (gas_vdisp / omega_dyn).to("pc")                           # gas_vdisp*1e5 / omega_dyn / const.pc.cgs  #SCALE HEIGHT units: pc
    rho          = (Zgas_mol  / H_gas).to("g/cm3")                            #Zgas_mol / H_gas * const.M_sun.cgs / (const.pc.cgs **3.) #[g cm^-3]
    tff          = (np.sqrt(3. * np.pi / rho /  32. / const.G.cgs)).to("Myr")
    Zgas_tff     = Zgas_mol / tff                                             #np.array(Zgas_mol / tff * 1e6 * year) # [M_sol pc^-2 Myr^-1]
   
    e_ff         = Zsfr / Zgas_tff # efficiency_per_freefall
     
    # ---- SFK 15 ---- 
    mach         = gas_vdisp/c_speed
    turb_term    = (1. + b_parameter**2. * mach**2.)**(3./8.)
    Zgas_multiff = Zgas_tff*turb_term
    
    # ---- SAF 20 ----
    #alpha_vir        = (gas_vdisp.to("pc/yr"))**2./(const.G.to("pc3/M_sun/yr2") * pixel_width * Zgas_mol)
    s_crit           = np.log((np.pi**2./5.) * phi_x**2. * alpha_vir * mach**2.)
    sigma_s_square   = np.log(1. + b_parameter**2. * mach**2.) #logarithmic density variance
    turb_term_ext    = 0.5 * (1. + erf((sigma_s_square - s_crit)/np.sqrt(2.* sigma_s_square)))
    Zgas_multiff_ext = turb_term_ext * Zgas_multiff 

    # ---- FG 13 -----
    Zsfr_FG13       = (np.sqrt(3.)/2. * gas_vdisp * omega_dyn * Zgas_mol/P_on_mstar).to("M_sun/(pc2 Myr)")
    
    # CALCULATE BASE LOSSES AND OFFSETS FOR FINAL SFR PARAMETERIZATION

    model_names      = ["K98",    "KDM12",  "SFK15",      "SAF20",          "FG13"   ] 
    sfrdesc_list     = [Zgas_mol, Zgas_tff, Zgas_multiff, Zgas_multiff_ext, Zsfr_FG13]
    gradients        = [1.4,      1.,       1.,           1.,               1.       ] 
    
    #if (output_log_space==True):
    logZsfr          = np.log10(Zsfr.to_value())
    log_sfrdesc_list = list(map(lambda g: np.log10(g.to_value()), sfrdesc_list))
    fit_params_list  = list(starmap((lambda y:(lambda x, g: fixed_slope_calc_base_residuals(x, y, g)))(logZsfr), zip(log_sfrdesc_list, gradients)))
    base_loss_list, base_r2_list, offset_fit_list, _  = map(list, zip(*fit_params_list))
    best_fit_logZsfr_vals  = list(map(lambda x, m, c: m*x+c, gradients[:-1], log_sfrdesc_list[:-1], offset_fit_list[:-1]))
    df_analytic_sfr_models = pd.DataFrame(np.vstack([logZsfr] + best_fit_logZsfr_vals + [np.log10(Zsfr_FG13.to_value())]).transpose(), columns=["DATA"] + model_names)
    
    return zip(model_names, base_loss_list), df_analytic_sfr_models

#### ============================================================== ####
#### ---------- FUNCTIONS FOR CREATING SYNTHETIC FG13 DATASET ---------- ####
#### ============================================================== ####
    
def create_synthetic_FG13_dataset(df, pixel_width=750*u.pc, output_log_space=True):
    
    ####### read in from data table ############
    #pixel_width    = pixel_width * u.pc
    pixel_area     = (pixel_width)**2.

    sfr            = np.array(df["SFR_ave_010"])/0.85*1e6 * u.M_sun/u.Myr     #[M_sol/Myr^-1] 
    sfr_long       = np.array(df["SFR_ave_100"])/0.70*1e6 * u.M_sun/u.Myr     #[M_sol/Myr^-1] from starburst 99
    gas_mass       = np.array(df["GasNeutMass"])          * u.M_sun
    
    Zsfr           = sfr      / pixel_area                               #[M_sol/pc^2/Myr^-1]
    Zgas_mol       = gas_mass / pixel_area                               #[M_sol/pc^2]     
    Zgas_cut       =  10.                                 * u.M_sun/u.pc**2 
    Zgas_mol[np.where(Zgas_mol < Zgas_cut)] = np.nan                       # apply gas cut 
    
    gas_vdisp      = np.array(df["GasNeutVDisp_z"])       * u.km/u.second     # units: [km/s]
    omega_dyn      = np.array(df["omega_dyn"])            * u.Gyr**(-1)             #[Gyr^-1] 
   
    _ , df_analytic_sfr_models = sfr_analytic_models(sfr, gas_mass, gas_vdisp, omega_dyn, pixel_width=pixel_width)
    
    n_points        = sfr.ravel().shape[0]
    
    FG13_var          = [gas_vdisp.ravel(), omega_dyn.ravel(),      Zgas_mol.ravel()]
    #uncertainties    =          [5km/s,             10%,                    5M_sol/pc^2] HI-->H_2 transition 10M_sol/pc^2
    uncertainties     = [5.*u.km/u.second,  np.mean(omega_dyn)*0.1, 5.*u.Msun/u.pc**2.]

    white_noise_FUNC   = (lambda mu, n: (lambda sig: np.random.normal(mu, sig.to_value(), n) * sig.unit))(0, n_points)
    
    noise_array        = list(map(white_noise_FUNC, uncertainties)) 
    synthetic_variable = np.squeeze(np.random.rand(1, n_points)) * u.M_sun/u.pc**2 #synthetic units similar to Sigma_gas to try to "trick" the algorithm?
    FG13_var_noise     = list(map(sum, zip(FG13_var, noise_array)))
    FG13_var_noise.append(synthetic_variable)
    FG13_var.append(synthetic_variable)
    
    column_names   = ["\sigma_{\mathrm{gas}}", "\Omega_{\mathrm{dyn}}", "\Sigma_{\mathrm{gas}}", "x_{\mathrm{noise}}"]
    
    Q_Zsfr               = QTable([Zsfr], names=["\Sigma_{\mathrm{SFR}}"])
    Q_ZsfrFG13           = QTable([10**(df_analytic_sfr_models["FG13"].values) * Zsfr.unit], names=["\Sigma_{\mathrm{SFR, FG13}}"])
    Q_ZsfrFG13_var       = QTable(FG13_var,       names=column_names)
    Q_ZsfrFG13_var_noise = QTable(FG13_var_noise, names=column_names) 
    
    Q_ZsfrFG13_var_noise["\Sigma_{\mathrm{gas}}"][np.where(Q_ZsfrFG13_var_noise["\Sigma_{\mathrm{gas}}"] < Zgas_cut)] = np.nan

    Q_LIST = [Q_ZsfrFG13_var, Q_ZsfrFG13_var_noise, Q_ZsfrFG13, Q_Zsfr]
    
    if (output_log_space==True):
    #OUTPUT IN LOG SPACE
        log_Q_LIST = list(map(lambda Q: log10_units_QTable(Q), Q_LIST))
        for i in range(2):
            log_Q_LIST[i].rename_column(log_Q_LIST[i].colnames[-1], "x_{\mathrm{noise}}")
        Q_LIST     = log_Q_LIST
    #return df_logZsfrFG13_var, df_logZsfrFG13_var_noise, df_logZsfrFG13, df_logZsfr
    return Q_LIST

#### ========================================================================= ####
#### ---------- FUNCTIONS TO SPLIT DATA TO TRAIN/TEST SETS W/O NANS ---------- ####
#### ========================================================================= ####

def clean_dataframes(X, y):
    combined_df = pd.concat([X, y], axis=1)
    combined_df.replace({np.inf:np.nan, -np.inf:np.nan}, inplace=True)
    combined_df.dropna(inplace=True)

    X_clean = combined_df.iloc[:,:-1]
    y_clean = combined_df.iloc[:,-1:]
    return X_clean, y_clean

def clean_train_test_split(X, y): #NOW TAKE INPUTS AS QTABLES
    #-------- STOCHASTIC SAMPLING ----------
    X_clean, y_clean = clean_dataframes(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size = 0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def isfinite_classification_train_test_aplit(X, y):
    combined_df        = pd.concat([X, y], axis=1)
    combined_df_isfinite = combined_df.apply(np.isfinite)
    combined_df_isfinite.replace({True:1, False:0}, inplace=True)
    
    X_isfinite = combined_df_isfinite.iloc[:,:-1]
    y_isfinite = combined_df_isfinite.iloc[:,-1:]

    X_train, X_test, y_train, y_test = train_test_split(X_isfinite, y_isfinite, test_size = 0.2, random_state=42) 
    return X_train, X_test, y_train, y_test

#### =========================================================== ####
#### ---------- RUN XGBOOST AND CALCULATE SHAP VALUES ---------- ####
#### =========================================================== ####    

def run_xgboost_shap_values(X, y, picklefile="XBG_model_shap.pickle"):
   
    if (isinstance(X, QTable) & isinstance(y, QTable)):
        X_units = [X[col].unit for col in X.columns if X[col].unit] #{col: Q_X[col].unit for col in Q_X.columns if Q_X[col].unit}
        y_units = y[y.colnames[0]].unit                             #{Q_y.colnames[0]: Q_y[Q_y.colnames[0]].unit} 
        X, y = X.to_pandas(), y.to_pandas()
    else: 
        X_units, y_units = None, None
    
    # --- CLASSIFICATION ---
    X_train_c, X_test_c, y_train_c, y_test_c = isfinite_classification_train_test_aplit(X, y)
    
    model_XBGcla  = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=1000, max_depth=7, subsample=0.7, colsample_bytree=0.8)
    model_XBGcla.fit(X_train_c, y_train_c)
    print("FINISHED FITTING XGBoost CLASSIFICATION MODEL")

    shap_explainer_cla = shap.TreeExplainer(model_XBGcla)
    shap_values_cla    = shap_explainer_cla.shap_values(X_test_c)
    print("FINISHED CALCULATING SHAP VALUES (CLASSIFICATION)")
    
    # --- REGRESSION ---
    X_train, X_test, y_train, y_test = clean_train_test_split(X, y)
    
    model_XBGreg = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model_XBGreg.fit(X_train, y_train)
    print("FINISHED FITTING XGBoost REGRESSION MODEL")

    shap_explainer_reg = shap.TreeExplainer(model_XBGreg)
    shap_values_reg    = shap_explainer_reg.shap_values(X_test)
    print("FINISHED CALCULATING SHAP VALUES (REGRESSION)")
    
    xgb_save_object = [[X_train,   X_test,   y_train,   y_test,   model_XBGreg, shap_values_reg],
                       [X_train_c, X_test_c, y_train_c, y_test_c, model_XBGcla, shap_values_cla], 
                       [X_units,             y_units                                           ]]
    
    with open(picklefile, "wb") as f:
        pickle.dump(xgb_save_object, f)

    print("FINISHED PICKLING " + picklefile) 
    return xgb_save_object

#### ======================================================== ####
#### ---------- FUNCTIONS TO CREATE ANALYSIS PLOTS ---------- ####
#### ======================================================== ####
def make_shap_plot(X, y, test_feature_name="\\log~Sigma_mol", picklefile="XBG_model_shap.pickle", pdf_savefile="shap_summary_plot_synthetic.pdf", log=False):

    if (os.path.isfile(picklefile)==True):
        with open(picklefile, "rb") as f:
            xbg_save_object     = pickle.load(f) 
    else: 
            xbg_save_object = run_xgboost_shap_values(X, y, picklefile=picklefile)
    
    [X_train,   X_test,   y_train,   y_test,   model_XBGreg, shap_values_reg] = xbg_save_object[0]
    [X_train_c, X_test_c, y_train_c, y_test_c, model_XBGcla, shap_values_cla] = xbg_save_object[1]

    ######## FIND SHAP VALUES #############
    #f_shap, ax_shap = plt.subplots(1, 3, figsize=(30, 8))
    f_shap  = plt.figure(figsize=(9,9))#, constrained_layout=True)
    
    #gs_whole       = GridSpec(2, 2, left=0.25, figure=f_shap)
    #gs_shap_c      = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[0,0])

    [left, right, space]   = [0.25, 0.95, 0.2]
    gs_shap_c      = GridSpec(nrows=1, ncols=1, left=left, right=(left + (right-left)/2 - space/2), bottom=0.5)#, hspace=0.3)
    ax_shap_c      = f_shap.add_subplot(gs_shap_c[0])
    ax_shap_c.set_title("CLASSIFICATION")    
    ax_shap_c.tick_params(axis='both', labelsize=5)

    #gs_shap_r      = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[0,1])
    gs_shap_r      = GridSpec(nrows=1, ncols=1, left=(right - (right-left)/2 + space/2), right=right, bottom=0.5)
    ax_shap_r      = f_shap.add_subplot(gs_shap_r[0]) 
    ax_shap_r.set_title("REGRESSION")   
    ax_shap_r.tick_params(axis='both', labelsize=5)

    gs_sfr  = GridSpec(nrows=1, ncols=2, left=left, right=right, bottom=0.1, top=0.4, hspace=0.001)
    ax_sfr  = f_shap.add_subplot(gs_sfr[0]) 
    #gs_sfr         = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_whole[1,:])    

    y_XBGreg    = model_XBGreg.predict(X_test)
    y_test_np   = np.squeeze(np.array(y_test))
    
    make_math_FUNC = lambda x: f"${x.replace('synthetic', 'noise')}$"   

    y_label     = make_math_FUNC((y_test.keys())[0])
    #y_label = y.keys()[0] #temporary fix
   

    replace_FUNC      = lambda s: s.replace("synthetic", "noise")
    X_test.columns    = X_test.columns.map(replace_FUNC)
    X_test_c.columns  = X_test_c.columns.map(replace_FUNC)

    plt.sca(ax_shap_c)
    #comment back in for proper column name in latex math mode
    shap_summary_fig = shap.summary_plot(shap_values_cla, X_test_c.rename(columns=make_math_FUNC), plot_size=None, show=False, color_bar=False)
    #shap_summary_fig = shap.summary_plot(shap_values_cla, X_test_c, plot_size=None, show=False, color_bar=False)
    ax_shap_c        = plt.gcf()
 
    plt.sca(ax_shap_r)
    shap_summary_fig = shap.summary_plot(shap_values_reg, X_test.rename(columns=make_math_FUNC), plot_size=None, show=False, color_bar=False)
    ax_shap_r = plt.gcf()
    #sfr_comparison_plots(f_shap, gs_sfr, y_test_np, y_XBGreg, y_label, model_name="XGBoost", log=log)
    test_feature_values = X_test[test_feature_name]
    
    #pdb.set_trace()
    sfr_comparison_plots(ax_sfr, y_test_np, y_XBGreg, y_label, test_feature_values, model_name="XGBoost", log=log)

    f_shap.savefig(pdf_savefile, bbox_inches='tight')
    print("SAVED " + pdf_savefile)
    return xbg_save_object

def latex_eqn_to_marker(latex_eqn):
    if ("\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn) and ("\\log \\Sigma_{*}" in latex_eqn) and ("\\log \\sigma_{\\mathrm{gas},z}" in latex_eqn):
        label, marker = "$\\log \\Sigma_{\\mathrm{gas}}$, $\\log \\Sigma_{*}$ and $\\log \\sigma_{\\mathrm{gas},z}$ in eqn" , "s" #square
    else:
        if ("\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn and "\\log \\Sigma_{*}" in latex_eqn) or ("\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn and "\\log f_{\\mathrm{gas}}" in latex_eqn):
            label, marker = "$\\log \\Sigma_{\\mathrm{gas}}$ and $\\log \\Sigma_{*}$ in eqn", "^" #triangle_up
        else: 
            if ("\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn) and ("\\log \\sigma_{\\mathrm{gas},z}" in latex_eqn):
                label, marker = "$\\log \\Sigma_{\\mathrm{gas}}$ and $\\log \\sigma_{\\mathrm{gas},z}$ in eqn", "v" #triangle_down
            else: 
                if ("\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn) and ("\log \Omega_{\mathrm{dyn}}" in latex_eqn):
                    label, marker = "$\\log \\Sigma_{\\mathrm{gas}}$ and $\log \Omega_{\mathrm{dyn}}$ in eqn" , "D" #diamond
                else: 
                    if "\\log \\Sigma_{\\mathrm{gas}}" in latex_eqn:
                        label, marker = "$\\log \\Sigma_{\\mathrm{gas}}$ only in eqn" , "o" #circle
                    else:
                        if "\\log \\Sigma_{*}" in latex_eqn:
                            label, marker = "$\\log \\Sigma_{*}$ only in eqn", "*"
                        else:
                            label, marker = "Constant","X" # x (filled)
    return label, marker

def sfr_comparison_plots(ax, y_test_np, y_model, y_label, test_feature, complexity=None, model_eqn=None, model_name=None, log=False):

    if log == True:
        y_test_np    = np.log10(y_test_np)
        y_model      = np.log10(y_model)
        y_label      = "log" + y_label
        where_finite = np.where(np.isfinite(y_test_np) & np.isfinite(y_model))
        y_test_np    = y_test_np[where_finite]
        y_model      = y_model[where_finite]    
     
    n_levels  = 4
    
    model_LOSS, model_R2 = r2_loss_calc(y_test_np, y_model)
    min_value = np.nanmin(y_test_np)-np.std(y_test_np)
    max_value = np.nanmax(y_test_np)+np.std(y_test_np)

    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_facecolor('lightgray')
    #ax.text(0.06, 0.7, "MSE Loss=%.{0}f".format(3)%model_LOSS, transform=ax.transAxes)
    ax.text(0.06, 0.6, r"$R^2=$%.{0}f".format(3)%model_R2,     transform=ax.transAxes) 
    
    if model_eqn != None:
        ax.text(0.06, 0.85, "$\log\Sigma_{\mathrm{SFR}}$ = " + model_eqn, fontsize = 9, fontweight ="bold", bbox=dict(facecolor='bisque'), transform=ax.transAxes)#, alpha=0.4))
    elif model_name != None:
        ax.text(0.06, 0.85, model_name, fontsize = 9, fontweight ="bold", bbox=dict(facecolor='bisque'), transform=ax.transAxes)#, alpha=0.4))
    z      = np.array(test_feature)
    #z_norm = (z - np.nanmin(z))/(np.nanmax(z) - np.nanmin(z))
    idx    = z.argsort() # Sort the points by density, so that the densest points are plotted last
    plot_sx, plot_sy, z = y_test_np[idx], y_model[idx], z[idx]
    sfr_scatter  = ax.scatter(   plot_sx,   plot_sy, c=z,      cmap="Greens", s=6)#, edgecolors='grey', alpha=0.7)
    sfr_contours = sns.kdeplot(x=plot_sx, y=plot_sy, c="grey", ax=ax,         bw_adjust=0.7, levels=[0.1, 0.5, 0.8])# levels=n_levels)
    #sns.kdeplot(x=plot_sx, y=plot_sy,          cmap="Accent_r", ax=ax, bw_adjust=0.5, levels=1)
    
    ax.plot([min_value,max_value], [min_value,max_value],  color='dimgray', linestyle='dashdot')

    if complexity != None:
        ax.text(0.06, 0.7, "Complexity="+complexity, transform=ax.transAxes)
    return sfr_scatter, sfr_contours

def make_sfrcomp_plot(model_trained, X_test, y_test, df_top_eqns, num_top_eqns=6, test_feature_name="\log \Sigma_{\mathrm{gas}}", savefile="FOUND_EQNS.pdf"): 

    no_columns           = X_test.columns.shape[0]
    temp_column_names    = list(map(lambda n: "x"+str(n), np.arange(no_columns)))
    X_test_tempname      = X_test.set_axis(temp_column_names, axis='columns') 
    
    model_complexity     = model_trained.equations_["complexity"]
     
    y_test_np            = np.squeeze(np.array(y_test))
    y_units              = "[\mathrm{M}_{\odot}\mathrm{pc}^{-2} \mathrm{Myr}^{-1}]"
    y_label              = r"$" + (y_test.keys())[0] + "$"
    
    # ---- DEFINE PLOT TO SHOW LOSS CURVES AND SFR COMPARISONS ---- ####
    f_pysr_results = plt.figure(figsize=(14, 10))
    n_cols         = 2
    n_rows         = (len(df_top_eqns))//2 #+ 1
    gs_whole       = GridSpec(n_rows, 6, figure=f_pysr_results)
    #gs_whole       = GridSpec(n_rows, n_cols, figure=f_pysr_results)
    f_pysr_results.text(0.05, 0.5, y_label + " (Found equation) "+ "$~"+y_units+"$", va='center', rotation='vertical', fontsize=16)
    
    gs_sfr         = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs_whole[:, 0:5], wspace=0, hspace=0)

    gs_cbar        = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_whole[:, -1], wspace=0, hspace=0)
    ax_cbar        = f_pysr_results.add_subplot(gs_cbar[:, 0])
    #ax_righthalf.axis("off")
    
    for n in range(len(df_top_eqns)):
    #### ---- plot PySR found equation ---- ####
        eqn_complexity   = int((df_top_eqns["complexity"])[n]) 
        model_index      = int((df_top_eqns["model_orig_index"])[n])
        eqn_latex        = (df_top_eqns["latex equation"])[n]
        eqn_colour       = complexity_colour(n, num_top_eqns=len(df_top_eqns))
        
        y_model_trained = model_trained.predict(X_test_tempname, index=model_index) 
        test_feature    = X_test[test_feature_name].to_numpy()
        ax_sfr          = f_pysr_results.add_subplot(gs_sfr[n//n_cols,  n%n_cols])
        
        if (n//n_cols == n_rows-1):
            ax_sfr.set_xlabel(y_label + "$~"+y_units+"$", fontsize=14)
        else:
            ax_sfr.xaxis.set_tick_params(labelbottom=False)
        #if (n//n_cols == n_rows-2) & (n%n_cols==0):
        #    ax_sfr.set_ylabel(y_label + " (Found equation) "+ "$~"+y_units+"$", fontsize=14)
        #else:
        if (n%n_cols==1):
            ax_sfr.yaxis.set_tick_params(labelleft=False)
        sfr_scatter, sfr_contours = sfr_comparison_plots(ax_sfr, y_test_np, y_model_trained, y_label, test_feature, complexity=str(eqn_complexity), model_eqn=r"$"+eqn_latex+"$")
    
    sfr_scatter.set_clim(np.nanmin(test_feature), np.nanmax(test_feature))
    cbar = f_pysr_results.colorbar(sfr_scatter, cax=ax_cbar, cmap="Greens", location='right')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("$\log \Sigma_{\mathrm{gas}} ~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$", size=14)
    f_pysr_results.savefig(savefile[:savefile.find(".")]+"_sfr.pdf", bbox_inches='tight')
    print("SAVED " + savefile)


