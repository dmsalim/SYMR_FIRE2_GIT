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

jl.seval(
    """
    try
        using StatsBase: percentile
    catch
        using Pkg: Pkg
        Pkg.add("StatsBase")
        using StatsBase: percentile
    end
    """
)

#from xgb_shap_ALL_FEATS import *
#def set_ax_ratio(ax, ratio=1):
#    x_left, x_right = ax.get_xlim()
#    y_low, y_high = ax.get_ylim()
#    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

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
    #return X_train, X_test, y_train, y_test, model_XBGreg, shap_values_reg

#### =================================================================================== ####
#### ---------- TRAIN PYSR WITH WARM START MODEL TO SAVE LOSSES AT EACH EPOCH ---------- ####
#### =================================================================================== ####
def save_into_final_loss_df(full_losses_epoch, MSE_losses_epoch, df_losses_all_eqns_all_epochs, complexity_columns, gamma=1): 
    df_full_loss_this_epoch       = pd.DataFrame(full_losses_epoch,       columns=complexity_columns)
    df_MSE_loss_this_epoch        = pd.DataFrame(MSE_losses_epoch,        columns=complexity_columns)
    df_quantile_loss_this_epoch   = (df_full_loss_this_epoch - df_MSE_loss_this_epoch)/gamma
    df_losses_this_epoch          = pd.concat([df_full_loss_this_epoch, df_MSE_loss_this_epoch, df_quantile_loss_this_epoch], axis=1, keys=["FULL_CUSTOM_LOSS", "MSE_LOSS", "QUANTILE_LOSS"])
    df_losses_all_eqns_all_epochs = pd.concat([df_losses_all_eqns_all_epochs, df_losses_this_epoch], ignore_index=True)
    return df_losses_all_eqns_all_epochs

def train_pysr_save_loss(X_train, X_test, y_train, y_test, X_units=None, y_units=None, n_epochs=10, n_saves=1, sfr_type=10, min_loss_sfr_laws=2, eqns_picklefile="FOUND_EQNS.pickle", eqns_picklefile_old=None):

    default_files_path = "/SH_SCRIPTS/DEFAULT_FILES"
    folder_indices     = [i for i, x in enumerate(eqns_picklefile) if x == "/"]
    exp_name           = eqns_picklefile[folder_indices[-1]:eqns_picklefile.find(".")]
    default_file = eqns_picklefile[:folder_indices[-2]]+default_files_path+exp_name+"_DEFAULT.csv"
    
    ######## MAKE LOSS TABLE ###########
    original_columns     = list((map(lambda n: n[1:-1], X_train.columns))) #CHANGE BACK TO X_train.columns for non-math column names
    no_columns           = X_train.columns.shape[0]
    temp_column_names    = list(map(lambda n: "x"+str(n),       np.arange(no_columns)))
    latex_column_names   = list(map(lambda n: "x_{"+str(n)+"}", np.arange(no_columns)))
    
    if eqns_picklefile_old != None:
        with open(eqns_picklefile_old, "rb") as f:
            eqns_save_object_old     = pickle.load(f)
        (model, df_train_losses_all_eqns_all_epochs, df_test_losses_all_eqns_all_epochs) = eqns_save_object_old
        n_index      = [i for i, x in enumerate(eqns_picklefile_old) if x == "_"][-1]
        n_epochs_old = int(eqns_picklefile_old[n_index+1 : eqns_picklefile_old.find("EPOCHS")]) 
        n_iterations = n_epochs - n_epochs_old
    else:
        #if (X_units!=None):
        #    dimensional_constraint_penalty=10**5
        #else: 
        #    dimensional_constraint_penalty=1000
        df_train_losses_all_eqns_all_epochs = pd.DataFrame()
        df_test_losses_all_eqns_all_epochs  = pd.DataFrame()
        n_epochs_old           = 0
        n_iterations           = n_epochs
        model = PySRRegressor(
            model_selection                = "best",
            niterations                    = 2,    #1,--> there's something that happens between iterations 
            populations                    = 128,  #40,  #running on 128 --> populations = 2*n_cores
            #dimensional_constraint_penalty = dimensional_constraint_penalty,
            ncycles_per_iteration          = 5000, # makes epochs longer but more efficiently using cores
            binary_operators               = ["plus", "pow", "mult"], 
            constraints                    = {"pow": (-1, 1), 'mult':(1, 1)},
            complexity_of_variables        = 2,
            unary_operators                = ["log10", "exp"], #, "abs"], 
            nested_constraints             = {"log10": {"log10": 0, "exp":0}, "exp":{"exp":0, "log10":0}}, #, "abs":{"abs":0}},
            batching                       = True,
            warm_start                     = True,
            batch_size                     = 256,
            equation_file                  = default_file)
 
        model.loss_function = (
            """
            function my_loss(
                tree::Node,
                dataset::Dataset{T,L},
                options::Options,
                idx = nothing,
            ) where {T,L}
                X = idx === nothing ? dataset.X : view(dataset.X, :, idx)
                y = idx === nothing ? dataset.y : view(dataset.y, idx)

                prediction, completed = eval_tree_array(tree, X, options)
                if !completed
                    return L(Inf)
                end

                predictive_loss = sum(i -> (prediction[i] - y[i])^2, eachindex(y)) / length(y)

                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                q_true = [percentile(y, q) for q in quantiles]
                q_pred = [percentile(prediction, q) for q in quantiles]

                quantile_loss = sum(i -> (q_true[i] - q_pred[i])^2, eachindex(quantiles)) / length(quantiles)

                return L(predictive_loss + quantile_loss)
            end
            """
        )
    save_epoch          = n_epochs//n_saves
    end_of_string_index = [i for i, x in enumerate(eqns_picklefile) if x == "_"][-1]
    name_savefile_FUNC  = (lambda p, i,: (lambda e: p[:i] + "_" + str(e) + "EPOCHS.pdf"))(eqns_picklefile, end_of_string_index)

    for k in range(n_iterations): 
        this_epoch = n_epochs_old + k + 1 
        # these functions are loss = f(i) where i is the index of the found equation 
        
        full_loss_FUNC     = (lambda t, m: (lambda X, y: (lambda i: full_custom_loss(np.squeeze(np.array(y)), m.predict(X.set_axis(t, axis='columns'), index=i)))))(temp_column_names, model)
        MSE_metrics_FUNC   = (lambda t, m: (lambda X, y: (lambda i: r2_loss_calc(np.squeeze(np.array(y)),     m.predict(X.set_axis(t, axis='columns'), index=i)))))(temp_column_names, model) 

        # ---- get TRAINING loss -----
        model.fit(X_train.set_axis(temp_column_names, axis='columns'), y_train)
        train_full_losses_epoch       = np.expand_dims(model.equations_["loss"].to_numpy(), axis=0) 
        found_eqns_indices            = np.arange(train_full_losses_epoch.shape[1]).tolist() 
        train_MSE_metrics             = list(map(MSE_metrics_FUNC(X_train, y_train), found_eqns_indices))
        train_MSE_losses_epoch , _    = map(lambda a: np.expand_dims(np.array(a), axis=0), zip(*train_MSE_metrics)) 
        
        # ---- get TEST loss -----
        test_full_losses_epoch   = np.expand_dims(np.array(list(map(full_loss_FUNC(X_test, y_test),   found_eqns_indices))), axis=0)
        test_MSE_metrics         = list(map(MSE_metrics_FUNC(X_test, y_test), found_eqns_indices))
        test_MSE_losses_epoch, _ = map(lambda a: np.expand_dims(np.array(a), axis=0), zip(*test_MSE_metrics))
        
        # ---- put train & test loss into respective dataframes ---- #
        df_train_losses_all_eqns_all_epochs = save_into_final_loss_df(train_full_losses_epoch, train_MSE_losses_epoch, df_train_losses_all_eqns_all_epochs, np.array(model.equations_["complexity"]))
        df_test_losses_all_eqns_all_epochs  = save_into_final_loss_df(test_full_losses_epoch,  test_MSE_losses_epoch,  df_test_losses_all_eqns_all_epochs,  np.array(model.equations_["complexity"]))

        if ((this_epoch%save_epoch == 0) & (this_epoch!=save_epoch*n_saves)):
            eqns_save_object    = (model, df_train_losses_all_eqns_all_epochs, df_test_losses_all_eqns_all_epochs) 
            savefile            = name_savefile_FUNC(this_epoch)
            this_epoch_pkl      = savefile[:savefile.find(".")] + ".pickle"
            with open(this_epoch_pkl, "wb") as f:
                pickle.dump(eqns_save_object, f)
            print("FINISHED PICKLING " + this_epoch_pkl) 
            #found_eqns_analysis_plots(eqns_save_object, X_train, X_test, y_train, y_test, sfr_type=sfr_type, max_loss=min_loss_sfr_laws, savefile = savefile) 
        print("INTERATION " + str(k))
        print("TRAINING LOSSES:")
        print(df_train_losses_all_eqns_all_epochs["FULL_CUSTOM_LOSS"])
        print("TEST LOSSES:")
        print(df_test_losses_all_eqns_all_epochs["FULL_CUSTOM_LOSS"])

    savefile         = name_savefile_FUNC(n_epochs)
    eqns_save_object = (model, df_train_losses_all_eqns_all_epochs, df_test_losses_all_eqns_all_epochs)
    with open(eqns_picklefile, "wb") as f:
        pickle.dump(eqns_save_object, f)
    #found_eqns_analysis_plots(eqns_save_object, X_train, X_test, y_train, y_test, sfr_type=sfr_type, max_loss=min_loss_sfr_laws, savefile = savefile) 
    print("FINISHED PICKLING " + eqns_picklefile)
    return eqns_save_object

def round_expr(expr, num_decimal_places):
    """Recursively round all floats in a SymPy expression to a specified number of decimal places."""
    return expr.xreplace({n: round(n, num_decimal_places) for n in expr.atoms(sympy.Float)})

def make_df_found_eqns(model, var_strings, param_strings):
    
    replacements = list(zip(var_strings, param_strings))

    
    df_found_eqns = pd.DataFrame(model.equations_).drop(columns=["sympy_format", "lambda_format"])
    latex_column  = []
    for n in range(len(df_found_eqns.index)):
        #latex_eqn_var_str   = latex(simplify(model.equations_.sympy_format[n]).evalf(3)) #model.latex(index=n)
        latex_eqn_var_str   = latex(round_expr(simplify(model.equations_.sympy_format[n]), 2))
        latex_eqn_param_str = reduce(lambda a, kv: a.replace(*kv), replacements, latex_eqn_var_str)
        latex_column.append(latex_eqn_param_str)
    df_found_eqns.insert(loc=len(df_found_eqns.columns), column='latex equation', value=latex_column)
    return df_found_eqns

def filter_top_eqns(model, X_train, max_loss, num_top_eqns=4):
    
    original_columns     = X_train.columns#list((map(lambda n: n[1:-1], X_train.columns)))
    no_columns           = X_train.columns.shape[0]
    latex_column_names   = list(map(lambda n: "x_{"+str(n)+"}", np.arange(no_columns)))
    
    df_found_eqns = make_df_found_eqns(model, latex_column_names, original_columns)
    
    #NEED TO DEFINE df_top_eqns
    df_top_eqns   = df_found_eqns.where(df_found_eqns.loss < max_loss).sort_values('score', ascending=False).dropna()
    if len(df_top_eqns)> num_top_eqns:
        df_top_eqns = df_top_eqns[:num_top_eqns]
    df_top_eqns = df_top_eqns.sort_values('complexity').reset_index(drop=True)
     
    model_complexity     = model.equations_["complexity"]
    where_top_eqn        = np.in1d(model_complexity.values, df_top_eqns["complexity"].values)
    model_indices        = model.equations_.index[where_top_eqn]
    
    df_top_eqns.insert(loc=1, column='model_orig_index', value=model_indices)
    
    return df_top_eqns, df_found_eqns

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

def plot_complexity_vs_r2(ax_train_metrics, ax_sd, model, df_found_eqns, X_test, y_test, r2_axes, used_markers, test_feature_temp_name ="x0", min_loss=None, max_r2=None, df_top_eqns=None, min_sd_res_zip=None, is_right=True):
     
    # ---- EXTRACT DESIRED METRICS FROM MODEL ---- #
    model_complexity          = model.equations_["complexity"]
    model_score               = model.equations_["score"]
    X_test_feature            = X_test[test_feature_temp_name]
    y_model_trained_list      = list(map((lambda X, m: (lambda n: m.predict(X, index=n)))(X_test, model),  model.equations_.index.tolist()))
    model_metrics             = list(map((lambda yt:   (lambda ym: r2_loss_calc(yt, ym)))(np.squeeze(y_test.to_numpy())), y_model_trained_list)) 
    model_mseloss, model_r2   = map(np.array, zip(*model_metrics))
    model_fullloss            = list(map((lambda yt:   (lambda ym: full_custom_loss(yt, ym)))(np.squeeze(y_test.to_numpy())), y_model_trained_list))
    model_latex               = df_found_eqns["latex equation"]
    latex_to_marker_results   = list(map(lambda e: latex_eqn_to_marker(e), model_latex.to_list()))
    model_label, model_marker = zip(*latex_to_marker_results)
    model_label               = list(model_label)
    model_marker              = list(model_marker)
    
    stddev_FUNC          = lambda y: np.nanstd(np.squeeze(np.array(y)))
    y_test_sd            = stddev_FUNC(y_test)
    y_label              = (y_test.keys())[0]
    y_model_sd           = list(map(stddev_FUNC, y_model_trained_list))
    
    model_testplane_m_sd = list(map((lambda xt:   (lambda ym: find_dispersion(xt, ym)))(X_test_feature.to_numpy()),  y_model_trained_list)) 
    
    data_testplane_m,  data_testplane_sd  = find_dispersion(X_test_feature.to_numpy(), np.squeeze(y_test.to_numpy()))
    model_testplane_m, model_testplane_sd = map(np.array, zip(*model_testplane_m_sd))

    m_label = "\mathrm{m}_{\mathrm{fit}}"

    metrics_list      = [model_score, model_r2,   model_fullloss,   model_testplane_m,        model_testplane_sd           ]#,            y_model_sd]
    metrics_standards = [None,        max_r2,     min_loss,         data_testplane_m,         data_testplane_sd            ]#,             y_test_sd]
    metrics_labels    = ["Score",     "$R^2$",    "Test Loss",      "$"+m_label+"$ in KS plane", "$\sigma_{\mathrm{KS, model}}$"]#, "$\mathrm{SD}_{"+y_label+"}$"]
    metric_plottype   = ["log",       None,       None,             None,                     None                         ]#,                          None     ]
    metric_linestyle  = ["None",      "None",     "solid",          "None",                   "None"                       ]#,                        "None"      ]
   
    metric_yrange     = [[5e-9, 5e0], [-1.5, 1.1], [0.1, 2.1],     [-0.1, 1.8],              [-0.001, 0.036]              ]#,                [-0.05, 0.85]]
    # ---- PLOT COMPLEXITY VS. LOSS & R^2 ---- #

    if not df_top_eqns.empty:
        top_eqns_complexity = df_top_eqns["complexity"]
        top_eqns_latex      = df_top_eqns["latex equation"] 
        
        top_eqns_latex_to_marker_results = list(map(lambda e: latex_eqn_to_marker(e), top_eqns_latex.to_list()))
        top_eqns_labels, top_eqns_markers = zip(*top_eqns_latex_to_marker_results)
        top_eqns_labels = list(top_eqns_labels)
        top_eqns_marker = list(top_eqns_markers)
        df_top_eqns_metric  = pd.DataFrame({"Complexity": top_eqns_complexity.sort_values()})
        df_top_eqns_metric["marker"] = top_eqns_markers
    
    for n, standard in enumerate(metrics_standards):
    #### ---- plot PySR found equation ---- ####
        # ---- define dataset to plot ----#
        if (n<=1):
            ax_metric = ax_train_metrics[n]
        else:
            ax_metric = ax_sd[n-2] 

        metric_plot = np.array(metrics_list[n])
        
        # ---- axes labes etc ---- #
        if is_right==False:
            ax_metric.set_ylabel(metrics_labels[n])
        else:
            ax_metric.yaxis.set_tick_params(labelleft=False)
        ax_metric.set_ylim(metric_yrange[n][0], metric_yrange[n][1]) 
        if metric_plottype[n] != None: 
            ax_metric.set_yscale(metric_plottype[n])

        # ---- plot some standard lines ---- #
        if n==3:
            ax_metric.axhline(1.4,  color='lightgrey', linestyle="dashdot", label="K98")
        if n==1:
            label_r2 = "Perfect $R^2$" if (is_right==True) else None
            ax_metric.axhline(1,    color='lightgrey', linestyle="dashdot", label=label_r2)
        #if (n==1 or n==3) & (is_right==False):
        #    ax_metric.legend(loc="lower right")
        if (n==4) & (min_sd_res_zip!=None):
            (min_sd_res, min_sd_res_model) = min_sd_res_zip
            ax_metric.axhline(min_sd_res, color='lightgrey', linestyle="dashdot", label=min_sd_res_model)
               # ---- plot for all found equations ---- # 
        
        #used_markers = set()
        for i, (complexity, metric, mark) in enumerate(zip(model_complexity, metric_plot, model_marker)):
            label = model_label[i] if mark not in used_markers else None
            used_markers.add(mark)
            ax_metric.plot(
                complexity, metric, 
                marker=mark, color="lightgray", 
                linestyle=metric_linestyle[n], 
                label=label) 
         
        # ---- plot top equations ---- #
        if not top_eqns_complexity.empty:
            where_top_eqn = np.in1d(model_complexity.values, top_eqns_complexity.values)
            plot_top_eqns_x = top_eqns_complexity.sort_values()
            plot_top_eqns_y = metric_plot[where_top_eqn]
            for i, marker in enumerate(top_eqns_markers):
                label = "Selected top eqn" if (n==1 and i == len(top_eqns_markers)-1 and is_right==True) else None
                ax_metric.plot(plot_top_eqns_x[i], plot_top_eqns_y[i], marker=marker, c="orange", linestyle='None', label=label)
            df_top_eqns_metric[metrics_labels[n]] = metric_plot[where_top_eqn] 
            if (n>=3): 
                metric_res = metric_plot - metrics_standards[n]
                residuals_top_eqns  = metric_res[where_top_eqn]# - metrics_standards[n]
                where_top_eqn_iloc  = int(np.where(np.abs(residuals_top_eqns) == np.nanmin(np.abs(residuals_top_eqns)))[0])
                min_res_eqn_comp    = df_top_eqns_metric["Complexity"].iloc[where_top_eqn_iloc]
                best_eqn_metric     = df_top_eqns_metric[metrics_labels[n]].iloc[where_top_eqn_iloc]
                ax_metric.plot(min_res_eqn_comp, best_eqn_metric,     marker = 'o', markeredgecolor="black", markerfacecolor="None", linestyle='None', label="Least residuals eqn")
        
        # ---- lines for comparison ---- #
        if metrics_standards[n] != None: 
        #    label = metrics_labels[n] 
            if (n==1) and (is_right==True):
                label = "XGBoost $R^2$"
            else:
                label = None
            #if (n>=3): 
               # label      = "FIRE test data"
            #else: 
                #label = None
            ax_metric.axhline(metrics_standards[n],  color='pink', linestyle="dashed", label=label) 
        # Only add label argument if label is not None
            #if label is not None:
            #    ax_metric.axhline(metrics_standards[n], color='pink', linestyle="dashed", label=label)
            #else:
            #    ax_metric.axhline(metrics_standards[n], color='pink', linestyle="dashed")
        
        # ---- x axes and legend ---- #
        if  (n==1) or (n == len(metrics_standards)-1): 
            ax_metric.set_xlabel("Equation Complexity")
            #if is_right==False:
            #    ax_metric.legend(loc="lower right", fontsize='small')
        #if n==2:
        #    pdb.set_trace()
        if (n==0) or (n==1):
            r2_axes.append(ax_metric)
     
    return df_top_eqns_metric, r2_axes, used_markers

def complexity_colour(top_eqn_index, num_top_eqns=6):
    norm         = mpl.colors.Normalize(vmin=0, vmax=num_top_eqns)
    palette      = sns.color_palette("hls", 8)
    palette_cmap = ListedColormap(palette)
    cmap         = mpl.cm.ScalarMappable(norm=norm, cmap=palette_cmap)
    cmap.set_array([])
    colour       = cmap.to_rgba(top_eqn_index)
    return colour

def make_loss_curves(ax_loss_train, ax_loss_test,          df_top_eqns, 
                     df_train_FULL_loss_top_eqns,          df_test_FULL_loss_top_eqns,
                     df_train_QUANTILE_loss_top_eqns=None, df_test_QUANTILE_loss_top_eqns=None,
                     min_loss=None, max_loss=1, max_loss_model=None, is_right=True): 
    
    n_epochs     = df_train_FULL_loss_top_eqns.index[-1] + 1  
    epochs_array = np.arange(n_epochs)+1
    
    if is_right==True:
        ax_loss_train.set_ylabel("Training Loss")
        ax_loss_test.set_ylabel("Test Loss")

    ax_loss_train.axhline(y=max_loss, color='lightgray', linestyle='dashed', label=max_loss_model)
    ax_loss_test.set_xlabel("Epochs")

    if min_loss != None: 
        ax_loss_train.axhline(y=min_loss, color='lightgray', linestyle='dashdot', label='XGBoost')
        ax_loss_test.axhline(y=min_loss, color='lightgray', linestyle='dashdot', label='XGBoost')

    for n in range(len(df_top_eqns)):
    #### ---- plot PySR found equation ---- ####
        eqn_complexity   = int((df_top_eqns["complexity"])[n]) 
        eqn_colour       = complexity_colour(n, num_top_eqns=len(df_top_eqns))
        train_FULL_loss_array = np.array(df_train_FULL_loss_top_eqns[eqn_complexity]) 
        test_FULL_loss_array  = np.array(df_test_FULL_loss_top_eqns[eqn_complexity])
        
        train_QUANTILE_loss_array = np.array(df_train_QUANTILE_loss_top_eqns[eqn_complexity]) 
        test_QUANTILE_loss_array  = np.array(df_test_QUANTILE_loss_top_eqns[eqn_complexity])
        
        label = "complexity = "+str(eqn_complexity)
        #else: 
            #label = None
        if (df_train_QUANTILE_loss_top_eqns is not None and not df_train_QUANTILE_loss_top_eqns.empty): #& (df_test_QUANTILE_loss_top_eqns is not None):
            ax_loss_train.loglog(epochs_array, train_FULL_loss_array,     c=eqn_colour, label=label)
            ax_loss_test.loglog(epochs_array,  test_FULL_loss_array,      c=eqn_colour, label=label)
            ax_loss_train.loglog(epochs_array, train_QUANTILE_loss_array, c=eqn_colour, label=label, linestyle="dashed")
            ax_loss_test.loglog(epochs_array,  test_QUANTILE_loss_array,  c=eqn_colour, label=label, linestyle="dashed")
        else:
            ax_loss_train.semilogx(epochs_array, train_FULL_loss_array, c=eqn_colour, label=label)
            ax_loss_test.semilogx(epochs_array,  test_FULL_loss_array,  c=eqn_colour, label=label)

    ax_loss_train.legend(loc="upper right", fontsize=10)


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

def eqn_behaviour(ax, equation, X_test, test_feature="\log\Sigma_{\mathrm{gas}}", comp_obj=None, comp_obj_colour=None, theory_line_label=None, plot_K98=True):
    # ---- DEFINE DATA FOR FIDUCIAL VALUES OF FEATURES ---- #
    galaxy_type_list = ["Local Dwarf",  "Local $L_*$", "ULIRG",       "High-z"]
    galaxy_type_col  = ["lightskyblue", "salmon",       "yellowgreen", "mediumpurple"]
    #                                      local local  ULIRG high-z
    #                                      dwarf spiral
    d1 = {"\Sigma_{\mathrm{gas}}"   : [4,     10,    250,   560], #Gas surface density, [M_sun pc^-2]
          "\Omega_{\mathrm{dyn}}"   : [60,    30,    1250,  30 ], #Dynamical frequency, [Gyr^-1] based on Krumholz Burkhart et al. 2018 t_orb times
          "\sigma_{\mathrm{gas},z}" : [6,     10,    60,    40 ], #Gas velocity dispersion [km s^-1]
          "f_{\mathrm{gas}}"        : [0.2,   0.2,   1,     1  ], #Gas fraction 
          "\Sigma_{*}"              : [1e4,   1e2,   1e5,   2e4], #Stellar surface density
          "\sigma_{*,z}"            : [50,    50,    50,    50 ]} #Stellar velocity dispersion PLACEHOLDER VALUES FOR NOW
    
    d2 = {"\log \Sigma_{\mathrm{gas}}_MIN" : [0.3, 0.3, 1.3, 2.5], 
          "\log \Sigma_{\mathrm{gas}}_MAX" : [0.9, 1.7, 3.5, 3  ]} 

    df_fiducial       = pd.DataFrame(data=d1, index=galaxy_type_list)
    log_rename_FUNC   = lambda x: f"\log {x}"
    df_log_fiducial   = df_fiducial.apply(np.log10).rename(columns=log_rename_FUNC)
    df_log_Zgas_range = pd.DataFrame(data=d2, index=galaxy_type_list)
    
    K98_FUNC           = lambda x: 1.4*x + np.log10(2.5e-4)
    test_feature_array = np.linspace(X_test[test_feature].min()-2*X_test[test_feature].std(), X_test[test_feature].max()+X_test[test_feature].std(), num=100)
    x                  = symbols('x')
    ax.set_xlim(test_feature_array.min(), test_feature_array.max())
      
    if comp_obj!=None:
        galaxy_type_list, galaxy_type_col = [comp_obj], [galaxy_type_col[int(np.where(np.array(galaxy_type_list) == comp_obj)[0])]]

    for i, galaxy_type in enumerate(galaxy_type_list): 
        df_fiducial_galaxy_type = df_log_fiducial.loc[galaxy_type].astype("string")
        df_fiducial_galaxy_type[test_feature] = "x"
        replacements           = list(zip(df_log_fiducial.columns, df_fiducial_galaxy_type.values))
        eqn_fiducial_sub       = reduce(lambda a, kv: a.replace(*kv), replacements, equation)
        eqn_fiducial_sub_sympy = latex2sympy(eqn_fiducial_sub)
    
        #sympify_func           = sympify(eqn_fiducial_sub)
        sympify_func           = sympify(str(eqn_fiducial_sub_sympy)).n()
        if (test_feature == "\log \Sigma_{\mathrm{gas}}") & (comp_obj == None):
            df_log_Zgas_range_galaxy_type = df_log_Zgas_range.loc[galaxy_type]
            logZgas_min = df_log_Zgas_range_galaxy_type["\log \Sigma_{\mathrm{gas}}_MIN"]
            logZgas_max = df_log_Zgas_range_galaxy_type["\log \Sigma_{\mathrm{gas}}_MAX"]
            plot_test_feature = test_feature_array[np.where((test_feature_array>=logZgas_min) & (test_feature_array<=logZgas_max))]
        else:
            plot_test_feature = test_feature_array
        
        #y_prediction       = np.array(list(map(lambda i: sympify_func.subs(x, i), plot_test_feature.tolist()))).real.astype("float")
        y_prediction       = np.array(list(map(lambda i: sympify_func.subs(x, i), plot_test_feature.tolist()))).real
        #print(y_prediction)
        #pdb.set_trace()
        if comp_obj_colour == None:
            comp_obj_colour   = galaxy_type_col[i]
        if theory_line_label == None:
            theory_line_label = galaxy_type
        try: 
            ax.plot(plot_test_feature, y_prediction, linewidth=3.0, label=theory_line_label, color=comp_obj_colour)
        except:
            print("NO LINE PLOTTED FOR " + test_feature)
    if (test_feature=="\log \Sigma_{\mathrm{gas}}") & (plot_K98==True): 
        ax.plot(test_feature_array, K98_FUNC(test_feature_array), color='dimgray', linestyle='dashed', linewidth=3.0, label = "K98")

def plot_eqn_behaviour(model_trained, X_test, y_test, df_top_eqns, plot_KS_contours=False, sfr_type=10, test_feature="\log \Sigma_{\mathrm{gas}}", test_feature_units="$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$", comp_obj="Local $L_*$",  savefile="FOUND_EQNS_behaviour.pdf"):  
    
    no_columns           = X_test.columns.shape[0]
    temp_column_names    = list(map(lambda n: "x"+str(n), np.arange(no_columns)))
    X_test_tempname      = X_test.set_axis(temp_column_names, axis='columns') 
    y_label              = r"$" + (y_test.keys())[0] + "$"
    y_units              = "$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}\mathrm{Myr}^{-1}]$"
    x_label              = r"$" + test_feature + "$" + test_feature_units
    #ylim                 = [-4.5, 1.2]
    
    model_complexity     = model_trained.equations_["complexity"]
     
    plot_x          = np.squeeze(X_test[test_feature].to_numpy())
    plot_y          = np.squeeze(y_test.to_numpy())
    perr_m_FIRE     = find_dispersion(plot_x, plot_y) 
    #n_levels       = 5
    n_levels        = [0.05, 0.2, 0.5, 0.8, 0.9999]
    n_levels0       = [n_levels[0]] 
    contour_colours = mpl.colormaps['Greys']([0.05, 0.25, 0.5, 0.75])#(n_levels[1:])
    contour_percent = list(map(lambda p: f"${p}\%$",[int(x) for x in np.ceil((1. - np.array(n_levels[:-1]))*100)])) #['$95\\%$', '$80\\%$', '$50\\%$', '$20\\%$']
    
    m_b         = 7100  # minimum baryonic mass [M_sol] 
    pixel_width = 750   # [pc]
    sfr_floor   = np.log10(m_b/((pixel_width**2)*sfr_type))
    gas_cut     = np.log10(10)    # [M_sol pc^-2]
    #ylim        = [sfr_floor-0.5, 1.2]
    ylim        = [-4.4, 1.2]

    # ---- DEFINE PLOT TO SHOW LOSS CURVES AND SFR COMPARISONS ---- #### 
    f_all_eqn, ax_all_eqn = plt.subplots(nrows=1, ncols=1, figsize = (6, 6))
    ax_all_eqn.set_ylabel(y_label + y_units, fontsize = 20)
    ax_all_eqn.set_xlabel(x_label, fontsize = 20)
    ax_all_eqn.set_ylim(ylim[0], ylim[1])
    ax_all_eqn.tick_params(axis='both', which='major', labelsize=20) 
    
    sns.kdeplot(x=plot_x, y=plot_y,          cmap="Greys", fill=True, ax=ax_all_eqn, bw_adjust=0.5, levels=n_levels)
    sns.kdeplot(x=plot_x, y=plot_y,          cmap="Accent_r",         ax=ax_all_eqn, bw_adjust=0.5, levels=n_levels0)
    ax_all_eqn.axhline(y=sfr_floor, c="lightgrey", linestyle="dashdot")
    ax_all_eqn.axvline(x=gas_cut,   c="lightgrey", linestyle="dashdot")
    
    f_pysr_eqn = plt.figure(figsize=(12, 16))
    n_cols     = 2
    n_rows     = 3# (len(df_top_eqns))//2 
    gs_whole   = GridSpec(n_rows, n_cols, figure=f_pysr_eqn, wspace=0, hspace=0)

    for n, eqn_latex in enumerate(df_top_eqns["latex equation"].values):
    #### ---- plot PySR found equation ---- ####
        eqn_complexity  = int((df_top_eqns["complexity"])[n])
        model_index     = int((df_top_eqns["model_orig_index"])[n])
        model_eqn       = "$\log\Sigma_{\mathrm{SFR}}$ = " + r"$"+eqn_latex+"$"
        y_model_trained = model_trained.predict(X_test_tempname, index=model_index) 
        eqn_colour      = complexity_colour(n, num_top_eqns=len(df_top_eqns))
        
        gs_eqn       = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[n//n_cols,  n%n_cols])
        ax_eqn       = f_pysr_eqn.add_subplot(gs_eqn[0])
        #ax_eqn.set_title(y_label + " = " + r"$"+eqn_latex+"$", fontsize = 9, fontweight ="bold", pad=12, bbox=dict(facecolor='yellow', alpha=0.4)) 
        ax_eqn.set_ylim(ylim[0], ylim[1])
        ax_eqn.text(0.06, 0.9, model_eqn, fontsize =10, fontweight ="bold", bbox=dict(facecolor='bisque'), transform=ax_eqn.transAxes)
        if (n//n_cols == n_rows-1):
            ax_eqn.set_xlabel(x_label, fontsize=14)
        else:
            ax_eqn.xaxis.set_tick_params(labelbottom=False) 
        if (n//n_cols == n_rows-2) & (n%n_cols == 0):
            ax_eqn.set_ylabel(y_label + y_units, fontsize=14)
        else:
            if (n%n_cols == 1):
                ax_eqn.yaxis.set_tick_params(labelleft=False)

        plot_K98 = False
        if plot_KS_contours == True:
            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Greys", fill=True, ax=ax_eqn, bw_adjust=0.5, levels=n_levels)
            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Accent_r",         ax=ax_eqn, bw_adjust=0.5, levels=n_levels0)
    
            popt_m_model, perr_m_model = find_dispersion(plot_x, y_model_trained)
            sns.kdeplot(x=plot_x, y=y_model_trained, cmap="Greens",     fill=True, ax=ax_eqn, bw_adjust=0.5, levels=n_levels)
            sns.kdeplot(x=plot_x, y=y_model_trained, cmap="Accent",                ax=ax_eqn, bw_adjust=0.5, levels=n_levels0)
            ax_eqn.axhline(y=sfr_floor, c="lightgrey", linestyle="dashdot")
            ax_eqn.axvline(x=gas_cut,   c="lightgrey", linestyle="dashdot")

            eqn_behaviour(ax_eqn, eqn_latex, X_test, test_feature=test_feature, comp_obj=comp_obj)
            ax_eqn.text(0.06, 0.8, "complexity="+str(eqn_complexity),                        transform=ax_eqn.transAxes)
            ax_eqn.text(0.06, 0.72, "$\sigma_{{KS, \mathrm{{model}}}}$={:.3f}".format(perr_m_model), transform=ax_eqn.transAxes)
            if (n == len(df_top_eqns)-1):
                plot_K98   = True 
                legend_top = (2.45, -2.35) #defined in data units
                box_height = 1.55
                box_width  = 0.45
                box_bottom = (legend_top[0]-0.1, legend_top[1]-box_height+0.25)
                rect       = patches.Rectangle(box_bottom, box_width, box_height, linewidth=2, edgecolor='lightgrey', facecolor='white', alpha=1)
                ax_eqn.add_patch(rect)
                for i, colour in enumerate(contour_colours): 
                    ax_eqn.scatter(legend_top[0],  legend_top[1]-0.35*i, c="dimgrey", marker="s", s=80)
                    ax_eqn.scatter(legend_top[0],  legend_top[1]-0.35*i, c=colour, marker="s", s=55)
                    ax_eqn.text(legend_top[0]+0.07, legend_top[1]-0.35*i-0.06, "= "+str(contour_percent[i]),  bbox=dict(facecolor='white', edgecolor='white'))
        plot_K98 = True if (n == len(df_top_eqns) - 1) else plot_K98
        eqn_behaviour(ax_all_eqn, eqn_latex, X_test, test_feature=test_feature, comp_obj=comp_obj, comp_obj_colour=eqn_colour, theory_line_label="Complexity = " + str(eqn_complexity), plot_K98=plot_K98) 
    ax_all_eqn.legend(loc="lower right")
    ax_eqn.legend(loc="lower right")
    f_pysr_eqn.savefig(savefile, bbox_inches='tight')
    f_all_eqn.savefig(savefile[:savefile.find(".")]+"_alleqns.pdf", bbox_inches='tight')
    print("SAVED " + savefile )


def plot_eqn_behaviour_all_features(model_trained, X_test, y_test, df_top_eqns, sfr_type=10, comp_obj="Local $L_*$",  savefile="FOUND_EQNS_behaviour.pdf"):  
    
    no_columns             = X_test.columns.shape[0]
    temp_column_names      = list(map(lambda n: "x"+str(n), np.arange(no_columns)))
    X_test_tempname        = X_test.set_axis(temp_column_names, axis='columns') 
    y_label                = r"$" + (y_test.keys())[0] + "$"
    y_units                = "$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}\mathrm{Myr}^{-1}]$"
    test_features_LIST     = ["\log \Sigma_{\mathrm{gas}}",              "\log \sigma_{\mathrm{gas},z}", "\log \Sigma_{*}"]
    test_features_units    = ["$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$", "$~[\mathrm{km~s}^{-1}]$",      "$~[\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$"]
    test_features_cmaps    = ["Greens",                                 "Purples",                      "Blues"]
    model_complexity       = model_trained.equations_["complexity"]
   
    plot_y          = np.squeeze(y_test.to_numpy())
    n_levels        = [0.05, 0.2, 0.5, 0.8, 0.9999]
    n_levels0       = [n_levels[0]] 
    contour_colours = mpl.colormaps['Greys']([0.05, 0.25, 0.5, 0.75])#(n_levels[1:])
    contour_percent = list(map(lambda p: f"${p}\%$",[int(x) for x in np.ceil((1. - np.array(n_levels[:-1]))*100)])) #['$95\\%$', '$80\\%$', '$50\\%$', '$20\\%$']
    
    m_b         = 7100  # minimum baryonic mass [M_sol] 
    pixel_width = 750   # [pc]
    sfr_floor   = np.log10(m_b/((pixel_width**2)*sfr_type))
    gas_cut     = np.log10(10)    # [M_sol pc^-2]
    ylim        = [sfr_floor-0.5, 1.2]
     
    f_all_eqn, ax_all_eqn = plt.subplots(nrows=1, ncols=1, figsize = (6, 6)) 
    ax_all_eqn.set_ylabel(y_label + y_units, fontsize = 20) 
    ax_all_eqn.set_ylim(ylim[0], ylim[1])
    ax_all_eqn.tick_params(axis='both', which='major', labelsize=20) 
 
    f_pysr_eqn = plt.figure(figsize=(12, 16))
    f_pysr_eqn.text(0.05, 0.5, y_label + y_units, va='center', rotation='vertical', fontsize=16)
    n_cols     = len(test_features_LIST)
    n_rows     = len(df_top_eqns)
    gs_whole   = GridSpec(n_rows, n_cols, figure=f_pysr_eqn, wspace=0, hspace=0)

    for i, test_feature in enumerate(test_features_LIST):
        plot_x             = np.squeeze(X_test[test_feature].to_numpy())
        perr_m_FIRE        = find_dispersion(plot_x, plot_y) 
        test_feature_unit  = test_features_units[i]
        test_features_cmap = test_features_cmaps[i]
        x_label            = r"$" + test_feature + "$" + test_feature_unit
        # ---- DEFINE PLOT TO SHOW LOSS CURVES AND SFR COMPARISONS ---- #### 
        if i==0:
            ax_all_eqn.set_xlabel(x_label, fontsize = 20)
            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Greys", fill=True, ax=ax_all_eqn, bw_adjust=0.5, levels=n_levels)
            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Accent_r",         ax=ax_all_eqn, bw_adjust=0.5, levels=n_levels0)
            ax_all_eqn.axhline(y=sfr_floor, c="lightgrey", linestyle="dashdot")
            ax_all_eqn.axvline(x=gas_cut,   c="lightgrey", linestyle="dashdot")
        
        for n, eqn_latex in enumerate(df_top_eqns["latex equation"].values):
        #### ---- plot PySR found equation ---- ####
            eqn_complexity  = int((df_top_eqns["complexity"])[n])
            model_index     = int((df_top_eqns["model_orig_index"])[n])
            model_eqn       = "$\log\Sigma_{\mathrm{SFR}}$ = " + r"$"+eqn_latex+"$"
            y_model_trained = model_trained.predict(X_test_tempname, index=model_index) 
            eqn_colour      = complexity_colour(n, num_top_eqns=len(df_top_eqns))
        
            #gs_eqn       = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[n//n_cols,  n%n_cols])
            gs_eqn       = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_whole[n, i])
            ax_eqn       = f_pysr_eqn.add_subplot(gs_eqn[0])
            #ax_eqn.set_title(y_label + " = " + r"$"+eqn_latex+"$", fontsize = 9, fontweight ="bold", pad=12, bbox=dict(facecolor='yellow', alpha=0.4)) 
            ax_eqn.set_ylim(ylim[0], ylim[1])
            if i==len(test_features_LIST)-1:
                ax_eqn.text(-1.94, 0.9, model_eqn, fontsize =10, fontweight ="bold", bbox=dict(facecolor='bisque'), transform=ax_eqn.transAxes)
            if n==n_rows-1:
                ax_eqn.set_xlabel(x_label, fontsize=14)
            else:
                ax_eqn.xaxis.set_tick_params(labelbottom=False) 
            #if i==0:
            #    ax_eqn.set_ylabel(y_label + y_units, fontsize=14)
            #else:
            if i != 0:
                ax_eqn.yaxis.set_tick_params(labelleft=False)

            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Greys", fill=True, ax=ax_eqn, bw_adjust=0.5, levels=n_levels)
            sns.kdeplot(x=plot_x, y=plot_y,          cmap="Accent_r",         ax=ax_eqn, bw_adjust=0.5, levels=n_levels0)
    
            popt_m_model, perr_m_model = find_dispersion(plot_x, y_model_trained)
            
            cmap_object = mpl.colormaps[test_features_cmap]
            sns.kdeplot(x=plot_x, y=y_model_trained, cmap=test_features_cmap,      fill=True, ax=ax_eqn, bw_adjust=0.5, levels=n_levels)
            sns.kdeplot(x=plot_x, y=y_model_trained, color=cmap_object(n_levels[-3]),            ax=ax_eqn, bw_adjust=0.5, levels=n_levels0)
            ax_eqn.axhline(y=sfr_floor, c="lightgrey", linestyle="dashdot")
            if test_feature == "\log \Sigma_{\mathrm{gas}}":
                ax_eqn.axvline(x=gas_cut,   c="lightgrey", linestyle="dashdot")

            #COMMENT BACK IN TO DRAW mODEL W FIDUCIAL VALUES
            ax_eqn.text(0.06, 0.8, "complexity="+str(eqn_complexity),                        transform=ax_eqn.transAxes)
            ax_eqn.text(0.06, 0.72, r"$\sigma_{{\mathrm{{plane}}}}$={:.3f}".format(perr_m_model), transform=ax_eqn.transAxes)
            plot_K98 = False
            
            
            if i==0:
                eqn_behaviour(ax_eqn, eqn_latex, X_test, test_feature=test_feature, comp_obj=comp_obj)    
                plot_K98 = True if (n == len(df_top_eqns)-1) else False
                eqn_behaviour(ax_all_eqn, eqn_latex, X_test, test_feature=test_feature, comp_obj=comp_obj, comp_obj_colour=eqn_colour, theory_line_label="Complexity = " + str(eqn_complexity), plot_K98=plot_K98) 
                if (n == len(df_top_eqns)-1):
                    plot_K98   = True 
                    #legend_top = (2.35, -1.5) #defined in data units
                    #box_height = 1.55
                    #box_width  = 0.60
                    #box_bottom = (legend_top[0]-0.1, legend_top[1]-box_height+0.25)
                    legend_top_axes = (0.755, 0.45)
                    box_height_axes = 0.31
                    box_width_axes  = 0.27
                    box_bottom_axes = (legend_top_axes[0] - 0.05, legend_top_axes[1] - box_height_axes + 0.05)
                    
                    rect       = patches.Rectangle(box_bottom_axes, box_width_axes, box_height_axes, transform=ax_eqn.transAxes, linewidth=2, edgecolor='lightgrey', facecolor='white', alpha=1)
                    ax_eqn.add_patch(rect)
                    for j, colour in enumerate(contour_colours): 
                        ax_eqn.scatter(legend_top_axes[0],  legend_top_axes[1]-0.07*j, transform=ax_eqn.transAxes, c="dimgrey", marker="s", s=80)
                        ax_eqn.scatter(legend_top_axes[0],  legend_top_axes[1]-0.07*j, transform=ax_eqn.transAxes, c=colour, marker="s", s=55)
                        ax_eqn.text(legend_top_axes[0]+0.045, legend_top_axes[1]-0.07*j-0.015, "= "+str(contour_percent[j]), transform=ax_eqn.transAxes, bbox=dict(facecolor='white', edgecolor='white'))
                
                    ax_eqn.legend(loc="lower right")
    ax_all_eqn.legend(loc="lower right")
    f_pysr_eqn.savefig(savefile, bbox_inches='tight')
    f_all_eqn.savefig(savefile[:savefile.find(".")]+"_alleqns.pdf", bbox_inches='tight')
    print("SAVED " + savefile )

# ---- MAIN FUNCTION TO MAKE ANALYSIS PLOTS FOR FOUND EQNS ---- #

def found_eqns_analysis_plots(eqns_save_obj, X_train, X_test, y_train, y_test, sfr_type=10, num_top_eqns=6, max_loss=1, savefile="FOUND_EQNS.pdf", test_feature = "\log \Sigma_{\mathrm{gas}}"): 
    
    # ---- EXTRACT TRAINING DATA AND RESULTANT EQUATIONS ---- ####
    #(model_trained, df_train_loss_all_eqns, df_test_loss_all_eqns) = eqns_save_obj
    (model_trained, _, _) = eqns_save_obj
    #n_epochs     = len(df_train_loss_all_eqns.index)
    #test_feature = "\log \Sigma_{\mathrm{gas}}"
    
    no_columns             = X_train.columns.shape[0] 
    temp_column_names      = list(map(lambda n: "x"+str(n), np.arange(no_columns)))
    test_feature_temp_name = temp_column_names[int(np.where(np.array(X_test.columns.to_list()) == test_feature)[0])]
    X_test_tempname        = X_test.set_axis(temp_column_names, axis='columns')   

    # ---- GET TOP EQNS ---- #    
    df_top_eqns, df_found_eqns   = filter_top_eqns(model_trained, X_train, max_loss, num_top_eqns=num_top_eqns)
    df_top_eqns   = df_top_eqns.reset_index(drop=True)
    df_top_eqns.to_csv(savefile[:savefile.find(".")]+"_TOPEQNS.csv")
    
    plot_eqn_behaviour_all_features(model_trained, X_test, y_test, df_top_eqns, sfr_type=sfr_type,  savefile=savefile[:savefile.find(".")]+"_behaviour_allfeats.pdf")
    
    #pdb.set_trace()
    #plot_eqn_behaviour(model_trained, X_test, y_test, df_top_eqns, sfr_type=sfr_type, test_feature=test_feature, savefile=savefile[:savefile.find(".")]+"_behaviour.pdf")

    #make_sfrcomp_plot(model_trained, X_test, y_test, df_top_eqns, num_top_eqns=num_top_eqns, test_feature_name=test_feature, savefile=savefile)

#### ======================================== ####
#### ---------- CALL MAIN FUNCTION ---------- ####
#### ======================================== ####

if __name__ == "__main__":

