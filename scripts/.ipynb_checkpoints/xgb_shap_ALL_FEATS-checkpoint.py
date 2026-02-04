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
from symr_fire2.functions_run_xgb_shap import *
from symr_fire2.functions_run_pysr import *
#from shaphypetune import BoostRFE, BoostBoruta

#### ================================================= ####
#### ---------- FUNCTIONS TO MAKE .SH FILES ---------- ####
#### ================================================= ####
class Cfile:
    #subclass file to have a more convienient use of writeline
    def __init__(self, name, mode='w'):
        self.file = open(name, mode, buffering =1) 
    def wl(self, string, newline=True, endfile=False):
        if newline:
            self.file.write(string + '\n')
        else:
            self.file.write(string)
        if endfile:
            self.file.close()

def write_sh_script(z_str, xgb_picklefile, sfr_type, redshift_index=None, pixel_width=750, n_epochs=10000, n_saves=5, disk_type=None, input_df_picklefile=None, savepath="~/SYMR_FIRE2_GIT/RUN_XGB_SHAP_PYSR_ZBINS/"):
    sh_directory  = savepath + "SH_SCRIPTS/"
    if disk_type != None:
        exp_directory = savepath + z_str[1:] + "/" + disk_type + "/"
        sh_id         = "run"+z_str + "_" + disk_type
    else:
        exp_directory = savepath + z_str[1:] + "/"
        sh_id         = "run"+z_str
    sh_filename   = sh_directory + sh_id +".sh"
    
    fid = Cfile(sh_filename, 'w')
    fid.wl("#!/bin/bash") 
    
    fid.wl("#SBATCH --job-name="+sh_id)
    #fid.wl("#SBATCH -p gpu -G 1 -c 16")
    #fid.wl("#SBATCH --gpus=4")
    fid.wl("#SBATCH -c 64") # 128")--> using 128 cores maxes out ur core usage so can't run training simiultaneously
    fid.wl("#SBATCH -n 1")
    fid.wl("#SBATCH --mem=750000          # total memory per node in MB (see also --mem-per-cpu)")
    fid.wl("#SBATCH -t 10080              # Runtime in minutes, 10080=7days")
    #fid.wl("#SBATCH -p gpu                # Partition")
    fid.wl("#SBATCH -o "+sh_directory + sh_id + ".out # Standard out goes to this file")
    fid.wl("#SBATCH -e "+sh_directory + sh_id + ".err # Standard err goes to this filehostname") 
    fid.wl("#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL")
    fid.wl("#SBATCH --mail-user=diane.m.salim@gmail.com")
    #fid.wl("#SBATCH --mem=500000          # total memory per node in MB (see also --mem-per-cpu)")
    
    fid.wl("module load slurm") 
    fid.wl("module load gcc/11")
    fid.wl("module load python3")
    fid.wl("module load julia")
    fid.wl("source ~/symr_sf_venv/bin/activate")
    fid.wl("cd ~/SYMR_FIRE2_GIT/")

    fid.wl('python run_pysr_ALL_FEATS.py' + ' --input_df_picklefile =' + '\"' + str(input_df_picklefile) + '\"'
                                          + ' --xgb_picklefile='       + '\"' + xgb_picklefile + '\"' 
                                          + ' --sfr_type='             + str(sfr_type)
                                          + ' --redshift_index='       + str(redshift_index)
                                          + ' --pixel_width='          + str(pixel_width)
                                          + ' --epochs='               + str(n_epochs)
                                          + ' --n_saves_train='        + str(n_saves)
                                          + ' --savepath='             + '\"' + exp_directory + '\"')

#### ==========================================================================####
#### ---------- FUNCTIONS TO EXTRACT DATA FROM FIRE GALAXY SNAPSHOTS --------- ####
#### ==========================================================================####

def make_data_redshift_bin(galmap_mdir       = "/mnt/home/morr/ceph/analysis/sfr/galmap/",
                           redshift_txt      = "/mnt/home/firesims/public_binder/fire2/fire2/public_release/core/snapshot_times_public.txt",
                           redshift_bins     = [0, 0.5, 1.0, 2.0],
                           pixel_width       = 750,
                           all_df_picklefile = "all_galaxies_all_params_redshift_bins_df.pickle",
                           m_number          = 12):
    
    #df_redshifts   = pd.read_fwf(redshift_txt, comment='#', names=["index", "scale-factor", "redshift", "time[Gyr]", "lookback-time[Gyr]", "time-width[Myr]"])
    
    df_redshifts = pd.read_csv(redshift_txt, delim_whitespace=True, comment="#", names=["index", "scale-factor", "redshift", "time[Gyr]", "lookback-time[Gyr]", "time-width[Myr]"])

    #pdb.set_trace()
    gal_index      = df_redshifts["index"].astype("int")
    gal_redshift   = df_redshifts["redshift"].map(lambda x: '%2.5f' % x).astype("float64")
    
    logfit_FUNC       = lambda x, a, c: a/x + c
    p0                = [600, 10]
    popt, pcov        = curve_fit(logfit_FUNC, gal_index, gal_redshift, p0)
    best_fit_redshift = logfit_FUNC(gal_index, *popt)
    # ---- see index-redshift relation ---- #
    f, ax = plt.subplots()
    ax.set_ylabel("redshift")
    ax.set_xlabel("galaxy snapshot index")
    ax.plot(gal_index, gal_redshift, label="txtfile data")
    ax.plot(gal_index, best_fit_redshift, label="curvefit")
    ax.axhline(2, color='lightgray', linestyle='dashed', label="z=2")
    ax.legend()
    f.savefig("galaxy_index_redshift.pdf", bbox_inches='tight')
    # ------------------------------------- #

    gal_index_0    = [np.array(gal_index.where(gal_redshift>=redshift_bins[0]).dropna())[-1]]
    gal_index_cuts = list(map(lambda r: np.array(gal_index.where(gal_redshift<=r).dropna()).astype("int")[0], redshift_bins[1:]))
    gal_index_bins = np.array(gal_index_0 + gal_index_cuts).astype("int") #[600, 382, 294, 172]

    m_number_folders_list = glob.glob(galmap_mdir + "m"+str(m_number)+"*_res7100/")

    all_redshifts_df_save_obj = []
    
    for i in range(len(redshift_bins)-1):
        redshift_bin_files            = np.array(list(map(lambda f: extract_galaxies_redshift_bin(f, gal_index_bins, i), m_number_folders_list))).ravel()
        df_all_snapshots_redshift_bin = make_df_all_params(redshift_bin_files, gal_index, gal_redshift, pixel_width=pixel_width) 
        print("FINISHED DATAFRAME " + str(redshift_bins[i]) + "<=z<" + str(redshift_bins[i+1]))
        all_redshifts_df_save_obj.append(df_all_snapshots_redshift_bin)

    with open(all_df_picklefile, "wb") as f:
        pickle.dump(all_redshifts_df_save_obj, f)
    
    print("FINISHED PICKLING " + all_df_picklefile) 
    return all_redshifts_df_save_obj

def extract_galaxies_redshift_bin(m12_folder, gal_index_bins, redshift_bin_index, pixel_width=750): #redshift_bin_index: 0, 1 or 2 
    res_extension      = "_"+str(pixel_width)+"pc.hdf5"
    snapshot_files     = np.array(sorted(glob.glob(m12_folder + "*" + res_extension), reverse=True))
    mnum_list          = np.array(list(map(lambda m : m[-len(res_extension)-3:-len(res_extension)], snapshot_files))).astype("int")
    #bin_cuts_index     = list(map(lambda r: int(np.where(mnum_list==r)[0]), gal_index_bins))
    bin_cuts_index     = [np.where(mnum_list == r)[0][0] for r in gal_index_bins if np.any(mnum_list == r)]


    if (redshift_bin_index == len(gal_index_bins)-2): 
        redshift_bin_files = snapshot_files[bin_cuts_index[redshift_bin_index]:]
    else:
        redshift_bin_files = snapshot_files[bin_cuts_index[redshift_bin_index]: bin_cuts_index[redshift_bin_index+1]]
    return redshift_bin_files

def index_to_redshift(mnum, gal_index, gal_redshift):
    diff = (int(mnum) - gal_index)
    if 0 in diff.values:
        redshift = (gal_redshift.where(gal_index==mnum).dropna()).to_numpy()
    else:
        #extract_min_FUNC = (lambda a: (lambda df: (df.where(a==a.min()).dropna())))(diff) 
        extract_min_FUNC = (lambda a: (lambda df: (df.where(a.where(a>0) == a.where(a>0).min())).dropna()))(diff)
        ind_1  = extract_min_FUNC(gal_index)
        z_1    = extract_min_FUNC(gal_redshift)
        ind_2  = gal_index[ind_1.index + 1].to_numpy()
        z_2    = gal_redshift[ind_1.index + 1].to_numpy()
        grad     = (z_2 - z_1.to_numpy())/(ind_2 - ind_1.to_numpy())
        offset   = z_2 - grad*ind_2
        pdb.set_trace()
        redshift = grad*np.array(mnum).astype("float64") + offset
        pdb.set_trace()
    return redshift
 
def make_inner_outer_disk_masks(pixel_width=750, total_num_pixels_width=40, inner_disk_r=2000, outer_disk_r1=7000, outer_disk_r2=9000): #distance units = pc
    total_num_pixels_width_half = total_num_pixels_width/2.
    x                           = np.linspace(-total_num_pixels_width_half, total_num_pixels_width_half, total_num_pixels_width)*pixel_width
    y                           = np.linspace(-total_num_pixels_width_half, total_num_pixels_width_half, total_num_pixels_width)*pixel_width
    xx, yy                      = np.meshgrid(x, y)
    dist_from_centre            = np.sqrt(xx**2. + yy**2.)
    
    disk_pixel_classification   = np.full([total_num_pixels_width, total_num_pixels_width], "OTHER_DISK")
    disk_pixel_classification[np.where(dist_from_centre <= inner_disk_r)]                                          = "INNER_DISK" #classify inner disk
    disk_pixel_classification[np.where((dist_from_centre <= outer_disk_r2) & (dist_from_centre >= outer_disk_r1))] = "OUTER_DISK" #classify outer disk
    
    return disk_pixel_classification.ravel() 
     
def make_df_all_params(snapshot_files, pixel_width=750): #, gal_index, gal_redshift):
    vdisp_comps = ['r', 'phi', 'z']
    df_ALL_SNAPSHOTS = pd.DataFrame()
    res_extension      = "_"+str(pixel_width)+"pc.hdf5" 
    mnum_list          = np.array(list(map(lambda m : m[-len(res_extension)-3:-len(res_extension)], snapshot_files))).astype("int")

    for i, datafile in enumerate(snapshot_files):
        mnum             = datafile[-len(res_extension)-3:-len(res_extension)]
        #redshift         = index_to_redshift(mnum, gal_index, gal_redshift)
        dataset          = h5py.File(datafile,'r')
        df_THIS_SNAPSHOT = pd.DataFrame()

        for n, key in enumerate(dataset.keys()):
            if len(dataset[key].shape)==2:
                df_THIS_SNAPSHOT.insert(loc=len(df_THIS_SNAPSHOT.columns), column=key, value=(np.array(dataset[key])).ravel())
            elif (dataset[key].shape)[-1]==3: #ADD VELOCITY DISPERSION COMPONENTS
                for r in range(3):
                    df_THIS_SNAPSHOT.insert(loc=len(df_THIS_SNAPSHOT.columns), column=key+"_"+vdisp_comps[r], value=(np.array(dataset[key][:,:,r])).ravel())
            elif (dataset[key].shape)[-1]==11: #ADD OVERALL METALLICITY
                df_THIS_SNAPSHOT.insert(loc=len(df_THIS_SNAPSHOT.columns), column=key, value=(np.array(dataset[key][:,:,0])).ravel())
            
        disk_pixel_classification = make_inner_outer_disk_masks(pixel_width=pixel_width)
        df_THIS_SNAPSHOT.insert(loc=len(df_THIS_SNAPSHOT.columns), column="DiskType", value=disk_pixel_classification) 
        
        df_ALL_SNAPSHOTS = pd.concat([df_ALL_SNAPSHOTS, df_THIS_SNAPSHOT])    
    df_ALL_SNAPSHOTS.reset_index(drop=True, inplace=True)
    return df_ALL_SNAPSHOTS

#### ======================================================== ####
#### --------- FUNCTIONS TO FILTER DESIRED FEATURES --------- ####
#### ======================================================== ####
def cut_unnecessary_features(df):
    #keep_keywords = np.array(["GasColdDens", "GasNeut",     "StarMass",  "StarVDisp",   "omega_dyn"])
    keep_keywords = np.array(["GasNeut",     "StarMass",    "StarVDisp", "omega_dyn"])
    cut_keywords  = np.array(["VDisp_phi",   "VDisp_m_phi", "_r",        "010"])
    
    all_features  = np.array(df.keys().tolist())
    keep_features = list(map(lambda keep: all_features[list(map(lambda s: keep in s, all_features))], keep_keywords))
    keep_features = np.array([item for sublist in keep_features for item in sublist])
    
    filtered_features = keep_features
    for i, cut in enumerate(cut_keywords):
        filtered_features = filtered_features[list(map(lambda s: cut not in s, filtered_features))]
    df_filtered = df[filtered_features].rename(columns=keyword_to_latex) 
    return df_filtered

def get_symbol_FUNC(d, keyword):
    key_list        = np.array(list(d.keys()))
    keys_to_extract = key_list[list(map(lambda s: s in keyword, key_list))]
    values          = [d[key] for key in keys_to_extract]
    full_symbol     = ','.join(values)
    return full_symbol

def keyword_to_latex(keyword):
    symbol_dict    = {"Mass":"M", "VDisp_":"\sigma", "V_":"V", "Z":"Z", "omega":"\Omega"}
    #subscript_dict = {"GasColdDens":"\mathrm{C.D.~gas}", "GasNeut":r"\mathrm{N.~gas}", "Star":"*", "_z":"z", "_phi":"\Phi", "_dyn":"\mathrm{dyn}" }
    subscript_dict = {"GasNeut":"\mathrm{gas}", "Star":"*", "_z":"z", "_phi":"\Phi", "_dyn":"\mathrm{dyn}" }

    terms           = list(map(lambda d: get_symbol_FUNC(d, keyword), [symbol_dict, subscript_dict]))  
    make_latex_var  = lambda t : f"{t[0]}_{{{t[1]}}}"
    latex_var       = make_latex_var(terms)
    return latex_var

def make_df_filt_feats(df_all_params, pixel_width=750, sfr_type=10):
    pixel_area    = pixel_width**2.
    
    if sfr_type   == 10:
        df_Zsfr   = pd.DataFrame(np.array(df_all_params["SFR_ave_010"])/0.85*1e6/pixel_area, columns=["\Sigma_{\mathrm{SFR, 10Myr}}"]) 
    if sfr_type   == 100:
        df_Zsfr   = pd.DataFrame(np.array(df_all_params["SFR_ave_100"])/0.70*1e6/pixel_area, columns=["\Sigma_{\mathrm{SFR, 100Myr}}"])
    df_filt_feats = cut_unnecessary_features(df_all_params)

    # ---- change "Mass" features to "Column Density"
    column_dens_FUNC          = (lambda a: (lambda m: m/a))(pixel_area)
    column_dense_key_FUNC     = lambda k: k.replace("M", "\Sigma") 
    mass_feats                = df_filt_feats.keys().where([(key.find("M") != -1) for key in df_filt_feats.keys()]).dropna() 
    df_filt_feats[mass_feats] = df_filt_feats[mass_feats].apply(column_dens_FUNC)
    df_filt_feats             = df_filt_feats.rename(columns=column_dense_key_FUNC)
   
    # ---- make cuts for \Sigma_gas<10
    dens_feats                    = np.array(list(map(column_dense_key_FUNC, mass_feats.tolist())))
    dens_gas_feats                = dens_feats[np.where([(feat.find("gas") != -1) for feat in dens_feats])].tolist()
    df_filt_feats[dens_gas_feats] = np.where(df_filt_feats[dens_gas_feats]<10, np.nan, df_filt_feats[dens_gas_feats])    

    # ---- add gas fraction as feature 
    df_f_gas = df_filt_feats["\Sigma_{\mathrm{gas}}"]/(df_filt_feats["\Sigma_{\mathrm{gas}}"]+df_filt_feats["\Sigma_{*}"])
    df_filt_feats.insert(loc=len(df_filt_feats.columns), column = "f_{\mathrm{gas}}", value=df_f_gas)

    # ---- output data in log space
    log_rename_FUNC        = lambda x: f"\log {x}"
    log_df_FUNC            = lambda d: d.apply(np.log10).rename(columns=log_rename_FUNC)

    df_log_filt_feats, df_log_Zsfr = list(map(log_df_FUNC, [df_filt_feats, df_Zsfr]))

    return df_log_filt_feats, df_log_Zsfr

#### =========================================== ####
#### ---------- CALLING MAIN FUNCTION ---------- #### 
#### =========================================== ####

if __name__ == "__main__":
    
    # -------- second argument in terminal call is index for redshift bin range
    # -------- eg. run shap_all_parameters 2
    ######## LOAD IN RAW DATA ########

    galmap_mdir       = "/mnt/home/morr/ceph/analysis/sfr/galmap/"
    redshift_txt      = "/mnt/home/chayward/firesims/fire2/public_release/core/snapshot_times_public.txt"
    redshift_bins     = [0, 0.5, 1, 2]
    pixel_width       = 750
    all_df_picklefile = "all_galaxies_all_params_redshift_bins_df.pickle"
    sfr_timescales    = [10, 100] #units: Myr
    disk_types        = ["INNER_DISK", "OUTER_DISK"]
    savepath          = "~/SYMR_FIRE2_GIT/RUN_XGB_SHAP_PYSR_ZBINS/"
    n_epochs_old      = 0
    n_epochs          = 10000
    n_saves           = 5
    
    if (os.path.isfile(all_df_picklefile)==True):
        with open(all_df_picklefile, "rb") as f:
            all_redshifts_df_save_obj     = pickle.load(f) 
    else: 
        all_redshifts_df_save_obj = make_data_redshift_bin(galmap_mdir=galmap_mdir, redshift_txt=redshift_txt, redshift_bins=redshift_bins, pixel_width=pixel_width, all_df_picklefile=all_df_picklefile) 
     
    filename_FUNC           = (lambda sp: (lambda n, ex: (lambda zs: sp + zs[1:] + "/" + n + zs + ex)))(savepath) 
    #eqns_epochs_file_FUNC   = lambda e: filename_FUNC("FOUND_EQNS", "_"+str(e)+"EPOCHS.pickle")  

    ######## RUN XGBOOST & SHAP ########

    for i, sfr_type in enumerate(sfr_timescales):
        sfr_str          = "_SFR_"+str(sfr_type)+"Myr" 
        redshift_bin_str = list(map(lambda s: s.replace(".", "pt"), ["_z"+str(redshift_bins[i])+"_z"+str(redshift_bins[i+1])+sfr_str for i in range(len(redshift_bins)-1)]))
        
        xgb_files_list   = list(map(filename_FUNC("xgb_shap"         , ".pickle"), redshift_bin_str))
        pdf_files_list   = list(map(filename_FUNC("shap_summary_plot", ".pdf"),    redshift_bin_str))

        filt_data_list   = list(map((lambda p, sf: (lambda d: make_df_filt_feats(d, p, sf)))(pixel_width, sfr_type), all_redshifts_df_save_obj))
        df_log_filt_feats_list, df_log_Zsfr_list = zip(*filt_data_list) 
        
        shap_plot_inputs       = zip(df_log_filt_feats_list, df_log_Zsfr_list, xgb_files_list, pdf_files_list)
        #xbg_allz_save_obj_LIST = list(map(lambda x: make_shap_plot(*x), shap_plot_inputs))
        
        for j, z_str in enumerate(redshift_bin_str): 
            print(j)
            print(z_str)
            print(xgb_files_list[j])
            write_sh_script(z_str, xgb_files_list[j], sfr_type, redshift_index=j, pixel_width=pixel_width, n_epochs=n_epochs, n_saves=n_saves, savepath=savepath)
            if j==0:
                for k, disk_type in enumerate(disk_types):
                    xgb_file_disk      = filename_FUNC(disk_type + "/xgb_shap",          "_" + disk_type + ".pickle")(z_str)
                    pdf_file_disk      = filename_FUNC(disk_type + "/shap_summary_plot", "_" + disk_type + ".pdf")(z_str)
                    write_sh_script(z_str, xgb_file_disk, sfr_type, redshift_index=j, pixel_width=pixel_width, n_epochs=n_epochs, n_saves=n_saves, disk_type=disk_type, savepath=savepath)

                    df_all_params_disk = all_redshifts_df_save_obj[j].where(all_redshifts_df_save_obj[j]["DiskType"]==disk_type).dropna()
                    df_log_filt_feats_disk, df_log_Zsfr_disk =  make_df_filt_feats(df_all_params_disk, pixel_width=pixel_width, sfr_type=sfr_type)
                    xbg_save_obj_disk  = make_shap_plot(df_log_filt_feats_disk, df_log_Zsfr_disk, xgb_file_disk, pdf_file_disk)
        
         




