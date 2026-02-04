#!/bin/bash
#SBATCH --job-name=run_z_ALL_SFR_100Myr
#SBATCH -c 64
#SBATCH -n 1
#SBATCH --mem=750000          # total memory per node in MB (see also --mem-per-cpu)
#SBATCH -t 10080              # Runtime in minutes, 10080=7days
#SBATCH -o /mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/SH_SCRIPTS/run_z_ALL_SFR_100Myr.out # Standard out goes to this file
#SBATCH -e /mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/SH_SCRIPTS/run_z_ALL_SFR_100Myr.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=diane.m.salim@gmail.com
module load slurm
module load gcc/11
module load python3
module load julia
source ~/symr_sf_venv/bin/activate
cd /mnt/home/dsalim/SYMR_STARFORMATION/
python run_pysr_ALL_FEATS.py --input_df_picklefile ="None" --xgb_picklefile="/mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/z_ALL_SFR_100Myr/m11_xgb_shap_z_ALL_SFR_100Myr.pickle" --sfr_type=100 --redshift_index=None --pixel_width=750 --epochs=4000 --n_saves_train=8 --savepath="/mnt/home/dsalim/SYMR_STARFORMATION/RUN_XGB_SHAP_PYSR_ZBINS_QUANTILE_LOSS/z_ALL_SFR_100Myr/"
