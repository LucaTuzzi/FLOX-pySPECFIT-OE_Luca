import os

#from src.AIRFLOX_processing_master import AIRFLOX_processing_master
from src.FLOX_processing_master import FLOX_processing_master
from src.FLOX_processing_master_nc import FLOX_processing_master_nc
#from src.FLOX_processing import FLOX_processing

#only variable to be modified by the user. The path should be relative to the script directory, 
#and should contain the subfolders with the data to be processed #(e.g., "test-data")
#data_folder=  "Version_3.1_test-data" 
#data_folder = r"G:\Il mio Drive\Data\2025_FRM4FLUO\Data\Storage-JB\SECOND_CAMPAIGN\SPECTROMETERS_DATA\VERSION_2\AIRFLOX DATA\UNIMIB"
data_folder = "dati_inc_nc"
data_nc = "RadIrrRef_FLOX_FLUO_2026-6-15_9_28_49.nc" #used only if nc_input is True. It should be the name of the .nc file to be processed, which should be in the data_folder

# Control flags
nc_input=True # in this case, data_nc is required to be the .nc file itself
uncertainty_as_input = True #if True, the uncertainty is read from the csv files in the data folder, if False it is computed. In case of nc_input=True, the uncertainty is read from the .nc file itself.
short_case = False #for testing purposes, if True it only run a subset of the data
SIF_unc_MC = True
parallel = False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #data_path = os.path.join(script_dir, "test_tds")
    data_path = os.path.join(script_dir, data_folder)
    cov_path = os.path.join(script_dir, "auxiliar_mat_files", "L2RM-AUX-V1-SIF_RHO_prior.nc")

    if uncertainty_as_input:
        uncertainty_path = data_path #path, for csv files with an absolute uncertainty for each spectrum
    else:
        uncertainty_path = os.path.join(script_dir, "auxiliar_mat_files", "uncertainty_FLOX_v2.mat")# relative uncertaint path, if the unc is not given as input

    if nc_input:
        FLOX_processing_master_nc(data_path, data_nc, cov_path, uncertainty_as_input, SIF_unc_MC, parallel, short_case,)
    else:
        FLOX_processing_master(data_path, uncertainty_path, cov_path, uncertainty_as_input, SIF_unc_MC, parallel, short_case,)

else:
    print("This script is meant to be run as the main program. Please run it directly.")

