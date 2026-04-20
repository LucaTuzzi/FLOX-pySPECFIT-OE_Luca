import os

#from src.AIRFLOX_processing_master import AIRFLOX_processing_master
from src.FLOX_processing_master import FLOX_processing_master
#from src.FLOX_processing import FLOX_processing

#only variable to be modified by the user. The path should be relative to the script directory, 
#and should contain the subfolders with the data to be processed #(e.g., "test-data")
data_folder=  "Version_3.1_test-data" 
#data_folder = r"G:\Il mio Drive\Data\2025_FRM4FLUO\Data\Storage-JB\SECOND_CAMPAIGN\SPECTROMETERS_DATA\VERSION_2\AIRFLOX DATA\UNIMIB"

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #data_path = os.path.join(script_dir, "test_tds")
    data_path = os.path.join(script_dir, data_folder)
    cov_path = os.path.join(script_dir, "auxiliar_mat_files", "L2RM-AUX-V1-SIF_RHO_prior.nc")
        
    uncertainty_as_input = True #if True, the uncertainty is read from the csv files in the data folder, if False it is computed
    short_case = False #for testing purposes, if True it run only a subset of the data
    SIF_unc_MC = True
    parallel = False

    if uncertainty_as_input:
        uncertainty_path = data_path #path, for csv files with an absolute uncertainty for each spectrum
    else:
        uncertainty_path = os.path.join(script_dir, "auxiliar_mat_files", "uncertainty_FLOX_v2.mat")# relative uncertaint path, if the unc is not given as input

    FLOX_processing_master(data_path, uncertainty_path, cov_path, uncertainty_as_input, SIF_unc_MC, parallel, short_case, )
else:
    print("This script is meant to be run as the main program. Please run it directly.")

