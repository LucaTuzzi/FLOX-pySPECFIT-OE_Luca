import os

#from src.AIRFLOX_processing_master import AIRFLOX_processing_master
from src.FLOX_processing_master import FLOX_processing_master
#from src.FLOX_processing_master_nc import FLOX_processing_master_nc
#from src.FLOX_processing import FLOX_processing

#only variable to be modified by the user. The path should be relative to the script directory, 
#and should contain the subfolders with the data to be processed #(e.g., "test-data")

#data_folder = r"G:\Il mio Drive\Data\2025_FRM4FLUO\Data\Storage-JB\SECOND_CAMPAIGN\SPECTROMETERS_DATA\VERSION_2\AIRFLOX DATA\UNIMIB"
#data_folder = "test_csv_no_unc"

#data_folder = "test_csv_unc"
#data_folder = "test_csv_teo"
data_folder = "test_nc"



# NetCDF input file (used only when nc_input = True)
data_nc = "RadIrrRef_FLOX_FLUO_2026-6-15_9_28_49.nc"

# Input data flags
nc_input = True                  # True: NetCDF input, False: CSV input
uncertainty_as_input = True      # True: read uncertainty from the input files # False: compute uncertainty from a fixed relative uncertainty
# Note: for NetCDF input, uncertainty is always read from the .nc file.

# PROCESSING OPTIONS
short_case = True # Process only a subset of the input data (100 spectra if nc, 1 file if csv) (used for testing)
SIF_unc_MC = True # Compute SIF uncertainty using Monte Carlo(TRUE) or analytical model (FALSE)

#Parallel Options
parallel = False # Enable parallel processing
n_jobs=-1 # Deafult: n_jobs= -1 ; ow but not suggested n_jobs = os.cpu_count()

# Apply a time/DOY mask defined in src/FLOX_mask.py
mask_time_doy = False #if False you can ignore the values of: date_range, time_ranges, doy_range

date_range = ("2026-04-24", "2026-04-26") # filtering by date range (start and end date) e.g ("2026-04-14", "2026-04-16")
time_ranges = [("10:00","12:00"), ("14:00","16:30")] # filtering by time intervals (can contain multiple ranges) e.g: [("12:00","12:05"), ("14:00","16:30")]
doy_range = None        #day-of-year range e.g: (124, 140)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #data_path = os.path.join(script_dir, "test_tds")
    data_path = os.path.join(script_dir, data_folder)
    cov_path = os.path.join(script_dir, "auxiliar_mat_files", "L2RM-AUX-V1-SIF_RHO_prior.nc")

    if uncertainty_as_input:
        uncertainty_path = data_path #path, for csv files with an absolute uncertainty for each spectrum
    else:
        uncertainty_path = os.path.join(script_dir, "auxiliar_mat_files", "uncertainty_FLOX_v2.mat")# relative uncertaint path, if the unc is not given as input

    try: data_nc
    except NameError:
        data_nc = "data_nc not defined"

    mask_time_opt = { "mask_time_doy": mask_time_doy,  "date_range": date_range, "time_ranges": time_ranges, "doy_range": doy_range,}

    FLOX_processing_master(data_path, uncertainty_path, data_nc, cov_path, nc_input, uncertainty_as_input, SIF_unc_MC, short_case, mask_time_opt, parallel, n_jobs)

    '''
    if nc_input:
        FLOX_processing_master_nc(data_path, data_nc, cov_path, uncertainty_as_input, SIF_unc_MC, parallel, short_case, mask_time_opt)
    else:
        FLOX_processing_master(data_path, uncertainty_path, cov_path, nc_input, uncertainty_as_input, SIF_unc_MC, parallel, short_case,)
    '''

else:
    print("This script is meant to be run as the main program. Please run it directly.")


#    CONFIG = {
#        "data_path": data_path,
#        "data_nc": data_nc,
#        "nc_input": nc_input,
#        "cov_path": cov_path,
#        "uncertainty_path": uncertainty_path,

#       "uncertainty_as_input": uncertainty_as_input,
#        "short_case": short_case,
#        "SIF_unc_MC": SIF_unc_MC,
#        "parallel": parallel,
#        "mask_time_doy": mask_time_doy,

#       "date_range": date_range,
#        "time_ranges": time_ranges,
#        "doy_range": doy_range,
#    }
