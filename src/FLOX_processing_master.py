import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
from scipy.io import loadmat
from src.FLOX_functions import write_results_to_netcdf, load_inverse_covariance, write_csv_with_headers
from src.FLOX_processing import FLOX_processing
#from src.FLOX_processing_NotParallel import FLOX_processing
import time


def FLOX_processing_master(data_path, uncertainty_path, cov_path, uncertainty_as_input=True, SIF_unc_MC=True, parallel=False, short_case=False,):
    """
    Method that orchestrates FLOX data retrieval from CSV files

    Args:
        data_path (str): Directory path where CSV data files are located
        uncertainty_path (str): Directory path where uncertainty data files are located
        cov_path (str): Path to the .mat file containing the inverse covariances
        SIF_unc_MC (bool): Whether to use Monte Carlo sampling for SIF uncertainty
        parallel (bool): Whether to use parallel processing
        short_case (bool): Only for testing purposes, if True to run only a subset of the data
    """

    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()

    # Initialize log file
    proc_time_all = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    logfile_name = os.path.join(output_path, f"{proc_time_all}_logfile.txt")

    logf = open(logfile_name, "w", encoding="utf-8")
    logf.write(f"FLOX_processing_master started at {proc_time_all}\n")
    logf.flush()

    # Suppress warnings:
    import warnings
    warnings.filterwarnings("ignore")

    # Log the input arguments:
    logf.write(f"data_path   = {data_path}\n")
    logf.write(f"uncertainty_path = {uncertainty_path}\n")
    logf.write(f"output_path = {output_path}\n")
    logf.write(f"cov_path         = {cov_path}\n")
    logf.write(f"uncertainty_as_input = {uncertainty_as_input}\n")
    logf.write(f"SIF_unc_MC = {SIF_unc_MC}\n")
    logf.write(f"parallel = {parallel}\n")
    logf.write(f"short_case = {short_case}\n")  
    logf.flush()

    sa, xa_mean = load_inverse_covariance(cov_path)
    # Adjust covariance matrix values
    sa[8:, :8]  = 1e-15
    sa[:8, 8:]  = 1e-5   #!!!
    sa[0, 3]    = 1e-15
    sa[3, 0]    = 1e-15
    sa[:2, 3:]  = 1e-15
    sa[3:, :2]  = 1e-15

    # Adjust variables before running the program
    wvlRet = [670, 780]
    
    # Init variables
    allm_specfit = []
    allf_specfit = None         # SIF 
    allf_unc_specfit = None     # SIF uncertainty 
    allr_specfit = None         # Reflectance    
    allar_unc = None            # Apparent Reflectance uncertainty 

    # Search for input files: AppReflectance as main data (so with uncertainty), Incoming radiance as atm fun (so no uncertainty).
    AppReflectance_pattern = os.path.join(data_path, "**", "Reflectance*FLUO*.csv")
    AppReflectance_list = sorted(glob.glob(AppReflectance_pattern, recursive=True))
    Lin_pattern = os.path.join(data_path, "**", "Incoming*FLUO*.csv")
    Lin_list = sorted(glob.glob(Lin_pattern, recursive=True))

    if uncertainty_as_input:
        unc_AppReflectance_pattern = os.path.join(uncertainty_path, "**", "Uncertainty*Reflectance*FLUO*.csv")
        unc_AppReflectance_list = sorted(glob.glob(unc_AppReflectance_pattern, recursive=True))
    else:
        unc_data = loadmat(uncertainty_path)

    #short version only for testing:
    if short_case:
        start_temp=0
        end_temp=1
        AppReflectance_list = AppReflectance_list[start_temp:end_temp]
        Lin_list= Lin_list[start_temp:end_temp]
        if uncertainty_as_input:
            unc_AppReflectance_list = unc_AppReflectance_list[start_temp:end_temp] 
        

    n_tables = len(AppReflectance_list)

    # Basic check confirming that the number of files matches:
    if uncertainty_as_input:
        valid = (len(Lin_list) == len(AppReflectance_list) == len(unc_AppReflectance_list))
    else:
        valid = (len(Lin_list) == len(AppReflectance_list))

    if not valid:
        msg = (
            "Warning: mismatch in number of files:\n"
            f"Incoming FLUO: {len(Lin_list)}\n"
            f"AppReflectance FLUO: {len(AppReflectance_list)}\n"
            + (f"Uncertainty AppReflectance FLUO: {len(unc_AppReflectance_list)}\n" if uncertainty_as_input else "" )
        )
        logf.write(msg + "\n")
        logf.flush()
        raise ValueError(msg)

    # Loop over each pair of CSV files

    for i_pair in range(n_tables):
        logf.write(f"\nProcessing file {i_pair+1} of {n_tables}\n")
        logf.flush()

        # Read the AppReflectance CSV
        fname_AppReflectance = AppReflectance_list[i_pair]
        logf.write(f"AppReflectance: {fname_AppReflectance}\n")
        with open(fname_AppReflectance, "r", encoding="utf-8") as f:
            sample = "".join([f.readline() for _ in range(2)])
            delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        data_AppReflectance = pd.read_csv(fname_AppReflectance,sep=delimiter,header=0)
        AppReflectance_table = data_AppReflectance.iloc[:, 1:].to_numpy()

        # Read the Incoming CSV
        fname_Lin = Lin_list[i_pair]
        logf.write(f"Incoming: {fname_Lin}\n")
        with open(fname_Lin, "r", encoding="utf-8") as f:
            sample = "".join([f.readline() for _ in range(2)])
            delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        data_Lin = pd.read_csv(fname_Lin, sep=delimiter, header=0)
        Lin_table = data_Lin.iloc[:, 1:].to_numpy()

        # Read the AppReflectance Uncertainty CSV
        if uncertainty_as_input:
            fname_unc_AppReflectance = unc_AppReflectance_list[i_pair]
            logf.write(f"Uncertainty: {fname_unc_AppReflectance}\n")
            with open(fname_unc_AppReflectance, "r", encoding="utf-8") as f:
                sample = "".join([f.readline() for _ in range(2)])
                delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
            data_unc_AppReflectance = pd.read_csv(fname_unc_AppReflectance, sep=delimiter, header=0)
            unc_AppReflectance_table = data_unc_AppReflectance.iloc[:, 1:].to_numpy()
        else:
            logf.write(f"Uncertainty: computed\n")

        #!!! Check consistency of file names: assume that the last 6 digits before .csv must be equal
        base_name_AppReflectance = os.path.basename(fname_AppReflectance)[-10:-4]  # Remove .csv extension
        base_name_Lin = os.path.basename(fname_Lin)[-10:-4]

        if uncertainty_as_input:
            base_name_unc_AppReflectance = os.path.basename(fname_unc_AppReflectance)[-10:-4]
            consistent = (base_name_AppReflectance == base_name_unc_AppReflectance == base_name_Lin)
        else:
            consistent = (base_name_AppReflectance == base_name_Lin)

        if not consistent:
            msg = f"Warning: Inconsistent file names for pair {i_pair+1}"
            logf.write(msg + "\n")
            logf.flush()
            raise ValueError(msg)

        #get only once:
        # #1) (from AppReflectance), assuming all files have the same wavelength (first column) and UTC timestamp (first row)
        # 2) (from uncertainty file, if not given as input) get the uncertainty values for the wavelength grid of the input spectra, to be used as input in the retrieval
        wvl_qepro = data_AppReflectance.iloc[:, 0].to_numpy()
        utc_column = data_AppReflectance.columns[1:]

        # Wavelength Definition
        lb = np.argmin(np.abs(wvl_qepro - wvlRet[0]))
        ub = np.argmin(np.abs(wvl_qepro - wvlRet[1]))
        wvl_sub = wvl_qepro[lb:ub+1]

        #read, mask and interpolate relative uncertainty values of Lin and Lout
        if not uncertainty_as_input:
            wl_unc = unc_data["wl_unc"].flatten()
            L_down_unc = unc_data["L_down_unc"].flatten()
            L_up_unc = unc_data["L_up_unc"].flatten()

            # Filter valid ranges [0, 5]
            mask_down = (L_down_unc >= 0) & (L_down_unc < 5)
            mask_up = (L_up_unc >= 0) & (L_up_unc < 5)

            wl_unc_down, val_down = wl_unc[mask_down], L_down_unc[mask_down]
            wl_unc_up, val_up = wl_unc[mask_up], L_up_unc[mask_up]

            flox_unc_down = np.interp(wvl_sub, wl_unc_down, val_down, left=np.nan, right=np.nan) \
                if wl_unc_down.size > 1 else np.full_like(wvl_sub, np.nan)
            flox_unc_up = np.interp(wvl_sub, wl_unc_up, val_up, left=np.nan, right=np.nan) \
                if wl_unc_up.size > 1 else np.full_like(wvl_sub, np.nan)

        # Spectral subset of input spectra to min_wvl - max_wvl range;  # Replace NaN and inf values with zeros (or a small number) to avoid issues in processing
        #!!! is it right to fix to 0 ?
        AppReflectance = AppReflectance_table[lb:ub+1, :]
        AppReflectance = np.nan_to_num(AppReflectance, nan=0.0, posinf=0.0, neginf=0.0) 
        Lin = Lin_table[lb:ub+1, :] * 1e3 #convert Lin from W to mW.
        Lin = np.nan_to_num(Lin, nan=0.0, posinf=0.0, neginf=0.0)

        valid_cols = np.all(AppReflectance >= 0, axis=0)
        logf.write(f"Spectra not processed (AppReflectance<0), indices: {np.where(~valid_cols)[0]}\n")
        logf.flush()
        
        AppReflectance = AppReflectance[:, valid_cols]
        Lin = Lin[:, valid_cols]

        if uncertainty_as_input:
            unc_AppReflectance = unc_AppReflectance_table[lb:ub+1, :]
            unc_AppReflectance = unc_AppReflectance[:, valid_cols]
        else:
            Lin_unc = flox_unc_down[:, np.newaxis] * Lin
            Lout_unc = flox_unc_up[:, np.newaxis] * Lin * AppReflectance
            #this is the relative unc propagation forumla, using only AppReflectance and Lin, so Lout = AppReflectance * Lin
            unc_AppReflectance = (np.sqrt(Lout_unc**2 + (AppReflectance * Lin_unc)**2 ))/Lin
        
        unc_AppReflectance = np.nan_to_num(unc_AppReflectance, nan=0.0, posinf=0.0, neginf=0.0)  

        # Compute UTC_time for the header information (from AppReflectance)
        utc_column = utc_column[valid_cols]
        utc_time = [header.strip() for header in utc_column]
        #folder_date = os.path.basename(os.path.dirname(fname_AppReflectance))

        doy_day_frac = []
        utc_datime_str = []

        for time_str in utc_time:
            dt_obj = datetime.strptime(time_str, "%d%m%y_%H%M%S")
            day_of_year = (dt_obj - datetime(dt_obj.year, 1, 1)).days + 1
            frac_day = (dt_obj - datetime(dt_obj.year,dt_obj.month, dt_obj.day)).seconds / 86400
            doy_day_frac.append(day_of_year + frac_day)
            utc_datime_str.append(dt_obj.strftime("%d-%b-%Y %H:%M:%S"))

        # === FLOX Data Processing ===
        (
            sif,                # Retrieved SIF spectrum (array)
            ref,                # Retrieved reflectance spectrum (array)
            fluo_unc,           # Uncertainty for SIF retrieval
            ref_unc,            # Uncertainty for reflectance retrieval
            wvl_out,            # Output wavelength grid

            # Key SIF metrics
            sif_r_max,          # Maximum SIF value in the red region
            sif_r_wl,           # Wavelength corresponding to red peak
            sif_o2b,            # SIF value at O2-B absorption band
            sif_fr_max,         # Maximum SIF value in the far-red region
            sif_fr_wl,          # Wavelength corresponding to far-red peak
            sif_o2a,            # SIF value at O2-A absorption band
            sif_int,            # Integrated SIF over the spectrum

            # Uncertainties for key bands
            sif_o2a_un,         # Uncertainty at O2-A band
            sif_o2b_un,         # Uncertainty at O2-B band

            app_ref_unc,         # Apparent reflectance uncertainty
            failed_spectra       # List of spectra indices that failed during retrieval
        ) = FLOX_processing(         
            wvl=wvl_sub,             # Wavelength subset for processing
            app_ref_array=AppReflectance,
            app_ref_unc_array = unc_AppReflectance,
            Lin_array=Lin ,        # Incident radiance (Lin) used only in arho_sim = rho + fluorescence / Lin
            sa=sa,  # covariance matrix  
            xa_mean=xa_mean,                # mean state vector
            SIF_unc_MC=SIF_unc_MC,
            parallel=parallel
        )
        n_spectra = AppReflectance.shape[1]

        # Accumulate the results
        out_arr_local = []
        for idx_col in range(n_spectra): 
            row = [
                doy_day_frac[idx_col],
                utc_datime_str[idx_col],
                idx_col+1,
                sif_fr_max[idx_col],
                sif_fr_wl[idx_col],
                sif_r_max[idx_col],
                sif_r_wl[idx_col],
                sif_o2b[idx_col],
                sif_o2a[idx_col],
                sif_int[idx_col],
                sif_o2b_un[idx_col],
                sif_o2a_un[idx_col]
            ]
            out_arr_local.append(row)

        # Store in a list:
        allm_specfit.extend(out_arr_local)

        # Also accumulate the SIF/reflectance spectra
        if allf_specfit is None:
            allf_specfit = sif
            allf_unc_specfit = fluo_unc
            allr_specfit = ref
            allar_unc = app_ref_unc
            all_utc_datetime_str = utc_datime_str
        else:
            # Concatenate horizontally
            allf_specfit = np.concatenate([allf_specfit, sif], axis=1)
            allf_unc_specfit = np.concatenate(
                [allf_unc_specfit, fluo_unc], axis=1)
            allr_specfit = np.concatenate([allr_specfit, ref], axis=1)
            allar_unc = np.concatenate([allar_unc, app_ref_unc], axis=1)
            all_utc_datetime_str.extend(utc_datime_str)

        logf.write(f"Finished file {i_pair+1} of {n_tables} . Processed {n_spectra} spectra.\nFailed {len(failed_spectra)}, indices: {[x + 1 for x in failed_spectra]} .\n")
        logf.flush()

    # write output files
    out_header = [
        "DOYdayfrac",
        "UTC_datetime",
        "filenum",
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2B",
        "SIF_O2A",
        "SIF_int",
        "SIF_O2B_un",
        "SIF_O2A_un"]

    final_sif_params_name = os.path.join(output_path,f"{proc_time_all}_pySPECFIT-OE_SIF_metrics.txt")
    write_csv_with_headers(final_sif_params_name, allm_specfit, out_header)

    col_headers = ["wvl"] + all_utc_datetime_str
    arr_f_specfit = np.column_stack([wvl_out, allf_specfit])
    final_sif_name = os.path.join(output_path, f"{proc_time_all}_pySPECFIT-OE_SIF_spectrum.txt")
    write_csv_with_headers(final_sif_name, arr_f_specfit, col_headers)

    arr_f_unc_specfit = np.column_stack([wvl_out, allf_unc_specfit])
    final_sif_unc_name = os.path.join(output_path, f"{proc_time_all}_pySPECFIT-OE_SIF_spectrum_uncertainty.txt")
    write_csv_with_headers(final_sif_unc_name, arr_f_unc_specfit, col_headers)

    arr_r_specfit = np.column_stack([wvl_out, allr_specfit])
    final_r_name = os.path.join(output_path, f"{proc_time_all}_pySPECFIT-OE_REFLECTANCE_spectrum.txt")
    write_csv_with_headers(final_r_name, arr_r_specfit, col_headers)

    arr_ar_unc = np.column_stack([wvl_out, allar_unc])
    final_r_name = os.path.join(output_path, f"{proc_time_all}_pySPECFIT-OE_APPARENT_REFLECTANCE_UNCERTAINTY.txt")
    write_csv_with_headers(final_r_name, arr_ar_unc, col_headers)
    

    write_results_to_netcdf(
        output_path=output_path,
        proc_time=proc_time_all,
        wavelengths=wvl_out,
        timestamps=all_utc_datetime_str,
        sif_spectrum=allf_specfit,
        sif_uncertainty=allf_unc_specfit,
        reflectance_spectrum=allr_specfit,
        apparent_reflectance_uncertainty=allar_unc,
        sif_metrics=allm_specfit
    )

    end_time = time.time()
    elapsed = end_time - start_time

    total_elapsed_seconds = int(elapsed)
    days, remainder_seconds = divmod(total_elapsed_seconds, 86400)  # 86400 seconds per day
    hours, remainder_seconds = divmod(remainder_seconds, 3600)
    minutes, seconds = divmod(remainder_seconds, 60)
    elapsed_str = f"{days}d {hours}h {minutes}m {seconds}s"

    # Scrivi nel log
    logf.write("\nFLOX_processing_master completed.\n")
    logf.write(f"Elapsed time: {elapsed_str}\n")
    logf.close()






