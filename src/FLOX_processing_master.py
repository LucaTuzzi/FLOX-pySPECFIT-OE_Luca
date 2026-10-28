import re
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
import xarray as xr
from src.FLOX_mask import FLOX_mask

def FLOX_processing_master(data_path, uncertainty_path, data_nc, cov_path, nc_input=True, uncertainty_as_input=True, SIF_unc_MC=True, short_case=False,  mask_time_opt=None, parallel=False, n_jobs=-1,):

    """
    Orchestrates the FLOX processing pipeline for NetCDF-based data retrieval.
    data_path : str                 Directory containing the input NetCDF data files.
    uncertainty_path : str          Path to the file containing uncertainty values.
    data_nc : str                   Name of the NetCDF file to be processed.
    cov_path : str                  Path to the auxiliary file containing inverse covariance matrices.
    nc_input : bool                 If True, input data is in NetCDF format. If False, input data is in CSV format.
    uncertainty_as_input : bool     If True, uncertainty values are read directly from the input data. If False, uncertainty is computed from a predefined model.
    SIF_unc_MC : bool               If True, Monte Carlo sampling is used for SIF uncertainty estimation. If False, an analytical approximation is used.
    parallel : bool                 If True, enables parallel processing to speed up computations.
    short_case : bool               If True, processes only a subset of the dataset for testing purposes.
    mask_time_opt : dict or None    Dictionary containing optional temporal filtering settings:
        - mask_time_doy (bool):     enable/disable DOY/time filtering
        - date_range (tuple):       start and end dates
        - time_ranges(list tuples): time intervals
        - doy_range (tuple):       day-of-year range
    """

    if mask_time_opt is None:
        mask_time_doy, date_range, time_ranges, doy_range = False, None, None, None
    else:
        mask_time_doy = mask_time_opt.get("mask_time_doy")
        date_range = mask_time_opt.get("date_range")
        time_ranges = mask_time_opt.get("time_ranges")
        doy_range = mask_time_opt.get("doy_range")

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
    logf.write(f"data_nc = {data_nc}\n")
    logf.write(f"uncertainty_as_input = {uncertainty_as_input}\n")
    logf.write(f"nc_input = {nc_input}\n")
    logf.write(f"SIF_unc_MC = {SIF_unc_MC}\n")
    logf.write(f"parallel = {parallel}\n")
    logf.write(f"short_case = {short_case}\n")
    logf.write(f"mask_time_doy = {mask_time_doy}\n")
    logf.flush()

    sa, xa_mean = load_inverse_covariance(cov_path)
    # Adjust covariance matrix values
    sa[8:, :8]  = 1e-15
    sa[:8, 8:]  = 1e-15 
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

    results = {"metrics": [], "sif": None, "sif_unc": None,
        "ref": None, "ref_unc": None, "timestamps": [], "wavelengths": None, }


    if nc_input:

        if not uncertainty_as_input:
            msg = ("Warning: proceeding with input uncertainty even though uncertainty_as_input = False, as it is included in the NetCDF file:\n")
            logf.write(msg + "\n")
            logf.flush()


        nc_path = os.path.join(data_path, data_nc)
        logf.write(f"\nProcessing file {nc_path} \n")
        logf.flush()

        ds = xr.open_dataset(nc_path) #if error, check the file path and name, and ensure that the file exists and is accessible.

        # Wavelength Definition
        wvl_qepro = ds["wavelength"].values
        lb = np.argmin(np.abs(wvl_qepro - wvlRet[0]))
        ub = np.argmin(np.abs(wvl_qepro - wvlRet[1]))
        wvl_sub = wvl_qepro[lb:ub+1]
        
        #Spectra Definition
        AppReflectance_ds = ds["R"]
        
        # check AppReflectance_ds.dims (wavelength, UTC_time)
        if not(AppReflectance_ds.dims[0] == "wavelength" and AppReflectance_ds.dims[1] == "UTC_time"):
            msg = ("Error: invalid data format\n")
            logf.write(msg + "\n")
            logf.flush()
            raise ValueError(msg)

        AppReflectance = AppReflectance_ds.values[lb:ub+1, :]
        AppReflectance = np.nan_to_num(AppReflectance, nan=0.0, posinf=0.0, neginf=0.0)
        valid_cols = np.all(AppReflectance >= 0, axis=0)
        logf.write(f"Spectra not processable, negative somewhere (exist 1 wavelength st AppReflectance<0), python indices (first is 0):\n{np.where(~valid_cols)[0]}")
        logf.flush()
        AppReflectance = AppReflectance[:, valid_cols]
        
        u_R_systematic = ds["u_R_systematic"].values[lb:ub+1, valid_cols]
        u_R_systematic = np.nan_to_num(u_R_systematic, nan=0.0, posinf=0.0, neginf=0.0) 
        u_R_random = ds["u_R_random"].values[lb:ub+1, valid_cols]
        u_R_random = np.nan_to_num(u_R_random, nan=0.0, posinf=0.0, neginf=0.0) 
        unc_AppReflectance = (u_R_systematic*u_R_systematic + u_R_random*u_R_random)**0.5
        unc_AppReflectance = np.nan_to_num(unc_AppReflectance, nan=0.0, posinf=0.0, neginf=0.0) 
        logf.write(f"\nComputing unc_AppReflectance as (u_R_systematic^2 + u_R_random^2)^0.5\n")
        logf.flush()

        Lout = ds["L"].values[lb:ub+1, valid_cols] * 1e3
        Lout = np.nan_to_num(Lout, nan=0.0, posinf=0.0, neginf=0.0)
        Lin = np.zeros_like(Lout, dtype=float)
        np.divide(Lout, AppReflectance, out=Lin, where=AppReflectance != 0)

        
        utc_time = ds["UTC_time"].values[valid_cols]
        dt_time = pd.to_datetime(utc_time)
        utc_datime_str = dt_time.strftime("%d-%b-%Y %H:%M:%S").to_numpy()
        frac_day = ((dt_time - dt_time.normalize()) / pd.Timedelta(days=1)).to_numpy()
        doy_day_frac = dt_time.dayofyear.astype(float) + frac_day
        
        ds.close()

        if short_case:
            start_temp=0
            end_temp = min(100, AppReflectance.shape[1]) 
            AppReflectance = AppReflectance[:,start_temp:end_temp] #end_temp=100
            Lin= Lin[:,start_temp:end_temp]
            unc_AppReflectance  = unc_AppReflectance[:,start_temp:end_temp]
            utc_datime_str = utc_datime_str[start_temp:end_temp]
            doy_day_frac = doy_day_frac[start_temp:end_temp]
            logf.write(f"\nShort case applied: columns {start_temp}:{end_temp-1} "f"(kept {end_temp - start_temp} elements)\n")
            logf.flush()


        if mask_time_doy:

            mask_temp = FLOX_mask(doy_day_frac, utc_datime_str, doy_range=doy_range, date_range = date_range, time_ranges=time_ranges)
            mask_cols = np.where(mask_temp)[0]

            AppReflectance = AppReflectance[:, mask_cols]
            Lin = Lin[:, mask_cols]
            unc_AppReflectance = unc_AppReflectance[:, mask_cols]
            utc_datime_str = utc_datime_str[mask_cols]
            doy_day_frac = doy_day_frac[mask_cols]

            logf.write(f"\nTime/DOY mask applied:\n"
                f"  date_range = {date_range}\n"
                f"  time_ranges = {time_ranges}\n"
                f"  remaining elements = {len(mask_cols)}\n")
            logf.flush()

        utc_datime_str = utc_datime_str.tolist()
        doy_day_frac = doy_day_frac.tolist()

        n_spectra = AppReflectance.shape[1]
        batch_size = 1000
        start_indices=range(0, n_spectra, batch_size)
        n_tables = len(start_indices)

        for i_pair, start in enumerate(start_indices):
            end = min(start + batch_size, n_spectra)
            logf.write(f"Processing NetCDF spectra {start}:{end}\n")
            logf.flush()
            
            filenum_offset=start
            
            results, failed_count, failed_indices = process_flox_block(
                AppReflectance=AppReflectance[:, start:end],
                unc_AppReflectance=unc_AppReflectance[:, start:end],
                Lin=Lin[:, start:end],
                wvl_sub=wvl_sub,
                utc_datime_str=utc_datime_str[start:end],
                doy_day_frac=doy_day_frac[start:end],
                sa=sa,
                xa_mean=xa_mean,
                SIF_unc_MC=SIF_unc_MC,
                parallel=parallel,
                n_jobs=n_jobs,
                results=results,
                filenum_offset=filenum_offset,
            )

            logf.write(f"Finished Batch {i_pair} of {n_tables}. Processed {end-start} spectra.\n"
                       f"Failed {failed_count}, indices (of the new list with only the processable spectra): \n"
                       f"{[x + 1 for x in failed_indices]}.\n")
            logf.flush()


    else:
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
            end_temp = 1 #csv file
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
            #AppReflectance_table = data_AppReflectance.iloc[:, 1:].to_numpy()
            AppReflectance_table = (data_AppReflectance.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))

            # Read the Incoming CSV
            fname_Lin = Lin_list[i_pair]
            logf.write(f"Incoming: {fname_Lin}\n")
            with open(fname_Lin, "r", encoding="utf-8") as f:
                sample = "".join([f.readline() for _ in range(2)])
                delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
            data_Lin = pd.read_csv(fname_Lin, sep=delimiter, header=0)
            #Lin_table = data_Lin.iloc[:, 1:].to_numpy()
            Lin_table = (data_Lin.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))

            # Read the AppReflectance Uncertainty CSV
            if uncertainty_as_input:
                fname_unc_AppReflectance = unc_AppReflectance_list[i_pair]
                logf.write(f"Uncertainty: {fname_unc_AppReflectance}\n")
                with open(fname_unc_AppReflectance, "r", encoding="utf-8") as f:
                    sample = "".join([f.readline() for _ in range(2)])
                    delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
                data_unc_AppReflectance = pd.read_csv(fname_unc_AppReflectance, sep=delimiter, header=0)
                #unc_AppReflectance_table = data_unc_AppReflectance.iloc[:, 1:].to_numpy()
                unc_AppReflectance_table = (data_unc_AppReflectance.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
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
            logf.write(f"Spectra not processable, negative somewhere (exist 1 wavelength st AppReflectance<0), python indices (first is 0):\n{np.where(~valid_cols)[0]}")
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
            utc_time = [header.strip().strip('"') for header in utc_column]
            folder_date = os.path.basename(os.path.dirname(fname_AppReflectance))

            doy_day_frac = []
            utc_datime_str = []

            if not utc_time:
                raise ValueError(f"No valid timestamps found in file: {fname_AppReflectance}")

            first_time = utc_time[0]
            
            if re.fullmatch(r"\d{6}_\d{6}", first_time):
                time_format = "%d%m%y_%H%M%S"# ddmmyy_HHMMSS, es. 150925_194229
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", first_time):
                time_format = "%Y-%m-%d_%H_%M_%S"  # yyyy-mm-dd_HH_MM_SS, es. 2026-03-15_04_00_59
            elif re.fullmatch(r"\d{2}_\d{2}_\d{2}", first_time):
                # HH_MM_SS, es. 08_01_12
                # The date is extracted from the parent folder (yymmdd, e.g. 220715)
                if not re.fullmatch(r"\d{6}", folder_date):  #folder name is: yymmdd es. 150925
                    raise ValueError(f"Folder name not valid for extracting the date in format: {folder_date!r} (file: {fname_AppReflectance})")
                date_str = datetime.strptime(folder_date, "%y%m%d").strftime("%Y-%m-%d")
                utc_time = [f"{date_str}_{time_str}" for time_str in utc_time]
                time_format = "%Y-%m-%d_%H_%M_%S"
            else:
                raise ValueError(f"Timestamp format not recognized: {first_time!r} (file: {fname_AppReflectance})")
            
            for time_str in utc_time:
                dt_obj = datetime.strptime(time_str, time_format)
                day_of_year = (dt_obj - datetime(dt_obj.year, 1, 1)).days + 1
                frac_day = (dt_obj - datetime(dt_obj.year,dt_obj.month, dt_obj.day)).seconds / 86400
                doy_day_frac.append(day_of_year + frac_day)
                utc_datime_str.append(dt_obj.strftime("%d-%b-%Y %H:%M:%S"))

            n_spectra = AppReflectance.shape[1]
            filenum_offset=0
            results, failed_count, failed_indices = process_flox_block(
                AppReflectance=AppReflectance,
                unc_AppReflectance=unc_AppReflectance,
                Lin=Lin,
                wvl_sub=wvl_sub,
                utc_datime_str=utc_datime_str,
                doy_day_frac=doy_day_frac,
                sa=sa,
                xa_mean=xa_mean,
                SIF_unc_MC=SIF_unc_MC,
                parallel=parallel,
                n_jobs=n_jobs,
                results=results,
                filenum_offset=filenum_offset,
            )

            logf.write(f"Finished file {i_pair+1} of {n_tables}. Processed {n_spectra} spectra.\n"
                       f"Failed {failed_count}, indices (of the new list with only the processable spectra):\n"
                       f"{[x + 1 for x in failed_indices]}.\n")
            logf.flush()

    # === SAVE FINAL RESULTS ===
    if results["sif"] is None: raise ValueError("No spectra available for processing after filtering.")

    allm_specfit = results["metrics"]
    allf_specfit = results["sif"]
    allf_unc_specfit = results["sif_unc"]
    allr_specfit = results["ref"]
    allar_unc = results["ref_unc"]
    all_utc_datetime_str = results["timestamps"]
    wvl_out = results["wavelengths"]

    # SIF metrics header for CSV output
    out_header = [
        "DOYdayfrac",
        "UTC_datetime",
        "filenum",
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2A",
        "SIF_O2B",
        "SIF_int",
        "SIF_FARRED_max_un",
        "SIF_FARRED_max_wvl_un",
        "SIF_RED_max_un",
        "SIF_RED_max_wvl_un",
        "SIF_O2A_un",
        "SIF_O2B_un",
        "SIF_int_un"]

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
    final_ar_name = os.path.join(output_path, f"{proc_time_all}_pySPECFIT-OE_APPARENT_REFLECTANCE_UNCERTAINTY.txt")
    write_csv_with_headers(final_ar_name, arr_ar_unc, col_headers)
    
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




def process_flox_block(AppReflectance,unc_AppReflectance,Lin,wvl_sub,utc_datime_str,doy_day_frac,
    sa,xa_mean,SIF_unc_MC,parallel,n_jobs,results,filenum_offset=0,):
    """Process a batch and accumulate the results."""
    (
        sif, ref, fluo_unc, ref_unc, wvl_out,
        sif_r_max, sif_r_wl, sif_o2b,
        sif_fr_max, sif_fr_wl, sif_o2a, sif_int,
        sif_r_max_un, sif_r_wl_un, sif_o2b_un,
        sif_fr_max_un, sif_fr_wl_un,
        sif_o2a_un, sif_int_un,
        app_ref_unc, failed_spectra
    ) = FLOX_processing(
        wvl=wvl_sub,
        app_ref_array=AppReflectance,
        app_ref_unc_array=unc_AppReflectance,
        Lin_array=Lin,
        sa=sa,
        xa_mean=xa_mean,
        SIF_unc_MC=SIF_unc_MC,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    n_spectra = AppReflectance.shape[1]

    for idx_col in range(n_spectra):
        results["metrics"].append([
            doy_day_frac[idx_col],
            utc_datime_str[idx_col],
            filenum_offset + idx_col + 1,
            sif_fr_max[idx_col],
            sif_fr_wl[idx_col],
            sif_r_max[idx_col],
            sif_r_wl[idx_col],
            sif_o2a[idx_col],
            sif_o2b[idx_col],
            sif_int[idx_col],
            sif_fr_max_un[idx_col],
            sif_fr_wl_un[idx_col],
            sif_r_max_un[idx_col],
            sif_r_wl_un[idx_col],
            sif_o2a_un[idx_col],
            sif_o2b_un[idx_col],
            sif_int_un[idx_col],
        ])

    # First batch
    if results["sif"] is None:
        results["sif"] = sif
        results["sif_unc"] = fluo_unc
        results["ref"] = ref
        results["ref_unc"] = app_ref_unc
        results["timestamps"] = list(utc_datime_str)
        results["wavelengths"] = wvl_out

    else:
        results["sif"] = np.concatenate([results["sif"], sif], axis=1)
        results["sif_unc"] = np.concatenate([results["sif_unc"], fluo_unc], axis=1)
        results["ref"] = np.concatenate([results["ref"], ref], axis=1)
        results["ref_unc"] = np.concatenate([results["ref_unc"], app_ref_unc], axis=1)
        results["timestamps"].extend(utc_datime_str)

    return results, len(failed_spectra), failed_spectra
