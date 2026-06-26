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

def FLOX_processing_master_nc(data_path, data_nc, cov_path, uncertainty_as_input=True, SIF_unc_MC=True, parallel=False, short_case=False,mask_time_doy = False,):
    """
    Method that orchestrates FLOX data retrieval from nc files

    Args:
        data_path (str): Directory path where CSV data files are located
        data_nc (str): Name of the .nc file containing the main data
        cov_path (str): Path to the .mat file containing the inverse covariances
        uncertainty_as_input (bool): Whether to use input uncertainty values
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
    logf.write(f"uncertainty_path = {data_path}\n")
    logf.write(f"output_path = {output_path}\n")
    logf.write(f"cov_path         = {cov_path}\n")
    logf.write(f"uncertainty_as_input = {uncertainty_as_input}\n")
    logf.write(f"data_nc = {data_nc}\n")
    logf.write(f"SIF_unc_MC = {SIF_unc_MC}\n")
    logf.write(f"parallel = {parallel}\n")
    logf.write(f"short_case = {short_case}\n")
    logf.write(f"mask_time_doy = {mask_time_doy}\n")
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

    if not uncertainty_as_input:
        msg = ("Warning: proceeding with input uncertainty even though uncertainty_as_input = False, as it is included in the NetCDF file:\n")
        logf.write(msg + "\n")
        logf.flush()


    nc_path = os.path.join(data_path, data_nc)
    logf.write(f"\nProcessing file {nc_path} \n")
    logf.flush()

    ds = xr.open_dataset(nc_path)

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
    logf.write(f"Spectra not processed (AppReflectance<0), indices:\n{np.where(~valid_cols)[0]}")
    logf.flush()
    AppReflectance = AppReflectance[:, valid_cols]
    
    u_R_systematic = ds["u_R_systematic"].values[lb:ub+1, valid_cols]
    u_R_systematic = np.nan_to_num(u_R_systematic, nan=0.0, posinf=0.0, neginf=0.0) 
    u_R_random = ds["u_R_random"].values[lb:ub+1, valid_cols]
    u_R_random = np.nan_to_num(u_R_random, nan=0.0, posinf=0.0, neginf=0.0) 
    unc_AppReflectance = (u_R_systematic*u_R_systematic + u_R_random*u_R_random)**0.5
    unc_AppReflectance = np.nan_to_num(unc_AppReflectance, nan=0.0, posinf=0.0, neginf=0.0) 
    logf.write(f"\nComputing unc_AppReflectance as (u_R_systematic**2 + u_R_random**2)**0.5\n")
    logf.flush()

    Lout = ds["L"].values[lb:ub+1, valid_cols] * 1e3
    Lin = Lout/AppReflectance
    
    utc_time = ds["UTC_time"].values[valid_cols]
    dt_time = pd.to_datetime(utc_time)
    utc_datime_str = dt_time.strftime("%d-%b-%Y %H:%M:%S").to_numpy()
    frac_day = ((dt_time - dt_time.normalize()) / pd.Timedelta(days=1)).to_numpy()
    doy_day_frac = dt_time.dayofyear.astype(float) + frac_day
    
    ds.close()

    if short_case:
        start_temp=0
        end_temp=230
        AppReflectance = AppReflectance[:,start_temp:end_temp]
        Lin= Lin[:,start_temp:end_temp]
        unc_AppReflectance  = unc_AppReflectance[:,start_temp:end_temp]
        utc_datime_str = utc_datime_str[start_temp:end_temp]
        doy_day_frac = doy_day_frac[start_temp:end_temp]
        logf.write(f"\nShort case applied: columns {start_temp}:{end_temp} "
               f"(kept {end_temp - start_temp} elements)\n")
        logf.flush()



    if mask_time_doy:
        date_range = ("2026-04-24", "2026-04-26")
        time_ranges=[("10:00", "10:30")]
        doy_range = None
        mask_temp = FLOX_mask(doy_day_frac, utc_datime_str, doy_range=doy_range, date_range = date_range, time_ranges=time_ranges)
        mask_cols = np.where(mask_temp)[0]

        AppReflectance = AppReflectance[:, mask_cols]
        Lin = Lin[:, mask_cols]
        unc_AppReflectance = unc_AppReflectance[:, mask_cols]
        utc_datime_str = utc_datime_str[mask_cols].tolist()
        doy_day_frac = doy_day_frac[mask_cols].tolist()

        logf.write(f"\nTime/DOY mask applied:\n"
            f"  date_range = {date_range}\n"
            f"  time_ranges = {time_ranges}\n"
            f"  remaining elements = {len(mask_cols)}\n")
        logf.flush()

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
        allf_unc_specfit = np.concatenate([allf_unc_specfit, fluo_unc], axis=1)
        allr_specfit = np.concatenate([allr_specfit, ref], axis=1)
        allar_unc = np.concatenate([allar_unc, app_ref_unc], axis=1)
        all_utc_datetime_str.extend(utc_datime_str)

    logf.write(f"Finished file {1} of {1}. Processed {n_spectra} spectra.\nFailed {len(failed_spectra)}, indices: \n{[x + 1 for x in failed_spectra]}.\n")
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






