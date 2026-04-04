import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
import h5py


def load_inverse_covariance(cov):
    """
    Load and adjust the inverse parameter covariance matrix from netcdf file.

    Parameters
    ----------
    cov : str
        Path to the HDF5 file containing 'invCOV_SIF_RHO' and 'xa_mean'.

    Returns
    -------
    sa : np.ndarray
        Adjusted inverse covariance matrix.
    xa_mean : np.ndarray
        Mean state vector.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required datasets are missing.
    ValueError
        If the covariance matrix size is smaller than expected.
    """

    # Check file existence
    if not os.path.isfile(cov):
        raise FileNotFoundError(f"Cannot find covariance file: {cov}")

    # Read data from HDF5 file
    with h5py.File(cov, 'r') as f:
        if 'invCOV_SIF_RHO' not in f:
            raise KeyError("invCOV_SIF_RHO not found in the provided file.")
        if 'xa_mean' not in f:
            raise KeyError("xa_mean not found in the provided file.")

        sa = np.array(f['invCOV_SIF_RHO']).astype(float)
        xa_mean = np.array(f['xa_mean']).astype(float).ravel()

    # Validate matrix size
    nparams = sa.shape[0]
    if nparams < 9:
        raise ValueError(f"invCOV_SIF_RHO matrix expected >= 26x26. Found {nparams}x{nparams}.")

    return sa, xa_mean



def write_csv_with_headers(filename, data2d, headers):
    """
    Utility method to write the CSV files.

    Args:
        filename (str): path to output CSV
        data2d (np.ndarray): 2D NumPy array to write
        headers (list): list of column headers (strings), length matches data2d.shape[1]
    """
    out_table = pd.DataFrame(data2d, columns=headers)
    out_table.to_csv(filename, sep=";", index=False,
                     na_rep="NaN", float_format="%.6f")
    



def write_results_to_netcdf(
    output_path,
    proc_time,
    wavelengths,
    timestamps,
    sif_spectrum,
    sif_uncertainty,
    reflectance_spectrum,
    apparent_reflectance_uncertainty,
    sif_metrics
):
    """
    Write FLOX processing results to a NetCDF file.

    Parameters
    ----------
    output_path : str
        Directory where the NetCDF file will be saved.
    proc_time : str
        Processing timestamp for naming and metadata.
    wavelengths : np.ndarray
        Wavelength grid (1D array).
    timestamps : list[str]
        List of UTC timestamps for each spectrum.
    sif_spectrum : np.ndarray
        SIF spectrum (2D array: wavelength x timestamp).
    sif_uncertainty : np.ndarray
        SIF uncertainty spectrum (same shape as sif_spectrum).
    reflectance_spectrum : np.ndarray
        Reflectance spectrum (same shape as sif_spectrum).
    reflectance_uncertainty : np.ndarray
        Apparent reflectance uncertainty (same shape as sif_spectrum).
    sif_metrics : dict
        Dictionary of SIF metrics:
        {
            "SIF_FARRED_max": float,
            "SIF_FARRED_max_wvl": float,
            "SIF_RED_max": float,
            "SIF_RED_max_wvl": float,
            "SIF_O2B": float,
            "SIF_O2A": float,
            "SIF_int": float,
            "SIF_O2B_un": float,
            "SIF_O2A_un": float
        }
    """
    netcdf_file = os.path.join(output_path, f"{proc_time}_pySPECFIT-OE_results.nc")

    metric_names = [
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2B",
        "SIF_O2A",
        "SIF_int",
        "SIF_O2B_un",
        "SIF_O2A_un"
    ]

    with Dataset(netcdf_file, "w", format="NETCDF4") as nc:
        # === Dimensions ===
        nc.createDimension("wavelength", len(wavelengths))
        nc.createDimension("timestamp", len(timestamps))

        # === Variables ===
        wvl_var = nc.createVariable("wavelength", "f4", ("wavelength",))
        wvl_var.units = "nm"
        wvl_var[:] = wavelengths

        time_var = nc.createVariable("timestamp", "S1", ("timestamp",))
        time_var[:] = np.array(timestamps, dtype="S")

        sif_var = nc.createVariable("SIF_spectrum", "f4", ("wavelength", "timestamp"), zlib=True)
        sif_var.units = "mW m-2 sr-1 nm-1"
        sif_var[:, :] = sif_spectrum

        sif_unc_var = nc.createVariable("SIF_uncertainty", "f4", ("wavelength", "timestamp"), zlib=True)
        sif_unc_var.units = "mW m-2 sr-1 nm-1"
        sif_unc_var[:, :] = sif_uncertainty

        refl_var = nc.createVariable("reflectance_spectrum", "f4", ("wavelength", "timestamp"), zlib=True)
        refl_var.units = "[-]"
        refl_var[:, :] = reflectance_spectrum

        refl_unc_var = nc.createVariable("apparent_reflectance_uncertainty", "f4", ("wavelength", "timestamp"), zlib=True)
        refl_unc_var.units = "[-]"
        refl_unc_var[:, :] = apparent_reflectance_uncertainty

        # === SIF metrics ===
        if isinstance(sif_metrics, list):
            metrics_array = np.array(sif_metrics)  # shape (n_timestamps, n_metrics)
            for i, name in enumerate(metric_names):
                var = nc.createVariable(name, "f4", ("timestamp",), zlib=True)


