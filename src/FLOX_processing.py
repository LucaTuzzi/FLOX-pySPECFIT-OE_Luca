import numpy as np
import os
from src.sif_retrieval_IPF_v2_1_modified import _l2b_regularized_cost_function_optimization
from tqdm import tqdm

def FLOX_processing(
            wvl,             # Wavelength subset for processing
            app_ref_array,
            app_ref_unc_array,
            Lin_array,        # Incident radiance (Lin) only in arho_sim = rho + fluorescence / Lin
            sa,         # covariance matrix
            xa_mean,
            SIF_unc_MC,                # mean state vector
            parallel = False
    ):

    if parallel:
        from joblib import Parallel, delayed
        from tqdm_joblib import tqdm_joblib
        

    """Method to compute apparent reflectance and call SIF retrieval function for FLOX data.
    Args:
        wvl (np.ndarray): Wavelength array for the spectra.
        app_ref_array (np.ndarray): 2D array of apparent reflectance spectra (n_wavelengths x n_spectra).
        app_ref_unc_array (np.ndarray): 2D array of apparent reflectance uncertainties (n_wavelengths x n_spectra).
        Lin (np.ndarray): 2D array of incident radiance spectra (n_wavelengths x n_spectra).
        cov (str): Path to the covariance file for retrieval.

    Returns:
        list[np.ndarray]: 
            sif_array               # SIF spectrum computed using SFM method
            ref_array               # Reflectance spectrum computed using SFM method
            sif_array_u             # SIF spectrum uncertainty
            ref_array_u             # Reflectance spectrum uncertainty
            wvl_out                 # Output wavelength array
            sif_red_peak            # Maximum SIF at red peak
            sif_red_peak_wl         # Wavelength of maximum SIF at red peak
            sif_o2b_band            # SIF at O₂-B absorption band
            sif_farred_peak         # Maximum SIF at far-red peak
            sif_farred_peak_wl      # Wavelength of maximum SIF at far-red peak
            sif_o2a_band            # SIF at O₂-A absorption band
            sif_integrated          # Spectrally integrated SIF
            sif_o2a_uncertainty     # Uncertainty of SIF at O₂-A band
            sif_o2b_uncertainty     # Uncertainty of SIF at O₂-B band
            app_ref_unc              # Uncertainty of apparent reflectance)
    """

    # Initialize output arrays
    n_wvl, n_spectra = app_ref_array.shape

    #in the current implementation we are using the same wavelength grid for input and output, but
    # we keep the possibility to have a different output grid for the future, if needed.
    l2b_wavelength_grid = np.copy(wvl)  # Wavelength grid for L2B processing
    wvl_out = np.copy(wvl)              # Output wavelength grid

    # Initialize output arrays
    sif_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    sif_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)

    #To keep track of spectra that fail during retrieval
    failed_spectra = [] 
   
    # SWITCH PARALLEL / SERIAL
    if parallel:
        # avoid oversubscription
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        with tqdm_joblib(tqdm(total=n_spectra, desc="Processing spectra", unit="spectra")):
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_spectrum)(
                    num_spec,
                    wvl,
                    Lin_array,
                    app_ref_array,
                    app_ref_unc_array,
                    xa_mean,
                    sa,
                    l2b_wavelength_grid,
                    SIF_unc_MC
                )
                for num_spec in range(n_spectra)
            )

        # results pick up and error handling    
        for num_spec, reflectance, sif, sif_unc, err in results:
            if err is not None:
                print(f"\n⚠️ Error on spectrum {num_spec}: {err}")
                failed_spectra.append(num_spec)
                continue

            ref_array[:, num_spec]   = reflectance
            sif_array[:, num_spec]   = sif
            sif_array_u[:, num_spec] = sif_unc

    else:
        for num_spec in tqdm(range(n_spectra), desc="Processing spectra", unit="spectra"):
            try:
                num_spec, reflectance, sif, sif_unc, err = process_spectrum(
                    num_spec,
                    wvl,
                    Lin_array,
                    app_ref_array,
                    app_ref_unc_array,
                    xa_mean,
                    sa,
                    l2b_wavelength_grid,
                    SIF_unc_MC
                )
                if err is not None:
                    raise err

                ref_array[:, num_spec]   = reflectance
                sif_array[:, num_spec]   = sif
                sif_array_u[:, num_spec] = sif_unc

            except Exception as e:
                print(f"\n⚠️ Error on spectrum {num_spec}: {e}")
                failed_spectra.append(num_spec)
                continue

    # errors report
    if failed_spectra:
        print("\nFailed spectra:", failed_spectra)

    # --- Compute SIF (Solar-Induced Fluorescence) Parameters ---
    (
            sif_red_peak,
            sif_red_peak_wl,
            sif_o2b_band,
            sif_farred_peak,
            sif_farred_peak_wl,
            sif_o2a_band,
            sif_integrated,
            sif_o2b_uncertainty,
            sif_o2a_uncertainty
    ) = sif_parms_flox(l2b_wavelength_grid, sif_array, sif_array_u)


    # --- Return all processed outputs ---
    return(
        sif_array,               # SIF spectrum computed using SFM method
        ref_array,               # Reflectance spectrum computed using SFM method
        sif_array_u,             # SIF spectrum uncertainty
        ref_array_u,             # Reflectance spectrum uncertainty #!!! it is emoty right now
        wvl_out,                 # Output wavelength array
        sif_red_peak,            # Maximum SIF at red peak
        sif_red_peak_wl,         # Wavelength of maximum SIF at red peak
        sif_o2b_band,            # SIF at O₂-B absorption band
        sif_farred_peak,         # Maximum SIF at far-red peak
        sif_farred_peak_wl,      # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,            # SIF at O₂-A absorption band
        sif_integrated,          # Spectrally integrated SIF
        sif_o2a_uncertainty,     # Uncertainty of SIF at O₂-A band
        sif_o2b_uncertainty,     # Uncertainty of SIF at O₂-B band
        app_ref_unc_array,       # Uncertainty of apparent reflectance
        failed_spectra       # List of spectra indices that failed during retrieval
    )

def sif_parms_flox(wl, sif, sif_un):
    """
    Extracts key SIF metrics (red peak, far-red peak,
    O2-B, O2-A, integrated SIF, etc.) from the retrieved fluorescence.
    Also includes uncertainties from sif_array_u

    Args:
        wl (np.ndarray): wavelengths
        sif (np.ndarray): solar induced fluorescence
        sif_un (np.ndarray): solar induced fluorescence uncertainties

    Returns:
        list[np.ndarray]: sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un
    """

    # Ensure inputs are numpy arrays
    wl = np.array(wl)
    sif = np.array(sif)
    sif_un = np.array(sif_un)

    # RED SIF
    # max
    red_indices = wl < 690
    sif_r = sif[red_indices, :]
    sif_r_max = np.max(sif_r, axis=0)
    id = np.argmax(sif_r, axis=0)
    sif_r_wl = wl[red_indices][id]

    # SIF at O2-B 687nm
    ii = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[ii, :]
    sif_o2b_un = sif_un[ii, :]

    # FAR-RED SIF
    far_red_indices = wl > 720
    sif_fr = sif[far_red_indices, :]
    sif_fr_max = np.max(sif_fr, axis=0)
    p = np.argmax(sif_fr, axis=0)
    x = wl[far_red_indices]
    sif_fr_wl = x[p]

    # SIF at O2-A 760nm
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii, :]
    sif_o2a_un = sif_un[ii, :]

    # Spectrally integrated SIF
    #compatibility check for numpy version.
    if hasattr(np, "trapezoid"):
        sif_int = np.trapezoid(sif, wl, axis=0)
    else:
        sif_int = np.trapz(sif, wl, axis=0)

    return (sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un)


def process_spectrum(num_spec,wvl,Lin_array,app_ref_array,app_ref_unc_array,xa_mean,sa,l2b_wavelength_grid,SIF_unc_MC):
    try:
        # --- estrazione dati ---
        Lin = Lin_array[:, num_spec]
        app_ref = app_ref_array[:, num_spec]
        app_ref_unc = app_ref_unc_array[:, num_spec]

        # --- covarianza ---
        app_ref_variance = app_ref_unc**2
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = 1e-15
            inv_app_ref_variance = np.where(app_ref_variance > eps,
                                            1.0 / app_ref_variance,
                                            0.0)
        sy = np.diag(inv_app_ref_variance)

        # --- inversione ---
        lmb = 1e-4

        reflectance, sif, sif_unc = _l2b_regularized_cost_function_optimization(
            wvl,
            app_ref,
            xa_mean,
            Lin,
            sa,
            sy,
            lmb,
            l2b_wavelength_grid,
            SIF_unc_MC,
            max_iter=100,
        )

        return num_spec, reflectance, sif, sif_unc, None

    except Exception as e:
        return num_spec, None, None, None, e