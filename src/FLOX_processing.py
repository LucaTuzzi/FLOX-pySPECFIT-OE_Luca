import numpy as np
from src.sif_retrieval_IPF_v2_1_modified import _l2b_regularized_cost_function_optimization
from tqdm import tqdm

def FLOX_processing(
            wvl,             # Wavelength subset for processing
            app_ref_array,
            app_ref_unc_array,
            Lin_array,        # Incident radiance (Lin) only in arho_sim = rho + fluorescence / Lin
            sa,         # covariance matrix
            xa_mean,    # mean state vector
            SIF_unc_MC,                
            parallel = False,
            n_jobs=-1
    ):

    if parallel:
        from joblib import Parallel, delayed, parallel_config
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
            ...
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

        with parallel_config(backend="loky", inner_max_num_threads=1):
            with tqdm_joblib(tqdm(total=n_spectra, desc="Processing spectra", unit="spectra")):
                results = Parallel(n_jobs=n_jobs,)(
                    delayed(process_spectrum)(
                        num_spec,
                        wvl,
                        Lin_array,
                        app_ref_array,
                        app_ref_unc_array,
                        xa_mean,
                        sa,
                        l2b_wavelength_grid,
                        SIF_unc_MC,
                        n_spectra,
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
                    SIF_unc_MC,
                    n_spectra,
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
            sif_red_peak_uncertainty,
            sif_red_peak_wl_uncertainty,
            sif_o2b_uncertainty,
            sif_farred_peak_uncertainty,
            sif_farred_peak_wl_uncertainty,
            sif_o2a_uncertainty,
            sif_integrated_uncertainty,
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
        sif_red_peak_uncertainty,
        sif_red_peak_wl_uncertainty,
        sif_o2b_uncertainty,
        sif_farred_peak_uncertainty,
        sif_farred_peak_wl_uncertainty,
        sif_o2a_uncertainty,
        sif_integrated_uncertainty,
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
        list[np.ndarray]: sif_r_max,    sif_r_wl,    sif_o2b,     sif_fr_max,    sif_fr_wl,    sif_o2a,    sif_int,
            sif_r_max_un, sif_r_wl_un, sif_o2b_un,  sif_fr_max_un, sif_fr_wl_un, sif_o2a_un, sif_int_un
    """

    # Ensure inputs are numpy arrays
    wl = np.array(wl)
    sif = np.array(sif)
    sif_un = np.array(sif_un)

    # RED SIF
    # max
    red_indices = wl < 690
    sif_r = sif[red_indices, :]
    idx_r_max = np.argmax(sif_r, axis=0)
    
    sif_r_wl  = wl[red_indices][idx_r_max]
    sif_r_max = sif_r[idx_r_max, np.arange(sif_r.shape[1])]
    sif_r_max_un = sif_un[red_indices][idx_r_max,np.arange(sif_r.shape[1])]

    # SIF at O2-B 687nm
    iio2b = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[iio2b, :]
    sif_o2b_un = sif_un[iio2b, :]
    
    # FAR-RED SIF
    # max
    far_red_indices = wl > 720
    sif_fr = sif[far_red_indices, :]
    idx_fr_max = np.argmax(sif_fr, axis=0)

    sif_fr_wl = wl[far_red_indices][idx_fr_max]
    sif_fr_max = sif_fr[idx_fr_max, np.arange(sif_fr.shape[1])]
    sif_fr_max_un = sif_un[far_red_indices][idx_fr_max, np.arange(sif_fr.shape[1])]

    # SIF at O2-A 760nm
    iio2a = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[iio2a, :]
    sif_o2a_un = sif_un[iio2a, :]

    # Spectrally integrated SIF
    #compatibility check for numpy version.
    if hasattr(np, "trapezoid"):
        sif_int = np.trapezoid(sif, wl, axis=0)
    else:
        sif_int = np.trapz(sif, wl, axis=0)
    # unc computation
    w = np.zeros_like(wl)
    w[0] = (wl[1] - wl[0])
    w[-1] = (wl[-1] - wl[-2])
    w[1:-1] = (wl[2:] - wl[:-2])
    w = w[:, None]
    sif_int_un = np.sqrt(np.sum((w * sif_un)**2, axis=0)) /2 #!!! to be done
    
    #unc on wl set to NAN
    sif_r_wl_un = np.full(sif_int_un.shape, np.nan)
    sif_fr_wl_un = np.full(sif_int_un.shape, np.nan)
    
    return (sif_r_max,    sif_r_wl,    sif_o2b,     sif_fr_max,    sif_fr_wl,    sif_o2a,    sif_int,
            sif_r_max_un, sif_r_wl_un, sif_o2b_un,  sif_fr_max_un, sif_fr_wl_un, sif_o2a_un, sif_int_un)


def process_spectrum(num_spec,wvl,Lin_array,app_ref_array,app_ref_unc_array,xa_mean,sa,l2b_wavelength_grid,SIF_unc_MC,n_spectra):
    try:
        #parameters
        lmb = 1e-4 # inversion parameter
        eps = 1e-15
        # spectra estraction
        Lin = Lin_array[:, num_spec]
        app_ref = app_ref_array[:, num_spec]
        app_ref_unc = app_ref_unc_array[:, num_spec]

        # covariance
        app_ref_variance = app_ref_unc**2
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_app_ref_variance = np.where(app_ref_variance > eps,
                1.0 / app_ref_variance, 0.0)
            
        sy = np.diag(inv_app_ref_variance)

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
    


    