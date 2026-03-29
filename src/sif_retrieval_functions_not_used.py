
def _get_red_sif(wl, sif):
    """
    Extract red fluorescence peak at 684 nm.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_R_max, SIF_R_wl) - Red peak intensity and wavelength
    """
    index = np.argmin(np.abs(wl - 684))  # TODO add consts
    sif_r_max = sif[index]
    if np.isnan(sif_r_max):
        sif_r_wl = np.nan
    else:
        sif_r_wl = wl[
            14
        ]  # TODO MAGIC shouldn't be here but local max too hard to find because second peak is hiding it.
    return sif_r_max, sif_r_wl


def _get_far_red_sif(wl, sif):
    """
    Extract far-red fluorescence peak (>720 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_FR_max, SIF_FR_wl) - Far-red peak intensity and wavelength
    """
    mask_far_red = wl > 720  # TODO add const
    max_far_red_index = np.argmax(sif[mask_far_red])
    sif_fr_max = sif[mask_far_red][max_far_red_index]
    if np.isnan(sif_fr_max):
        sif_fr_wl = np.nan
    else:
        sif_fr_wl = wl[mask_far_red][max_far_red_index]
    return sif_fr_max, sif_fr_wl


def _get_o2a_sif(wl, sif):
    """
    Extract SIF value at O2-A absorption line (760 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 760 nm
    """
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii]
    return sif_o2a


def _get_o2b_sif(wl, sif):
    """
    Extract SIF value at O2-B absorption line (687 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 687 nm
    """
    index = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[index]
    return sif_o2b


def _get_spectrally_integrated_sif(wl, sif):
    """
    Calculate total SIF by spectral integration.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        Integrated SIF value [W m⁻² sr⁻¹]
    """
    sifint = np.trapezoid(sif, wl)
    return sifint


def _reflectance_concatenation(
    floris_app_refl_map, floris_wv_merged, rhomin_wl, wvl_1nm, min_wl
):
    """
    Concatenate interpolated reflectance with minimum wavelength data.

    Parameters
    ----------
    floris_app_refl_map : numpy.ndarray
        FLORIS apparent reflectance values
    FLORIS_wv_merged : numpy.ndarray
        FLORIS wavelength grid [nm]
    RHOmin_wl : numpy.ndarray
        Reflectance data for minimum wavelength range
    wvl_1nm : numpy.ndarray
        Target 1nm wavelength grid [nm]
    min_wl : float
        Minimum wavelength threshold [nm]

    Returns
    -------
    numpy.ndarray
        Concatenated reflectance spectrum
    """
    # filter out zero and nan values from floris_wv_merged
    idx = (floris_wv_merged != 0) & (~np.isnan(floris_wv_merged))

    # interpolate floris_app_refl_map at points wvl_1nm using filtered floris_wv_merged
    tmp_rho = np.interp(wvl_1nm, floris_wv_merged[idx], floris_app_refl_map[idx])

    # concatenate values where wvl_1nm < min_wl with rhomin_wl
    y = np.concatenate((tmp_rho[wvl_1nm < min_wl], rhomin_wl))
    return y