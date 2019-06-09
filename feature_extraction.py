###############################################################################
#                            Import Statements                                #
###############################################################################

import pywt
import numpy as np
from malt_logger import*


###############################################################################
#                       Feature Extraction Functions                          #
###############################################################################



def get_wavelet_feature(lightcurve):
    """
        Returns wavelet coefficients for a given lightcurve object.

        Param
        ------
        lightcurve : Lightcurve object
            An instance of the Lightcurve class

    """
    interp_flux = lightcurve.interp_flux

    mlev =  pywt.swt_max_level(len(interp_flux))
    coeffs=np.array(pywt.swt(interp_flux, 'sym2', level=mlev))

    npoints=len(coeffs[0, 0, :])
    c=coeffs.reshape(mlev*2, npoints).T
    wavout = c.flatten('F')
    return wavout
