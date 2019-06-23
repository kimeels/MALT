from malt_logger import*
from interpolator import*
from feature_extraction import*

class Lightcurve:
    def __init__(self, filepath, interpolate = True,
                 interp_func = get_gp, ini_t = 'rand', timescale = 8/24,
                 sample_size = 100):
        """
            Initialises an instance of the Lightcurve class.

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.
            filepath: str
                The path to a data file.
            interpolate: boolean
                If True will use interp_func() to interpolate  the loaded data.
            interp_func: python function
                A python function that takes in a lightcurve and interpolates it.
            ini_t: str or float
                Initial time to start sampling.
            timescale: float
                The total length of the interpolated lightcurve.
            sample_size: int
                Number of data points in interpolated lightcurve.

        """
        try:
            t, flux, flux_err, ra_dec, clss = self.loadfile(filepath)

            self.time = t
            self.flux = flux
            self.flux_err = flux_err
            self.ra_dec = ra_dec
            self.type = clss

            filename = os.path.basename(filepath)
            self.filename = filename

        except IOError as io:
            print('An error occured trying to read the file.')
            lc_logger.exception("An error occured trying to read the file.")
        except Exception as e:
            print("An error has occured")
            lc_logger.exception("Exception occurred")

        if interpolate == True:
            try:
                self.interpolate(interp_func, ini_t, timescale, sample_size)
            except Exception as e:
                print("An error has occured while performing interpolation")
                lc_logger.exception("Exception occurred while performing interpolation")
        else:
            self.interp_flux = [np.nan]

        self.features = np.nan





    # def loadfile(self, filename):
    #     """
    #         Loads the 2 different types of datasets
    #
    #         Param
    #         ------
    #
    #         filename: path to dataset
    #
    #     """
    #
    #     if "GBI" in filename:
    #         data = np.loadtxt(filename)
    #         x,y,err = data[:,0],data[:,3],data[:,6]
    #         y_eq_minus1 = np.where(y == -1)
    #         y = np.delete(y,y_eq_minus1)
    #         x = np.delete(x,y_eq_minus1)
    #         err =  np.delete(err,y_eq_minus1)
    #
    #     else:
    #         data = np.loadtxt(filename)
    #         x,y,err = data[:,0],data[:,1],data[:,2]
    #
    #     return x,y,err


    def loadfile(self, filename):
        """
            Loads file to extract time, flux, flux_err  ra_dec and class

            Param
            ------

            filename: path to dataset

        """
        data = []
        with open(filename) as my_file:
            for line in my_file:
                data.append(line)

        x = np.array(data[0][:-2].split(),dtype = float)
        y = np.array(data[1][:-2].split(),dtype = float)
        err = np.array(data[2][:-2].split(),dtype = float)

        ra_dec = int(data[3][:-2])
        typ = data[4][:-2]

        return x, y, err, ra_dec, typ

    def interpolate(self,interp_func = get_gp, ini_t = 'rand', timescale = 8/24,
                    sample_size = 100):
        """
            Interpolates the given lightcurve with assigned interpolation function

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.
            interp_func: python function
                A python function that takes in a lightcurve and interpolates it.
            ini_t: str or float
                Initial time to start sampling.
            timescale: float
                The total length of the interpolated lightcurve.
            sample_size: int
                Number of data points in interpolated lightcurve.

        """
        if np.isnan(self.interp_flux[0]):
            if ini_t == 'rand':
                t0 = np.random.uniform(np.min(self.time),np.max(self.time)-timescale)
            else:
                t0 = ini_t

            try:
                interp = interp_func(self, t0, timescale, sample_size)
                self.interp_flux = interp
            except Exception as e:
                print("An error has occured while performing interpolation")
                lc_logger.exception("Exception occurred while performing interpolation")
        else:
            lc_logger.info(self.filename + " is already interpolated")

    def feat_extract(self, feat_ex_method= get_wavelet_feature ):
        """
            Extracts features from the given lightcurve with assigned
            feature extraction method.

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.
            feat_ex_method: python function
                Function to use for the feature extraction.

        """

        try:
            if np.isnan(self.interp_flux[0]):
                print("No interpolated flux. Please run interpolate() first.")
            else:
                feats = feat_ex_method(self)
                self.features = feats
        except Exception as e:
            print("An error has occured while performing feature extraction")
            lc_logger.exception("Exception occurred while performing feature extraction")
