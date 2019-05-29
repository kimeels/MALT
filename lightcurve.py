from malt_logger import*
from interpolator import*

class Lightcurve:
    def __init__(self, filename, interpolate = True, interp_func = get_gp):
        try:
            t,flux,flux_err = self.loadfile(filename)

            self.t = t
            self.flux = flux
            self.flux_err = flux_err
            self.filename = filename

        except IOError as io:
            print('An error occured trying to read the file.')
            lc_logger.exception("An error occured trying to read the file.")
        except Exception as e:
            print("An error has occured")
            lc_logger.exception("Exception occurred")

        if interpolate == True:
            try:
                interp = interp_func(self)
                self.interp = interp
            except Exception as e:
                print("An error has occured while performing interpolation")
                lc_logger.exception("Exception occurred while performing interpolation")
        else:
            self.interp = np.nan





    def loadfile(self, filename):
        """
            Loads the 2 different types of datasets

            Param
            ------

            filename: path to dataset

        """

        if "GBI" in filename:
            data = np.loadtxt(filename)
            x,y,err = data[:,0],data[:,3],data[:,6]
            y_eq_minus1 = np.where(y == -1)
            y = np.delete(y,y_eq_minus1)
            x = np.delete(x,y_eq_minus1)
            err =  np.delete(err,y_eq_minus1)

        else:
            data = np.loadtxt(filename)
            x,y,err = data[:,0],data[:,1],data[:,2]

        return x,y,err

    def interpolate(self,interp_func = get_gp):
        """
            Interpolates the given lightcurve with assigned interpolation function

            Param
            ------
            self : Lightcurve object
            interp_func: python function
                A python function that takes in a lightcurve and interpolates it.

        """
        try:
            interp = interp_func(self)
            self.interp = interp
        except Exception as e:
            print("An error has occured while performing interpolation")
            lc_logger.exception("Exception occurred while performing interpolation")
