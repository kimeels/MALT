###############################################################################
#                            Import Statements                                #
###############################################################################

import numpy as np
import george
from george import kernels
import scipy.optimize as optimize
from astropy.stats import median_absolute_deviation, sigma_clip
import os
import pickle
from .malt_logger import*
from matplotlib import pylab as pl

import seaborn as sns
color_palette = sns.color_palette("Set3",12)
sns.set_palette(color_palette)


###############################################################################
#                          Interpolator Functions                             #
###############################################################################

def get_gp(lightcurve, t0, obs_time, sample_size, aug_num):
    """
        Returns a Gaussian Process (george) object marginalised on the data
        in file.

        Param
        ------
        lightcurve : Lightcurve object
            An instance of the Lightcurve class.
        t0: float
            Initial time to start sampling.
        obs_time: float
            The total length of the interpolated lightcurve.
        sample_size: int
            Number of data points in interpolated lightcurve.


    """
    root_dir = "./saved/gps/"
    if os.path.isdir(root_dir) == False:
            os.makedirs(root_dir)
            lc_logger.info("Func get_gp() made directory "+root_dir)

    filename = lightcurve.filename

    from_save = False
    for file in os.listdir(root_dir):
        if file == filename+".gpsave":
            from_save = True
            break

    if from_save == True:
        gp = pickle.load(open(root_dir + filename+".gpsave", 'rb'))
        lc_logger.info("Func get_gp() used saved gp: "+root_dir + filename+".gpsave")

    else:
        lc_logger.info("Func get_gp() created new gp and saved it to: "
                       +root_dir + filename+".gpsave")

        def get_sigma_clipped_fluxes(raw_fluxes):
            """
            Uses the astropy sigma_clip function to try and get rid of outliers.

            Returns a masked array where all outlier values are masked.
            """

            # First we try to run sigma clip with the defaults - hopefully this will
            # iterate until it converges:
            clipped_fluxes = sigma_clip(raw_fluxes, maxiters=None,
                                        stdfunc=median_absolute_deviation)
            # If it fails (number of unmasked values <3),
            # then we just accept the result from a single iteration:
            if len(clipped_fluxes.compressed()) < 3:
                logger.warning("Sigma clipping did not converge, "
                               "using single iteration")
                clipped_fluxes = sigma_clip(raw_fluxes, maxiters=1,
                                            stdfunc=median_absolute_deviation)
            return clipped_fluxes

        def get_ls(x,y,err):
            """
                Returns a length scale of a peak in the dataset

                Params
                ------

                x   : time axis data
                y   : Flux axis data
                err : error on the flux measurements
            """
            clipped_fluxes = get_sigma_clipped_fluxes(y)
            background = np.ma.median(clipped_fluxes)
            noise = median_absolute_deviation(clipped_fluxes)

            rise_threshold = background + 5 * noise
            fall_threshold = background #+ 1 * noise
            flux_plus_err = y + err

            trigger = np.where(flux_plus_err > rise_threshold)[0]
            if len(trigger) == 0:
                trigger = np.where(flux_plus_err == np.max(flux_plus_err))[0][0]
            else:
                trigger = trigger[0]

            indexes = np.where(y < fall_threshold)[0]
            if len(indexes) == 0:
                indexes = np.where(y < background)[0]

            fall_indexes = np.array([indexes[i] for i in range(len(indexes)-1) if y[indexes[i]] > y[indexes[i+1]]])

            fall = np.where(fall_indexes > trigger)[0]
            if len(fall) == 0:
                fall_idx = len(y) - 1
            else:
                fall_idx = fall_indexes[np.where(fall_indexes > trigger)][0]

            rise = np.where(indexes <= trigger)[0]
            if len(rise) == 0:
                rise_idx = trigger
            else:
                rise_idx = indexes[np.where(indexes < trigger)][-1]
            return  ((x[fall_idx] - x[rise_idx]))/(2*np.sqrt(2*np.log(2)))

        def kernel1(data):
            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y)
            try:
                x,y,err = data
                ls = get_ls(x,y,err)

                k = kernels.ExpSquaredKernel(ls**2)

                gp = george.GP(k,fit_mean=True, white_noise=np.max(err)**2,
                               fit_white_noise=True)

                gp.compute(x,err)
                results = optimize.minimize(neg_ln_like, gp.get_parameter_vector(),
                                      jac=grad_neg_ln_like, method="L-BFGS-B",
                                      tol = 1e-5)

                # Update the kernel and print the final log-likelihood.
                gp.set_parameter_vector(results.x)
            except Exception as e:
                print("An error has occured")
                print(e)

            return gp,results

        def kernel2(data):
            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y)
            try:
                x,y,err = data
                ls = get_ls(x,y,err)
                k =  np.var(y)*kernels.ExpSquaredKernel(ls**2)
                k2 = kernels.ExpKernel(ls)

                kernel = k + k2

                gp = george.GP(kernel,fit_mean=True, white_noise=np.max(err)**2,
                               fit_white_noise=True)

                gp.compute(x,err)
                results = optimize.minimize(neg_ln_like, gp.get_parameter_vector(),
                                      jac=grad_neg_ln_like, method="L-BFGS-B",
                                      tol = 1e-5)
                # Update the kernel and print the final log-likelihood.
                gp.set_parameter_vector(results.x)
            except:
                gp,results = kernel1(data)

            return gp,results

        def kernel3(data):
            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y)
            try:
                x,y,err = data
                ls = get_ls(x,y,err)
                kernel =  np.var(y)*kernels.ExpSquaredKernel(ls**2)

                gp = george.GP(kernel,fit_mean=True, white_noise=np.max(err)**2,
                               fit_white_noise=True)

                gp.compute(x,err)
                results = optimize.minimize(neg_ln_like, gp.get_parameter_vector(),
                                      jac=grad_neg_ln_like, method="L-BFGS-B",
                                      tol = 1e-5)
                # Update the kernel and print the final log-likelihood.
                gp.set_parameter_vector(results.x)

            except:
                gp,results = kernel1(data)

            return gp,results

        def kernel4(data):
            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y)

            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y)
            try:
                x,y,err = data
                ls = get_ls(x,y,err)
                k =  np.var(y)*kernels.ExpSquaredKernel(ls**2)
                k2 = kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=ls,log_period=ls)

                kernel = k + k2

                gp = george.GP(kernel,fit_mean=True, white_noise=np.max(err)**2,
                               fit_white_noise=True)

                gp.compute(x,err)
                results = optimize.minimize(neg_ln_like, gp.get_parameter_vector(),
                                      jac=grad_neg_ln_like, method="L-BFGS-B",
                                      tol = 1e-5)
                # Update the kernel and print the final log-likelihood.
                gp.set_parameter_vector(results.x)

            except:
                gp,results = kernel1(data)

            return gp,results

        data = [lightcurve.time, lightcurve.flux, lightcurve.flux_err]

        gp1,results1  = kernel2(data)
        gp2,results2 = kernel3(data)
        gp3,results3 = kernel4(data)

        gpes = np.array([gp1,gp2,gp3])
        loss = [results1.fun,results2.fun,results3.fun]

        ind = np.where(loss == np.min(loss))

        gp = gpes[ind[0][0]]

        save_loc = root_dir + filename + ".gpsave"

        pickle.dump(gp, open(save_loc, 'wb'))


    xsample = np.linspace(t0,t0+obs_time,sample_size)
    ysample = gp.sample_conditional(lightcurve.flux,xsample,aug_num)

    if aug_num == 1:
        ysample = np.array([ysample])

    return ysample


# def plot_gp(lightcurve, gp_error = True, show = True, save = False):
#     """
#         Plots a lightcurve with overlaid GP
#
#         Params
#         -------
#
#         lightcurve: Lightcurve object
#             An instance of the Lightcurve class
#         gp_error: boolean
#             2 sigma error bounds.
#         show: boolean
#             Prints the plot to screen.
#         save: boolean
#             Saves plot to location save_loc.
#         save_loc: str
#             Name of plot and save location.
#
#
#     """
#     if lightcurve.interp_flux == np.nan:
#         print("No gp available. Please run gp_reg() first.")
#         lc_logger.warning("Attempted to plot gp with no gp defined")
#         return
#     else:
#
#         root_dir = "./saved/gp_plots/"
#         if os.path.isdir(root_dir) == False:
#                 os.makedirs(root_dir)
#                 lc_logger.info("Func get_gp() made directory "+root_dir)
#
#         save_name = root_dir+lightcurve.filename+"_gp.pdf"
#         lc_logger.info("Plotted gp for "+lightcurve.filename)
#
#         gp = lightcurve.interp_flux
#
#         x,y,err = lightcurve.time, lightcurve.flux, lightcurve.flux_err
#
#
#         t = np.linspace(np.min(x), np.max(x), 500)
#         mu, var = gp.predict(y, t,return_var=True)
#
#
#         fs = 17
#         pl.figure(figsize=(12,8))
#         pl.errorbar(x,y,err,fmt='o',color = '#808080',capsize=5,capthick=1,
#                     label = 'Data')
#         pl.plot(t,mu, color = np.array(color_palette[4]),linewidth=4,
#                 label = 'GP mean')
#         pl.title(lightcurve.filename+'_gp',fontsize=fs)
#
#         sns.despine()
#         pl.legend(fontsize = 17)
#         pl.xlabel("Time   [Days]",fontsize = fs, fontweight = 'bold')
#         pl.ylabel("Flux   [Jy]",fontsize = fs, fontweight = 'bold')
#
#         ax = pl.gca()
#         for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(fs)
#         for tick in ax.yaxis.get_major_ticks():
#             tick.label.set_fontsize(fs)
#
#         if gp_error == True:
#             pl.fill_between(t, mu - np.sqrt(var), mu + np.sqrt(var),
#                             color=.9*np.array(color_palette[4]), alpha=0.35)
#
#         if show == True:
#             pl.show()
#
#         if save == True:
#             pl.savefig(save_name,bbox_inches='tight')
#             lc_logger.info("Saved plot of gp to "+save_name)
#
#         return
