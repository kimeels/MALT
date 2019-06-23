from malt_logger import*
from lightcurve import*
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

class Dataset:
    def __init__(self, filepaths, obj_types, interpolate = False, interp_func = get_gp,
                 ini_t = 'rand', timescale = 8/24, sample_size = 100):

        """
            Initialises an instance of the Lightcurve class.

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.
            filepaths: list
                List containing the paths to the data files.
            obj_types: list
                List of types corresponding to each Lightcurve object.
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
            lcs = [Lightcurve(file,interpolate,interp_func,ini_t,timescale,
                              sample_size) for file in filepaths]

            self.lightcurves = lcs
            self.pca = [[[np.nan]]]

        except Exception as e:
            print("An error has occured while reading in dataset")
            lc_logger.exception("Exception occurred while reading in dataset")

        if interpolate == True:
            try:
                self.interpolate(interp_func, ini_t, timescale, sample_size)
            except Exception as e:
                print("An error has occured while performing interpolation")
                lc_logger.exception("Exception occurred while performing interpolation")

    def interpolate(self,interp_func = get_gp, ini_t = 'rand', timescale = 8/24,
                    sample_size = 100):
        """
            Interpolates all the lightcurves in the given dataset with
            assigned interpolation function.

            Param
            ------
            self : Dataset object
                An instance of the Dataset class containing instances of the
                Lightcurve class.
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
            interp = [lc.interpolate(interp_func, ini_t, timescale,sample_size)
                      for lc in tqdm(self.lightcurves)]
        except Exception as e:
            print("An error has occured while interpolating the dataset")
            lc_logger.exception("Exception occurred while interpolating the dataset")

    def feat_extract(self, feat_ex_method= get_wavelet_feature ):
        """
            Extracts features from all the lightcurves in the given dataset with
            assigned feature extraction method.

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.
            feat_ex_method: python function
                Function to use for the feature extraction.
        """

        try:
            for lc in tqdm(self.lightcurves):
                if np.isnan(lc.interp_flux[0]):
                    print("No interpolated flux. Please run interpolate() first.")
                else:
                    feats = feat_ex_method(lc)
                    lc.features = feats
        except Exception as e:
            print("An error has occured while performing feature extraction")
            lc_logger.exception("Exception occurred while performing feature extraction")

    def get_pca(self,n_components = 20):
        """
        Performs PCA decomposition of a feature array X.

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        n_components: int
            Number of pca components to keep.
        """

        X = np.array([lc.features for lc in tqdm(self.lightcurves)])
        X=X.transpose()

        mn=np.mean(X, axis=1)
        mn.shape=(len(mn), 1)
        X=X-mn
        C=np.cov(X)
        vals, vec = np.linalg.eigh(C)

        inds=np.argsort(vals)[::-1]

        eigvals = vals[inds]
        eigvecs = vec[:, inds]
        self.pca = [eigvecs[:,:n_components], mn[:,0] , eigvals]

    def project_pca(self):
        """
        Projects self.features onto  calculated PCA axis from self.pca

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.

        """

        eigvecs, mn, eigvals = self.pca
        lcs = self.lightcurves
        feat_matrix = np.array([l.features for l in lcs])

        proj_feat = np.matmul(feat_matrix - mn, eigvecs)

        return proj_feat

    def train(self,ml_method = RandomForestClassifier(),
              hyperparams = {"n_estimators": np.arange(70,90),
                             "criterion": ["gini", "entropy"]},
              verbose = 1, n_jobs=7, pca = True):
        """


        Params:
        -------
        self : Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        ml_method: sklearn object.
            The machine learning classifier to use.
        hyperparams: dictionary
            The hyperparams to optimize the ml_method over.
        verbose : How much information to print out.
        n_jobs  : Number of jobs to run in parallel. If set to -1 will use all cores.
        """

        lcs = self.lightcurves

        if pca == True:
            if np.isnan(self.pca[0][0][0]):
                    print("No Principle components. Please run get_pca() first.")
            else:
                feat_matrix = self.project_pca()
        else:
            feat_matrix = np.array([l.features for l in lcs])

        coords = np.array([[l.ra_dec] for l in lcs])

        X_train = np.append(feat_matrix,coords, axis = 1)
        y_train = np.array([l.type for l in lcs])


        GSclf = GridSearchCV(ml_method, param_grid=hyperparams,verbose = verbose,
                             n_jobs=n_jobs,cv = 3)
        start = time.time()
        GSclf.fit(X_train, y_train)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time.time() - start, len(GSclf.cv_results_['params'])))

        clf = GSclf.best_estimator_
        return clf


    # def extra_features(self, ex_feats, from_text = False):
    # """
    #
    #
    # Params:
    # -------
    # self : Dataset object
    #     An instance of the Dataset class containing instances of the
    #     Lightcurve class.
    #
    # """
    # if from_text == True:
    #     self.extra_features = np.loadtxt(ex_feats)
    # else:
    #     self.extra_features = ex_feats
