from malt_logger import*
from lightcurve import Lightcurve
import interpolator
import feature_extraction
import machine_learning
from tqdm.autonotebook import tqdm
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import configparser

class Dataset:
    def __init__(self, configFile = '',
                 feat_ex_method = feature_extraction.get_wavelet_feature,
                 interpolate = False,
                 interp_func = interpolator.get_gp, ini_t = 'rand',
                 obs_time = 8/24, sample_size = 100,
                 ml_method = machine_learning.RFclassifier,
                 hyperparams = {"n_estimators": np.arange(70,90),
                                "criterion": ["gini", "entropy"]},
                 n_jobs=7, pca = True, n_components = 20):

        """
            Initialises an instance of the Dataset class.

            Param
            ------
            self: Database object
                An instance of the Database class.
            feat_ex_method: python function
                Function to use for the feature extraction.
            interpolate: boolean
                If True will use interp_func() to interpolate  the loaded data.
            interp_func: python function
                A python function that takes in a lightcurve and interpolates it.
            ini_t: str or float
                Initial time to start sampling.
            obs_time: float
                The total length of the interpolated lightcurve.
            sample_size: int
                Number of data points in interpolated lightcurve.
            ml_method: sklearn object.
                The machine learning classifier to use.
            hyperparams: dictionary
                The hyperparams to optimize the ml_method over.
            n_jobs: Number of jobs to run in parallel. If set to -1 will use
                      all cores.
            n_components: int
                Number of pca components to keep.

        """
        if configFile:
            try:
                lc_logger.info("Initialising Dataset with parameters from config file: "+configFile)

                config = configparser.ConfigParser()
                config.read(configFile)

                interp_params = config['INTERPOLATOR']

                self._interpolate = interp_params['interpolate']
                interp_func = getattr(interpolator, interp_params['interp_func'], None)

                self._interp_func = interp_func
                ini_t = interp_params['ini_t']
                if 'os' in ini_t:
                    raise ValueError('Can not use os module in ini_t!')
                else:
                    self._ini_t = eval(ini_t)
                self._obs_time = float(interp_params['obs_time'])
                self._sample_size = int(interp_params['sample_size'])

                ml_params = config['ML']

                feat_ex_method = getattr(feature_extraction,ml_params['feat_ex_method'], None)
                self._feat_ex_method = feat_ex_method

                ml_method = getattr(machine_learning,ml_params['ml_method'], None)
                self._ml_method = ml_method

                hyperparams = ml_params['hyperparams']
                if 'os' in hyperparams:
                    raise ValueError('Can not use os module in hyperparams!')
                else:
                    self._hyperparams = eval(hyperparams)

                self._n_jobs = int(ml_params['n_jobs'])
                self._pca = ml_params['pca']
                self._n_components = int(ml_params['n_components'])


            except IOError as io:
                print('An error occured trying to read the configFile.')
                lc_logger.exception("An error occured trying to read the configFile.")
            except Exception as e:
                print("An error has occured trying to read the configFile")
                lc_logger.exception("Exception occurred trying to read the configFile")

        else:
            lc_logger.info("Initialising Dataset with default parameters")

            self._interpolate = interpolate
            self._interp_func = interp_func
            self._ini_t = ini_t
            self._obs_time = obs_time
            self._sample_size = sample_size

            self._feat_ex_method = feat_ex_method

            self._ml_method = ml_method
            self._hyperparams = hyperparams
            self._n_jobs = n_jobs
            self._pca = pca
            self._n_components = n_components


    def populate(self,filepaths):
        """
            Initialises an instance of the Dataset class.

            Param
            ------
            self : Database object
                An instance of the Database class.
            filepaths: list
                List containing the paths to the data files.
        """
        interpolate = self._interpolate
        interp_func = self._interp_func
        ini_t = self._ini_t
        obs_time = self._obs_time
        sample_size = self._sample_size

        try:
            lcs = [Lightcurve(file,interpolate,interp_func,ini_t,obs_time,
                              sample_size) for file in tqdm(filepaths,desc="Loading lightcurves", unit="lcs")]

            self.lightcurves = lcs


        except Exception as e:
            print("An error has occured while reading in dataset")
            lc_logger.exception("Exception occurred while reading in dataset")

        if interpolate == 'True':
            try:
                self.interpolate()
            except Exception as e:
                print("An error has occured while performing interpolation")
                lc_logger.exception("Exception occurred while performing interpolation")

    def types(self):
        """
        Prints out the counts of each object type stored in the dataset.

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.

        """
        try:
            lcs = self.lightcurves
            classes = [l.type for l in lcs]
            labels,counts = np.unique(classes, return_counts = True)

            for i in range(len(labels)):
                print("{a:12s}: {p:3d}".format(a=labels[i], p=counts[i]))

            print('-'*18)
            print("{a:12s}: {p:3d}".format(a='Total', p=np.sum(counts)))

        except Exception as e:
            print("An error has occured.")
            lc_logger.exception("Exception occurred.")



    def interpolate(self):
        """
            Interpolates all the lightcurves in the given dataset with
            assigned interpolation function.

            Param
            ------
            self : Dataset object
                An instance of the Dataset class containing instances of the
                Lightcurve class.

        """
        interp_func = self._interp_func
        ini_t = self._ini_t
        obs_time = self._obs_time
        sample_size = self._sample_size

        try:
            interp = [lc.interpolate(interp_func, ini_t, obs_time,sample_size)
                      for lc in tqdm(self.lightcurves,desc="Interpolating lightcurves", unit="lcs")]
        except Exception as e:
            print("An error has occured while interpolating the dataset")
            lc_logger.exception("Exception occurred while interpolating the dataset")

    def extract_features(self):
        """
            Extracts features from all the lightcurves in the given dataset with
            assigned feature extraction method.

            Param
            ------
            self : Lightcurve object
                An instance of the Lightcurve class.

        """
        feat_ex_method = self._feat_ex_method

        try:
            for lc in tqdm(self.lightcurves,desc="Extracting features", unit="lcs"):
                lc.extract_features(feat_ex_method)
        except Exception as e:
            print("An error has occured while performing feature extraction")
            lc_logger.exception("Exception occurred while performing feature extraction")

    def get_pca(self):
        """
        Performs PCA decomposition of a feature array X.

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.

        """
        n_components = self._n_components

        X = np.array([lc.features for lc in tqdm(self.lightcurves,desc="Stacking features", unit="lcs")])
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

    def project_pca(self, dataset = None,lightcurve = None):
        """
        Projects self.features onto  calculated PCA axis from self.pca

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.

        """
        if not dataset:
            eigvecs, mn, eigvals = self.pca
        else:
            eigvecs, mn, eigvals = dataset.pca

        if not lightcurve:
            lcs = self.lightcurves
            feats = np.array([l.features for l in lcs])
        else:
            feats = lightcurve.features

        proj_feat = np.matmul(feats - mn, eigvecs)

        return proj_feat

    def train(self, verbose = 1):
        """
         Trains a ML algorithm on the Dataset with the parameters specified on
         initialisation.
        Params:
        -------
        self : Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        verbose: How much information to print out.

        """

        lcs = self.lightcurves


        if self._pca == 'True':
            self.get_pca()
            feat_matrix = self.project_pca()
        else:
            feat_matrix = np.array([l.features for l in lcs])

        coords = np.array([[l.ra_dec] for l in lcs])

        X_train = np.append(feat_matrix,coords, axis = 1)
        y_train = np.array([l.type for l in lcs])


        GSclf = GridSearchCV(self._ml_method(), param_grid=self._hyperparams,
                             verbose = verbose, n_jobs=self._n_jobs, cv = 3)
        start = time.time()
        GSclf.fit(X_train, y_train)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time.time() - start, len(GSclf.cv_results_['params'])))

        clf = GSclf.best_estimator_
        self.classifier = clf

    def predict(self, lightcurve, show_prob = False):
        """
         Predicts the type of given lightcurve object using classifier trained
         on Dataset.
        Params:
        -------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        lightcurve: Lightcurve object
            Lightcurve object for which to predict
        show_prob: boolean.
            If True will print full output from predict_proba()
        """
        proj_feats = self.project_pca(lightcurve = lightcurve)
        coords = lightcurve.ra_dec

        X_test = np.append(proj_feats,coords)
        y_test = lightcurve.type

        clf = self.classifier
        classes = clf.classes_

        probs = clf.predict_proba([X_test])[0]
        label = clf.predict([X_test])[0]
        p = probs[np.where(classes == label)][0]

        if show_prob == True:
            for i in range(len(classes)):
                print("{c:12s}: {p:>2.2f}%".format(c=classes[i], p=probs[i]*100))

        return p,label


    def add(self, new_lightcurve):
        """
         Adds new lightcurve to the Dataset then retrains Dataset.
        Params:
        -------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        lightcurve: Lightcurve object
            Lightcurve object to add to dataset.
        """
        self.lightcurves.append(new_lightcurve)
        self.interpolate()
        self.extract_features()
        self.train()
