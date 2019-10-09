from .malt_logger import*
from .lightcurve import Lightcurve
from . import interpolator
from . import feature_extraction
from . import machine_learning
from .diagnostic import Diagnostic
from tqdm.autonotebook import tqdm
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import configparser
import pickle

class Dataset:
    def __init__(self, configFile = '',
                 feat_ex_method = feature_extraction.get_wavelet_feature,
                 interpolate = True,
                 interp_func = interpolator.get_gp, ini_t = 'rand',
                 obs_time = 8/24, sample_size = 100, aug_num = 1,
                 ml_method = machine_learning.RFclassifier,
                 hyperparams = {"n_estimators": np.arange(70,90),
                                "criterion": ["gini", "entropy"]},
                 n_jobs=-1, pca = True, n_components = 20):

        """
            Initialises an instance of the Dataset class.

            Params
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
                self._aug_num = int(ml_params['aug_num'])
                self._test_split = float(ml_params['test_split'])
                self._num_of_runs = int(ml_params['num_of_runs'])

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
            self._aug_num = aug_num
            self._test_split = .25
            self._num_of_runs = 20

        self._did_interpolation = False
        self._did_feat_extraction = False


    def populate(self,filepaths):
        """
            Initialises an instance of the Dataset class.

            Param
            ------
            self: Database object
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


            classes = [l.type for l in lcs]
            labels,counts = np.unique(classes, return_counts = True)
            type_dict = dict(zip(labels, counts))
            type_dict['Total'] = np.sum(counts)

            self._type_dict = type_dict

        except Exception as e:
            print("An error has occured while reading in dataset")
            lc_logger.exception("Exception occurred while reading in dataset")


    def types(self,show_aug_num = False):
        """
        Prints out the counts of each object type stored in the dataset.

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        show_aug_num: boolean
            Use augmented lightcurve when counting type numbers.

        """
        try:
            if show_aug_num == True:
                aug = self._aug_num
            else:
                aug = 1
            for key in self._type_dict:
                if key != 'Total':
                    print("{a:12s}: {p:5d}".format(a=key, p=self._type_dict[key]*aug))

            print('-'*20)
            print("{a:12s}: {p:5d}".format(a='Total', p=self._type_dict['Total']*aug))

        except Exception as e:
            print("An error has occured.")
            lc_logger.exception("Exception occurred.")



    def interpolate(self):
        """
            Interpolates all the lightcurves in the given dataset with
            assigned interpolation function.

            Param
            ------
            self: Dataset object
                An instance of the Dataset class containing instances of the
                Lightcurve class.

        """
        interp_func = self._interp_func
        ini_t = self._ini_t
        obs_time = self._obs_time
        sample_size = self._sample_size
        aug_num = self._aug_num

        try:
            interp = [lc.interpolate(interp_func, ini_t, obs_time,sample_size,aug_num)
                      for lc in tqdm(self.lightcurves,desc="Interpolating lightcurves", unit="lcs")]
        except Exception as e:
            print("An error has occured while interpolating the dataset")
            lc_logger.exception("Exception occurred while interpolating the dataset")

        self._did_interpolation = True

    def extract_features(self):
        """
            Extracts features from all the lightcurves in the given dataset with
            assigned feature extraction method.

            Param
            ------
            self: Dataset object
                An instance of the Dataset class containing instances of the
                Lightcurve class.
        """
        feat_ex_method = self._feat_ex_method

        try:
            for lc in tqdm(self.lightcurves,desc="Extracting features", unit="lcs"):
                lc.extract_features(feat_ex_method)
        except Exception as e:
            print("An error has occured while performing feature extraction")
            lc_logger.exception("Exception occurred while performing feature extraction")
        self._did_feat_extraction = True

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

        all_lc_feats = np.array([lc.features for lc in tqdm(self.lightcurves,desc="Stacking features", unit="lcs")])
        shape = all_lc_feats.shape

        X = all_lc_feats.reshape(shape[0]*shape[1],shape[2])

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

    def project_pca(self,lightcurve = None):
        """
        Projects self.features onto  calculated PCA axis from self.pca

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.

        """

        eigvecs, mn, eigvals = self.pca

        #If Lightcurve is not given, will project all lightcurves in Database
        #onto PCA components.
        if not lightcurve:
            all_lc_feats = np.array([lc.features for lc in tqdm(self.lightcurves,desc="Stacking features", unit="lcs")])
            shape = all_lc_feats.shape

            feats = all_lc_feats.reshape(shape[0]*shape[1],shape[2])

        #If lightcurve is given, will only project that lightcurve onto PCA components.
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
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        verbose: How much information to print out.

        """

        lcs = self.lightcurves
        aug_num = self._aug_num
        if self._interpolate == 'True':
            if self._did_interpolation == False:
                self.interpolate()

        if self._did_feat_extraction == False:
            self.extract_features()

        if self._pca == 'True':
            self.get_pca()
            feat_matrix = self.project_pca()
        else:
            feat_matrix = np.array([l.features for l in lcs])

        coords = np.array([[l.ra_dec]*aug_num for l in lcs])
        coords = coords.reshape(coords.shape[0]*coords.shape[1],1)

        X_train = np.append(feat_matrix,coords, axis = 1)
        y = np.array([[l.type]*aug_num for l in lcs])

        y_train = y.reshape(y.shape[0]*y.shape[1])


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

        interp_func = self._interp_func
        ini_t = self._ini_t
        obs_time = self._obs_time
        sample_size = self._sample_size
        feat_ex_method = self._feat_ex_method

        lightcurve.interpolate(interp_func, ini_t, obs_time,sample_size)
        lightcurve.extract_features(feat_ex_method)

        proj_feats = self.project_pca(lightcurve = lightcurve)
        coords = lightcurve.ra_dec

        X_test = np.append(proj_feats,coords)

        clf = self.classifier
        classes = clf.classes_

        probs = clf.predict_proba([X_test])[0]
        label = clf.predict([X_test])[0]
        p = probs[np.where(classes == label)][0]

        if show_prob == True:
            for i in range(len(classes)):
                print("{c:12s}: {p:>2.2f}%".format(c=classes[i], p=probs[i]*100))

        return p,label

    def run_diagnostic(self):
        """
        Runs the Diagnostic test which trains n classifiers on different subsets
        of the Dataset to test how well it can classify objects.

        Params
        ------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        """
        diag = Diagnostic(self)
        return diag


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

        new_type = new_lightcurve.type
        if new_type in self._type_dict.keys():
            self._type_dict[new_type] += 1
        else:
            self._type_dict[new_type] = 1
        self._type_dict['Total'] += 1

        self._did_interpolation = False
        self._did_feat_extraction = False

        self.train()

    def save(self,filename = 'saved_dataset'):
        """
            Saves a Dataset instance using a pickle dump
        Params:
        -------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        filename: str
            filename under which to store the Dataset instance
        """
        pickle.dump(self,open(filename+'.save',"wb"))
        lc_logger.info("Saved Dataset to "+filename+'.save')
        print("Saved Dataset to "+filename+'.save')

    @classmethod
    def load_from_save(cls,filename):
        """
            Returns a saved Dataset instance using pickle
        Params:
        -------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        filename: str
            filename under which the Dataset instance was saved.
        """
        if '.save' not in filename:
            filename = filename+'.save'

        l_dataset = pickle.load(open(filename,"rb"))
        lc_logger.info("Loaded Dataset from "+filename+'.save')
        print("Loaded Dataset to "+filename+'.save')
        return l_dataset
