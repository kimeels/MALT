import numpy as np
import random
from sklearn.metrics import confusion_matrix
from tqdm.autonotebook import tqdm
from sklearn.model_selection import GridSearchCV
import time
import pylab as pl
import itertools



class Diagnostic:

    def __init__(self, dataset):
        """
        Initialises an instance of the Diagnostic class.

        Params
        ------
        self: Diagnostic object
            An instance of the Diagnostic class.
        dataset: Database object
            An instance of the Database class.
        """

        self._dataset = dataset

        self._test_split = self._dataset._test_split
        self._num_of_runs = self._dataset._num_of_runs
        self._aug_num = self._dataset._aug_num
        self._pca = dataset._pca
        self._ml_method = dataset._ml_method
        self._hyperparams = dataset._hyperparams


        self.classifiers = []
        self.conf_matrices = []
        self.types = None

        self.run_diagnostic()

    def run_diagnostic(self):
        """
        Runs the Diagnostic test which trains n classifiers on different subsets
        of the Dataset to test how well it can classify objects. 

        Params
        ------
        self: Diagnostic object
            An instance of the Diagnostic class.

        """
        dataset = self._dataset
        num_of_runs = self._num_of_runs
        test_split = self._test_split

        if dataset._interpolate == 'True':
            if dataset._did_interpolation == False:
                dataset.interpolate()

        if dataset._did_feat_extraction == False:
            dataset.extract_features()

        lcs = dataset.lightcurves
        type_dict = dataset._type_dict


        keys = [key for key in type_dict if key!='Total']
        self.types = keys

        num_in_test_set = [int(np.ceil(type_dict[key]*test_split))
                   for key in keys]

        fnames_per_type = [np.array([l.filename for l in lcs if l.type == key])
                           for key in keys]

        for i in tqdm(range(num_of_runs),desc="Training Classifier", unit="runs"):
            rand_nums = [random.sample(range(len(fnames_per_type[i])),  num_in_test_set[i])
                         for i in range(len(num_in_test_set))]

            fnames_in_test_set = [fnames_per_type[i][rand_nums[i]] for i in range(len(rand_nums))]
            fnames_in_test_set = np.hstack(fnames_in_test_set)

            lcs_test = [l for l in lcs if l.filename in fnames_in_test_set]
            lcs_train = [l for l in lcs if l.filename not in fnames_in_test_set]

            self._train(lcs_train,lcs_test)

    def plot_confusion_matrix(self, cm, normalize=False, title = None, save = False,
                              name = 'conf_matrix.pdf'):
        """
        This function plots the confusion matrix.

        Params
        ------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        cm: 2d numpy array
            The confusion matrix
        normalize: boolean
            Option to normalize the confusion matrix
        title: str
            Title of plot
        save: boolean
            Option to save plot
        name: str
            Name of saved plot
        """

        acc = np.trace(cm)/np.sum(cm)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            if title == None:
                title = "Normalized confusion matrix: Accuracy = "+ str(acc)
        else:
            if title == None:
                title = 'Confusion matrix, without normalization ' +str(acc)

        classes = self.types
        pl.figure(figsize=(12,9))
        pl.imshow(cm, interpolation='nearest', cmap=pl.cm.Blues)
        pl.title(title,fontsize = 17,fontweight = 'bold')
        #cbar = pl.colorbar()
        tick_marks = np.arange(len(classes))
        pl.xticks(tick_marks, classes, rotation=60,fontweight = 'bold')
        pl.yticks(tick_marks, classes,fontweight = 'bold')

        ax = pl.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        #cbar.ax.tick_params(labelsize=13)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            text = format(cm[i, j], fmt)
            if text == '0.00':
                text = '0.0'
            pl.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     weight = 'medium',
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize = 15,
                   )

        pl.grid(False)
        pl.ylabel('True label',fontsize = 18)
        pl.xlabel('Predicted label',fontsize = 18)
        # get rid of the frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        # remove all the ticks and directly label each bar with respective value
        pl.tick_params(top=False, bottom=False, left=False, right=False,
                       labelleft=True, labelbottom=True)


        if save != False:
            pl.savefig(name,bbox_inches="tight")

    def plot_results(self, normalize=True, save = False, save_root = './'):
        """
        This function plots the results of the diagnostic run. It plots the
        confusion matrices with the maximum, minimum and average overall accuracies.

        Params
        ------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        normalize: boolean
            Option to normalize the confusion matrix.
        save: boolean
            Option to save plot.
        save_root: str
            file path for saved plots.
        """

        cnfs = np.array(self.conf_matrices)

        traces = np.trace(cnfs,axis1=1, axis2=2)
        sums = np.array([np.sum(cnfs[i]) for i in range(len(cnfs))])
        accs = traces/sums

        title_min ='Min Confusion matrix: Accuracy = '+str(np.round(np.min(accs)*100,1)) + '%'

        title_max ='Max Confusion matrix: Accuracy = '+str(np.round(np.max(accs)*100,1)) + '%'

        title_avg ='Avg Confusion matrix: Accuracy = '+str(np.round(np.average(accs)*100,1)) + '%'

        names = self.types

        self.plot_confusion_matrix(cnfs[np.where(accs == np.min(accs))[0][0]],
                                   normalize=normalize,
                                   title= title_min,
                                   save=save, name=save_root+'Min_conf_matrix.pdf')

        self.plot_confusion_matrix(cnfs[np.where(accs == np.max(accs))[0][0]],
                                   normalize = normalize, title = title_max,
                                   save = save, name=save_root+'Max_conf_matrix.pdf')


        self.plot_confusion_matrix(np.sum(cnfs,axis=0), normalize=normalize,
                                   title= title_avg, save=save,
                                   name=save_root+'Average_conf_matrix.pdf')



    def _train(self,lcs_train,lcs_test):
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

        aug_num = self._dataset._aug_num


        if self._dataset._pca == 'True':
            pca = self._get_pca(lcs_train)
            feat_matrix_train = self._project_pca(pca,lcs_train)
            feat_matrix_test = self._project_pca(pca,lcs_test)

        else:
            feat_matrix_train = np.array([l.features for l in lcs_train])
            feat_matrix_test = np.array([l.features for l in lcs_test])

        coords_train = np.array([[l.ra_dec]*aug_num for l in lcs_train])
        coords_train = coords_train.reshape(coords_train.shape[0]*coords_train.shape[1],1)
        X_train = np.append(feat_matrix_train,coords_train, axis = 1)

        coords_test = np.array([[l.ra_dec]*aug_num for l in lcs_test])
        coords_test = coords_test.reshape(coords_test.shape[0]*coords_test.shape[1],1)
        X_test = np.append(feat_matrix_test,coords_test, axis = 1)

        y_train = np.array([[l.type]*aug_num for l in lcs_train])
        y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1])

        y_test = np.array([[l.type]*aug_num for l in lcs_test])
        y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1])


        GSclf = GridSearchCV(self._dataset._ml_method(),
                             param_grid=self._dataset._hyperparams,
                             n_jobs=self._dataset._n_jobs, cv = 3, verbose = 0)
        start = time.time()
        GSclf.fit(X_train, y_train)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time.time() - start, len(GSclf.cv_results_['params'])))

        clf = GSclf.best_estimator_

        cnf_matrix = confusion_matrix(y_test,clf.predict(X_test),labels=self.types)

        self.conf_matrices.append(cnf_matrix)
        self.classifiers.append(clf)

    def _get_pca(self,lcs):
        """
        Performs PCA decomposition of a feature array X.

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        lcs: list
            List of Lightcurve objects.

        Returns
        --------
        pca: list
            List containing arrays from PCA calculation.
        """
        n_components = self._dataset._n_components

        all_lc_feats = np.array([lc.features for lc in lcs])
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
        pca = [eigvecs[:,:n_components], mn[:,0] , eigvals]
        return pca

    def _project_pca(self,pca,lcs):
        """
        Projects self.features onto calculated PCA axis from self.pca

        Parameters
        ----------
        self: Dataset object
            An instance of the Dataset class containing instances of the
            Lightcurve class.
        pca: list
            Output from self.pca

        Returns
        --------
        proj_feat: array
            Array of projected features.
        """

        eigvecs, mn, eigvals = pca

        all_lc_feats = np.array([lc.features for lc in lcs])
        shape = all_lc_feats.shape

        feats = all_lc_feats.reshape(shape[0]*shape[1],shape[2])

        proj_feat = np.matmul(feats - mn, eigvecs)

        return proj_feat
