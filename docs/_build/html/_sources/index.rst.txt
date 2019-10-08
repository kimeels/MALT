.. MALT documentation master file, created by
   sphinx-quickstart on Wed Sep 18 13:56:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**MALT**: Machine Learning for Transients
******************************************
MALT is a classification pipeline based on the paper by
`Sooknunan et al. (2018) <https://arxiv.org/abs/1811.08446>`_. It is a framework
which allows the user to classify time series data. The user is free to choose
the interpolation technique, feature extraction method, and the machine learning
classifier to use. The default pipeline is shown below. It uses Gaussian processes
to interpolate the data, a wavelet feature extraction method and a random forest
classifier.

.. image:: _static/pipeline.png
   :scale: 70 %
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   api
