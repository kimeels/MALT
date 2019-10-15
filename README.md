# MALT: Machine Learning for Transients


MALT is a classification pipeline based on the paper "Classification of
Multiwavelength Transients with Machine Learning by [Sooknunan et al. (2018)](https://arxiv.org/abs/1811.08446). It is a framework which allows the user to classify time series data. The user is free to choose the interpolation technique, feature extraction method, and the machine learning classifier to use.


## How to install MALT


First clone the git repo and install virtualenv:
  ```
  git clone https://github.com/kimeels/MALT.git

  python3 -m pip install --user virtualenv
  ```
Change directories into MALT and create a virtual environment:
  ```
  cd MALT

  python3 -m venv malt_env
  ```

Start the virtual env and install the necessary packages using the requirements file:
  ```
  source malt_env/bin/activate

  pip3 install -r requirements.txt
  ```
