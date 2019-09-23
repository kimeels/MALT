How to install MALT
========================

First clone the git repo and install virtualenv::

  git clone https://github.com/kimeels/MALT.git

  python3 -m pip install --user virtualenv

Change directories into MALT and create a virtual environment::

  cd MALT

  python3 -m venv malt_env


Start the virtual env and install the necessary packages using the requirements file::

  source malt_env/bin/activate
  
  pip3 install -r requirements.txt
