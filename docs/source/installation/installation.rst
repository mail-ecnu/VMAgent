Installation
============

Install from Source
-------------------

First clone our git repo:

.. code:: bash

   git clone git@github.com:mail-ecnu/VMAgent.git
   cd VMAgent

Then create the virtual environment to satisfy dependency with conda:

.. code:: bash

   conda env create -f conda_env.yml
   conda activate VMAgent-dev

Finally Install our simulator:

.. code:: bash

   python3 setup.py develop
