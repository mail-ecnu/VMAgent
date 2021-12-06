Visualization Usage
===================

Install requirements
--------------------

.. code:: bash

   pip install -r requirements.txt

Start server
------------

You can either run a development server or deploy it to a production environment.

Run a development server
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python run.py

Deploy to production via gunicorn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``gunicorn``, and run the following command.

.. code:: bash

   gunicorn dashboard.wsgi

Upload your data
----------------

After starting the service, visit and upload the data (pickle file) you want to visualize through the web page.

For the format of the file, please refer to the `data format <./format.md>`__.

|image1| |image2| |image3|

.. |image1| image:: ../images/rec-small.gif
.. |image2| image:: ../images/rec-large.gif
.. |image3| image:: ../images/exp-large.gif
