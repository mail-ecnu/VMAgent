# Visualization Usage

## Install requirements

```bash
pip install -r requirements.txt
```

## Start server

You can either run a development server or deploy it to a production environment. 

### Run a development server

```bash
python run.py
```

### Deploy to production via gunicorn

Install `gunicorn`, and run the following command.

```bash
gunicorn dashboard.wsgi
```

## Upload your data

After starting the service, visit <localhost:8080> and upload the data (pickle file) you want to visualize through the web page.

For the format of the file, please refer to the [data format](./format.md).

![](../images/rec-small.gif)
![](../images/rec-large.gif)
![](../images/exp-large.gif)
