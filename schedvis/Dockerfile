FROM python:3.7

WORKDIR /app
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD . .
EXPOSE 8000
CMD ["gunicorn", "dashboard.wsgi", "-b", ":8000"]
