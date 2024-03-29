FROM python:3.10

COPY app.py ./app.py
COPY requirements.txt ./requirements.txt
COPY clfmodel.pkl ./clfmodel.pkl

RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app