FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py .
COPY predictor.py .
COPY model.cbm .
COPY static static
COPY templates templates

EXPOSE 5000

CMD gunicorn app:"app" -b 0.0.0.0:5000 --reload  
# CMD python3 /app.py