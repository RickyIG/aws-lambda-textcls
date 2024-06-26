FROM python:3.10.12

WORKDIR /app

ADD . .

RUN pip install -r requirements.txt

RUN python preprocess.py

CMD ["python", "app.py"]
