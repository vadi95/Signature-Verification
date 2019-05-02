from python:2.7

WORKDIR /var/app

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

CMD python run.py
