# dockerfile para prediccion
FROM python:3.8.12-slim-buster
COPY ./code /opt/wcontrol
COPY ./requirements.txt /opt/wcontrol
COPY ./experiments/experimentoZP1/pmodel /opt/modelo-wc/pmodel
COPY ./experiments/experimentoZP1/zmodel /opt/modelo-wc/zmodel

RUN apt-get update
#RUN apt-get install -y git
RUN apt-get install -y vim

WORKDIR /opt/wcontrol
RUN pip3 install -r requirements.txt
CMD python3 main_api.py
