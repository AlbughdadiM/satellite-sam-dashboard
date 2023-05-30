FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get install -y wget
RUN apt-get install -y python3-pip software-properties-common 
RUN apt -y update
RUN add-apt-repository ppa:ubuntugis/ppa -y
RUN apt -y update
RUN apt install -y gdal-bin python3-gdal libgdal-dev libspatialindex-dev
RUN mkdir /satellite-sam-dashboar
COPY . /satellite-sam-dashboard/
WORKDIR  /satellite-sam-dashboard/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --ignore-installed --timeout=1000
WORKDIR  /satellite-sam-dashboard/src
CMD exec gunicorn --bind :8080 --log-level info --workers 1 --threads 8 --timeout 0 app:server




