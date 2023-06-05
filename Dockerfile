FROM python:3.10-slim-buster

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget gdal-bin libgdal-dev libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /satellite-sam-dashboard
COPY . /satellite-sam-dashboard/
WORKDIR /satellite-sam-dashboard

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

WORKDIR /satellite-sam-dashboard/src
RUN mkdir weights \
    && wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

CMD exec gunicorn --bind :8080 --log-level info --workers 1 --threads 8 --timeout 0 app:server
