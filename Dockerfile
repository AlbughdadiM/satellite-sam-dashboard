FROM python3.10:slim
RUN apt-get -y update
RUN apt-get install -y wget
RUN mkdir /satellite-sam-dashboar
COPY . /satellite-sam-dashboard/
WORKDIR  /satellite-sam-dashboard/ 
RUN pip3 install -r requirements.txt

CMD exec gunicorn --bind :$PORT --log-level info --workers 1 --threads 8 --timeout 0 src/app:server




