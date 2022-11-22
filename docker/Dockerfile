#FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

USER root
ENV http_proxy 'http://proxy.uec.ac.jp:8080'
ENV https_proxy 'http://proxy.uec.ac.jp:8080'
ENV ftp_proxy 'http://proxy.uec.ac.jp:8080'

ARG NB_USER='inoue'
ARG NB_UID="20010"
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER


RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

ENV TZ=Asia/Tokyo
RUN apt-get install -y --no-install-recommends python3 python3-pip emacs

COPY requirements.txt .
RUN pip3 install -U setuptools pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install flax
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN jupyter lab --generate-config
#RUN echo -e "c.NotebookApp.ip = '0.0.0.0'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port=8888\nc.NotebookApp.token='argon2:$argon2id$v=19$m=10240,t=10,p=8$OzNC9DwSCyg5ACdhmNjMdQ$cnFZQ8SWBHF33BVQLv0CUqcmT+fLypyV0BMf3xNgRdo'" > ~/.jupyter/jupyter_lab_config.py


WORKDIR /home/inoue/work
EXPOSE 8888
EXPOSE 6006