FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter
ARG DOWNLOAD_ALL=0

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# Python, don't write bytecode!
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt update --fix-missing && apt install -y wget unzip vim tmux git graphviz graphviz-dev make

# -- Install Pipenv:
RUN apt install -y python3.7-dev libffi-dev
RUN curl --silent https://bootstrap.pypa.io/get-pip.py | python3.7

# Backwards compatility.
RUN rm -fr /usr/bin/python3 && ln /usr/bin/python3.7 /usr/bin/python3

RUN pip3 install pipenv

RUN mkdir /project
WORKDIR /project

# RUN adduser --disabled-password --gecos '' nncuser
# WORKDIR /home/nncuser/
# USER nncuser
# COPY build.sh .
# RUN chown nncuser build.sh

RUN git clone https://github.com/dfilan/clusterability_in_neural_networks.git
WORKDIR clusterability_in_neural_networks
RUN git checkout main

RUN echo "Installing python dependencies..." && pipenv install --system
RUN echo "Patching scipy and sklearn..." \
    && python -m site | grep /usr/local/lib/python3.7/dist-packages \
    || { echo "Cannot find dist-packages directory for patching of scipy and \
    sklearn"; exit 1; }
COPY patches/* /usr/local/lib/python3.7/dist-packages/

RUN if [ "$DOWNLOAD_ALL" -gt 0 ]; then echo "Downloading models folder, \
    including checkpoints. Note that this may not include all models used \
    in the paper. Hold tight, this might take a while..." \
    && aws s3 cp --no-sign-request --region=us-west-1 --recursive \
    s3://clusterability-in-neural-networks/public/models models; \
    else echo "Downloading models folder, excluding checkpoints. Note that \
    this might not include all models used in the paper..." \
    && aws s3 cp --no-sign-request --region=us-west-1 --recursive \
    --exclude "*.ckpt" \
    s3://clusterability-in-neural-networks/public/models models; fi
RUN echo "Downloading datasets folder..." \
    && aws s3 cp --no-sign-request --region=us-west-1 --recursive \
    s3://clusterability-in-neural-networks/public/datasets datasets
