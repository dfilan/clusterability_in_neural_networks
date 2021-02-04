FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

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
COPY build.sh /project/build.sh
WORKDIR /project

# RUN adduser --disabled-password --gecos '' nncuser
# WORKDIR /home/nncuser/
# USER nncuser
# COPY build.sh .
# RUN chown nncuser build.sh
