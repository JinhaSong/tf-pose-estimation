FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git wget python3-dev python3-pip apt-utils \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y swig
RUN apt-get install -y libglib2* libsm6 libxrender1 libxext6
RUN apt-get install -y vim ssh
RUN rm -rf /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip
RUN pip install setuptools cython numpy opencv-python
RUN pip install tensorflow-gpu

WORKDIR /workspace
ADD . .

RUN chmod -R a+w /workspace
RUN pip install -r requirements.txt