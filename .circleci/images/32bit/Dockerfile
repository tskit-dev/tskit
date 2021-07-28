FROM i386/python:3.7-buster

RUN apt-get update && apt-get install -y sudo rustc libhdf5-dev libgsl-dev
RUN adduser --disabled-password --gecos "" circleci
RUN echo 'circleci     ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER circleci
