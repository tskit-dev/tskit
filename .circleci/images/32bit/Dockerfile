FROM i386/python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y sudo rustc cargo libhdf5-dev libgsl-dev pkg-config libssl-dev llvm build-essential
RUN adduser --disabled-password --gecos "" circleci
RUN echo 'circleci     ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER circleci
