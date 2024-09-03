FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /root
COPY . /root/
RUN apt-get update && apt-get install --yes --no-install-recommends \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cd /root && pip install -r requirements.docker.txt
