# Dockerfile
ARG FUNCTION_DIR="/function"

FROM nvcr.io/nvidia/tensorrt:21.09-py3 as build-image

RUN apt update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}

COPY . ${FUNCTION_DIR}

RUN pip install \
        --target ${FUNCTION_DIR} \
        awslambdaric \
        numpy==1.20 \
        soundfile \
        librosa \
        boto3

FROM nvcr.io/nvidia/tensorrt:21.09-py3

RUN apt update && \
  apt-get install -y \
  libsndfile1

ENV NUMBA_CACHE_DIR=/tmp/
ENV AWS_ACCESS_KEY_ID="AKIASVQOX5XZJMYPBX3T"
ENV AWS_SECRET_ACCESS_KEY="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"

ARG FUNCTION_DIR

WORKDIR ${FUNCTION_DIR}

COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

RUN mkdir -p /opt/ml/processing/output

COPY . /usr/src/app
