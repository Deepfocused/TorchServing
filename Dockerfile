# syntax = docker/dockerfile:experimental
#
# This file can build images for cpu and gpu env. By default it builds image for CPU.
# Use following option to build image for cuda/GPU: --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Here is complete command for GPU/cuda - 
# $ DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:18.04
FROM ${BASE_IMAGE} AS compile-image
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG CUDA_VERSION=""

RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    TORCH_VISION_VER=$(curl --silent --location https://pypi.org/pypi/torchvision/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            pip install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION torchvision==$TORCH_VISION_VER+$CUDA_VERSION -f https://download.pytorch.org/whl/torch_stable.html; \
        # Install the binary with the latest CUDA version support
        else \
            pip install --no-cache-dir torch torchvision; \
        fi \
    # Install the CPU binary
    else \
        pip install --no-cache-dir torch==$TORCH_VER+cpu torchvision==$TORCH_VISION_VER+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    fi
RUN pip install --no-cache-dir captum torchtext torchserve torch-model-archiver

# RUN pip install --no-cache-dir torch==1.7.1 torchvision==0.8.2
# RUN pip install --no-cache-dir captum torchtext torchserve torch-model-archiver

# Final image for production
FROM ${BASE_IMAGE} AS runtime-image

ENV PYTHONUNBUFFERED TRUE

# https://vsupalov.com/buildkit-cache-mount-dockerfile/
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3 \
    python3-distutils \
    python3-dev \
    openjdk-11-jre-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m hae \
    && mkdir -p /home/hae
COPY --chown=hae --from=compile-image /home/venv /home/venv
ENV PATH="/home/venv/bin:$PATH"
RUN chown -R hae /home/hae

EXPOSE 8080 8081 8082 7070 7071

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir opencv-python

USER hae
WORKDIR /home/hae/models
CMD ["torchserve", "--start", "--ncs", "--ts-config", "config.properties", "--model-store", ".", "--models", "facedetector.mar", "--foreground"]
