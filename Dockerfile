ARG BASE_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04@sha256:0725ed044e80c230fcd54218ae3edc2855897ef7813b20868bdb53b03b99ea1c
FROM ${BASE_IMAGE}

ARG CUTLASS_COMMIT=e05f953a5b3d38adc240df2ff928e0421c2abba3
ARG PYTORCH_VERSION=2.10.0+cu130
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG CUDA_PYTHON_VERSION=13.0.3
ARG MAX_BUILD_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_JOBS=${MAX_BUILD_JOBS} \
    MAKEFLAGS=-j${MAX_BUILD_JOBS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git make python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}

RUN git init -q /opt/cutlass \
    && git -C /opt/cutlass remote add origin https://github.com/NVIDIA/cutlass.git \
    && git -C /opt/cutlass fetch --depth 1 origin "${CUTLASS_COMMIT}" \
    && git -C /opt/cutlass checkout -q FETCH_HEAD \
    && bash /opt/cutlass/python/CuTeDSL/setup.sh --cu13

RUN python3 -m pip install "cuda-python==${CUDA_PYTHON_VERSION}" "cuda-bindings==${CUDA_PYTHON_VERSION}" \
    && python3 -m pip install --index-url "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"

WORKDIR /workspace
