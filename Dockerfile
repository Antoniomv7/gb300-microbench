# make image supplies the pinned versions from VERSIONS.env.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUTLASS_COMMIT
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX_URL
ARG CUDA_PYTHON_VERSION
ARG MAX_BUILD_JOBS

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
