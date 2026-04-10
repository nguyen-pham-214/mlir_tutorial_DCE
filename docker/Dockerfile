# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=22.04
ARG LLVM_VERSION=llvmorg-18.1.8

FROM ubuntu:${UBUNTU_VERSION} AS builder
ARG LLVM_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build \
    wget \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev \
    libncurses5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch ${LLVM_VERSION} https://github.com/llvm/llvm-project.git
WORKDIR /opt/llvm-project/build
RUN cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm
RUN ninja -j2 install

FROM ubuntu:${UBUNTU_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    ninja-build \
    python3 \
    python3-pip \
    cmake \
    zlib1g \
    libedit2 \
    libxml2 \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llvm /opt/llvm
ENV LLVM_BUILD_DIR=/opt/llvm \
    MLIR_DIR=/opt/llvm/lib/cmake/mlir \
    LLVM_DIR=/opt/llvm/lib/cmake/llvm \
    PATH=/opt/llvm/bin:${PATH}

WORKDIR /workspace
CMD ["/bin/bash"]