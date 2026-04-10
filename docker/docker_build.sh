#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME=${IMAGE_NAME:-mlir-tutorial:latest}
LLVM_VERSION=${LLVM_VERSION:-llvmorg-18.1.8}
UBUNTU_VERSION=${UBUNTU_VERSION:-22.04}

exec docker build \
    --build-arg LLVM_VERSION="${LLVM_VERSION}" \
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"