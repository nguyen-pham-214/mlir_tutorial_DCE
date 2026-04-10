# Docker environment

This image builds LLVM/MLIR from source so the tutorial can be configured without touching your host setup.

## Build the image

```bash
cd docker
./docker_build.sh
```

Environment variables:
- `IMAGE_NAME` - tag for the produced image (default `mlir_tutorial:latest`).
- `LLVM_VERSION` - git tag for llvm-project (default `llvmorg-18.1.8`).
- `UBUNTU_VERSION` - base image version (default `22.04`).

## Run the docker

```bash
./scripts/docker_run.sh
```
