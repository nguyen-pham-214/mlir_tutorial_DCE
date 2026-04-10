#!/bin/bash

CONTAINER_NAME="mlir_tutorial_${USER}"

docker run -it --name "${CONTAINER_NAME}" -v ${PWD}:/workspace mlir-tutorial:latest /bin/bash