# mlir_tutorial

A tiny project skeleton that demonstrates how to build an MLIR-based toy dialect, write passes, and run a minimal lowering pipeline. Everything is intentionally stubbed with TODO markers so learners can complete each step while following the docs under `docs/`.

## Project Layout

- `toy/` – Dialect sources, TableGen definitions, and registration hooks.
- `passes/` – Example pass stubs (shape inference & canonicalization).
- `examples/` – Toy source snippet plus helper script for the pipeline.
- `docs/` – Tutorial chapters explaining each stage of the journey.
- `docker/` – Self-contained build environment powered by LLVM/MLIR.
- `CMakeLists.txt` – Configures the project against an existing MLIR build.

## Prerequisites

- LLVM/MLIR build (provide `LLVM_DIR`/`MLIR_DIR` via CMake cache or environment).
- CMake ≥ 3.20 and Ninja (already bundled in the docker image).

## Configure & Build

```bash
apt-get update
apt-get install -y zlib1g-dev libtinfo-dev libedit-dev libxml2-dev
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
ninja
```

The configuration step automatically adds the Toy dialect library (`toy-dialect`) and pass library (`toy-passes`).

## Run the Example Pipeline

1. Build the project (generates `build/tools/toy-opt`).
2. (Optional) Source your MLIR environment for extra tooling like `mlir-translate`.
3. Execute the example script:
   ```bash
   ./examples/pipeline.sh
   ```
  The script feeds `examples/intro.toy` into `toy-opt`, executes the Toy pipeline (`toy-shape-inference`, `toy-canonicalize`, `toy-constant-fold`, `toy-dce`, `toy-cse`), and prints each IR stage. Extend it to finish the lowering story.

## Docker Workflow

```bash
cd docker
./docker_build.sh
# optional: docker run --rm -it -v $(pwd)/..:/workspace mlir-tutorial /bin/bash
```

Inside the container, configure & build exactly as above.

## Learning Objectives

- Understand how MLIR dialects are defined via TableGen and C++ scaffolding.
- Implement operation builders, printers, and verifiers.
- Write analysis / transformation passes and chain them into pipelines.
- Lower custom dialect IR to the LinAlg/Arith dialects.
- Use MLIR tooling (`mlir-opt`, `mlir-translate`, pass pipelines) for debugging.

Dive into `docs/00_overview.md` to get started!
