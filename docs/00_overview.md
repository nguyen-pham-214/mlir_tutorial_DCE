# 00 – Overview

This tutorial walks through the essential MLIR concepts using the Toy dialect skeleton in this repository.

## What is MLIR?

**MLIR (Multi-Level Intermediate Representation)** is a compiler infrastructure that enables the design of reusable and extensible compiler architectures. Unlike traditional compilers that typically have a single IR level (like LLVM IR), MLIR supports multiple levels of abstraction through **dialects**.

### Core Concepts

#### 1. Operations (Ops)
Operations are the fundamental units of computation in MLIR. Each operation:
- Has a unique name (e.g., `toy.add`, `arith.addf`)
- Takes **operands** (input values) and produces **results** (output values)
- Can have **attributes** (compile-time constants like shape, data type)
- Can have **regions** containing nested operations (for control flow)
- Must belong to a specific **dialect**

Example:
```mlir
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
```

#### 2. Dialects
A **dialect** is a namespace for a cohesive set of operations, types, and attributes. Dialects allow you to:
- Model domain-specific abstractions (e.g., `toy`, `linalg`, `tensor`)
- Progressively lower high-level concepts to low-level representations
- Mix multiple dialects in a single module

Common MLIR dialects:
- `func` – Functions and control flow
- `arith` – Arithmetic operations
- `tensor` – Tensor operations
- `memref` – Memory references (buffers)
- `llvm` – LLVM IR operations

#### 3. Types
Types describe the data flowing through operations:
- **Built-in types**: `i32`, `f64`, `tensor<2x3xf64>`, `memref<10xf32>`
- **Dialect-specific types**: Custom types defined by dialects
- **Type inference**: MLIR can infer result types based on operands

#### 4. Regions and Blocks
- **Region**: Container for a CFG (Control Flow Graph) of blocks
- **Block**: A sequence of operations with a single entry point
- Used for structured control flow (loops, conditionals)

#### 5. Passes
Passes are transformations or analyses that operate on MLIR IR:
- **Analysis passes**: Gather information (e.g., shape inference)
- **Transformation passes**: Modify IR (e.g., constant folding, optimization)
- **Lowering passes**: Convert between dialects (e.g., Toy → Arith → LLVM)

#### 6. Pass Manager
The **Pass Manager** orchestrates the execution of passes:
- Executes passes in a pipeline
- Manages pass dependencies
- Supports nested pass managers for different operation types

## Goals

- Understand why MLIR introduces multiple IR levels.
- See how dialects capture domain-specific abstractions.
- Learn how passes and pipelines transform IR toward lower levels like LLVM.
- Master the fundamentals of writing custom dialects and passes.

## MLIR Architecture Overview

```
High-Level Dialect (Toy)
        ↓
  Shape Inference
  Canonicalization
  ConstantFold
  DeadCodeElimination
  CommonSubexpressionElimination
        ↓
Mid-Level Dialects (Tensor, Linalg)
        ↓
  Bufferization
        ↓
Low-Level Dialects (MemRef, Arith, SCF)
        ↓
  Conversion to LLVM
        ↓
    LLVM Dialect
        ↓
  Translation to LLVM IR
        ↓
   Machine Code
```

## Reading Order

1. `01_dialect.md` – Dialect basics and TableGen.
2. `02_ops.md` – Defining operations, types, and inference hooks.
3. `03_passes.md` – Writing passes and chaining them.
4. `04_lowering.md` – Mapping Toy IR to standard / LLVM.
5. `05_debugging.md` – Tooling tips.
6. `06_exercises.md` – Guided exercises to extend the project.

Throughout the docs you will find **TODO** callouts referencing files under `toy/`, `passes/`, and `examples/`.

## Project Structure

```
mlir_tutorial/
├── toy/              # Dialect definition (TableGen + C++)
│   ├── ToyDialect.td  # Dialect metadata
│   └── ToyOps.td      # Operation definitions
├── passes/           # Transformation and analysis passes
│   ├── ShapeInferencePass.cpp
│   ├── CanonicalizePass.cpp
│   ├── ToyConstantFoldPass.cpp
│   ├── ToyCSEPass.cpp
│   └── ToyDCEPass.cpp
├── include/          # Header files
├── tools/            # toy-opt tool for testing
└── examples/         # Example files
    ├── intro.toy     # NOTE: This is actually MLIR IR, not Toy source!
    └── pipeline.sh   # Test pipeline script
```

## File Extensions: `.toy` vs `.mlir`

**`.toy`** = Custom high-level Toy language source (**Python-like** syntax: `def`, `var`, etc.)  
**`.mlir`** = MLIR Intermediate Representation (IR syntax: `func.func`, `%0 =`, etc.)

**In this tutorial**: We use **`.mlir`** format (MLIR IR). Files in `examples/` contain MLIR IR and can be processed directly with `toy-opt` or `mlir-opt`.
