# 01 – Dialect Basics

MLIR dialects package custom operations, types, and attributes, letting you model domain abstractions without lowering immediately.

## Understanding Dialects

A **dialect** in MLIR is a logical grouping of operations, types, and attributes that represent a specific domain or abstraction level. Dialects provide:

1. **Namespace isolation**: Each dialect has a unique namespace prefix (e.g., `toy.`, `arith.`, `func.`)
2. **Custom semantics**: Domain-specific operations with specialized behavior
3. **Progressive lowering**: High-level dialects can be transformed to lower-level ones
4. **Modularity**: Mix and match dialects as needed

### Dialect Components

#### Operations (Ops)
Define the computations available in your dialect:
- Input/output types (operands and results)
- Traits (e.g., `Pure`, `Commutative`)
- Custom verification logic
- Print/parse methods for textual IR

#### Types
Custom data types specific to your dialect:
- Example: `tensor<*xf64>` (unranked tensor), `!toy.struct<{...}>`
- Type constraints ensure type safety

#### Attributes
Compile-time constant metadata:
- Examples: shapes, constant values, flags
- Attached to operations or function arguments

## Files of Interest

- `toy/ToyDialect.td` – Dialect metadata (name, namespace, C++ class)
- `toy/ToyOps.td` – Operation definitions using TableGen
- `toy/DialectRegistration.cpp` – Hooks to register the dialect with MLIR
- `toy/Ops.cpp` – Builders, printers, verifiers, and implementation details

## TableGen Deep Dive

MLIR uses **TableGen** (from LLVM) to generate boilerplate C++ code from declarative specifications.

### Dialect Definition

```tablegen
def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "A toy dialect for tutorial purposes";
  let description = [{
    This dialect demonstrates MLIR concepts through simple
    tensor operations.
  }];
}
```

### Operation Definition

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition";
  let description = [{
    Adds two tensors element-wise.
    Example: %result = toy.add %lhs, %rhs : tensor<2x3xf64>
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  
  // Optional: custom assembly format
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}
```

### Operation Traits

Traits provide compile-time guarantees about operation behavior:

- **Pure**: No side effects (enables CSE, DCE optimizations)
- **Commutative**: Operands can be reordered
- **Idempotent**: Multiple applications have same effect as one
- **SameOperandsAndResultType**: All types must match

## Current Implementation

### ToyDialect.td
Defines the `toy` dialect with its namespace and basic metadata.

### ToyOps.td
Defines operations like:
- `Toy_AddOp` – Element-wise addition
- `Toy_MulOp` – Element-wise multiplication

Both are tagged with the `Pure` trait, making them eligible for optimization passes like:
- **Common Subexpression Elimination (CSE)**
- **Dead Code Elimination (DCE)**
- **Constant Folding**

## Build System Integration

After editing `.td` files:

1. **Regenerate TableGen outputs**:
   ```bash
   cd build
   ninja
   ```

2. **Generated files** (in `build/toy/`):
   - `ToyDialect.h.inc` / `ToyDialect.cpp.inc`
   - `ToyOps.h.inc` / `ToyOps.cpp.inc`

3. **Include generated code** in your C++ files:
   ```cpp
   #include "toy/ToyDialect.h"
   #include "toy/ToyOps.h"
   ```

## TODO Ideas

### Basic Extensions
- [ ] Restrict operand/result types to ranked tensors of `f64` using type constraints
- [ ] Add attributes to operations (e.g., `axis` for reduction operations)
- [ ] Implement custom verifiers checking shape compatibility

### New Operations
- [ ] Add `Toy_SubOp` (subtraction)
- [ ] Add `Toy_TransposeOp` with axis permutation attribute
- [ ] Add `Toy_ConstantOp` to represent constant tensors
- [ ] Add `Toy_ReshapeOp` to change tensor shapes

### Type System
- [ ] Define custom tensor type with static shape information
- [ ] Add verification that shapes are compatible across operations
- [ ] Implement type inference interfaces

## References

- [MLIR Dialect Definition](https://mlir.llvm.org/docs/DefiningDialects/)
- [Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [TableGen Documentation](https://llvm.org/docs/TableGen/)
