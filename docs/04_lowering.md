# 04 – Lowering Toy to Linalg

## 1. What “Lowering” Means in MLIR

*Lowering* is the process of rewriting operations from one dialect into operations of another dialect that sits "closer" to the eventual execution model. In MLIR this always happens **between dialects**, not directly to machine code. Each lowering step:

- Preserves the program’s semantics.
- Chooses a more concrete abstraction level (e.g., from Toy math ops to structured Linalg ops).
- Relies on rewrite patterns so the IR remains valid MLIR after every transformation.

Think of lowering as building a staircase of dialects. We decide where to stop depending on which abstractions we still want to reason about. This chapter focuses on the first backend-facing step: **Toy → Linalg**.

---

## 2. Why Toy → Linalg Is a Natural Boundary

Toy is a high-level, pedagogical dialect with tensor semantics but no explicit execution strategy. Linalg, on the other hand, offers:

- Structured ops (`linalg.generic`, named matmul, etc.) that describe computations over tensors.
- Rich metadata (indexing maps, iterator types) to express how elements are traversed.
- A mature ecosystem of tiling, vectorization, and bufferization passes.

Lowering Toy into Linalg keeps us **within tensor semantics** while unlocking powerful downstream optimizations. Stopping here avoids committing to loops, memory layouts, or hardware yet.

---

## 3. ToyToLinalgPass Overview

File: `passes/ToyToLinalgPass.cpp`

Key characteristics:

- Runs as a `func::FuncOp` pass (nested inside the pipeline).
- Registers rewrite patterns for `toy.add` and `toy.mul`.
- Emits `linalg::GenericOp` with tensor outputs (still on-buffer).
- Declares dependencies on the `arith`, `linalg`, and `tensor` dialects.

Pipeline placement (see `passes/Passes.cpp`):

1. Frontend passes (shape inference, canonicalize, DCE, CSE, constant fold).
2. `toy::createToyToLinalgPass()`.
3. Bufferization passes (handled later).

---

## 4. Pattern-Based Lowering: OpRewritePattern + PatternRewriter

MLIR lowering is typically implemented by registering `OpRewritePattern`s:

```cpp
struct LowerToyAddPattern : public OpRewritePattern<toy::AddOp> {
	LogicalResult matchAndRewrite(toy::AddOp op,
																PatternRewriter &rewriter) const override {
		return lowerPointwiseBinaryOp(
				op.getOperation(), ValueRange{op.getLhs(), op.getRhs()}, rewriter,
				[](OpBuilder &b, Location loc, Value lhs, Value rhs) {
					return b.create<arith::AddFOp>(loc, lhs, rhs);
				});
	}
};
```

A helper (`lowerPointwiseBinaryOp`) encapsulates shared logic:

1. Validate the result type (must be a ranked tensor of floats).
2. Materialize an output tensor via `tensor::EmptyOp`.
3. Build identity indexing maps and parallel iterator metadata.
4. Create a `linalg::GenericOp` and emit the scalar body using the lambda.
5. Replace the Toy op with the new result.

`PatternRewriter` guarantees the IR stays consistent: it tracks replacements, erases old ops, and updates users for you.

---

## 5. Worked Example: `toy.add` → `linalg.generic`

**Input IR (after frontend passes):**

```mlir
func.func @main(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<4xf64> {
	%0 = toy.add %lhs, %rhs : tensor<4xf64>
	return %0 : tensor<4xf64>
}
```

**After `toy-to-linalg` lowering:**

```mlir
func.func @main(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<4xf64> {
	%init = tensor.empty() : tensor<4xf64>
	%res = linalg.generic {
					 indexing_maps = [affine_map<(i) -> (i)>,
														affine_map<(i) -> (i)>,
														affine_map<(i) -> (i)>],
					 iterator_types = ["parallel"]
				 } ins(%lhs, %rhs : tensor<4xf64>, tensor<4xf64>)
					 outs(%init : tensor<4xf64>) {
		^bb0(%a: f64, %b: f64, %c: f64):
			%sum = arith.addf %a, %b : f64
			linalg.yield %sum : f64
	} -> tensor<4xf64>
	return %res : tensor<4xf64>
}
```

Observations:

- The body region now operates on scalar block arguments (`%a`, `%b`).
- `arith.addf` performs the scalar computation.
- `linalg.yield` returns the scalar result to the enclosing generic op.
- No loops or buffers are visible yet — everything remains tensor-level.

---

## 6. Anatomy of the Generated `linalg::GenericOp`

### 6.1 `indexing_maps`

- One affine map per operand (two inputs + one output in this example).
- Identity maps `affine_map<(i) -> (i)>` state that all tensors share the same indexing scheme.
- More complex lowerings can encode broadcasting or permutations by using non-identity maps.

### 6.2 `iterator_types`

- A list of strings (or `IteratorType` enums) describing loop semantics for each dimension.
- Here every dimension is `"parallel"`, meaning the computation has no carried dependencies and can be tiled/vectorized freely.

### 6.3 `tensor.empty` as the output buffer

- Linalg ops require explicit output tensors.
- `tensor.empty` creates an SSA value with the right shape and element type, serving as a pristine output.
- Later bufferization passes turn `tensor.empty` into actual memory allocations (e.g., via `alloc` or stack slots).

---

## 7. Why Use `linalg.generic` Instead of Named Ops?

- `toy.add` and `toy.mul` are **elementwise**; `linalg.generic` natively expresses arbitrary elementwise patterns.
- Named ops (`linalg.add`, `linalg.matmul`, …) come with stricter arity and indexing requirements; many do not exist yet for simple pointwise math.
- Generic keeps the lowering extensible: adding new Toy elementwise ops just means swapping the scalar lambda.
- Downstream transformations (fusion, tiling, vectorization) operate uniformly on `linalg.generic`, so no optimization opportunity is lost.

---

## 8. Preconditions Guaranteed by Frontend Passes

Before the lowering pass runs we rely on invariants established earlier:

1. **Static shapes** – Shape inference ensures every tensor has a concrete shape.
2. **Canonical IR** – Canonicalization normalizes operand ordering and removes trivial patterns.
3. **No dead ops** – DCE strips unused computations, so lowering only touches live values.
4. **No redundant work** – CSE + constant folding keep the IR minimal.

These guarantees let the Toy→Linalg pass focus purely on structural translation, without re-validating the same conditions.

---

## 9. Testing the Lowering Pass

### Test File Structure

The lowering pass has a dedicated test file: `test/toy-lowering.mlir`

This test verifies that Toy operations are correctly transformed into Linalg generic operations with proper indexing maps, iterator types, and scalar computations.

### Running Lowering Tests

**Run the lowering pass:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg
```

**Run the complete lowering pipeline:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir \
  -toy-shape-inference \
  -toy-canonicalize \
  -toy-to-linalg
```

**Run all passes before lowering:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir \
  --pass-pipeline='builtin.module(toy-shape-inference,toy-canonicalize,toy-constant-fold,toy-cse,toy-dce,toy-to-linalg)'
```

### Example Test Cases

**Test 1: Basic AddOp Lowering**
```mlir
// RUN: toy-opt %s -toy-to-linalg | FileCheck %s --check-prefix=CHECK-ADD

// CHECK-ADD-LABEL: func.func @test_add_lowering
func.func @test_add_lowering(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<4xf64> {
  // CHECK-ADD: %[[INIT:.*]] = tensor.empty() : tensor<4xf64>
  // CHECK-ADD: %[[RES:.*]] = linalg.generic
  // CHECK-ADD-SAME: indexing_maps = [#map, #map, #map]
  // CHECK-ADD-SAME: iterator_types = ["parallel"]
  // CHECK-ADD-SAME: ins(%arg0, %arg1 : tensor<4xf64>, tensor<4xf64>)
  // CHECK-ADD-SAME: outs(%[[INIT]] : tensor<4xf64>)
  // CHECK-ADD: ^bb0(%[[A:.*]]: f64, %[[B:.*]]: f64, %[[C:.*]]: f64):
  // CHECK-ADD:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f64
  // CHECK-ADD:   linalg.yield %[[SUM]] : f64
  %0 = "toy.add"(%lhs, %rhs) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  // CHECK-ADD: return %[[RES]]
  return %0 : tensor<4xf64>
}
```

**Test 2: Basic MulOp Lowering**
```mlir
// CHECK-MUL-LABEL: func.func @test_mul_lowering
func.func @test_mul_lowering(%lhs: tensor<2x3xf64>, %rhs: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // CHECK-MUL: %[[INIT:.*]] = tensor.empty() : tensor<2x3xf64>
  // CHECK-MUL: %[[RES:.*]] = linalg.generic
  // CHECK-MUL-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-MUL: ^bb0(%[[A:.*]]: f64, %[[B:.*]]: f64, %[[C:.*]]: f64):
  // CHECK-MUL:   %[[PROD:.*]] = arith.mulf %[[A]], %[[B]] : f64
  // CHECK-MUL:   linalg.yield %[[PROD]] : f64
  %0 = "toy.mul"(%lhs, %rhs) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
```

**Test 3: Chained Operations**
```mlir
// CHECK-CHAIN-LABEL: func.func @test_chained_lowering
func.func @test_chained_lowering(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  // First operation: add
  // CHECK-CHAIN: %[[INIT1:.*]] = tensor.empty() : tensor<4xf64>
  // CHECK-CHAIN: %[[ADD:.*]] = linalg.generic
  // CHECK-CHAIN:   arith.addf
  %0 = "toy.add"(%arg0, %arg1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  
  // Second operation: mul using result of add
  // CHECK-CHAIN: %[[INIT2:.*]] = tensor.empty() : tensor<4xf64>
  // CHECK-CHAIN: %[[MUL:.*]] = linalg.generic
  // CHECK-CHAIN:   arith.mulf
  %1 = "toy.mul"(%0, %arg1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  
  // CHECK-CHAIN: return %[[MUL]]
  return %1 : tensor<4xf64>
}
```

**Test 4: Higher-Rank Tensors**
```mlir
// CHECK-RANK-LABEL: func.func @test_higher_rank
func.func @test_higher_rank(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  // CHECK-RANK: tensor.empty() : tensor<2x3x4xf64>
  // CHECK-RANK: linalg.generic
  // CHECK-RANK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
  %0 = "toy.add"(%arg0, %arg1) : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}
```

### Key Verification Points

When testing lowering, verify these critical elements:

1. **tensor.empty creation**: Each linalg.generic must have an output tensor
   ```mlir
   // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<...>
   ```

2. **Indexing maps**: Identity maps for elementwise operations
   ```mlir
   // CHECK: indexing_maps = [#map, #map, #map]
   ```

3. **Iterator types**: Parallel for all dimensions in elementwise ops
   ```mlir
   // CHECK: iterator_types = ["parallel", "parallel"]
   ```

4. **Scalar body**: Correct arithmetic operation with proper types
   ```mlir
   // CHECK: arith.addf %{{.*}}, %{{.*}} : f64
   // CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
   ```

5. **linalg.yield**: Proper yielding of computed values
   ```mlir
   // CHECK: linalg.yield %{{.*}} : f64
   ```

### Testing Best Practices

**1. Test Different Tensor Ranks**
```mlir
// 1D tensors
func.func @test_1d(%a: tensor<10xf64>, %b: tensor<10xf64>) -> tensor<10xf64>

// 2D tensors
func.func @test_2d(%a: tensor<3x4xf64>, %b: tensor<3x4xf64>) -> tensor<3x4xf64>

// 3D tensors
func.func @test_3d(%a: tensor<2x3x4xf64>, %b: tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
```

**2. Test Operation Combinations**
```mlir
// Mix add and mul operations
func.func @test_mixed(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>) -> tensor<4xf64> {
  %0 = "toy.add"(%a, %b) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  %1 = "toy.mul"(%0, %c) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %1 : tensor<4xf64>
}
```

**3. Test with Frontend Passes**

Always test lowering in combination with frontend passes:
```bash
# Run shape inference first
./build/tools/toy-opt test/toy-lowering.mlir \
  -toy-shape-inference \
  -toy-to-linalg

# Run full frontend pipeline
./build/tools/toy-opt test/toy-lowering.mlir \
  -toy-shape-inference \
  -toy-canonicalize \
  -toy-constant-fold \
  -toy-cse \
  -toy-dce \
  -toy-to-linalg
```

### Debugging Lowering Issues

**1. Check if Toy ops are recognized:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg --debug
```

**2. Verify indexing maps are created correctly:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg \
  --mlir-print-ir-after-all 2>&1 | grep "indexing_maps"
```

**3. Examine the scalar computation region:**
```bash
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg \
  --mlir-print-ir-after-all 2>&1 | grep -A 5 "^bb0"
```

**4. Validate linalg.generic structure:**
```bash
# Check if linalg ops are valid
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg --verify-each
```

**5. Compare with expected output:**
```bash
# Generate expected IR manually
cat > expected.mlir << 'EOF'
// Expected linalg IR here
EOF

# Compare with actual output
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg > actual.mlir
diff expected.mlir actual.mlir
```

### Common Issues and Solutions

**Issue 1: Missing tensor.empty**
```
Error: linalg.generic expects explicit output tensor
```
**Solution**: Ensure the pattern creates tensor.empty before linalg.generic

**Issue 2: Incorrect iterator types**
```
Error: iterator_types size doesn't match rank
```
**Solution**: Match iterator_types count with tensor rank (one "parallel" per dimension)

**Issue 3: Indexing map mismatch**
```
Error: indexing_maps size doesn't match operand count
```
**Solution**: Create one affine_map per input + one for output (e.g., 3 maps for binary ops)

**Issue 4: Type mismatch in scalar body**
```
Error: block argument type doesn't match tensor element type
```
**Solution**: Ensure block argument types match tensor element types (f64 for tensor<...xf64>)

### Quick Reference: Lowering Test Commands

```bash
# Basic lowering test
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg

# With frontend passes
./build/tools/toy-opt test/toy-lowering.mlir \
  -toy-shape-inference -toy-to-linalg

# Full pipeline
./build/tools/toy-opt test/toy-lowering.mlir \
  --pass-pipeline='builtin.module(toy-shape-inference,toy-canonicalize,toy-to-linalg)'

# Debug mode
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg --debug

# With verification
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg --verify-each

# Print IR after transformation
./build/tools/toy-opt test/toy-lowering.mlir -toy-to-linalg \
  --mlir-print-ir-after-all
```

---

## 10. Summary Checklist

- Understand that lowering is a dialect-to-dialect rewrite, not a jump to machine code.
- Treat Linalg as the first backend layer that still manipulates tensors.
- Implement lowering via targeted `OpRewritePattern`s and a shared helper.
- Ensure `indexing_maps`, `iterator_types`, and `tensor.empty` are set coherently.
- Prefer `linalg.generic` for elementwise Toy ops to keep the lowering general.
- Run frontend passes first to provide the invariants the lowering assumes.
- **Write comprehensive tests** to verify correct transformation to Linalg ops.
- **Test different tensor ranks** and operation combinations.
- **Debug systematically** using MLIR's verification and printing tools.

With Toy→Linalg implemented and thoroughly tested, the IR is ready for bufferization, tiling, and eventually lower dialects. The next stages will move from tensors to buffers and, eventually, to hardware-friendly representations.
