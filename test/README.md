# MLIR Pattern Tests

This directory contains comprehensive test files for MLIR transformation passes in the Toy dialect.

## Test Files

### 1. toy-shape-inference.mlir
Tests for the Shape Inference Pass (`-toy-shape-inference`)

**Purpose**: Verifies that the shape inference pass correctly propagates tensor shapes from operands to results.

**Test Cases**:
- `test_add_shape_inference`: Basic AddOp with unranked result
- `test_mul_shape_inference`: Basic MulOp with unranked result
- `test_chained_operations`: Multiple operations with shape propagation
- `test_1d_tensors`: Shape inference for 1D tensors
- `test_higher_rank_tensors`: Shape inference for 3D tensors
- `test_already_ranked`: Verify that ranked results remain unchanged
- `test_complex_computation`: Complex computation graph with mixed ranked/unranked types

**Run**: 
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference
```

### 2. toy-canonicalize.mlir
Tests for the Canonicalize Pass (`-toy-canonicalize`)

**Purpose**: Verifies that the canonicalize pass standardizes the operand order for commutative operations (AddOp, MulOp) based on SSA value pointers.

**Test Cases**:
- `test_add_canonicalization`: Canonicalize AddOp operand order
- `test_mul_canonicalization`: Canonicalize MulOp operand order
- `test_multiple_ops`: Multiple operations requiring canonicalization
- `test_intermediate_values`: Canonicalization with intermediate SSA values
- `test_same_operand`: Operations with identical operands (should remain unchanged)
- `test_chained_commutative`: Chain of commutative operations
- `test_different_shapes`: Canonicalization works with various tensor shapes
- `test_idempotent`: Multiple operations in same function

**Run**: 
```bash
./build/tools/toy-opt test/toy-canonicalize.mlir -toy-canonicalize
```

### 3. toy-cse.mlir
Tests for Common Subexpression Elimination Pass (`-toy-cse`)

### 4. toy-dce.mlir
Tests for Dead Code Elimination Pass (`-toy-dce`)

### 5. toy-constant-fold.mlir
Tests for Constant Folding Pass (`-toy-constant-fold`)

### 6. toy-lowering.mlir
Tests for lowering passes to Arith and Linalg dialects

## Running All Tests

Use the provided test runner script:

```bash
cd test
./run_tests.sh
```

This script will:
1. Check if `toy-opt` is built
2. Run all test files with appropriate passes
3. Report success/failure for each test suite

## Writing New Tests

### FileCheck Syntax

Tests use LLVM's FileCheck utility for pattern matching:

- `// RUN:` - Command to run the test
- `// CHECK:` - Pattern that must appear in output
- `// CHECK-LABEL:` - Marks the start of a new test case
- `%[[VAR]]` - Captures a value for later reference
- `{{.*}}` - Matches any string

### Example Test Structure

```mlir
// RUN: toy-opt %s -toy-shape-inference | FileCheck %s

// CHECK-LABEL: func.func @my_test
func.func @my_test(%arg0: tensor<2x2xf64>) -> tensor<*xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<2x2xf64>
  return %0 : tensor<*xf64>
}
```

## Test Guidelines

1. **Descriptive Names**: Use clear function names that describe what is being tested
2. **Single Responsibility**: Each test function should verify one specific behavior
3. **CHECK Labels**: Always use `CHECK-LABEL` to separate test cases
4. **Coverage**: Test edge cases, boundary conditions, and typical use cases
5. **Comments**: Add comments explaining complex test scenarios

## Integration with CI/CD

These tests can be integrated into GitHub Actions workflows for continuous testing:

```yaml
- name: Run MLIR Tests
  run: |
    cd test
    ./run_tests.sh
```

## Debugging Failed Tests

If a test fails:

1. Run the specific test manually:
   ```bash
   ./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference
   ```

2. Compare actual output with expected CHECK patterns

3. Use `-debug` flag for verbose output:
   ```bash
   ./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference -debug
   ```

4. Use `--print-ir-after-all` to see IR after each transformation:
   ```bash
   ./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference --print-ir-after-all
   ```

## Additional Resources

- [MLIR Testing Guide](https://mlir.llvm.org/docs/TestingGuide/)
- [FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
