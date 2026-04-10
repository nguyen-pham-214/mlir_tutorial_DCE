# 03 – Passes & Pipelines

Passes let you analyze or transform MLIR modules. This project demonstrates several types of passes for the Toy dialect.

## Understanding MLIR Passes

A **pass** is a unit of compilation that performs analysis or transformation on MLIR IR. Passes are the building blocks of compiler pipelines.

### Types of Passes

#### 1. Analysis Passes
Gather information about the IR without modifying it:
- **Shape Inference**: Deduce tensor shapes from operations
- **Alias Analysis**: Determine which memory references may overlap
- **Liveness Analysis**: Track variable lifetimes

#### 2. Transformation Passes
Modify the IR to optimize or prepare for lowering:
- **Constant Folding**: Evaluate constant expressions at compile time
- **Common Subexpression Elimination (CSE)**: Remove duplicate computations
- **Dead Code Elimination (DCE)**: Remove unused operations
- **Canonicalization**: Normalize IR to a canonical form

#### 3. Conversion/Lowering Passes
Transform operations from one dialect to another:
- Toy → Linalg
- Linalg → SCF + MemRef
- MemRef → LLVM

### Pass Infrastructure

MLIR provides base classes for different pass types:

```cpp
// Function-level pass
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    // Transform the function
  }
};

// Module-level pass
struct MyModulePass : public PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Transform the entire module
  }
};
```

## Project Passes

This tutorial includes five custom passes:

### 1. Shape Inference Pass
**File**: `passes/ShapeInferencePass.cpp`  
**Header**: `include/ToyShapeInferencePass.h`

**Purpose**: Infer and update tensor shapes throughout the IR.

**Key Concepts**:
- Walks the IR in topological order
- Propagates shape information from inputs to outputs
- Updates unranked tensors with concrete shapes

**Example**:
```mlir
// Before
func.func @example(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %0 = toy.add %arg0, %arg0 : tensor<*xf64>
  return %0 : tensor<*xf64>
}

// After shape inference
func.func @example(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %0 = toy.add %arg0, %arg0 : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
```

### 2. Canonicalization Pass
**File**: `passes/CanonicalizePass.cpp`  
**Header**: `include/ToyCanonicalizePass.h`

**Purpose**: Normalize IR using rewrite patterns.

**Key Concepts**:
- Pattern-based rewriting
- `RewritePatternSet` to collect patterns
- `applyPatternsAndFoldGreedily` to apply patterns iteratively

**Common Patterns**:
- Commutative operation normalization
- Folding identity operations
- Algebraic simplifications

### 3. Constant Fold Pass
**File**: `passes/ToyConstantFoldPass.cpp`  
**Header**: `include/ToyConstantFoldPass.h`

**Purpose**: Evaluate operations on constants at compile time.

**Example**:
```mlir
// Before
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
%1 = toy.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
%2 = toy.add %0, %1 : tensor<2x2xf64>

// After constant folding
%2 = toy.constant dense<[[6.0, 8.0], [10.0, 12.0]]> : tensor<2x2xf64>
```

### 4. Common Subexpression Elimination (CSE)
**File**: `passes/ToyCSEPass.cpp`  
**Header**: `include/ToyCSEPass.h`

**Purpose**: Eliminate redundant computations.

**Example**:
```mlir
// Before
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %0, %arg2 : tensor<2x3xf64>
%2 = toy.add %arg0, %arg1 : tensor<2x3xf64>  // Duplicate!
%3 = toy.mul %2, %arg3 : tensor<2x3xf64>

// After CSE
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %0, %arg2 : tensor<2x3xf64>
%3 = toy.mul %0, %arg3 : tensor<2x3xf64>  // Reuses %0
```

**Requirements**:
- Operations must have `Pure` trait (no side effects)
- Identical operands and attributes

### 5. Dead Code Elimination (DCE)
**File**: `passes/ToyDCEPass.cpp`  
**Header**: `include/ToyDCEPass.h`

**Purpose**: Remove operations whose results are never used.

**Example**:
```mlir
// Before
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %arg0, %arg1 : tensor<2x3xf64>  // Unused!
return %0 : tensor<2x3xf64>

// After DCE
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
return %0 : tensor<2x3xf64>
```

## Key Files

### Headers
- `include/ToyShapeInferencePass.h`
- `include/ToyCanonicalizePass.h`
- `include/ToyConstantFoldPass.h`
- `include/ToyCSEPass.h`
- `include/ToyDCEPass.h`
- `include/toy/Transforms/Passes.h` – Centralized pass declarations

### Implementation
- `passes/ShapeInferencePass.cpp`
- `passes/CanonicalizePass.cpp`
- `passes/ToyConstantFoldPass.cpp`
- `passes/ToyCSEPass.cpp`
- `passes/ToyDCEPass.cpp`
- `passes/Passes.cpp` – Pass registration and pipelines

## Pass Registration

Passes must be registered to be used:

```cpp
// In Passes.cpp
void toy::registerToyPasses() {
  detail::registerToyShapeInferencePass();
  detail::registerToyCanonicalizePass();
  detail::registerToyConstantFoldPass();
  detail::registerToyCSEPass();
  detail::registerToyDCEPass();
}
```

## Pass Pipelines

A **pipeline** is an ordered sequence of passes:

```cpp
void registerToyPassPipelines() {
  PassPipelineRegistration<>("toy-full",
    "Complete Toy optimization pipeline",
    [](OpPassManager &pm) {
      pm.addPass(createToyShapeInferencePass());
      pm.addPass(createToyCanonicalizePass());
      pm.addPass(createToyConstantFoldPass());
      pm.addPass(createToyCSEPass());
      pm.addPass(createToyDCEPass());
    });
}
```

### Running Pipelines

```bash
# Run individual pass
./build/tools/toy-opt examples/intro.toy --toy-shape-inference

# Run complete pipeline
./examples/pipeline.sh
```

## Writing a New Pass

### Step 1: Create Header File

Create `include/MyNewPass.h`:
```cpp
#ifndef TOY_MY_NEW_PASS_H
#define TOY_MY_NEW_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {
std::unique_ptr<mlir::Pass> createMyNewPass();

namespace detail {
void registerMyNewPass();
}
}

#endif
```

### Step 2: Implement the Pass

Create `passes/MyNewPass.cpp`:
```cpp
#include "MyNewPass.h"
#include "toy/ToyOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace toy;

namespace {
struct MyNewPass : public PassWrapper<MyNewPass, 
                                      OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyNewPass)
  
  StringRef getArgument() const final { return "my-new-pass"; }
  StringRef getDescription() const final { 
    return "Description of my pass"; 
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    
    // Walk all operations in the function
    func.walk([&](Operation *op) {
      // Your transformation logic here
    });
  }
};
}

std::unique_ptr<Pass> toy::createMyNewPass() {
  return std::make_unique<MyNewPass>();
}

void toy::detail::registerMyNewPass() {
  PassRegistration<MyNewPass>();
}
```

### Step 3: Register the Pass

Update `passes/Passes.cpp`:
```cpp
#include "MyNewPass.h"

void toy::registerToyPasses() {
  // ... existing passes ...
  detail::registerMyNewPass();
}
```

### Step 4: Update CMakeLists.txt

Add your new file to `passes/CMakeLists.txt`:
```cmake
add_mlir_library(toy-passes
  # ... existing files ...
  MyNewPass.cpp
)
```

### Step 5: Build and Test

```bash
cd build
ninja
./tools/toy-opt --help | grep my-new-pass
```

## Pattern Rewriting Example

For canonicalization and optimization passes:

```cpp
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
// Pattern to fold mul by 1.0
struct FoldMulByOne : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MulOp op, 
                                PatternRewriter &rewriter) const override {
    // Check if one operand is constant 1.0
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    if (lhsConst && isOne(lhsConst)) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (rhsConst && isOne(rhsConst)) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    
    return failure();
  }
};

void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  patterns.add<FoldMulByOne>(&getContext());
  
  if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                          std::move(patterns)))) {
    signalPassFailure();
  }
}
}
```

## TODO Exercises

- [ ] Implement `runOnOperation()` in ShapeInferencePass with shape propagation logic
- [ ] Add canonicalization patterns for commutative operations
- [ ] Implement constant folding for Add and Mul operations
- [ ] Test CSE with duplicate operations
- [ ] Verify DCE removes unused operations

## Testing Your Passes

### Test Infrastructure

This project includes comprehensive test files for all major passes:

| Pass                | Test File                       | Purpose                                    |
| ------------------- | ------------------------------- | ------------------------------------------ |
| Shape Inference     | `test/toy-shape-inference.mlir` | Verify tensor shape propagation            |
| Canonicalization    | `test/toy-canonicalize.mlir`    | Verify operand ordering and normalization  |
| Constant Folding    | `test/toy-constant-fold.mlir`   | Verify compile-time constant evaluation    |
| CSE                 | `test/toy-cse.mlir`             | Verify duplicate computation elimination   |
| DCE                 | `test/toy-dce.mlir`             | Verify unused operation removal            |

### Running Tests

**Run individual test:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference
```

**Run all tests:**
```bash
./scripts/run_tests.sh
```

### Writing Test Cases

Tests use **FileCheck** syntax to verify transformations. FileCheck is LLVM's pattern matching tool for test files.

**Basic FileCheck Example:**
```mlir
// RUN: toy-opt %s -toy-shape-inference | FileCheck %s

// CHECK-LABEL: func.func @my_test
func.func @my_test(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
```

**FileCheck Directives:**
- `// RUN:` – Command to execute for this test
- `// CHECK:` – Pattern that must appear in output
- `// CHECK-LABEL:` – Marks start of a test case (resets pattern matching)
- `%[[VAR]]` – Captures a value for later reference
- `{{.*}}` – Matches any string (wildcard)
- `// CHECK-NOT:` – Pattern that must NOT appear
- `// CHECK-NEXT:` – Pattern must appear on immediately next line

### Test Guidelines

Follow these best practices when writing tests:

1. **Descriptive Names**: Use clear function names that describe what is being tested
   ```mlir
   func.func @test_add_with_constants() { ... }
   func.func @test_shape_inference_chained_ops() { ... }
   ```

2. **Single Responsibility**: Each test function should verify one specific behavior
   ```mlir
   // Good: Tests one thing
   func.func @test_add_canonicalization() { ... }
   
   // Avoid: Tests too many things
   func.func @test_everything() { ... }
   ```

3. **CHECK Labels**: Always use `CHECK-LABEL` to separate test cases
   ```mlir
   // CHECK-LABEL: func.func @test_one
   func.func @test_one() { ... }
   
   // CHECK-LABEL: func.func @test_two
   func.func @test_two() { ... }
   ```

4. **Coverage**: Test edge cases, boundary conditions, and typical use cases
   - Empty tensors
   - 1D, 2D, 3D+ tensors
   - Chained operations
   - Already optimized IR (no-op tests)

5. **Documentation**: Add comments explaining complex test scenarios
   ```mlir
   // Test that shape inference works through multiple operations
   // where intermediate results have unranked types
   ```

### Shape Inference Test Patterns

The shape inference pass converts unranked tensor types (`tensor<*xf64>`) to ranked types based on operand shapes.

**Example test case:**
```mlir
// CHECK-LABEL: func.func @test_add_shape_inference
func.func @test_add_shape_inference(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // Operation has unranked result, but operands have concrete shapes
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  %0 = "toy.add"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
```

**Key patterns to test:**
- Basic AddOp and MulOp with unranked results
- Chained operations with shape propagation
- Various tensor ranks (1D, 2D, 3D, higher dimensions)
- Already ranked results (should remain unchanged)
- Complex computation graphs with mixed ranked/unranked types

**Test cases in `toy-shape-inference.mlir`:**
```mlir
test_add_shape_inference        // Basic AddOp shape inference
test_mul_shape_inference        // Basic MulOp shape inference
test_chained_operations         // Multiple ops in sequence
test_1d_tensors                 // 1D tensor shapes
test_higher_rank_tensors        // 3D and higher dimensions
test_already_ranked             // No-op when already ranked
test_complex_computation        // Complex computation graph
```

### Canonicalize Test Patterns

The canonicalize pass normalizes IR by standardizing operand order for commutative operations.

**Example test case:**
```mlir
// CHECK-LABEL: func.func @test_add_canonicalization
func.func @test_add_canonicalization(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // Operands may be reordered based on SSA value pointers
  // CHECK: %[[RESULT:.*]] = "toy.add"({{%arg[0-1]}}, {{%arg[0-1]}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg1, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<2x2xf64>
}
```

**Key patterns to test:**
- Operand reordering for AddOp and MulOp (commutative operations)
- Multiple operations requiring canonicalization
- Intermediate SSA values
- Same-operand operations (idempotent - should remain unchanged)
- Chained commutative operations
- Various tensor shapes

**Test cases in `toy-canonicalize.mlir`:**
```mlir
test_add_canonicalization       // AddOp operand reordering
test_mul_canonicalization       // MulOp operand reordering
test_multiple_ops               // Multiple ops in one function
test_intermediate_values        // Canonicalize with SSA values
test_same_operand               // Operations like add(%0, %0)
test_chained_commutative        // Chain of commutative operations
test_different_shapes           // Various tensor dimensions
test_idempotent                 // Multiple operations per function
```

### Constant Folding Test Patterns

**Example test case:**
```mlir
// CHECK-LABEL: func.func @test_constant_fold_add
func.func @test_constant_fold_add() -> tensor<2x2xf64> {
  %c0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %c1 = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
  // CHECK: %[[RESULT:.*]] = arith.constant dense<{{\[\[}}6.{{.*}}, 8.{{.*}}{{\]\]}}> : tensor<2x2xf64>
  %sum = "toy.add"(%c0, %c1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[RESULT]]
  return %sum : tensor<2x2xf64>
}
```

### CSE Test Patterns

**Example test case:**
```mlir
// CHECK-LABEL: func.func @test_cse_duplicate_add
func.func @test_cse_duplicate_add(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // First operation
  // CHECK: %[[ADD:.*]] = "toy.add"(%arg0, %arg1)
  %0 = "toy.add"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %1 = "toy.mul"(%0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // Duplicate operation - should be eliminated
  // CHECK-NOT: "toy.add"(%arg0, %arg1)
  %2 = "toy.add"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // Should reuse %0
  // CHECK: "toy.mul"(%[[ADD]], %arg1)
  %3 = "toy.mul"(%2, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  return %3 : tensor<2x2xf64>
}
```

### DCE Test Patterns

**Example test case:**
```mlir
// CHECK-LABEL: func.func @test_dce_unused_op
func.func @test_dce_unused_op(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %0 = "toy.add"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // This operation is unused and should be eliminated
  // CHECK-NOT: "toy.mul"(%arg0, %arg1)
  %1 = "toy.mul"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return
  return %0 : tensor<2x2xf64>
}
```

### Debugging Failed Tests

When a test fails, follow these steps:

**1. Run the test manually and examine output:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference
```

**2. Use debug flags for verbose output:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference -debug
```

**3. Print IR after each transformation:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference \
  --mlir-print-ir-after-all
```

**4. Print IR before and after:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all
```

**5. Use diff to compare expected vs actual:**
```bash
# Generate expected output
cat > expected.mlir << 'EOF'
// Expected IR here
EOF

# Run pass and compare
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference > actual.mlir
diff expected.mlir actual.mlir
```

**6. Enable pass statistics:**
```bash
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference \
  --mlir-pass-statistics
```

### Quick Reference: Test Commands

```bash
# Run single test
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference

# Run all tests
./test/run_tests.sh

# Run with debug output
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference -debug

# Run with IR printing
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference \
  --mlir-print-ir-after-all

# Run with pass statistics
./build/tools/toy-opt test/toy-shape-inference.mlir -toy-shape-inference \
  --mlir-pass-statistics

# Test specific pass pipeline
./build/tools/toy-opt test/toy-shape-inference.mlir \
  --pass-pipeline='builtin.module(toy-shape-inference,toy-canonicalize)'
```

For more details on test structure and FileCheck syntax, see `test/README.md`.

## Debugging Passes

### Print IR Before/After

```cpp
void runOnOperation() override {
  auto func = getOperation();
  
  llvm::outs() << "Before transformation:\n";
  func.dump();
  
  // Your transformation
  
  llvm::outs() << "After transformation:\n";
  func.dump();
}
```

### Enable Diagnostics

```cpp
if (someConditionFails) {
  return op->emitError("Detailed error message here");
}
```

## References

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Diagnostic Infrastructure](https://mlir.llvm.org/docs/Diagnostics/)
