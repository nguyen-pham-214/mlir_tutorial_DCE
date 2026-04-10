// RUN: toy-opt %s -toy-canonicalize | FileCheck %s

// Test canonicalization of AddOp - should reorder operands based on SSA value
// CHECK-LABEL: func.func @test_add_canonicalization
func.func @test_add_canonicalization(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // The operands should be reordered if %arg1's pointer > %arg0's pointer
  // CHECK: %[[RESULT:.*]] = "toy.add"({{%arg[0-1]}}, {{%arg[0-1]}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg1, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<2x2xf64>
}

// Test canonicalization of MulOp - should reorder operands based on SSA value
// CHECK-LABEL: func.func @test_mul_canonicalization
func.func @test_mul_canonicalization(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // The operands should be reordered if %arg1's pointer > %arg0's pointer
  // CHECK: %[[RESULT:.*]] = "toy.mul"({{%arg[0-1]}}, {{%arg[0-1]}}) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %0 = "toy.mul"(%arg1, %arg0) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<3x3xf64>
}

// Test multiple operations that need canonicalization
// CHECK-LABEL: func.func @test_multiple_ops
func.func @test_multiple_ops(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: %[[ADD:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg2, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: %[[MUL:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %1 = "toy.mul"(%arg1, %0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[MUL]]
  return %1 : tensor<2x2xf64>
}

// Test canonicalization with intermediate values
// CHECK-LABEL: func.func @test_intermediate_values
func.func @test_intermediate_values(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK: %[[ADD1:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  %0 = "toy.add"(%arg0, %arg1) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  // CHECK: %[[MUL:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  %1 = "toy.mul"(%0, %arg0) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  // CHECK: %[[ADD2:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  %2 = "toy.add"(%1, %0) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  // CHECK: return %[[ADD2]]
  return %2 : tensor<4x4xf64>
}

// Test that same-operand operations remain unchanged
// CHECK-LABEL: func.func @test_same_operand
func.func @test_same_operand(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg0, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<2x2xf64>
}

// Test chained commutative operations
// CHECK-LABEL: func.func @test_chained_commutative
func.func @test_chained_commutative(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>, %arg2: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // All operations should be canonicalized to have deterministic ordering
  // CHECK: %[[ADD1:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %0 = "toy.add"(%arg2, %arg1) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK: %[[MUL1:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %1 = "toy.mul"(%0, %arg0) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK: %[[ADD2:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %2 = "toy.add"(%1, %arg1) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK: %[[MUL2:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %3 = "toy.mul"(%2, %0) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  // CHECK: return %[[MUL2]]
  return %3 : tensor<3x3xf64>
}

// Test canonicalization with different tensor shapes
// CHECK-LABEL: func.func @test_different_shapes
func.func @test_different_shapes(%arg0: tensor<1x5xf64>, %arg1: tensor<1x5xf64>) -> tensor<1x5xf64> {
  // CHECK: %[[ADD:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<1x5xf64>
  %0 = "toy.add"(%arg1, %arg0) : (tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<1x5xf64>
  // CHECK: %[[MUL:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<1x5xf64>
  %1 = "toy.mul"(%0, %arg0) : (tensor<1x5xf64>, tensor<1x5xf64>) -> tensor<1x5xf64>
  // CHECK: return %[[MUL]]
  return %1 : tensor<1x5xf64>
}

// Test that operations are idempotent after canonicalization
// CHECK-LABEL: func.func @test_idempotent
func.func @test_idempotent(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2x2xf64>) {
  // CHECK: %[[ADD:.*]] = "toy.add"({{.*}}, {{.*}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg1, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: %[[MUL:.*]] = "toy.mul"({{.*}}, {{.*}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %1 = "toy.mul"(%arg1, %arg0) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[ADD]], %[[MUL]]
  return %0, %1 : tensor<2x2xf64>, tensor<2x2xf64>
}
