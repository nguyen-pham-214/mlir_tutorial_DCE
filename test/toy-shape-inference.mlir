// RUN: toy-opt %s -toy-shape-inference | FileCheck %s

// Test shape inference for AddOp with unranked tensors
// CHECK-LABEL: func.func @test_add_shape_inference
func.func @test_add_shape_inference(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<*xf64> {
  %0 = "toy.add"(%arg0, %arg1)
      : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  return %0 : tensor<*xf64>
}


// Test shape inference for MulOp with unranked tensors
// CHECK-LABEL: func.func @test_mul_shape_inference
func.func @test_mul_shape_inference(%arg0: tensor<4x5xf64>, %arg1: tensor<4x5xf64>) -> tensor<*xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.mul"(%arg0, %arg1) : (tensor<4x5xf64>, tensor<4x5xf64>) -> tensor<4x5xf64>
  %0 = "toy.mul"(%arg0, %arg1) : (tensor<4x5xf64>, tensor<4x5xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<4x5xf64>
  return %0 : tensor<*xf64>
}

// Test shape inference with multiple operations
// CHECK-LABEL: func.func @test_chained_operations
func.func @test_chained_operations(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>, %arg2: tensor<3x3xf64>) -> tensor<*xf64> {
  // CHECK: %[[ADD:.*]] = "toy.add"(%arg0, %arg1) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %0 = "toy.add"(%arg0, %arg1) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<*xf64>
  // CHECK: %[[MUL:.*]] = "toy.mul"(%[[ADD]], %arg2) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  %1 = "toy.mul"(%0, %arg2) : (tensor<*xf64>, tensor<3x3xf64>) -> tensor<*xf64>
  // CHECK: return %[[MUL]] : tensor<3x3xf64>
  return %1 : tensor<*xf64>
}

// Test shape inference with 1D tensors
// CHECK-LABEL: func.func @test_1d_tensors
func.func @test_1d_tensors(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<*xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg1) : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xf64>
  %0 = "toy.add"(%arg0, %arg1) : (tensor<10xf64>, tensor<10xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<10xf64>
  return %0 : tensor<*xf64>
}

// Test shape inference with higher rank tensors
// CHECK-LABEL: func.func @test_higher_rank_tensors
func.func @test_higher_rank_tensors(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<*xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.mul"(%arg0, %arg1) : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  %0 = "toy.mul"(%arg0, %arg1) : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<*xf64>
  // CHECK: return %[[RESULT]] : tensor<2x3x4xf64>
  return %0 : tensor<*xf64>
}

// Test that already ranked results remain unchanged
// CHECK-LABEL: func.func @test_already_ranked
func.func @test_already_ranked(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: %[[RESULT:.*]] = "toy.add"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %0 = "toy.add"(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[RESULT]] : tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

// Test complex computation with mixed ranked/unranked
// CHECK-LABEL: func.func @test_complex_computation
func.func @test_complex_computation(%arg0: tensor<5x5xf64>) -> tensor<*xf64> {
  // CHECK: %[[ADD1:.*]] = "toy.add"(%arg0, %arg0) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
  %0 = "toy.add"(%arg0, %arg0) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<*xf64>
  // CHECK: %[[MUL1:.*]] = "toy.mul"(%[[ADD1]], %arg0) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
  %1 = "toy.mul"(%0, %arg0) : (tensor<*xf64>, tensor<5x5xf64>) -> tensor<*xf64>
  // CHECK: %[[ADD2:.*]] = "toy.add"(%[[MUL1]], %[[ADD1]]) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
  %2 = "toy.add"(%1, %0) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  // CHECK: return %[[ADD2]] : tensor<5x5xf64>
  return %2 : tensor<*xf64>
}
