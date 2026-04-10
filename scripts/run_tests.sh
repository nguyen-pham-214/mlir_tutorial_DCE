#!/bin/bash

# Script to run all MLIR pattern tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_DIR="$PROJECT_ROOT/test"

# Check if toy-opt exists
if [ ! -f "$BUILD_DIR/tools/toy-opt" ]; then
    echo "Error: toy-opt not found. Please build the project first."
    exit 1
fi

echo "Running MLIR pattern tests..."
echo "=============================="

# Run Shape Inference tests
echo ""
echo "Testing Shape Inference Pass..."
echo "-------------------------------"
$BUILD_DIR/tools/toy-opt $TEST_DIR/toy-shape-inference.mlir -toy-shape-inference
if [ $? -eq 0 ]; then
    echo "✓ Shape Inference tests passed"
else
    echo "✗ Shape Inference tests failed"
    exit 1
fi

# Run Canonicalize tests
echo ""
echo "Testing Canonicalize Pass..."
echo "----------------------------"
$BUILD_DIR/tools/toy-opt $TEST_DIR/toy-canonicalize.mlir -toy-canonicalize
if [ $? -eq 0 ]; then
    echo "✓ Canonicalize tests passed"
else
    echo "✗ Canonicalize tests failed"
    exit 1
fi

# Run CSE tests
echo ""
echo "Testing CSE Pass..."
echo "-------------------"
if [ -f "$TEST_DIR/toy-cse.mlir" ]; then
    $BUILD_DIR/tools/toy-opt $TEST_DIR/toy-cse.mlir -toy-cse
    if [ $? -eq 0 ]; then
        echo "✓ CSE tests passed"
    else
        echo "✗ CSE tests failed"
        exit 1
    fi
fi

# Run DCE tests
echo ""
echo "Testing DCE Pass..."
echo "-------------------"
if [ -f "$TEST_DIR/toy-dce.mlir" ]; then
    $BUILD_DIR/tools/toy-opt $TEST_DIR/toy-dce.mlir -toy-dce
    if [ $? -eq 0 ]; then
        echo "✓ DCE tests passed"
    else
        echo "✗ DCE tests failed"
        exit 1
    fi
fi

# Run Constant Fold tests
echo ""
echo "Testing Constant Fold Pass..."
echo "-----------------------------"
if [ -f "$TEST_DIR/toy-constant-fold.mlir" ]; then
    $BUILD_DIR/tools/toy-opt $TEST_DIR/toy-constant-fold.mlir -toy-constant-fold
    if [ $? -eq 0 ]; then
        echo "✓ Constant Fold tests passed"
    else
        echo "✗ Constant Fold tests failed"
        exit 1
    fi
fi

echo ""
echo "=============================="
echo "All tests completed successfully!"
