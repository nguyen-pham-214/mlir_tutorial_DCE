include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy dialect for tutorial purposes";
  let description = [{
    The Toy dialect is a simple dialect for demonstrating MLIR concepts.
  }];
  let cppNamespace = "::toy";
}
