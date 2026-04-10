// Toy CSE pass placeholder test.
// Expected behavior after implementation: identical toy.add ops should share the first result.

func.func @cse_example(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
  %a0 = "toy.add"(%arg0, %arg1) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %a1 = "toy.add"(%arg0, %arg1) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %r = "toy.mul"(%a0, %a1) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %r : tensor<?xf64>
}
