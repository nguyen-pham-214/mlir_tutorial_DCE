module {
  func.func @main(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<4xf64> {
    %0 = "toy.add"(%lhs, %rhs)
         : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %1 = "toy.mul"(%0, %rhs)
         : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %1 : tensor<4xf64>
  }
}
