// Toy constant folding pass placeholder test.
// Expected behavior after implementation: elementwise ops fed by constants should fold into a single constant tensor.
// module {
//   func.func @constant_fold_example() -> tensor<2x2xf64> {
//     %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
//     %cst_0 = arith.constant dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
//     %cst_1 = arith.constant dense<[[6.000000e+00, 8.000000e+00], [1.000000e+01, 1.200000e+01]]> : tensor<2x2xf64>
//     return %cst_1 : tensor<2x2xf64>
//   }
// }

func.func @constant_fold_example() -> tensor<2x2xf64> {
  %c0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %c1 = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
  %sum = "toy.add"(%c0, %c1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  return %sum : tensor<2x2xf64>
}
