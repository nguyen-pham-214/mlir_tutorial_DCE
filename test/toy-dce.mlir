// Test 1: dead chain
// Expected: remove toy.mul and then toy.add after iterative DCE,
// because toy.add becomes dead once toy.mul is erased.
func.func @dead_chain(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %a = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %b = "toy.mul"(%a, %a) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %arg0 : tensor<?xf64>
}

// Test 2: live producer
// Expected: keep both toy.add and toy.mul because the final result is returned.
func.func @live_chain(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %a = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %b = "toy.mul"(%a, %a) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %b : tensor<?xf64>
}

// Test 3: one dead op, one live op
// Expected: remove the second toy.add only; keep the first because it is returned.
func.func @mixed_live_dead(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %live = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %dead = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %live : tensor<?xf64>
}

// Test 4: multiple dead ops in one pass
// Expected: remove both toy.mul and toy.add because neither has users.
func.func @multiple_dead(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %keep = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %dead0 = "toy.mul"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %dead1 = "toy.add"(%dead0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %keep : tensor<?xf64>
}

// Test 5: deeper iterative case
// Expected: first remove %c, then %b becomes dead, then %a becomes dead.
// Final function should just return %arg0.
func.func @iterative_three_step(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %a = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %b = "toy.mul"(%a, %a) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %c = "toy.add"(%b, %b) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %arg0 : tensor<?xf64>
}