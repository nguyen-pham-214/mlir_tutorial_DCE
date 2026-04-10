This project extends the original mlir_tutorial repository (https://github.com/huydang-bossemi/mlir_tutorial) with a simple Dead Code Elimination (DCE) pass.

## Goal
The pass removes operations that are considered dead code by repeatedly checking:

1. Is the operation a terminator?
  If yes, do not erase it.
2. Does removing it preserve SSA correctness?
  An operation can only be removed when all of its results have no uses.
3. Repeat until fixed point
  After one dead operation is removed, other operations may become dead too, so the pass loops again and again until no more dead operations remain.

## Example
Before
```
%0 = toy.add %a, %b
%1 = toy.mul %0, %c
return %a
```

After first iteration
```
%0 = toy.add %a, %b
return %a
```

After second iteration
```
return %a
```

**Note**: you can check the code in 2 files here: 
- passes/ToyDCEPass.cpp
- test/toy-dce.mlir
