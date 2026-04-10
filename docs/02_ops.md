# 02 – Operations

Operations carry the semantics of your dialect. Each operation in Toy has:

- **Operands / Results** described in TableGen.
- **Builders** (C++) that construct operations with inferred types.
- **Printers/Parsers** controlling textual syntax.

## Current State

Builders in `toy/Ops.cpp` simply forward the left-hand-side type. This keeps the code compiling but hides real inference logic.

## TODOs

1. Implement shape inference in the builders or via dedicated interfaces.
2. Add custom parsers/printers that expose literals (e.g., `toy.add %a, %b : tensor<*xf64>`).
3. Introduce attributes to carry metadata such as constant folding hints.

Remember to regenerate TableGen output after editing `.td` files by rerunning CMake / Ninja.
