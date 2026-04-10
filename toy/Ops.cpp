#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace toy;

namespace toy {
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
      >();
}
} // namespace toy

#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"

#include "ToyDialect.cpp.inc"
