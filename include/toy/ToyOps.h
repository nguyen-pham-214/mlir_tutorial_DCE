#ifndef TOY_TOYOPS_H
#define TOY_TOYOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace toy {
class ToyDialect;
}

#define GET_OP_CLASSES
#include "ToyOps.h.inc"

#endif // TOY_TOYOPS_H
