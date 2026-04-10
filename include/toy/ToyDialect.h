#ifndef TOY_TOYDIALECT_H
#define TOY_TOYDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
class DialectRegistry;
}

namespace toy {
    void registerToyDialect(mlir::DialectRegistry &registry);
} // namespace toy

#include "ToyDialect.h.inc"

#endif // TOY_TOYDIALECT_H

