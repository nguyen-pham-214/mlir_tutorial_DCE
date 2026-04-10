#ifndef TOY_TO_ARITH_PASS_H
#define TOY_TO_ARITH_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyToArithPass();

namespace detail {
void registerToyToArithPass();
}

} // namespace toy

#endif // TOY_TO_ARITH_PASS_H
