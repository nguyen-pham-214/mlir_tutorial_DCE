#ifndef TOY_CONSTANT_FOLD_PASS_H
#define TOY_CONSTANT_FOLD_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyConstantFoldPass();

namespace detail {
void registerToyConstantFoldPass();
}

} // namespace toy

#endif // TOY_CONSTANT_FOLD_PASS_H
