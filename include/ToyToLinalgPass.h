#ifndef TOY_TO_LINALG_PASS_H
#define TOY_TO_LINALG_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyToLinalgPass();

namespace detail {
void registerToyToLinalgPass();
}

} // namespace toy

#endif // TOY_TO_LINALG_PASS_H
