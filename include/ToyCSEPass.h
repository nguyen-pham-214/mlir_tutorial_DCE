#ifndef TOY_CSE_PASS_H
#define TOY_CSE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyCSEPass();

namespace detail {
void registerToyCSEPass();
}

} // namespace toy

#endif // TOY_CSE_PASS_H
