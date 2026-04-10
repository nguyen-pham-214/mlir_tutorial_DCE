#ifndef TOY_CANONICALIZE_PASS_H
#define TOY_CANONICALIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyCanonicalizePass();

namespace detail {
void registerToyCanonicalizePass();
}

} // namespace toy

#endif // TOY_CANONICALIZE_PASS_H
