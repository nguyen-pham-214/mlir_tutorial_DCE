#ifndef TOY_DCE_PASS_H
#define TOY_DCE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyDCEPass();

namespace detail {
void registerToyDCEPass();
}

} // namespace toy

#endif // TOY_DCE_PASS_H
