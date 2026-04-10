#ifndef TOY_SHAPE_INFERENCE_PASS_H
#define TOY_SHAPE_INFERENCE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

std::unique_ptr<mlir::Pass> createToyShapeInferencePass();

namespace detail {
void registerToyShapeInferencePass();
}

} // namespace toy

#endif // TOY_SHAPE_INFERENCE_PASS_H
