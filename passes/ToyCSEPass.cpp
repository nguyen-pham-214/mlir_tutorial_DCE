#include "ToyCSEPass.h"

#include "toy/ToyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace toy;

namespace {

struct ToyCSEPass
    : public mlir::PassWrapper<ToyCSEPass,
                              mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyCSEPass)

  mlir::StringRef getArgument() const final { return "toy-cse"; }
  mlir::StringRef getDescription() const final {
    return "Skeleton pass for Toy common subexpression elimination";
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    (void)funcOp;

    // TODO: Identify pure operations (op->hasTrait<mlir::OpTrait::Pure>()).
    // TODO: Compare operations using `op->isIdenticalTo(other)` to detect
    // structurally equivalent Toy ops beyond SSA value equality.
    // TODO: Replace redundant operations with the previously seen instance via
    // `replaceAllUsesWith`.
    // TODO: Erase redundant operations in a safe order (e.g., reverse order) to
    // avoid mutating the list while iterating.
  }
};

} // namespace

std::unique_ptr<mlir::Pass> toy::createToyCSEPass() {
  return std::make_unique<ToyCSEPass>();
}


void toy::detail::registerToyCSEPass() {
  static mlir::PassRegistration<ToyCSEPass> pass;
  (void)pass;
}
