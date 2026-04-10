#include "ToyDCEPass.h"

#include "toy/ToyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace toy;

namespace {

struct ToyDCEPass
    : public PassWrapper<ToyDCEPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyDCEPass)

  StringRef getArgument() const final { return "toy-dce"; }

  StringRef getDescription() const final {
    return "Iterative dead code elimination for Toy";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    llvm::outs() << "\n=== Processing function: "
                 << funcOp.getSymName() << " ===\n";

    bool changed = true;

    while (changed) {
      changed = false;
      SmallVector<Operation *, 8> deadOps;

      for (Block &block : funcOp) {
        for (Operation &opRef : llvm::reverse(block)) {
          Operation *op = &opRef;

          if (op == funcOp.getOperation())
            continue;

          if (op->hasTrait<OpTrait::IsTerminator>())
            continue;

          if (op->use_empty()) {
            llvm::outs() << "Dead op: ";
            op->print(llvm::outs());
            llvm::outs() << "\n";
            deadOps.push_back(op);
          }
        }
      }

      if (!deadOps.empty()) {
        changed = true;
        for (Operation *op : deadOps)
          op->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> toy::createToyDCEPass() {
  return std::make_unique<ToyDCEPass>();
}

void toy::detail::registerToyDCEPass() {
  static PassRegistration<ToyDCEPass> pass;
  (void)pass;
}